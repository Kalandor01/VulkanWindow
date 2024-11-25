using Silk.NET.Assimp;
using Silk.NET.Core;
using Silk.NET.Core.Native;
using Silk.NET.Input;
using Silk.NET.Maths;
using Silk.NET.Vulkan;
using Silk.NET.Vulkan.Extensions.EXT;
using Silk.NET.Vulkan.Extensions.KHR;
using Silk.NET.Windowing;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Buffer = Silk.NET.Vulkan.Buffer;
using File = System.IO.File;
using Image = Silk.NET.Vulkan.Image;
using Semaphore = Silk.NET.Vulkan.Semaphore;

// From the tutorial at "https://github.com/dfkeenan/SilkVulkanTutorial".
namespace VulkanWindow
{
    struct QueueFamilyIndices
    {
        public uint? GraphicsFamily { get; set; }
        public uint? PresentFamily { get; set; }

        public bool IsComplete()
        {
            return GraphicsFamily.HasValue && PresentFamily.HasValue;
        }
    }

    struct SwapChainSupportDetails
    {
        public SurfaceCapabilitiesKHR Capabilities;
        public SurfaceFormatKHR[] Formats;
        public PresentModeKHR[] PresentModes;
    }

    struct Vertex
    {
        public Vector3D<float> pos;
        public Vector3D<float> color;
        public Vector2D<float> textCoord;

        public static VertexInputBindingDescription GetBindingDescription()
        {
            VertexInputBindingDescription bindingDescription = new()
            {
                Binding = 0,
                Stride = (uint)Unsafe.SizeOf<Vertex>(),
                InputRate = VertexInputRate.Vertex,
            };

            return bindingDescription;
        }

        public static VertexInputAttributeDescription[] GetAttributeDescriptions()
        {
            var attributeDescriptions = new[]
            {
                new VertexInputAttributeDescription()
                {
                    Binding = 0,
                    Location = 0,
                    Format = Format.R32G32B32Sfloat,
                    Offset = (uint)Marshal.OffsetOf<Vertex>(nameof(pos)),
                },
                new VertexInputAttributeDescription()
                {
                    Binding = 0,
                    Location = 1,
                    Format = Format.R32G32B32Sfloat,
                    Offset = (uint)Marshal.OffsetOf<Vertex>(nameof(color)),
                },
                new VertexInputAttributeDescription()
                {
                    Binding = 0,
                    Location = 2,
                    Format = Format.R32G32Sfloat,
                    Offset = (uint)Marshal.OffsetOf<Vertex>(nameof(textCoord)),
                }
            };
            return attributeDescriptions;
        }
    }

    struct UniformBufferObject
    {
        public Matrix4X4<float> model;
        public Matrix4X4<float> view;
        public Matrix4X4<float> proj;
    }

    public class VulkanWindow : IDisposable
    {
        #region Test constants
        private const PolygonMode POLYGON_MODE = PolygonMode.Fill;
        private const CullModeFlags CULLING_MODE = CullModeFlags.BackBit;

        const string MODEL_PATH = @"Assets\Objects\viking_room.obj";
        const string TEXTURE_PATH = @"Assets\Textures\viking_room.png";
        #endregion

        #region Fields
        private IWindow? window;
        private Vk? vulkan;
        private Instance vulkanInstance;
        private ExtDebugUtils? debugUtils;
        private DebugUtilsMessengerEXT debugMessenger;
        private PhysicalDevice physicalDevice;
        private SampleCountFlags msaaSamples = SampleCountFlags.Count1Bit;
        private Device device;
        private Queue graphicsQueue;
        private KhrSurface? khrSurface;
        private SurfaceKHR surface;
        private Queue presentQueue;

        private KhrSwapchain? khrSwapChain;
        private SwapchainKHR swapChain;
        private Image[]? swapChainImages;
        private Format swapChainImageFormat;
        private Extent2D swapChainExtent;
        private ImageView[]? swapChainImageViews;
        private Framebuffer[]? swapChainFramebuffers;

        private RenderPass renderPass;
        private DescriptorSetLayout descriptorSetLayout;
        private PipelineLayout pipelineLayout;
        private Pipeline graphicsPipeline;

        private CommandPool commandPool;

        private Image colorImage;
        private DeviceMemory colorImageMemory;
        private ImageView colorImageView;

        private Image depthImage;
        private DeviceMemory depthImageMemory;
        private ImageView depthImageView;

        private uint mipLevels;
        private Image textureImage;
        private DeviceMemory textureImageMemory;
        private ImageView textureImageView;
        private Sampler textureSampler;

        private Buffer vertexBuffer;
        private DeviceMemory vertexBufferMemory;
        private Buffer indexBuffer;
        private DeviceMemory indexBufferMemory;

        private Buffer[]? uniformBuffers;
        private DeviceMemory[]? uniformBuffersMemory;

        private DescriptorPool descriptorPool;
        private DescriptorSet[]? descriptorSets;

        private CommandBuffer[]? commandBuffers;

        private Semaphore[]? imageAvailableSemaphores;
        private Semaphore[]? renderFinishedSemaphores;
        private Fence[]? inFlightFences;
        private Fence[]? imagesInFlight;
        private int currentFrame = 0;

        private bool frameBufferResized = false;

        private Vertex[]? vertices;

        private uint[]? indices;

        bool EnableValidationLayers = true;

        private readonly string[] validationLayers =
        [
            "VK_LAYER_KHRONOS_validation"
        ];

        private readonly string[] deviceExtensions =
        [
            KhrSwapchain.ExtensionName,
        ];

        const int MAX_FRAMES_IN_FLIGHT = 2;
        #endregion

        public VulkanWindow(int width, int height, bool enableValidation)
        {
            EnableValidationLayers = enableValidation;

            //Create a window.
            var options = WindowOptions.DefaultVulkan;
            options.Size = new Vector2D<int>(width, height);
            options.Title = "test";

            window = Window.Create(options);

            //Assign events.
            window.Load += OnLoad;
            window.Update += OnUpdate;
            window.Render += OnRender;
            window.FramebufferResize += OnFramebufferResize;
            window.Resize += FramebufferResizeCallback;

            window.Initialize();

            if (window.VkSurface is null)
            {
                throw new Exception("Windowing platform doesn't support Vulkan.");
            }

            InitVulkan();
        }

        #region Public methods
        public void Display()
        {
            window!.Run();
            vulkan!.DeviceWaitIdle(device);
        }

        public void Dispose()
        {
            DebugLog($"PHASE: {nameof(Dispose)}");

            CleanUpSwapChain();
            unsafe
            {
                vulkan!.DestroySampler(device, textureSampler, null);
                vulkan!.DestroyImageView(device, textureImageView, null);

                vulkan!.DestroyImage(device, textureImage, null);
                vulkan!.FreeMemory(device, textureImageMemory, null);

                vulkan!.DestroyDescriptorSetLayout(device, descriptorSetLayout, null);

                vulkan!.DestroyBuffer(device, indexBuffer, null);
                vulkan!.FreeMemory(device, indexBufferMemory, null);

                vulkan!.DestroyBuffer(device, vertexBuffer, null);
                vulkan!.FreeMemory(device, vertexBufferMemory, null);

                for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
                {
                    vulkan!.DestroySemaphore(device, renderFinishedSemaphores![i], null);
                    vulkan!.DestroySemaphore(device, imageAvailableSemaphores![i], null);
                    vulkan!.DestroyFence(device, inFlightFences![i], null);
                }

                vulkan!.DestroyCommandPool(device, commandPool, null);
                vulkan!.DestroyDevice(device, null);
            }

            if (EnableValidationLayers)
            {
                unsafe
                {
                    debugUtils!.DestroyDebugUtilsMessenger(vulkanInstance, debugMessenger, null);
                }
            }

            unsafe
            {
                khrSurface!.DestroySurface(vulkanInstance, surface, null);
                vulkan!.DestroyInstance(vulkanInstance, null);
            }
            vulkan!.Dispose();

            window?.Dispose();
        }
        #endregion

        #region Private methods
        #region Other
        private unsafe void CleanUpSwapChain()
        {
            vulkan!.DestroyImageView(device, depthImageView, null);
            vulkan!.DestroyImage(device, depthImage, null);
            vulkan!.FreeMemory(device, depthImageMemory, null);

            vulkan!.DestroyImageView(device, colorImageView, null);
            vulkan!.DestroyImage(device, colorImage, null);
            vulkan!.FreeMemory(device, colorImageMemory, null);

            foreach (var framebuffer in swapChainFramebuffers!)
            {
                vulkan!.DestroyFramebuffer(device, framebuffer, null);
            }

            fixed (CommandBuffer* commandBuffersPtr = commandBuffers)
            {
                vulkan!.FreeCommandBuffers(device, commandPool, (uint)commandBuffers!.Length, commandBuffersPtr);
            }

            vulkan!.DestroyPipeline(device, graphicsPipeline, null);
            vulkan!.DestroyPipelineLayout(device, pipelineLayout, null);
            vulkan!.DestroyRenderPass(device, renderPass, null);

            foreach (var imageView in swapChainImageViews!)
            {
                vulkan!.DestroyImageView(device, imageView, null);
            }

            khrSwapChain!.DestroySwapchain(device, swapChain, null);

            for (int i = 0; i < swapChainImages!.Length; i++)
            {
                vulkan!.DestroyBuffer(device, uniformBuffers![i], null);
                vulkan!.FreeMemory(device, uniformBuffersMemory![i], null);
            }

            vulkan!.DestroyDescriptorPool(device, descriptorPool, null);
        }

        private QueueFamilyIndices FindQueueFamilies(PhysicalDevice device)
        {
            var indices = new QueueFamilyIndices();

            uint queueFamilityCount = 0;

            unsafe
            {
                vulkan!.GetPhysicalDeviceQueueFamilyProperties(device, ref queueFamilityCount, null);

                var queueFamilies = new QueueFamilyProperties[queueFamilityCount];
                fixed (QueueFamilyProperties* queueFamiliesPtr = queueFamilies)
                {
                    vulkan!.GetPhysicalDeviceQueueFamilyProperties(device, ref queueFamilityCount, queueFamiliesPtr);
                }

                uint i = 0;
                foreach (var queueFamily in queueFamilies)
                {
                    if (queueFamily.QueueFlags.HasFlag(QueueFlags.GraphicsBit))
                    {
                        indices.GraphicsFamily = i;
                    }

                    khrSurface!.GetPhysicalDeviceSurfaceSupport(device, i, surface, out var presentSupport);

                    if (presentSupport)
                    {
                        indices.PresentFamily = i;
                    }

                    if (indices.IsComplete())
                    {
                        break;
                    }

                    i++;
                }
            }

            return indices;
        }

        private SwapChainSupportDetails QuerySwapChainSupport(PhysicalDevice physicalDevice)
        {
            var details = new SwapChainSupportDetails();

            khrSurface!.GetPhysicalDeviceSurfaceCapabilities(physicalDevice, surface, out details.Capabilities);

            unsafe
            {
                uint formatCount = 0;
                khrSurface.GetPhysicalDeviceSurfaceFormats(physicalDevice, surface, ref formatCount, null);

                if (formatCount != 0)
                {
                    details.Formats = new SurfaceFormatKHR[formatCount];
                    fixed (SurfaceFormatKHR* formatsPtr = details.Formats)
                    {
                        khrSurface.GetPhysicalDeviceSurfaceFormats(physicalDevice, surface, ref formatCount, formatsPtr);
                    }
                }
                else
                {
                    details.Formats = [];
                }

                uint presentModeCount = 0;
                khrSurface.GetPhysicalDeviceSurfacePresentModes(physicalDevice, surface, ref presentModeCount, null);

                if (presentModeCount != 0)
                {
                    details.PresentModes = new PresentModeKHR[presentModeCount];
                    fixed (PresentModeKHR* formatsPtr = details.PresentModes)
                    {
                        khrSurface.GetPhysicalDeviceSurfacePresentModes(physicalDevice, surface, ref presentModeCount, formatsPtr);
                    }
                }
                else
                {
                    details.PresentModes = [];
                }
            }

            return details;
        }
        #endregion

        #region Create instance
        private unsafe bool CheckValidationLayerSupport()
        {
            uint layerCount = 0;
            vulkan!.EnumerateInstanceLayerProperties(ref layerCount, null);
            var availableLayers = new LayerProperties[layerCount];
            fixed (LayerProperties* availableLayersPtr = availableLayers)
            {
                vulkan!.EnumerateInstanceLayerProperties(ref layerCount, availableLayersPtr);
            }

            var availableLayerNames = availableLayers.Select(layer => Marshal.PtrToStringAnsi((IntPtr)layer.LayerName)).ToHashSet();

            return validationLayers.All(availableLayerNames.Contains);
        }

        private unsafe string[] GetRequiredExtensions()
        {
            var glfwExtensions = window!.VkSurface!.GetRequiredExtensions(out var glfwExtensionCount);
            var extensions = SilkMarshal.PtrToStringArray((nint)glfwExtensions, (int)glfwExtensionCount);

            if (EnableValidationLayers)
            {
                return [.. extensions, ExtDebugUtils.ExtensionName];
            }

            return extensions;
        }

        private void CreateInstance()
        {
            vulkan = Vk.GetApi();

            if (EnableValidationLayers && !CheckValidationLayerSupport())
            {
                throw new Exception("validation layers requested, but not available!");
            }

            unsafe
            {
                var appInfo = new ApplicationInfo()
                {
                    SType = StructureType.ApplicationInfo,
                    PApplicationName = (byte*)Marshal.StringToHGlobalAnsi("Hello Triangle"),
                    ApplicationVersion = new Version32(1, 0, 0),
                    PEngineName = (byte*)Marshal.StringToHGlobalAnsi("No Engine"),
                    EngineVersion = new Version32(1, 0, 0),
                    ApiVersion = Vk.Version13,
                };

                var createInfo = new InstanceCreateInfo()
                {
                    SType = StructureType.InstanceCreateInfo,
                    PApplicationInfo = &appInfo
                };


                var extensions = GetRequiredExtensions();
                createInfo.EnabledExtensionCount = (uint)extensions.Length;
                createInfo.PpEnabledExtensionNames = (byte**)SilkMarshal.StringArrayToPtr(extensions);

                if (EnableValidationLayers)
                {
                    createInfo.EnabledLayerCount = (uint)validationLayers.Length;
                    createInfo.PpEnabledLayerNames = (byte**)SilkMarshal.StringArrayToPtr(validationLayers);

                    var debugCreateInfo = new DebugUtilsMessengerCreateInfoEXT();
                    PopulateDebugMessengerCreateInfo(ref debugCreateInfo);
                    createInfo.PNext = &debugCreateInfo;
                }
                else
                {
                    createInfo.EnabledLayerCount = 0;
                    createInfo.PNext = null;
                }

                if (vulkan.CreateInstance(in createInfo, null, out vulkanInstance) != Result.Success)
                {
                    throw new Exception("failed to create instance!");
                }


                Marshal.FreeHGlobal((IntPtr)appInfo.PApplicationName);
                Marshal.FreeHGlobal((IntPtr)appInfo.PEngineName);
                SilkMarshal.Free((nint)createInfo.PpEnabledExtensionNames);

                if (EnableValidationLayers)
                {
                    SilkMarshal.Free((nint)createInfo.PpEnabledLayerNames);
                }
            }
        }
        #endregion

        #region Debugger
        private void DebugLog(string message)
        {
            if (EnableValidationLayers)
            {
                Console.WriteLine($"Debug: " + message);
                //System.Diagnostics.Debug.WriteLine($"Debug: " + message);
            }
        }

        private unsafe uint DebugCallback(
            DebugUtilsMessageSeverityFlagsEXT messageSeverity,
            DebugUtilsMessageTypeFlagsEXT messageTypes,
            DebugUtilsMessengerCallbackDataEXT* pCallbackData,
            void* pUserData
        )
        {
            Console.WriteLine($"Debug: " + Marshal.PtrToStringAnsi((nint)pCallbackData->PMessage));
            //System.Diagnostics.Debug.WriteLine($"Debug: " + Marshal.PtrToStringAnsi((nint)pCallbackData->PMessage));
            return Vk.False;
        }

        private void PopulateDebugMessengerCreateInfo(ref DebugUtilsMessengerCreateInfoEXT createInfo)
        {
            createInfo.SType = StructureType.DebugUtilsMessengerCreateInfoExt;
            createInfo.MessageSeverity = DebugUtilsMessageSeverityFlagsEXT.VerboseBitExt |
                                         DebugUtilsMessageSeverityFlagsEXT.WarningBitExt |
                                         DebugUtilsMessageSeverityFlagsEXT.ErrorBitExt;
            createInfo.MessageType = DebugUtilsMessageTypeFlagsEXT.GeneralBitExt |
                                     DebugUtilsMessageTypeFlagsEXT.PerformanceBitExt |
                                     DebugUtilsMessageTypeFlagsEXT.ValidationBitExt;
            unsafe
            {
                createInfo.PfnUserCallback = (DebugUtilsMessengerCallbackFunctionEXT)DebugCallback;
            }
        }

        private void SetupDebugMessenger()
        {
            if (!EnableValidationLayers) return;

            //TryGetInstanceExtension equivilant to method CreateDebugUtilsMessengerEXT from original tutorial.
            if (!vulkan!.TryGetInstanceExtension(vulkanInstance, out debugUtils)) return;

            DebugUtilsMessengerCreateInfoEXT createInfo = new();
            PopulateDebugMessengerCreateInfo(ref createInfo);

            unsafe
            {
                if (debugUtils!.CreateDebugUtilsMessenger(vulkanInstance, in createInfo, null, out debugMessenger) != Result.Success)
                {
                    throw new Exception("failed to set up debug messenger!");
                }
            }
        }
        #endregion

        #region Pick phisical device
        private bool CheckDeviceExtensionsSupport(PhysicalDevice device)
        {
            uint extentionsCount = 0;
            HashSet<string?>? availableExtensionNames = null;
            unsafe
            {
                vulkan!.EnumerateDeviceExtensionProperties(device, (byte*)null, ref extentionsCount, null);

                var availableExtensions = new ExtensionProperties[extentionsCount];
                fixed (ExtensionProperties* availableExtensionsPtr = availableExtensions)
                {
                    vulkan!.EnumerateDeviceExtensionProperties(device, (byte*)null, ref extentionsCount, availableExtensionsPtr);
                }

                availableExtensionNames = availableExtensions.Select(extension => Marshal.PtrToStringAnsi((IntPtr)extension.ExtensionName)).ToHashSet();
            }

            if (availableExtensionNames is null)
            {
                return false;
            }

            return deviceExtensions.All(availableExtensionNames.Contains);
        }

        private bool IsDeviceSuitable(PhysicalDevice device)
        {
            var indices = FindQueueFamilies(device);

            var extensionsSupported = CheckDeviceExtensionsSupport(device);

            var swapChainAdequate = false;
            if (extensionsSupported)
            {
                var swapChainSupport = QuerySwapChainSupport(device);
                swapChainAdequate = swapChainSupport.Formats.Length != 0 && swapChainSupport.PresentModes.Length != 0;
            }

            vulkan!.GetPhysicalDeviceFeatures(device, out PhysicalDeviceFeatures supportedFeatures);
            return indices.IsComplete() && extensionsSupported && swapChainAdequate && supportedFeatures.SamplerAnisotropy;
        }

        private SampleCountFlags GetMaxUsableSampleCount()
        {
            vulkan!.GetPhysicalDeviceProperties(physicalDevice, out var physicalDeviceProperties);

            var counts = physicalDeviceProperties.Limits.FramebufferColorSampleCounts & physicalDeviceProperties.Limits.FramebufferDepthSampleCounts;

            return counts switch
            {
                var c when (c & SampleCountFlags.Count64Bit) != 0 => SampleCountFlags.Count64Bit,
                var c when (c & SampleCountFlags.Count32Bit) != 0 => SampleCountFlags.Count32Bit,
                var c when (c & SampleCountFlags.Count16Bit) != 0 => SampleCountFlags.Count16Bit,
                var c when (c & SampleCountFlags.Count8Bit) != 0 => SampleCountFlags.Count8Bit,
                var c when (c & SampleCountFlags.Count4Bit) != 0 => SampleCountFlags.Count4Bit,
                var c when (c & SampleCountFlags.Count2Bit) != 0 => SampleCountFlags.Count2Bit,
                _ => SampleCountFlags.Count1Bit
            };
        }

        private void PickPhysicalDevice()
        {
            var devices = vulkan!.GetPhysicalDevices(vulkanInstance);

            foreach (var device in devices)
            {
                if (IsDeviceSuitable(device))
                {
                    physicalDevice = device;
                    msaaSamples = GetMaxUsableSampleCount();
                    break;
                }
            }

            if (physicalDevice.Handle == 0)
            {
                throw new Exception("failed to find a suitable GPU!");
            }
        }
        #endregion

        #region Create logical device
        private unsafe void CreateLogicalDevice()
        {
            var indices = FindQueueFamilies(physicalDevice);

            var uniqueQueueFamilies = new[] { indices.GraphicsFamily!.Value, indices.PresentFamily!.Value };
            uniqueQueueFamilies = uniqueQueueFamilies.Distinct().ToArray();

            using var mem = GlobalMemory.Allocate(uniqueQueueFamilies.Length * sizeof(DeviceQueueCreateInfo));
            var queueCreateInfos = (DeviceQueueCreateInfo*)Unsafe.AsPointer(ref mem.GetPinnableReference());


            var queuePriority = 1.0f;
            for (int i = 0; i < uniqueQueueFamilies.Length; i++)
            {
                queueCreateInfos[i] = new()
                {
                    SType = StructureType.DeviceQueueCreateInfo,
                    QueueFamilyIndex = uniqueQueueFamilies[i],
                    QueueCount = 1,
                    PQueuePriorities = &queuePriority
                };
            }

            var deviceFeatures = new PhysicalDeviceFeatures()
            {
                SamplerAnisotropy = true,
            };

            var createInfo = new DeviceCreateInfo()
            {
                SType = StructureType.DeviceCreateInfo,
                QueueCreateInfoCount = (uint)uniqueQueueFamilies.Length,
                PQueueCreateInfos = queueCreateInfos,
                PEnabledFeatures = &deviceFeatures,
                EnabledExtensionCount = (uint)deviceExtensions.Length,
                PpEnabledExtensionNames = (byte**)SilkMarshal.StringArrayToPtr(deviceExtensions),
            };

            if (EnableValidationLayers)
            {
                createInfo.EnabledLayerCount = (uint)validationLayers.Length;
                createInfo.PpEnabledLayerNames = (byte**)SilkMarshal.StringArrayToPtr(validationLayers);
            }
            else
            {
                createInfo.EnabledLayerCount = 0;
            }

            if (vulkan!.CreateDevice(physicalDevice, in createInfo, null, out device) != Result.Success)
            {
                throw new Exception("failed to create logical device!");
            }

            vulkan!.GetDeviceQueue(device, indices.GraphicsFamily!.Value, 0, out graphicsQueue);
            vulkan!.GetDeviceQueue(device, indices.PresentFamily!.Value, 0, out presentQueue);

            if (EnableValidationLayers)
            {
                SilkMarshal.Free((nint)createInfo.PpEnabledLayerNames);
            }

            SilkMarshal.Free((nint)createInfo.PpEnabledExtensionNames);
        }
        #endregion

        #region Create Surface
        private void CreateSurface()
        {
            if (!vulkan!.TryGetInstanceExtension<KhrSurface>(vulkanInstance, out khrSurface))
            {
                throw new NotSupportedException("KHR_surface extension not found.");
            }

            unsafe
            {
                surface = window!.VkSurface!.Create<AllocationCallbacks>(vulkanInstance.ToHandle(), null).ToSurface();
            }
        }
        #endregion

        #region Create swap chain
        private SurfaceFormatKHR ChooseSwapSurfaceFormat(IReadOnlyList<SurfaceFormatKHR> availableFormats)
        {
            foreach (var availableFormat in availableFormats)
            {
                if (availableFormat.Format == Format.B8G8R8A8Srgb && availableFormat.ColorSpace == ColorSpaceKHR.SpaceSrgbNonlinearKhr)
                {
                    return availableFormat;
                }
            }

            return availableFormats[0];
        }

        private PresentModeKHR ChoosePresentMode(IReadOnlyList<PresentModeKHR> availablePresentModes)
        {
            foreach (var availablePresentMode in availablePresentModes)
            {
                if (availablePresentMode == PresentModeKHR.MailboxKhr)
                {
                    return availablePresentMode;
                }
            }

            return PresentModeKHR.FifoKhr;
        }

        private Extent2D ChooseSwapExtent(SurfaceCapabilitiesKHR capabilities)
        {
            if (capabilities.CurrentExtent.Width != uint.MaxValue)
            {
                return capabilities.CurrentExtent;
            }
            else
            {
                var framebufferSize = window!.FramebufferSize;

                Extent2D actualExtent = new()
                {
                    Width = (uint)framebufferSize.X,
                    Height = (uint)framebufferSize.Y
                };

                actualExtent.Width = Math.Clamp(actualExtent.Width, capabilities.MinImageExtent.Width, capabilities.MaxImageExtent.Width);
                actualExtent.Height = Math.Clamp(actualExtent.Height, capabilities.MinImageExtent.Height, capabilities.MaxImageExtent.Height);

                return actualExtent;
            }
        }

        private void CreateSwapChain()
        {
            var swapChainSupport = QuerySwapChainSupport(physicalDevice);

            var surfaceFormat = ChooseSwapSurfaceFormat(swapChainSupport.Formats);
            var presentMode = ChoosePresentMode(swapChainSupport.PresentModes);
            var extent = ChooseSwapExtent(swapChainSupport.Capabilities);

            var imageCount = swapChainSupport.Capabilities.MinImageCount + 1;
            if (swapChainSupport.Capabilities.MaxImageCount > 0 && imageCount > swapChainSupport.Capabilities.MaxImageCount)
            {
                imageCount = swapChainSupport.Capabilities.MaxImageCount;
            }

            SwapchainCreateInfoKHR creatInfo = new()
            {
                SType = StructureType.SwapchainCreateInfoKhr,
                Surface = surface,

                MinImageCount = imageCount,
                ImageFormat = surfaceFormat.Format,
                ImageColorSpace = surfaceFormat.ColorSpace,
                ImageExtent = extent,
                ImageArrayLayers = 1,
                ImageUsage = ImageUsageFlags.ColorAttachmentBit,
            };

            var indices = FindQueueFamilies(physicalDevice);

            unsafe
            {
                var queueFamilyIndices = stackalloc[] { indices.GraphicsFamily!.Value, indices.PresentFamily!.Value };

                if (indices.GraphicsFamily != indices.PresentFamily)
                {
                    creatInfo = creatInfo with
                    {
                        ImageSharingMode = SharingMode.Concurrent,
                        QueueFamilyIndexCount = 2,
                        PQueueFamilyIndices = queueFamilyIndices,
                    };
                }
                else
                {
                    creatInfo.ImageSharingMode = SharingMode.Exclusive;
                }
            }

            creatInfo = creatInfo with
            {
                PreTransform = swapChainSupport.Capabilities.CurrentTransform,
                CompositeAlpha = CompositeAlphaFlagsKHR.OpaqueBitKhr,
                PresentMode = presentMode,
                Clipped = true,
            };

            if (khrSwapChain is null)
            {
                if (!vulkan!.TryGetDeviceExtension(vulkanInstance, device, out khrSwapChain))
                {
                    throw new NotSupportedException("VK_KHR_swapchain extension not found.");
                }
            }

            unsafe
            {
                if (khrSwapChain!.CreateSwapchain(device, in creatInfo, null, out swapChain) != Result.Success)
                {
                    throw new Exception("failed to create swap chain!");
                }

                khrSwapChain.GetSwapchainImages(device, swapChain, ref imageCount, null);
                swapChainImages = new Image[imageCount];
                fixed (Image* swapChainImagesPtr = swapChainImages)
                {
                    khrSwapChain.GetSwapchainImages(device, swapChain, ref imageCount, swapChainImagesPtr);
                }
            }

            swapChainImageFormat = surfaceFormat.Format;
            swapChainExtent = extent;
        }

        private void CreateUniformBuffers()
        {
            var bufferSize = (ulong)Unsafe.SizeOf<UniformBufferObject>();

            uniformBuffers = new Buffer[swapChainImages!.Length];
            uniformBuffersMemory = new DeviceMemory[swapChainImages!.Length];

            for (int i = 0; i < swapChainImages.Length; i++)
            {
                CreateBuffer(bufferSize, BufferUsageFlags.UniformBufferBit, MemoryPropertyFlags.HostVisibleBit | MemoryPropertyFlags.HostCoherentBit, ref uniformBuffers[i], ref uniformBuffersMemory[i]);
            }
        }

        private void RecreateSwapChain()
        {
            Vector2D<int> framebufferSize = window!.FramebufferSize;

            while (framebufferSize.X == 0 || framebufferSize.Y == 0)
            {
                framebufferSize = window.FramebufferSize;
                window.DoEvents();
            }

            vulkan!.DeviceWaitIdle(device);

            CleanUpSwapChain();

            CreateSwapChain();
            CreateImageViews();
            CreateRenderPass();
            CreateGraphicsPipeline();
            CreateColorResources();
            CreateDepthResources();
            CreateFramebuffers();
            CreateUniformBuffers();
            CreateDescriptorPool();
            CreateDescriptorSets();
            CreateCommandBuffers();

            imagesInFlight = new Fence[swapChainImages!.Length];
        }
        #endregion

        #region Create image views
        private ImageView CreateImageView(Image image, Format format, ImageAspectFlags aspectFlags, uint mipLevels)
        {
            var createInfo = new ImageViewCreateInfo()
            {
                SType = StructureType.ImageViewCreateInfo,
                Image = image,
                ViewType = ImageViewType.Type2D,
                Format = format,
                //Components =
                //{
                //    R = ComponentSwizzle.Identity,
                //    G = ComponentSwizzle.Identity,
                //    B = ComponentSwizzle.Identity,
                //    A = ComponentSwizzle.Identity,
                //},
                SubresourceRange =
                {
                    AspectMask = aspectFlags,
                    BaseMipLevel = 0,
                    LevelCount = mipLevels,
                    BaseArrayLayer = 0,
                    LayerCount = 1,
                }
            };

            unsafe
            {
                if (vulkan!.CreateImageView(device, in createInfo, null, out var imageView) != Result.Success)
                {
                    throw new Exception("failed to create image views!");
                }

                return imageView;
            }
        }

        private void CreateImageViews()
        {
            swapChainImageViews = new ImageView[swapChainImages!.Length];

            for (int x = 0; x < swapChainImages.Length; x++)
            {
                swapChainImageViews[x] = CreateImageView(swapChainImages[x], swapChainImageFormat, ImageAspectFlags.ColorBit, 1);
            }
        }
        #endregion

        #region Create render pass
        private void CreateRenderPass()
        {
            var colorAttachment = new AttachmentDescription()
            {
                Format = swapChainImageFormat,
                Samples = msaaSamples,
                LoadOp = AttachmentLoadOp.Clear,
                StoreOp = AttachmentStoreOp.Store,
                StencilLoadOp = AttachmentLoadOp.DontCare,
                InitialLayout = ImageLayout.Undefined,
                FinalLayout = ImageLayout.ColorAttachmentOptimal,
            };

            var depthAttachment = new AttachmentDescription()
            {
                Format = FindDepthFormat(),
                Samples = msaaSamples,
                LoadOp = AttachmentLoadOp.Clear,
                StoreOp = AttachmentStoreOp.DontCare,
                StencilLoadOp = AttachmentLoadOp.DontCare,
                StencilStoreOp = AttachmentStoreOp.DontCare,
                InitialLayout = ImageLayout.Undefined,
                FinalLayout = ImageLayout.DepthStencilAttachmentOptimal,
            };

            var colorAttachmentResolve = new AttachmentDescription()
            {
                Format = swapChainImageFormat,
                Samples = SampleCountFlags.Count1Bit,
                LoadOp = AttachmentLoadOp.DontCare,
                StoreOp = AttachmentStoreOp.Store,
                StencilLoadOp = AttachmentLoadOp.DontCare,
                StencilStoreOp = AttachmentStoreOp.DontCare,
                InitialLayout = ImageLayout.Undefined,
                FinalLayout = ImageLayout.PresentSrcKhr,
            };

            var colorAttachmentRef = new AttachmentReference()
            {
                Attachment = 0,
                Layout = ImageLayout.ColorAttachmentOptimal,
            };

            var depthAttachmentRef = new AttachmentReference()
            {
                Attachment = 1,
                Layout = ImageLayout.DepthStencilAttachmentOptimal,
            };

            var colorAttachmentResolveRef = new AttachmentReference()
            {
                Attachment = 2,
                Layout = ImageLayout.ColorAttachmentOptimal,
            };

            SubpassDescription subpass;
            unsafe
            {
                subpass = new SubpassDescription()
                {
                    PipelineBindPoint = PipelineBindPoint.Graphics,
                    ColorAttachmentCount = 1,
                    PColorAttachments = &colorAttachmentRef,
                    PDepthStencilAttachment = &depthAttachmentRef,
                    PResolveAttachments = &colorAttachmentResolveRef,
                };
            }

            var dependency = new SubpassDependency()
            {
                SrcSubpass = Vk.SubpassExternal,
                DstSubpass = 0,
                SrcStageMask = PipelineStageFlags.ColorAttachmentOutputBit | PipelineStageFlags.EarlyFragmentTestsBit,
                SrcAccessMask = 0,
                DstStageMask = PipelineStageFlags.ColorAttachmentOutputBit | PipelineStageFlags.EarlyFragmentTestsBit,
                DstAccessMask = AccessFlags.ColorAttachmentWriteBit | AccessFlags.DepthStencilAttachmentWriteBit
            };

            var attachments = new[] { colorAttachment, depthAttachment, colorAttachmentResolve };

            unsafe
            {
                fixed (AttachmentDescription* attachmentsPtr = attachments)
                {
                    var renderPassInfo = new RenderPassCreateInfo()
                    {
                        SType = StructureType.RenderPassCreateInfo,
                        AttachmentCount = (uint)attachments.Length,
                        PAttachments = attachmentsPtr,
                        SubpassCount = 1,
                        PSubpasses = &subpass,
                        DependencyCount = 1,
                        PDependencies = &dependency,
                    };

                    if (vulkan!.CreateRenderPass(device, in renderPassInfo, null, out renderPass) != Result.Success)
                    {
                        throw new Exception("failed to create render pass!");
                    }
                }
            }
        }
        #endregion

        #region Create descriptor set layout
        private void CreateDescriptorSetLayout()
        {
            var uboLayoutBinding = new DescriptorSetLayoutBinding()
            {
                Binding = 0,
                DescriptorCount = 1,
                DescriptorType = DescriptorType.UniformBuffer,
                PImmutableSamplers = null,
                StageFlags = ShaderStageFlags.VertexBit,
            };

            var samplerLayoutBinding = new DescriptorSetLayoutBinding()
            {
                Binding = 1,
                DescriptorCount = 1,
                DescriptorType = DescriptorType.CombinedImageSampler,
                PImmutableSamplers = null,
                StageFlags = ShaderStageFlags.FragmentBit,
            };

            var bindings = new DescriptorSetLayoutBinding[] { uboLayoutBinding, samplerLayoutBinding };

            unsafe
            {
                fixed (DescriptorSetLayoutBinding* bindingsPtr = bindings)
                fixed (DescriptorSetLayout* descriptorSetLayoutPtr = &descriptorSetLayout)
                {
                    var layoutInfo = new DescriptorSetLayoutCreateInfo()
                    {
                        SType = StructureType.DescriptorSetLayoutCreateInfo,
                        BindingCount = (uint)bindings.Length,
                        PBindings = bindingsPtr,
                    };

                    if (vulkan!.CreateDescriptorSetLayout(device, in layoutInfo, null, descriptorSetLayoutPtr) != Result.Success)
                    {
                        throw new Exception("failed to create descriptor set layout!");
                    }
                }
            }
        }
        #endregion

        #region Create graphics pipeline
        private ShaderModule CreateShaderModule(byte[] code)
        {
            var createInfo = new ShaderModuleCreateInfo()
            {
                SType = StructureType.ShaderModuleCreateInfo,
                CodeSize = (nuint)code.Length,
            };

            ShaderModule shaderModule;
            unsafe
            {
                fixed (byte* codePtr = code)
                {
                    createInfo.PCode = (uint*)codePtr;

                    if (vulkan!.CreateShaderModule(device, in createInfo, null, out shaderModule) != Result.Success)
                    {
                        throw new Exception();
                    }
                }
            }
            return shaderModule;
        }

        private unsafe void CreateGraphicsPipeline()
        {
            var vertShaderCode = File.ReadAllBytes("Shaders/vert.spv");
            var fragShaderCode = File.ReadAllBytes("Shaders/frag.spv");

            var vertShaderModule = CreateShaderModule(vertShaderCode);
            var fragShaderModule = CreateShaderModule(fragShaderCode);

            var vertShaderStageInfo = new PipelineShaderStageCreateInfo()
            {
                SType = StructureType.PipelineShaderStageCreateInfo,
                Stage = ShaderStageFlags.VertexBit,
                Module = vertShaderModule,
                PName = (byte*)SilkMarshal.StringToPtr("main")
            };

            var fragShaderStageInfo = new PipelineShaderStageCreateInfo()
            {
                SType = StructureType.PipelineShaderStageCreateInfo,
                Stage = ShaderStageFlags.FragmentBit,
                Module = fragShaderModule,
                PName = (byte*)SilkMarshal.StringToPtr("main")
            };

            var shaderStages = stackalloc[]
            {
                vertShaderStageInfo,
                fragShaderStageInfo
            };

            var bindingDescription = Vertex.GetBindingDescription();
            var attributeDescriptions = Vertex.GetAttributeDescriptions();

            fixed (VertexInputAttributeDescription* attributeDescriptionsPtr = attributeDescriptions)
            fixed (DescriptorSetLayout* descriptorSetLayoutPtr = &descriptorSetLayout)
            {
                var vertexInputInfo = new PipelineVertexInputStateCreateInfo()
                {
                    SType = StructureType.PipelineVertexInputStateCreateInfo,
                    VertexBindingDescriptionCount = 1,
                    VertexAttributeDescriptionCount = (uint)attributeDescriptions.Length,
                    PVertexBindingDescriptions = &bindingDescription,
                    PVertexAttributeDescriptions = attributeDescriptionsPtr,
                };

                var inputAssembly = new PipelineInputAssemblyStateCreateInfo()
                {
                    SType = StructureType.PipelineInputAssemblyStateCreateInfo,
                    Topology = PrimitiveTopology.TriangleList,
                    PrimitiveRestartEnable = false,
                };

                var viewport = new Viewport()
                {
                    X = 0,
                    Y = 0,
                    Width = swapChainExtent.Width,
                    Height = swapChainExtent.Height,
                    MinDepth = 0,
                    MaxDepth = 1,
                };

                var scissor = new Rect2D()
                {
                    Offset = { X = 0, Y = 0 },
                    Extent = swapChainExtent,
                };

                var viewportState = new PipelineViewportStateCreateInfo()
                {
                    SType = StructureType.PipelineViewportStateCreateInfo,
                    ViewportCount = 1,
                    PViewports = &viewport,
                    ScissorCount = 1,
                    PScissors = &scissor,
                };

                var rasterizer = new PipelineRasterizationStateCreateInfo()
                {
                    SType = StructureType.PipelineRasterizationStateCreateInfo,
                    DepthClampEnable = false,
                    RasterizerDiscardEnable = false,
                    PolygonMode = POLYGON_MODE,
                    LineWidth = 1,
                    CullMode = CULLING_MODE,
                    FrontFace = FrontFace.CounterClockwise,
                    DepthBiasEnable = false,
                };

                var multisampling = new PipelineMultisampleStateCreateInfo()
                {
                    SType = StructureType.PipelineMultisampleStateCreateInfo,
                    SampleShadingEnable = false,
                    RasterizationSamples = msaaSamples,
                };

                var depthStencil = new PipelineDepthStencilStateCreateInfo()
                {
                    SType = StructureType.PipelineDepthStencilStateCreateInfo,
                    DepthTestEnable = true,
                    DepthWriteEnable = true,
                    DepthCompareOp = CompareOp.Less,
                    DepthBoundsTestEnable = false,
                    StencilTestEnable = false,
                };

                var colorBlendAttachment = new PipelineColorBlendAttachmentState()
                {
                    ColorWriteMask = ColorComponentFlags.RBit | ColorComponentFlags.GBit | ColorComponentFlags.BBit | ColorComponentFlags.ABit,
                    BlendEnable = false,
                };

                var colorBlending = new PipelineColorBlendStateCreateInfo()
                {
                    SType = StructureType.PipelineColorBlendStateCreateInfo,
                    LogicOpEnable = false,
                    LogicOp = LogicOp.Copy,
                    AttachmentCount = 1,
                    PAttachments = &colorBlendAttachment,
                };

                colorBlending.BlendConstants[0] = 0;
                colorBlending.BlendConstants[1] = 0;
                colorBlending.BlendConstants[2] = 0;
                colorBlending.BlendConstants[3] = 0;

                var pipelineLayoutInfo = new PipelineLayoutCreateInfo()
                {
                    SType = StructureType.PipelineLayoutCreateInfo,
                    PushConstantRangeCount = 0,
                    SetLayoutCount = 1,
                    PSetLayouts = descriptorSetLayoutPtr
                };

                if (vulkan!.CreatePipelineLayout(device, in pipelineLayoutInfo, null, out pipelineLayout) != Result.Success)
                {
                    throw new Exception("failed to create pipeline layout!");
                }

                var pipelineInfo = new GraphicsPipelineCreateInfo()
                {
                    SType = StructureType.GraphicsPipelineCreateInfo,
                    StageCount = 2,
                    PStages = shaderStages,
                    PVertexInputState = &vertexInputInfo,
                    PInputAssemblyState = &inputAssembly,
                    PViewportState = &viewportState,
                    PRasterizationState = &rasterizer,
                    PMultisampleState = &multisampling,
                    PDepthStencilState = &depthStencil,
                    PColorBlendState = &colorBlending,
                    Layout = pipelineLayout,
                    RenderPass = renderPass,
                    Subpass = 0,
                    BasePipelineHandle = default
                };

                if (vulkan!.CreateGraphicsPipelines(device, default, 1, in pipelineInfo, null, out graphicsPipeline) != Result.Success)
                {
                    throw new Exception("failed to create graphics pipeline!");
                }
            }

            vulkan!.DestroyShaderModule(device, fragShaderModule, null);
            vulkan!.DestroyShaderModule(device, vertShaderModule, null);

            SilkMarshal.Free((nint)vertShaderStageInfo.PName);
            SilkMarshal.Free((nint)fragShaderStageInfo.PName);
        }
        #endregion

        #region Create frame buffers
        private void CreateFramebuffers()
        {
            swapChainFramebuffers = new Framebuffer[swapChainImageViews!.Length];

            for (var x = 0; x < swapChainImageViews.Length; x++)
            {
                var attachments = new[] { colorImageView, depthImageView, swapChainImageViews[x] };

                unsafe
                {
                    fixed (ImageView* attachmentsPtr = attachments)
                    {
                        var framebufferInfo = new FramebufferCreateInfo()
                        {
                            SType = StructureType.FramebufferCreateInfo,
                            RenderPass = renderPass,
                            AttachmentCount = (uint)attachments.Length,
                            PAttachments = attachmentsPtr,
                            Width = swapChainExtent.Width,
                            Height = swapChainExtent.Height,
                            Layers = 1,
                        };

                        if (vulkan!.CreateFramebuffer(device, in framebufferInfo, null, out swapChainFramebuffers[x]) != Result.Success)
                        {
                            throw new Exception("failed to create framebuffer!");
                        }
                    }
                }
            }
        }
        #endregion

        #region Create command pool
        private void CreateCommandPool()
        {
            var queueFamiliyIndicies = FindQueueFamilies(physicalDevice);

            var poolInfo = new CommandPoolCreateInfo()
            {
                SType = StructureType.CommandPoolCreateInfo,
                QueueFamilyIndex = queueFamiliyIndicies.GraphicsFamily!.Value,
            };

            unsafe
            {
                if (vulkan!.CreateCommandPool(device, in poolInfo, null, out commandPool) != Result.Success)
                {
                    throw new Exception("failed to create command pool!");
                }
            }
        }
        #endregion

        #region Create color resources
        private void CreateColorResources()
        {
            Format colorFormat = swapChainImageFormat;

            CreateImage(swapChainExtent.Width, swapChainExtent.Height, 1, msaaSamples, colorFormat, ImageTiling.Optimal, ImageUsageFlags.TransientAttachmentBit | ImageUsageFlags.ColorAttachmentBit, MemoryPropertyFlags.DeviceLocalBit, ref colorImage, ref colorImageMemory);
            colorImageView = CreateImageView(colorImage, colorFormat, ImageAspectFlags.ColorBit, 1);
        }
        #endregion

        #region Create depth resources
        private Format FindSupportedFormat(IEnumerable<Format> candidates, ImageTiling tiling, FormatFeatureFlags features)
        {
            foreach (var format in candidates)
            {
                vulkan!.GetPhysicalDeviceFormatProperties(physicalDevice, format, out var props);

                if (tiling == ImageTiling.Linear && (props.LinearTilingFeatures & features) == features)
                {
                    return format;
                }
                else if (tiling == ImageTiling.Optimal && (props.OptimalTilingFeatures & features) == features)
                {
                    return format;
                }
            }

            throw new Exception("failed to find supported format!");
        }

        private Format FindDepthFormat()
        {
            return FindSupportedFormat([Format.D32Sfloat, Format.D32SfloatS8Uint, Format.D24UnormS8Uint], ImageTiling.Optimal, FormatFeatureFlags.DepthStencilAttachmentBit);
        }

        private void CreateDepthResources()
        {
            var depthFormat = FindDepthFormat();

            CreateImage(swapChainExtent.Width, swapChainExtent.Height, 1, msaaSamples, depthFormat, ImageTiling.Optimal, ImageUsageFlags.DepthStencilAttachmentBit, MemoryPropertyFlags.DeviceLocalBit, ref depthImage, ref depthImageMemory);
            depthImageView = CreateImageView(depthImage, depthFormat, ImageAspectFlags.DepthBit, 1);
        }
        #endregion

        #region Create texture image
        private void GenerateMipMaps(Image image, Format imageFormat, uint width, uint height, uint mipLevels)
        {
            vulkan!.GetPhysicalDeviceFormatProperties(physicalDevice, imageFormat, out var formatProperties);

            if ((formatProperties.OptimalTilingFeatures & FormatFeatureFlags.SampledImageFilterLinearBit) == 0)
            {
                throw new Exception("texture image format does not support linear blitting!");
            }

            var commandBuffer = BeginSingleTimeCommands();

            var barrier = new ImageMemoryBarrier()
            {
                SType = StructureType.ImageMemoryBarrier,
                Image = image,
                SrcQueueFamilyIndex = Vk.QueueFamilyIgnored,
                DstQueueFamilyIndex = Vk.QueueFamilyIgnored,
                SubresourceRange =
                {
                    AspectMask = ImageAspectFlags.ColorBit,
                    BaseArrayLayer = 0,
                    LayerCount = 1,
                    LevelCount = 1,
                }
            };

            var mipWidth = width;
            var mipHeight = height;

            for (uint i = 1; i < mipLevels; i++)
            {
                barrier.SubresourceRange.BaseMipLevel = i - 1;
                barrier.OldLayout = ImageLayout.TransferDstOptimal;
                barrier.NewLayout = ImageLayout.TransferSrcOptimal;
                barrier.SrcAccessMask = AccessFlags.TransferWriteBit;
                barrier.DstAccessMask = AccessFlags.TransferReadBit;

                unsafe
                {
                    vulkan!.CmdPipelineBarrier(
                        commandBuffer, PipelineStageFlags.TransferBit, PipelineStageFlags.TransferBit, 0,
                        0, null,
                        0, null,
                        1, in barrier
                    );
                }

                ImageBlit blit = new()
                {
                    SrcOffsets =
                    {
                        Element0 = new Offset3D(0,0,0),
                        Element1 = new Offset3D((int)mipWidth, (int)mipHeight, 1),
                    },
                    SrcSubresource =
                    {
                        AspectMask = ImageAspectFlags.ColorBit,
                        MipLevel = i - 1,
                        BaseArrayLayer = 0,
                        LayerCount = 1,
                    },
                    DstOffsets =
                    {
                        Element0 = new Offset3D(0,0,0),
                        Element1 = new Offset3D((int)(mipWidth > 1 ? mipWidth / 2 : 1), (int)(mipHeight > 1 ? mipHeight / 2 : 1),1),
                    },
                    DstSubresource =
                    {
                        AspectMask = ImageAspectFlags.ColorBit,
                        MipLevel = i,
                        BaseArrayLayer = 0,
                        LayerCount = 1,
                    },
                };

                vulkan!.CmdBlitImage(commandBuffer,
                    image, ImageLayout.TransferSrcOptimal,
                    image, ImageLayout.TransferDstOptimal,
                    1, in blit,
                    Filter.Linear);

                barrier.OldLayout = ImageLayout.TransferSrcOptimal;
                barrier.NewLayout = ImageLayout.ShaderReadOnlyOptimal;
                barrier.SrcAccessMask = AccessFlags.TransferReadBit;
                barrier.DstAccessMask = AccessFlags.ShaderReadBit;

                unsafe
                {
                    vulkan!.CmdPipelineBarrier(
                        commandBuffer, PipelineStageFlags.TransferBit, PipelineStageFlags.FragmentShaderBit, 0,
                        0, null,
                        0, null,
                        1, in barrier
                    );
                }

                if (mipWidth > 1) mipWidth /= 2;
                if (mipHeight > 1) mipHeight /= 2;
            }

            barrier.SubresourceRange.BaseMipLevel = mipLevels - 1;
            barrier.OldLayout = ImageLayout.TransferDstOptimal;
            barrier.NewLayout = ImageLayout.ShaderReadOnlyOptimal;
            barrier.SrcAccessMask = AccessFlags.TransferWriteBit;
            barrier.DstAccessMask = AccessFlags.ShaderReadBit;

            unsafe
            {
                vulkan!.CmdPipelineBarrier(
                    commandBuffer, PipelineStageFlags.TransferBit, PipelineStageFlags.FragmentShaderBit, 0,
                    0, null,
                    0, null,
                    1, in barrier
                );
            }

            EndSingleTimeCommands(commandBuffer);
        }

        private void CreateTextureImage()
        {
            using var img = SixLabors.ImageSharp.Image.Load<SixLabors.ImageSharp.PixelFormats.Rgba32>(TEXTURE_PATH);

            var imageSize = (ulong)(img.Width * img.Height * img.PixelType.BitsPerPixel / 8);
            mipLevels = (uint)(Math.Floor(Math.Log2(Math.Max(img.Width, img.Height))) + 1);

            var stagingBuffer = default(Buffer);
            var stagingBufferMemory = default(DeviceMemory);
            CreateBuffer(imageSize, BufferUsageFlags.TransferSrcBit, MemoryPropertyFlags.HostVisibleBit | MemoryPropertyFlags.HostCoherentBit, ref stagingBuffer, ref stagingBufferMemory);

            unsafe
            {
                void* data;
                vulkan!.MapMemory(device, stagingBufferMemory, 0, imageSize, 0, &data);
                img.CopyPixelDataTo(new Span<byte>(data, (int)imageSize));
                vulkan!.UnmapMemory(device, stagingBufferMemory);
            }

            CreateImage(
                (uint)img.Width,
                (uint)img.Height,
                mipLevels,
                SampleCountFlags.Count1Bit,
                Format.R8G8B8A8Srgb,
                ImageTiling.Optimal,
                ImageUsageFlags.TransferSrcBit | ImageUsageFlags.TransferDstBit | ImageUsageFlags.SampledBit,
                MemoryPropertyFlags.DeviceLocalBit,
                ref textureImage,
                ref textureImageMemory
            );

            TransitionImageLayout(textureImage, Format.R8G8B8A8Srgb, ImageLayout.Undefined, ImageLayout.TransferDstOptimal, mipLevels);
            CopyBufferToImage(stagingBuffer, textureImage, (uint)img.Width, (uint)img.Height);
            //Transitioned to VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL while generating mipmaps

            unsafe
            {
                vulkan!.DestroyBuffer(device, stagingBuffer, null);
                vulkan!.FreeMemory(device, stagingBufferMemory, null);
            }

            GenerateMipMaps(textureImage, Format.R8G8B8A8Srgb, (uint)img.Width, (uint)img.Height, mipLevels);
        }

        private void CreateImage(
            uint width,
            uint height,
            uint mipLevels,
            SampleCountFlags numSamples,
            Format format,
            ImageTiling tiling,
            ImageUsageFlags usage,
            MemoryPropertyFlags properties,
            ref Image image,
            ref DeviceMemory imageMemory
        )
        {
            var imageInfo = new ImageCreateInfo()
            {
                SType = StructureType.ImageCreateInfo,
                ImageType = ImageType.Type2D,
                Extent =
                {
                    Width = width,
                    Height = height,
                    Depth = 1,
                },
                MipLevels = mipLevels,
                ArrayLayers = 1,
                Format = format,
                Tiling = tiling,
                InitialLayout = ImageLayout.Undefined,
                Usage = usage,
                Samples = numSamples,
                SharingMode = SharingMode.Exclusive,
            };

            unsafe
            {
                fixed (Image* imagePtr = &image)
                {
                    if (vulkan!.CreateImage(device, in imageInfo, null, imagePtr) != Result.Success)
                    {
                        throw new Exception("failed to create image!");
                    }
                }
            }

            vulkan!.GetImageMemoryRequirements(device, image, out MemoryRequirements memRequirements);

            var allocInfo = new MemoryAllocateInfo()
            {
                SType = StructureType.MemoryAllocateInfo,
                AllocationSize = memRequirements.Size,
                MemoryTypeIndex = FindMemoryType(memRequirements.MemoryTypeBits, properties),
            };

            unsafe
            {
                fixed (DeviceMemory* imageMemoryPtr = &imageMemory)
                {
                    if (vulkan!.AllocateMemory(device, in allocInfo, null, imageMemoryPtr) != Result.Success)
                    {
                        throw new Exception("failed to allocate image memory!");
                    }
                }
            }

            vulkan!.BindImageMemory(device, image, imageMemory, 0);
        }

        private void TransitionImageLayout(Image image, Format format, ImageLayout oldLayout, ImageLayout newLayout, uint mipLevels)
        {
            var commandBuffer = BeginSingleTimeCommands();

            var barrier = new ImageMemoryBarrier()
            {
                SType = StructureType.ImageMemoryBarrier,
                OldLayout = oldLayout,
                NewLayout = newLayout,
                SrcQueueFamilyIndex = Vk.QueueFamilyIgnored,
                DstQueueFamilyIndex = Vk.QueueFamilyIgnored,
                Image = image,
                SubresourceRange =
                {
                    AspectMask = ImageAspectFlags.ColorBit,
                    BaseMipLevel = 0,
                    LevelCount = mipLevels,
                    BaseArrayLayer = 0,
                    LayerCount = 1,
                }
            };

            PipelineStageFlags sourceStage;
            PipelineStageFlags destinationStage;

            if (oldLayout == ImageLayout.Undefined && newLayout == ImageLayout.TransferDstOptimal)
            {
                barrier.SrcAccessMask = 0;
                barrier.DstAccessMask = AccessFlags.TransferWriteBit;

                sourceStage = PipelineStageFlags.TopOfPipeBit;
                destinationStage = PipelineStageFlags.TransferBit;
            }
            else if (oldLayout == ImageLayout.TransferDstOptimal && newLayout == ImageLayout.ShaderReadOnlyOptimal)
            {
                barrier.SrcAccessMask = AccessFlags.TransferWriteBit;
                barrier.DstAccessMask = AccessFlags.ShaderReadBit;

                sourceStage = PipelineStageFlags.TransferBit;
                destinationStage = PipelineStageFlags.FragmentShaderBit;
            }
            else
            {
                throw new Exception("unsupported layout transition!");
            }

            unsafe
            {
                vulkan!.CmdPipelineBarrier(commandBuffer, sourceStage, destinationStage, 0, 0, null, 0, null, 1, in barrier);
            }

            EndSingleTimeCommands(commandBuffer);
        }

        private void CopyBufferToImage(Buffer buffer, Image image, uint width, uint height)
        {
            var commandBuffer = BeginSingleTimeCommands();

            var region = new BufferImageCopy()
            {
                BufferOffset = 0,
                BufferRowLength = 0,
                BufferImageHeight = 0,
                ImageSubresource =
                {
                    AspectMask = ImageAspectFlags.ColorBit,
                    MipLevel = 0,
                    BaseArrayLayer = 0,
                    LayerCount = 1,
                },
                ImageOffset = new Offset3D(0, 0, 0),
                ImageExtent = new Extent3D(width, height, 1),

            };

            vulkan!.CmdCopyBufferToImage(commandBuffer, buffer, image, ImageLayout.TransferDstOptimal, 1, in region);

            EndSingleTimeCommands(commandBuffer);
        }
        #endregion

        #region Create texture image view
        private void CreateTextureImageView()
        {
            textureImageView = CreateImageView(textureImage, Format.R8G8B8A8Srgb, ImageAspectFlags.ColorBit, mipLevels);
        }
        #endregion

        #region Create texture sampler
        private void CreateTextureSampler()
        {
            vulkan!.GetPhysicalDeviceProperties(physicalDevice, out PhysicalDeviceProperties properties);

            var samplerInfo = new SamplerCreateInfo()
            {
                SType = StructureType.SamplerCreateInfo,
                MagFilter = Filter.Linear,
                MinFilter = Filter.Linear,
                AddressModeU = SamplerAddressMode.Repeat,
                AddressModeV = SamplerAddressMode.Repeat,
                AddressModeW = SamplerAddressMode.Repeat,
                AnisotropyEnable = true,
                MaxAnisotropy = properties.Limits.MaxSamplerAnisotropy,
                BorderColor = BorderColor.IntOpaqueBlack,
                UnnormalizedCoordinates = false,
                CompareEnable = false,
                CompareOp = CompareOp.Always,
                MipmapMode = SamplerMipmapMode.Linear,
                MinLod = 0,
                MaxLod = mipLevels,
                MipLodBias = 0,
            };

            unsafe
            {
                fixed (Sampler* textureSamplerPtr = &textureSampler)
                {
                    if (vulkan!.CreateSampler(device, in samplerInfo, null, textureSamplerPtr) != Result.Success)
                    {
                        throw new Exception("failed to create texture sampler!");
                    }
                }
            }
        }
        #endregion

        #region Load model
        private unsafe void LoadModel()
        {
            using var assimp = Assimp.GetApi();
            var scene = assimp.ImportFile(MODEL_PATH, (uint)PostProcessPreset.TargetRealTimeMaximumQuality);

            var vertexMap = new Dictionary<Vertex, uint>();
            var vertices = new List<Vertex>();
            var indices = new List<uint>();

            VisitSceneNode(scene->MRootNode);

            assimp.ReleaseImport(scene);

            this.vertices = [.. vertices];
            this.indices = [.. indices];

            void VisitSceneNode(Node* node)
            {
                for (var m = 0; m < node->MNumMeshes; m++)
                {
                    var mesh = scene->MMeshes[node->MMeshes[m]];

                    for (var f = 0; f < mesh->MNumFaces; f++)
                    {
                        var face = mesh->MFaces[f];

                        for (var x = 0; x < face.MNumIndices; x++)
                        {
                            uint index = face.MIndices[x];

                            var position = mesh->MVertices[index];
                            var texture = mesh->MTextureCoords[0][(int)index];

                            var vertex = new Vertex()
                            {
                                pos = new Vector3D<float>(position.X, position.Y, position.Z),
                                color = new Vector3D<float>(1, 1, 1),
                                //Flip Y for OBJ in Vulkan
                                textCoord = new Vector2D<float>(texture.X, 1.0f - texture.Y)
                            };

                            if (vertexMap.TryGetValue(vertex, out var meshIndex))
                            {
                                indices.Add(meshIndex);
                            }
                            else
                            {
                                indices.Add((uint)vertices.Count);
                                vertexMap[vertex] = (uint)vertices.Count;
                                vertices.Add(vertex);
                            }
                        }
                    }
                }

                for (int c = 0; c < node->MNumChildren; c++)
                {
                    VisitSceneNode(node->MChildren[c]);
                }
            }
        }
        #endregion

        #region Create vertex buffer
        private uint FindMemoryType(uint typeFilter, MemoryPropertyFlags properties)
        {
            PhysicalDeviceMemoryProperties memProperties;
            unsafe
            {
                vulkan!.GetPhysicalDeviceMemoryProperties(physicalDevice, out memProperties);
            }

            for (var x = 0; x < memProperties.MemoryTypeCount; x++)
            {
                if ((typeFilter & (1 << x)) != 0 && (memProperties.MemoryTypes[x].PropertyFlags & properties) == properties)
                {
                    return (uint)x;
                }
            }

            throw new Exception("failed to find suitable memory type!");
        }

        private void CreateBuffer(ulong size, BufferUsageFlags usage, MemoryPropertyFlags properties, ref Buffer buffer, ref DeviceMemory bufferMemory)
        {
            var bufferInfo = new BufferCreateInfo()
            {
                SType = StructureType.BufferCreateInfo,
                Size = size,
                Usage = usage,
                SharingMode = SharingMode.Exclusive,
            };

            unsafe
            {
                fixed (Buffer* bufferPtr = &buffer)
                {
                    if (vulkan!.CreateBuffer(device, in bufferInfo, null, bufferPtr) != Result.Success)
                    {
                        throw new Exception("failed to create vertex buffer!");
                    }
                }
            }

            var memRequirements = new MemoryRequirements();
            vulkan!.GetBufferMemoryRequirements(device, buffer, out memRequirements);

            var allocateInfo = new MemoryAllocateInfo()
            {
                SType = StructureType.MemoryAllocateInfo,
                AllocationSize = memRequirements.Size,
                MemoryTypeIndex = FindMemoryType(memRequirements.MemoryTypeBits, properties),
            };

            unsafe
            {
                fixed (DeviceMemory* bufferMemoryPtr = &bufferMemory)
                {
                    if (vulkan!.AllocateMemory(device, in allocateInfo, null, bufferMemoryPtr) != Result.Success)
                    {
                        throw new Exception("failed to allocate vertex buffer memory!");
                    }
                }
            }

            vulkan!.BindBufferMemory(device, buffer, bufferMemory, 0);
        }

        private CommandBuffer BeginSingleTimeCommands()
        {
            var allocateInfo = new CommandBufferAllocateInfo()
            {
                SType = StructureType.CommandBufferAllocateInfo,
                Level = CommandBufferLevel.Primary,
                CommandPool = commandPool,
                CommandBufferCount = 1,
            };

            vulkan!.AllocateCommandBuffers(device, in allocateInfo, out var commandBuffer);

            var beginInfo = new CommandBufferBeginInfo()
            {
                SType = StructureType.CommandBufferBeginInfo,
                Flags = CommandBufferUsageFlags.OneTimeSubmitBit,
            };

            vulkan!.BeginCommandBuffer(commandBuffer, in beginInfo);
            return commandBuffer;
        }

        private void EndSingleTimeCommands(CommandBuffer commandBuffer)
        {
            vulkan!.EndCommandBuffer(commandBuffer);

            SubmitInfo submitInfo;
            unsafe
            {
                submitInfo = new SubmitInfo()
                {
                    SType = StructureType.SubmitInfo,
                    CommandBufferCount = 1,
                    PCommandBuffers = &commandBuffer,
                };
            }

            vulkan!.QueueSubmit(graphicsQueue, 1, in submitInfo, default);
            vulkan!.QueueWaitIdle(graphicsQueue);

            vulkan!.FreeCommandBuffers(device, commandPool, 1, in commandBuffer);
        }

        private void CopyBuffer(Buffer srcBuffer, Buffer dstBuffer, ulong size)
        {
            CommandBuffer commandBuffer = BeginSingleTimeCommands();

            var copyRegion = new BufferCopy()
            {
                Size = size,
            };

            vulkan!.CmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, in copyRegion);

            EndSingleTimeCommands(commandBuffer);
        }

        private void CreateVertexBuffer()
        {
            var bufferSize = (ulong)(Unsafe.SizeOf<Vertex>() * vertices!.Length);

            Buffer stagingBuffer = default;
            DeviceMemory stagingBufferMemory = default;
            CreateBuffer(bufferSize, BufferUsageFlags.TransferSrcBit, MemoryPropertyFlags.HostVisibleBit | MemoryPropertyFlags.HostCoherentBit, ref stagingBuffer, ref stagingBufferMemory);

            unsafe
            {
                void* data;
                vulkan!.MapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
                vertices.AsSpan().CopyTo(new Span<Vertex>(data, vertices.Length));
                vulkan!.UnmapMemory(device, stagingBufferMemory);

                CreateBuffer(bufferSize, BufferUsageFlags.TransferDstBit | BufferUsageFlags.VertexBufferBit, MemoryPropertyFlags.DeviceLocalBit, ref vertexBuffer, ref vertexBufferMemory);

                CopyBuffer(stagingBuffer, vertexBuffer, bufferSize);

                vulkan!.DestroyBuffer(device, stagingBuffer, null);
                vulkan!.FreeMemory(device, stagingBufferMemory, null);
            }
        }
        #endregion

        #region Create index buffer
        private void CreateIndexBuffer()
        {
            var bufferSize = (ulong)(Unsafe.SizeOf<uint>() * indices!.Length);

            Buffer stagingBuffer = default;
            DeviceMemory stagingBufferMemory = default;
            CreateBuffer(bufferSize, BufferUsageFlags.TransferSrcBit, MemoryPropertyFlags.HostVisibleBit | MemoryPropertyFlags.HostCoherentBit, ref stagingBuffer, ref stagingBufferMemory);

            unsafe
            {
                void* data;
                vulkan!.MapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
                indices.AsSpan().CopyTo(new Span<uint>(data, indices.Length));
                vulkan!.UnmapMemory(device, stagingBufferMemory);

                CreateBuffer(bufferSize, BufferUsageFlags.TransferDstBit | BufferUsageFlags.IndexBufferBit, MemoryPropertyFlags.DeviceLocalBit, ref indexBuffer, ref indexBufferMemory);

                CopyBuffer(stagingBuffer, indexBuffer, bufferSize);

                vulkan!.DestroyBuffer(device, stagingBuffer, null);
                vulkan!.FreeMemory(device, stagingBufferMemory, null);
            }
        }
        #endregion

        #region Create descriptor pool
        private void CreateDescriptorPool()
        {
            var poolSizes = new DescriptorPoolSize[]
            {
                new()
                {
                    Type = DescriptorType.UniformBuffer,
                    DescriptorCount = (uint)swapChainImages!.Length,
                },
                new()
                {
                    Type = DescriptorType.CombinedImageSampler,
                    DescriptorCount = (uint)swapChainImages!.Length,
                }
            };

            unsafe
            {
                fixed (DescriptorPoolSize* poolSizesPtr = poolSizes)
                fixed (DescriptorPool* descriptorPoolPtr = &descriptorPool)
                {
                    var poolInfo = new DescriptorPoolCreateInfo()
                    {
                        SType = StructureType.DescriptorPoolCreateInfo,
                        PoolSizeCount = (uint)poolSizes.Length,
                        PPoolSizes = poolSizesPtr,
                        MaxSets = (uint)swapChainImages!.Length,
                    };

                    if (vulkan!.CreateDescriptorPool(device, in poolInfo, null, descriptorPoolPtr) != Result.Success)
                    {
                        throw new Exception("failed to create descriptor pool!");
                    }
                }
            }
        }
        #endregion

        #region Create descriptor sets
        private void CreateDescriptorSets()
        {
            var layouts = new DescriptorSetLayout[swapChainImages!.Length];
            Array.Fill(layouts, descriptorSetLayout);

            unsafe
            {
                fixed (DescriptorSetLayout* layoutsPtr = layouts)
                {
                    var allocateInfo = new DescriptorSetAllocateInfo()
                    {
                        SType = StructureType.DescriptorSetAllocateInfo,
                        DescriptorPool = descriptorPool,
                        DescriptorSetCount = (uint)swapChainImages!.Length,
                        PSetLayouts = layoutsPtr,
                    };

                    descriptorSets = new DescriptorSet[swapChainImages.Length];
                    fixed (DescriptorSet* descriptorSetsPtr = descriptorSets)
                    {
                        if (vulkan!.AllocateDescriptorSets(device, in allocateInfo, descriptorSetsPtr) != Result.Success)
                        {
                            throw new Exception("failed to allocate descriptor sets!");
                        }
                    }
                }
            }

            for (var x = 0; x < swapChainImages.Length; x++)
            {
                var bufferInfo = new DescriptorBufferInfo()
                {
                    Buffer = uniformBuffers![x],
                    Offset = 0,
                    Range = (ulong)Unsafe.SizeOf<UniformBufferObject>(),
                };

                var imageInfo = new DescriptorImageInfo()
                {
                    ImageLayout = ImageLayout.ShaderReadOnlyOptimal,
                    ImageView = textureImageView,
                    Sampler = textureSampler,
                };

                unsafe
                {
                    var descriptorWrites = new WriteDescriptorSet[]
                    {
                    new()
                    {
                        SType = StructureType.WriteDescriptorSet,
                        DstSet = descriptorSets[x],
                        DstBinding = 0,
                        DstArrayElement = 0,
                        DescriptorType = DescriptorType.UniformBuffer,
                        DescriptorCount = 1,
                        PBufferInfo = &bufferInfo,
                    },
                    new()
                    {
                        SType = StructureType.WriteDescriptorSet,
                        DstSet = descriptorSets[x],
                        DstBinding = 1,
                        DstArrayElement = 0,
                        DescriptorType = DescriptorType.CombinedImageSampler,
                        DescriptorCount = 1,
                        PImageInfo = &imageInfo,
                    }
                    };

                    fixed (WriteDescriptorSet* descriptorWritesPtr = descriptorWrites)
                    {
                        vulkan!.UpdateDescriptorSets(device, (uint)descriptorWrites.Length, descriptorWritesPtr, 0, null);
                    }
                }
            }
        }
        #endregion

        #region Create command buffers
        private void CreateCommandBuffers()
        {
            commandBuffers = new CommandBuffer[swapChainFramebuffers!.Length];

            var allocInfo = new CommandBufferAllocateInfo()
            {
                SType = StructureType.CommandBufferAllocateInfo,
                CommandPool = commandPool,
                Level = CommandBufferLevel.Primary,
                CommandBufferCount = (uint)commandBuffers.Length,
            };

            unsafe
            {
                fixed (CommandBuffer* commandBuffersPtr = commandBuffers)
                {
                    if (vulkan!.AllocateCommandBuffers(device, in allocInfo, commandBuffersPtr) != Result.Success)
                    {
                        throw new Exception("failed to allocate command buffers!");
                    }
                }
            }

            for (var x = 0; x < commandBuffers.Length; x++)
            {
                var beginInfo = new CommandBufferBeginInfo()
                {
                    SType = StructureType.CommandBufferBeginInfo,
                };

                if (vulkan!.BeginCommandBuffer(commandBuffers[x], in beginInfo) != Result.Success)
                {
                    throw new Exception("failed to begin recording command buffer!");
                }

                var renderPassInfo = new RenderPassBeginInfo()
                {
                    SType = StructureType.RenderPassBeginInfo,
                    RenderPass = renderPass,
                    Framebuffer = swapChainFramebuffers[x],
                    RenderArea =
                    {
                        Offset = { X = 0, Y = 0 },
                        Extent = swapChainExtent,
                    }
                };

                var clearValues = new ClearValue[]
                {
                    new()
                    {
                        Color = new (){ Float32_0 = 0, Float32_1 = 0, Float32_2 = 0, Float32_3 = 1 },
                    },
                    new()
                    {
                        DepthStencil = new () { Depth = 1, Stencil = 0 }
                    }
                };

                unsafe
                {
                    fixed (ClearValue* clearValuesPtr = clearValues)
                    {
                        renderPassInfo.ClearValueCount = (uint)clearValues.Length;
                        renderPassInfo.PClearValues = clearValuesPtr;

                        vulkan!.CmdBeginRenderPass(commandBuffers[x], &renderPassInfo, SubpassContents.Inline);
                    }
                }

                vulkan!.CmdBindPipeline(commandBuffers[x], PipelineBindPoint.Graphics, graphicsPipeline);

                var vertexBuffers = new Buffer[] { vertexBuffer };
                var offsets = new ulong[] { 0 };

                unsafe
                {
                    fixed (ulong* offsetsPtr = offsets)
                    fixed (Buffer* vertexBuffersPtr = vertexBuffers)
                    {
                        vulkan!.CmdBindVertexBuffers(commandBuffers[x], 0, 1, vertexBuffersPtr, offsetsPtr);
                    }
                }

                vulkan!.CmdBindIndexBuffer(commandBuffers[x], indexBuffer, 0, IndexType.Uint32);

                unsafe
                {
                    vulkan!.CmdBindDescriptorSets(
                        commandBuffers[x],
                        PipelineBindPoint.Graphics,
                        pipelineLayout,
                        0,
                        1,
                        in descriptorSets![x],
                        0,
                        null
                    );
                }

                vulkan!.CmdDrawIndexed(commandBuffers[x], (uint)indices!.Length, 1, 0, 0, 0);
                vulkan!.CmdEndRenderPass(commandBuffers[x]);

                if (vulkan!.EndCommandBuffer(commandBuffers[x]) != Result.Success)
                {
                    throw new Exception("failed to record command buffer!");
                }
            }
        }
        #endregion

        #region Create sync object
        private void CreateSyncObjects()
        {
            imageAvailableSemaphores = new Semaphore[MAX_FRAMES_IN_FLIGHT];
            renderFinishedSemaphores = new Semaphore[MAX_FRAMES_IN_FLIGHT];
            inFlightFences = new Fence[MAX_FRAMES_IN_FLIGHT];
            imagesInFlight = new Fence[swapChainImages!.Length];

            var semaphoreInfo = new SemaphoreCreateInfo()
            {
                SType = StructureType.SemaphoreCreateInfo,
            };

            var fenceInfo = new FenceCreateInfo()
            {
                SType = StructureType.FenceCreateInfo,
                Flags = FenceCreateFlags.SignaledBit,
            };

            for (var i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
            {
                unsafe
                {
                    if (vulkan!.CreateSemaphore(device, in semaphoreInfo, null, out imageAvailableSemaphores[i]) != Result.Success ||
                        vulkan!.CreateSemaphore(device, in semaphoreInfo, null, out renderFinishedSemaphores[i]) != Result.Success ||
                        vulkan!.CreateFence(device, in fenceInfo, null, out inFlightFences[i]) != Result.Success)
                    {
                        throw new Exception("failed to create synchronization objects for a frame!");
                    }
                }
            }
        }
        #endregion

        private void InitVulkan()
        {
            DebugLog($"PHASE: {nameof(CreateInstance)}");
            CreateInstance();
            DebugLog($"PHASE: {nameof(SetupDebugMessenger)}");
            SetupDebugMessenger();
            DebugLog($"PHASE: {nameof(CreateSurface)}");
            CreateSurface();
            DebugLog($"PHASE: {nameof(PickPhysicalDevice)}");
            PickPhysicalDevice();
            DebugLog($"PHASE: {nameof(CreateLogicalDevice)}");
            CreateLogicalDevice();
            DebugLog($"PHASE: {nameof(CreateSwapChain)}");
            CreateSwapChain();
            DebugLog($"PHASE: {nameof(CreateImageViews)}");
            CreateImageViews();
            DebugLog($"PHASE: {nameof(CreateRenderPass)}");
            CreateRenderPass();
            DebugLog($"PHASE: {nameof(CreateDescriptorSetLayout)}");
            CreateDescriptorSetLayout();
            DebugLog($"PHASE: {nameof(CreateGraphicsPipeline)}");
            CreateGraphicsPipeline();
            DebugLog($"PHASE: {nameof(CreateCommandPool)}");
            CreateCommandPool();
            DebugLog($"PHASE: {nameof(CreateColorResources)}");
            CreateColorResources();
            DebugLog($"PHASE: {nameof(CreateDepthResources)}");
            CreateDepthResources();
            DebugLog($"PHASE: {nameof(CreateFramebuffers)}");
            CreateFramebuffers();
            DebugLog($"PHASE: {nameof(CreateTextureImage)}");
            CreateTextureImage();
            DebugLog($"PHASE: {nameof(CreateTextureImageView)}");
            CreateTextureImageView();
            DebugLog($"PHASE: {nameof(CreateTextureSampler)}");
            CreateTextureSampler();
            DebugLog($"PHASE: {nameof(LoadModel)}");
            LoadModel();
            DebugLog($"PHASE: {nameof(CreateVertexBuffer)}");
            CreateVertexBuffer();
            DebugLog($"PHASE: {nameof(CreateIndexBuffer)}");
            CreateIndexBuffer();
            DebugLog($"PHASE: {nameof(CreateUniformBuffers)}");
            CreateUniformBuffers();
            DebugLog($"PHASE: {nameof(CreateDescriptorPool)}");
            CreateDescriptorPool();
            DebugLog($"PHASE: {nameof(CreateDescriptorSets)}");
            CreateDescriptorSets();
            DebugLog($"PHASE: {nameof(CreateCommandBuffers)}");
            CreateCommandBuffers();
            DebugLog($"PHASE: {nameof(CreateSyncObjects)}");
            CreateSyncObjects();
        }
        #endregion

        #region Events
        private double prewTime = 0;
        private Vector3D<float> moveVector = new(0, 0, 1);
        private float moveSpeed = 90;
        private Matrix4X4<float> prewMatrix = Matrix4X4.CreateFromAxisAngle(new Vector3D<float>(0, 0, 1), 1f);

        private float camreaDistanceKindOf = 2;
        private double cameraSpeed = 0;

        public void OnLoad()
        {
            //Set-up input context.
            var input = window!.CreateInput();
            for (int x = 0; x < input.Keyboards.Count; x++)
            {
                input.Keyboards[x].KeyChar += CharPressed;
                input.Keyboards[x].KeyDown += KeyDown;
            }
        }

        private void UpdateUniformBuffer(uint currentImage, double deltaTime)
        {
            //Silk Window has timing information so we are skipping the time code.
            var time = window!.Time;

            var newMatrix = prewMatrix * Matrix4X4.CreateFromAxisAngle(moveVector, (float)(deltaTime * Scalar.DegreesToRadians(moveSpeed)));
            prewMatrix = newMatrix;
            
            camreaDistanceKindOf = (float)(camreaDistanceKindOf + cameraSpeed);

            var ubo = new UniformBufferObject()
            {
                model = Matrix4X4<float>.Identity * newMatrix,
                view = Matrix4X4.CreateLookAt(new Vector3D<float>(camreaDistanceKindOf, camreaDistanceKindOf, camreaDistanceKindOf), new Vector3D<float>(0, 0, 0), new Vector3D<float>(0, 0, 1)),
                proj = Matrix4X4.CreatePerspectiveFieldOfView(Scalar.DegreesToRadians(45.0f), (float)swapChainExtent.Width / swapChainExtent.Height, 0.1f, 10.0f),
            };
            ubo.proj.M22 *= -1;

            unsafe
            {
                void* data;
                vulkan!.MapMemory(
                    device,
                    uniformBuffersMemory![currentImage],
                    0,
                    (ulong)Unsafe.SizeOf<UniformBufferObject>(),
                    0,
                    &data
                );
                new Span<UniformBufferObject>(data, 1)[0] = ubo;
                vulkan!.UnmapMemory(device, uniformBuffersMemory![currentImage]);
            }

            prewTime = time;
        }

        public void OnRender(double deltaTime)
        {
            vulkan!.WaitForFences(device, 1, in inFlightFences![currentFrame], true, ulong.MaxValue);

            uint imageIndex = 0;
            var result = khrSwapChain!.AcquireNextImage(device, swapChain, ulong.MaxValue, imageAvailableSemaphores![currentFrame], default, ref imageIndex);

            if (result == Result.ErrorOutOfDateKhr)
            {
                RecreateSwapChain();
                return;
            }
            else if (result != Result.Success && result != Result.SuboptimalKhr)
            {
                throw new Exception("failed to acquire swap chain image!");
            }

            UpdateUniformBuffer(imageIndex, deltaTime);

            if (imagesInFlight![imageIndex].Handle != default)
            {
                vulkan!.WaitForFences(device, 1, in imagesInFlight[imageIndex], true, ulong.MaxValue);
            }
            imagesInFlight[imageIndex] = inFlightFences[currentFrame];

            var submitInfo = new SubmitInfo()
            {
                SType = StructureType.SubmitInfo,
            };

            PresentInfoKHR presentInfo;
            unsafe
            {
                var waitSemaphores = stackalloc[] { imageAvailableSemaphores[currentFrame] };
                var waitStages = stackalloc[] { PipelineStageFlags.ColorAttachmentOutputBit };

                var buffer = commandBuffers![imageIndex];

                submitInfo = submitInfo with
                {
                    WaitSemaphoreCount = 1,
                    PWaitSemaphores = waitSemaphores,
                    PWaitDstStageMask = waitStages,

                    CommandBufferCount = 1,
                    PCommandBuffers = &buffer
                };

                var signalSemaphores = stackalloc[] { renderFinishedSemaphores![currentFrame] };
                submitInfo = submitInfo with
                {
                    SignalSemaphoreCount = 1,
                    PSignalSemaphores = signalSemaphores,
                };

                vulkan!.ResetFences(device, 1, in inFlightFences[currentFrame]);

                if (vulkan!.QueueSubmit(graphicsQueue, 1, in submitInfo, inFlightFences[currentFrame]) != Result.Success)
                {
                    throw new Exception("failed to submit draw command buffer!");
                }

                var swapChains = stackalloc[] { swapChain };
                presentInfo = new PresentInfoKHR()
                {
                    SType = StructureType.PresentInfoKhr,

                    WaitSemaphoreCount = 1,
                    PWaitSemaphores = signalSemaphores,

                    SwapchainCount = 1,
                    PSwapchains = swapChains,

                    PImageIndices = &imageIndex
                };
            }

            result = khrSwapChain.QueuePresent(presentQueue, in presentInfo);

            if (
                result == Result.ErrorOutOfDateKhr ||
                result == Result.SuboptimalKhr ||
                frameBufferResized
            )
            {
                if (frameBufferResized || window!.WindowState != WindowState.Normal)
                {
                    frameBufferResized = false;
                    RecreateSwapChain();
                }
            }
            else if (result != Result.Success)
            {
                throw new Exception("failed to present swap chain image!");
            }

            currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
        }

        private void FramebufferResizeCallback(Vector2D<int> newSize)
        {
            frameBufferResized = newSize.Y > 0 && newSize.X > 0;
        }

        public void OnUpdate(double deltaTime)
        {
            //Here all updates to the program should be done.
        }

        public void OnFramebufferResize(Vector2D<int> newSize)
        {
            //Update aspect ratios, clipping regions, viewports, etc.
        }

        /// <summary>
        /// Called every frame the key is pressed.
        /// </summary>
        /// <param name="keyboard"></param>
        /// <param name="pressedKey"></param>
        public void CharPressed(IKeyboard keyboard, char pressedKey)
        {
            //moveSpeed += 10;
        }

        /// <summary>
        /// Called only once per press.
        /// </summary>
        /// <param name="keyboard"></param>
        /// <param name="pressedKey"></param>
        /// <param name="arg3"></param>
        public void KeyDown(IKeyboard keyboard, Key pressedKey, int arg3)
        {
            switch (pressedKey)
            {
                case Key.Escape:
                    window!.Close();
                    break;
                case Key.Q:
                    cameraSpeed -= 0.1;
                    break;
                case Key.E:
                    cameraSpeed += 0.1;
                    break;
                case Key.W:
                    moveVector = new Vector3D<float>(0, 1, 0);
                    break;
                case Key.S:
                    moveVector = new Vector3D<float>(0, -1, 0);
                    break;
                case Key.A:
                    moveVector = new Vector3D<float>(0, 0, -1);
                    break;
                case Key.D:
                    moveVector = new Vector3D<float>(0, 0, 1);
                    break;
                case Key.Up:
                    moveSpeed += 10;
                    break;
                case Key.Down:
                    moveSpeed -= 10;
                    break;
                case Key.R:
                    cameraSpeed = 0;
                    camreaDistanceKindOf = 2;
                    moveSpeed = 90;
                    moveVector = new Vector3D<float>(0, 0, 1);
                    prewMatrix = Matrix4X4.CreateFromAxisAngle(moveVector, (float)(0.1 * Scalar.DegreesToRadians(moveSpeed)));
                    break;
            }
        }
        #endregion
    }
}
