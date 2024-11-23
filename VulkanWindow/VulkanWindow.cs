﻿using Silk.NET.Core;
using Silk.NET.Core.Native;
using Silk.NET.Input;
using Silk.NET.Maths;
using Silk.NET.Vulkan;
using Silk.NET.Vulkan.Extensions.EXT;
using Silk.NET.Vulkan.Extensions.KHR;
using Silk.NET.Windowing;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
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

    public class VulkanWindow : IDisposable
    {
        #region Fields
        private IWindow? window;
        private Vk? vulkan;
        private Instance vulkanInstance;
        private ExtDebugUtils? debugUtils;
        private DebugUtilsMessengerEXT debugMessenger;
        private PhysicalDevice physicalDevice;
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

        private PipelineLayout pipelineLayout;
        private RenderPass renderPass;
        private Pipeline graphicsPipeline;
        private Framebuffer[]? swapChainFramebuffers;

        private CommandPool commandPool;
        private CommandBuffer[]? commandBuffers;

        private Semaphore[]? imageAvailableSemaphores;
        private Semaphore[]? renderFinishedSemaphores;
        private Fence[]? inFlightFences;
        private Fence[]? imagesInFlight;
        private int currentFrame = 0;

        bool EnableValidationLayers = true;

        private readonly string[] validationLayers = [
            "VK_LAYER_KHRONOS_validation"
        ];

        private readonly string[] deviceExtensions = [
            KhrSwapchain.ExtensionName,
        ];

        const int MAX_FRAMES_IN_FLIGHT = 2;
        #endregion

        public VulkanWindow(int width, int height)
        {
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
            unsafe
            {
                for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
                {
                    vulkan!.DestroySemaphore(device, renderFinishedSemaphores![i], null);
                    vulkan!.DestroySemaphore(device, imageAvailableSemaphores![i], null);
                    vulkan!.DestroyFence(device, inFlightFences![i], null);
                }

                vulkan!.DestroyCommandPool(device, commandPool, null);

                foreach (var framebuffer in swapChainFramebuffers!)
                {
                    vulkan!.DestroyFramebuffer(device, framebuffer, null);
                }

                vulkan!.DestroyPipeline(device, graphicsPipeline, null);
                vulkan!.DestroyPipelineLayout(device, pipelineLayout, null);
                vulkan!.DestroyRenderPass(device, renderPass, null);

                foreach (var imageView in swapChainImageViews!)
                {
                    vulkan!.DestroyImageView(device, imageView, null);
                }

                khrSwapChain!.DestroySwapchain(device, swapChain, null);
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
        #region Common-er
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

            return indices.IsComplete() && extensionsSupported && swapChainAdequate;
        }

        private void PickPhysicalDevice()
        {
            var devices = vulkan!.GetPhysicalDevices(vulkanInstance);

            foreach (var device in devices)
            {
                if (IsDeviceSuitable(device))
                {
                    physicalDevice = device;
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

            var deviceFeatures = new PhysicalDeviceFeatures();

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

                OldSwapchain = default
            };

            if (!vulkan!.TryGetDeviceExtension(vulkanInstance, device, out khrSwapChain))
            {
                throw new NotSupportedException("VK_KHR_swapchain extension not found.");
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
        #endregion

        #region Create image views
        private void CreateImageViews()
        {
            swapChainImageViews = new ImageView[swapChainImages!.Length];

            for (int i = 0; i < swapChainImages.Length; i++)
            {
                ImageViewCreateInfo createInfo = new()
                {
                    SType = StructureType.ImageViewCreateInfo,
                    Image = swapChainImages[i],
                    ViewType = ImageViewType.Type2D,
                    Format = swapChainImageFormat,
                    Components =
                {
                    R = ComponentSwizzle.Identity,
                    G = ComponentSwizzle.Identity,
                    B = ComponentSwizzle.Identity,
                    A = ComponentSwizzle.Identity,
                },
                    SubresourceRange =
                {
                    AspectMask = ImageAspectFlags.ColorBit,
                    BaseMipLevel = 0,
                    LevelCount = 1,
                    BaseArrayLayer = 0,
                    LayerCount = 1,
                }

                };

                unsafe
                {
                    if (vulkan!.CreateImageView(device, in createInfo, null, out swapChainImageViews[i]) != Result.Success)
                    {
                        throw new Exception("failed to create image views!");
                    }
                }
            }
        }
        #endregion

        #region Create render pass
        private void CreateRenderPass()
        {
            var colorAttachment = new AttachmentDescription()
            {
                Format = swapChainImageFormat,
                Samples = SampleCountFlags.Count1Bit,
                LoadOp = AttachmentLoadOp.Clear,
                StoreOp = AttachmentStoreOp.Store,
                StencilLoadOp = AttachmentLoadOp.DontCare,
                InitialLayout = ImageLayout.Undefined,
                FinalLayout = ImageLayout.PresentSrcKhr,
            };

            var colorAttachmentRef = new AttachmentReference()
            {
                Attachment = 0,
                Layout = ImageLayout.ColorAttachmentOptimal,
            };

            var dependency = new SubpassDependency()
            {
                SrcSubpass = Vk.SubpassExternal,
                DstSubpass = 0,
                SrcStageMask = PipelineStageFlags.ColorAttachmentOutputBit,
                SrcAccessMask = 0,
                DstStageMask = PipelineStageFlags.ColorAttachmentOutputBit,
                DstAccessMask = AccessFlags.ColorAttachmentWriteBit
            };

            unsafe
            {
                var subpass = new SubpassDescription()
                {
                    PipelineBindPoint = PipelineBindPoint.Graphics,
                    ColorAttachmentCount = 1,
                    PColorAttachments = &colorAttachmentRef,
                };

                var renderPassInfo = new RenderPassCreateInfo()
                {
                    SType = StructureType.RenderPassCreateInfo,
                    AttachmentCount = 1,
                    PAttachments = &colorAttachment,
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

        private void CreateGraphicsPipeline()
        {
            var vertShaderCode = File.ReadAllBytes("Shaders/vert.spv");
            var fragShaderCode = File.ReadAllBytes("Shaders/frag.spv");

            var vertShaderModule = CreateShaderModule(vertShaderCode);
            var fragShaderModule = CreateShaderModule(fragShaderCode);

            var vertexInputInfo = new PipelineVertexInputStateCreateInfo()
            {
                SType = StructureType.PipelineVertexInputStateCreateInfo,
                VertexBindingDescriptionCount = 0,
                VertexAttributeDescriptionCount = 0,
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

            var rasterizer = new PipelineRasterizationStateCreateInfo()
            {
                SType = StructureType.PipelineRasterizationStateCreateInfo,
                DepthClampEnable = false,
                RasterizerDiscardEnable = false,
                PolygonMode = PolygonMode.Fill,
                LineWidth = 1,
                CullMode = CullModeFlags.BackBit,
                FrontFace = FrontFace.Clockwise,
                DepthBiasEnable = false,
            };

            var multisampling = new PipelineMultisampleStateCreateInfo()
            {
                SType = StructureType.PipelineMultisampleStateCreateInfo,
                SampleShadingEnable = false,
                RasterizationSamples = SampleCountFlags.Count1Bit,
            };

            var colorBlendAttachment = new PipelineColorBlendAttachmentState()
            {
                ColorWriteMask = ColorComponentFlags.RBit | ColorComponentFlags.GBit | ColorComponentFlags.BBit | ColorComponentFlags.ABit,
                BlendEnable = false,
            };

            var pipelineLayoutInfo = new PipelineLayoutCreateInfo()
            {
                SType = StructureType.PipelineLayoutCreateInfo,
                SetLayoutCount = 0,
                PushConstantRangeCount = 0,
            };

            unsafe
            {
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

                var viewportState = new PipelineViewportStateCreateInfo()
                {
                    SType = StructureType.PipelineViewportStateCreateInfo,
                    ViewportCount = 1,
                    PViewports = &viewport,
                    ScissorCount = 1,
                    PScissors = &scissor,
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

                vulkan!.DestroyShaderModule(device, fragShaderModule, null);
                vulkan!.DestroyShaderModule(device, vertShaderModule, null);

                SilkMarshal.Free((nint)vertShaderStageInfo.PName);
                SilkMarshal.Free((nint)fragShaderStageInfo.PName);
            }
        }
        #endregion

        #region Create frame buffer
        private void CreateFramebuffers()
        {
            swapChainFramebuffers = new Framebuffer[swapChainImageViews!.Length];

            for (int i = 0; i < swapChainImageViews.Length; i++)
            {
                var attachment = swapChainImageViews[i];

                unsafe
                {
                    var framebufferInfo = new FramebufferCreateInfo()
                    {
                        SType = StructureType.FramebufferCreateInfo,
                        RenderPass = renderPass,
                        AttachmentCount = 1,
                        PAttachments = &attachment,
                        Width = swapChainExtent.Width,
                        Height = swapChainExtent.Height,
                        Layers = 1,
                    };

                    if (vulkan!.CreateFramebuffer(device, in framebufferInfo, null, out swapChainFramebuffers[i]) != Result.Success)
                    {
                        throw new Exception("failed to create framebuffer!");
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

                var clearColor = new ClearValue()
                {
                    Color = new() { Float32_0 = 0, Float32_1 = 0, Float32_2 = 0, Float32_3 = 1 },
                };

                renderPassInfo.ClearValueCount = 1;

                unsafe
                {
                    renderPassInfo.PClearValues = &clearColor;

                    vulkan!.CmdBeginRenderPass(commandBuffers[x], &renderPassInfo, SubpassContents.Inline);
                }

                vulkan!.CmdBindPipeline(commandBuffers[x], PipelineBindPoint.Graphics, graphicsPipeline);

                vulkan!.CmdDraw(commandBuffers[x], 3, 1, 0, 0);

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
            DebugLog($"PHASE: {nameof(CreateGraphicsPipeline)}");
            CreateGraphicsPipeline();
            DebugLog($"PHASE: {nameof(CreateFramebuffers)}");
            CreateFramebuffers();
            DebugLog($"PHASE: {nameof(CreateCommandPool)}");
            CreateCommandPool();
            DebugLog($"PHASE: {nameof(CreateCommandBuffers)}");
            CreateCommandBuffers();
            DebugLog($"PHASE: {nameof(CreateSyncObjects)}");
            CreateSyncObjects();
        }
        #endregion

        #region Events
        public void OnLoad()
        {
            //Set-up input context.
            var input = window!.CreateInput();
            for (int i = 0; i < input.Keyboards.Count; i++)
            {
                input.Keyboards[i].KeyDown += KeyDown;
            }
        }

        public void OnRender(double delta)
        {
            vulkan!.WaitForFences(device, 1, in inFlightFences![currentFrame], true, ulong.MaxValue);

            uint imageIndex = 0;
            khrSwapChain!.AcquireNextImage(device, swapChain, ulong.MaxValue, imageAvailableSemaphores![currentFrame], default, ref imageIndex);

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

            khrSwapChain.QueuePresent(presentQueue, in presentInfo);

            currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
        }

        public void OnUpdate(double obj)
        {
            //Here all updates to the program should be done.
        }

        public void OnFramebufferResize(Vector2D<int> newSize)
        {
            //Update aspect ratios, clipping regions, viewports, etc.
        }

        public void KeyDown(IKeyboard arg1, Key arg2, int arg3)
        {
            //Check to close the window on escape.
            if (arg2 == Key.Escape)
            {
                window!.Close();
            }
        }
        #endregion
    }
}
