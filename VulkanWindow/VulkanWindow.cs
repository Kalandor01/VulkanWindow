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

        bool EnableValidationLayers = true;

        private readonly string[] validationLayers = [
            "VK_LAYER_KHRONOS_validation"
        ];

        private readonly string[] deviceExtensions = [
            KhrSwapchain.ExtensionName
        ];
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
        }

        public void Dispose()
        {
            unsafe
            {
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

        private void InitVulkan()
        {
            CreateInstance();
            SetupDebugMessenger();
            CreateSurface();
            PickPhysicalDevice();
            CreateLogicalDevice();
            CreateSwapChain();
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

        public void OnRender(double obj)
        {
            //Here all rendering should be done.
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
