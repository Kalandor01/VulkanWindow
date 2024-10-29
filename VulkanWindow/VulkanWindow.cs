using Silk.NET.Core;
using Silk.NET.Core.Native;
using Silk.NET.Input;
using Silk.NET.Maths;
using Silk.NET.Vulkan;
using Silk.NET.Vulkan.Extensions.EXT;
using Silk.NET.Windowing;
using System.Runtime.InteropServices;

namespace VulkanWindow
{
    public class VulkanWindow : IDisposable
    {
        #region Fields
        private IWindow window;
        private Vk vulkan;
        private Instance vulkanInstance;
        private ExtDebugUtils? debugUtils;
        private DebugUtilsMessengerEXT debugMessenger;

        bool EnableValidationLayers = true;

        private readonly string[] validationLayers = new[]
        {
            "VK_LAYER_KHRONOS_validation"
        };
        #endregion

        public VulkanWindow(int width, int height)
        {
            //Create a window.
            var options = WindowOptions.Default;
            options.API = GraphicsAPI.DefaultVulkan;
            options.Size = new Vector2D<int>(width, height);
            options.Title = "test";

            window = Window.Create(options);

            window = Window.Create(options);
            window.Initialize();

            if (window.VkSurface is null)
            {
                throw new Exception("Platform doesn't support Vulkan.");
            }

            CreateInstance();
            SetupDebugMessenger();

            //Assign events.
            window.Load += OnLoad;
            window.Update += OnUpdate;
            window.Render += OnRender;
            window.FramebufferResize += OnFramebufferResize;
        }

        #region Public methods
        public void Display()
        {
            window.Run();
        }

        public void Dispose()
        {
            if (EnableValidationLayers)
            {
                unsafe
                {
                    debugUtils!.DestroyDebugUtilsMessenger(vulkanInstance, debugMessenger, null);
                }
            }

            unsafe
            {
                vulkan!.DestroyInstance(vulkanInstance, null);
            }
            vulkan!.Dispose();

            window?.Dispose();
        }
        #endregion

        #region Private methods
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
                    ApiVersion = Vk.Version11
                };

                var createInfo = new InstanceCreateInfo()
                {
                    SType = StructureType.InstanceCreateInfo,
                    PApplicationInfo = &appInfo
                };

                var extensions = GetRequiredExtensions();
                createInfo.EnabledExtensionCount = (uint)extensions.Length;
                createInfo.PpEnabledExtensionNames = (byte**)SilkMarshal.StringArrayToPtr(extensions); ;

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

        #region Debugger
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

        private unsafe uint DebugCallback(DebugUtilsMessageSeverityFlagsEXT messageSeverity, DebugUtilsMessageTypeFlagsEXT messageTypes, DebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData)
        {
            Console.WriteLine($"validation layer:" + Marshal.PtrToStringAnsi((nint)pCallbackData->PMessage));

            return Vk.False;
        }
        #endregion
        #endregion

        #region Events
        public void OnLoad()
        {
            //Set-up input context.
            IInputContext input = window.CreateInput();
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
                window.Close();
            }
        }
        #endregion
    }
}
