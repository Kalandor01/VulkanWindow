namespace VulkanWindow
{
    internal class Program
    {
#if DEBUG
        private const bool debug = true;
#else
        private const bool debug = false;
#endif

        private static void Main(string[] args)
        {
            var window = new VulkanWindow(800, 600, debug);
            window.Display();
            window.Dispose();
        }
    }
}
