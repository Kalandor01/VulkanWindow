namespace VulkanWindow
{
    internal class Program
    {
        private static void Main(string[] args)
        {
            var window = new VulkanWindow(800, 600);
            window.Display();
            window.Dispose();
        }
    }
}
