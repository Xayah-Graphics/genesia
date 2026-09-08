export module genesia.editor.ui.renderer;
import genesia.editor.platform.window;
import genesia.editor.runtime.device;
import genesia.editor.runtime.resources;
import genesia.generation.output;
import std;
import vulkan;

export namespace genesia::editor {
    struct Renderer final {
        WindowPlatform& window;
        runtime::Instance instance;
        vk::raii::SurfaceKHR surface{nullptr};
        runtime::Device device;
        runtime::Resources resources;
        vk::Extent2D extent{};
        float dpi{1};
        bool visible{true};
        int hand_cursor{-1};
        double last_frame_seconds{};

        explicit Renderer(WindowPlatform& window);
        ~Renderer();
        bool begin();
        void present();
        std::uint64_t texture(vk::Extent2D extent, vk::Format format = vk::Format::eR8G8B8A8Srgb);
        std::uint64_t upload(const Image& image);
        void upload(std::uint64_t id, const void* rgba, int width, int height, bool initial);
        void copy(std::uint64_t id, const runtime::Buffer& source, vk::Semaphore semaphore, std::uint64_t ready);
        void discard(vk::Semaphore semaphore, std::uint64_t ready);
        void retire(std::uint64_t id);

    private:
        struct Texture final {
            runtime::Image image;
            bool initialized{};
        };
        struct Frame final {
            runtime::Buffer vertices;
            runtime::Buffer indices;
            vk::raii::Semaphore available{nullptr};
            vk::raii::Fence finished{nullptr};
            std::vector<runtime::Buffer> uploads;
            std::vector<Texture> retired;
            std::vector<std::uint32_t> recycled;
        };
        std::array<Frame, 2> frames;
        vk::raii::CommandPool pool{nullptr};
        vk::raii::CommandBuffers commands{nullptr};
        vk::raii::SwapchainKHR swapchain{nullptr};
        std::vector<vk::Image> images;
        std::vector<vk::raii::ImageView> views;
        std::vector<vk::raii::Semaphore> rendered;
        vk::raii::ShaderEXTs shaders{nullptr};
        std::map<std::uint64_t, Texture> textures;
        std::vector<std::uint32_t> free_descriptors;
        std::vector<vk::SemaphoreSubmitInfo> waits;
        std::vector<vk::SemaphoreSubmitInfo> signals;
        std::size_t frame_index{};
        std::uint32_t image_index{};
        std::chrono::steady_clock::time_point frame_started;

        void recreate();
        void update_fonts();
        void draw();
    };
} // namespace genesia::editor
