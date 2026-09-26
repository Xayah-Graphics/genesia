export module tools.editor.graphics.renderer;
import tools.editor.platform.window;
import tools.editor.graphics.device;
import tools.editor.graphics.resources;
import std;
import vulkan;

export namespace tools::editor {
    struct Renderer final {
        WindowPlatform& window;
        graphics::Instance instance;
        vk::raii::SurfaceKHR surface{nullptr};
        graphics::Device device;
        graphics::Resources resources;
        vk::Extent2D extent{};
        float dpi{1};
        bool visible{true};

        explicit Renderer(WindowPlatform& window);
        ~Renderer();
        bool begin();
        void present();

    private:
        struct Frame final {
            graphics::Buffer vertices;
            graphics::Buffer indices;
            vk::raii::Semaphore available{nullptr};
            vk::raii::Fence finished{nullptr};
            std::vector<graphics::Buffer> uploads;
            std::vector<graphics::Image> retired;
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
        std::map<std::uint64_t, graphics::Image> textures;
        std::vector<std::uint32_t> free_descriptors;
        std::size_t frame_index{};
        std::uint32_t image_index{};

        std::uint64_t texture(vk::Extent2D extent);
        void upload(std::uint64_t id, const void* pixels, int width, int height, bool initial);
        void retire(std::uint64_t id);
        void recreate();
        void update_fonts();
        void draw();
    };
} // namespace tools::editor
