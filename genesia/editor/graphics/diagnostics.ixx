export module genesia.editor.graphics.diagnostics;
import std;

export namespace genesia::editor::graphics {
    struct Diagnostics final {
        const std::filesystem::path directory;
        std::atomic_bool lost{};
        std::array<std::vector<std::uint32_t>, 2> shaders;

        Diagnostics();
        ~Diagnostics();
        void capture(std::string_view reason);

    private:
        std::mutex mutex;
        std::exception_ptr callback_error;
        std::uint64_t files{};
        void save(std::string_view extension, const void* data, std::uint32_t size) noexcept;
    };
} // namespace genesia::editor::graphics
