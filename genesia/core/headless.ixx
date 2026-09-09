export module genesia.headless;

export import genesia.generation.defaults;
export import genesia.prompt.preset;
export import genesia.generation.repaint;
import std;

export namespace genesia::headless {
    struct Options final {
        int count{1};
        std::optional<std::uint64_t> first_seed;
        std::filesystem::path source;
        std::filesystem::path mask;
        std::optional<std::array<int, 4>> region;
        std::optional<float> denoise;
        RepaintSettings repaint;
    };

    void run(const std::optional<prompt::Preset>& preset, std::shared_ptr<const prompt::Catalog> catalog, const Options& options);
}
