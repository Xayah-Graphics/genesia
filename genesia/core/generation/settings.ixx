export module genesia.generation.settings;
export import genesia.generation.defaults;
import std;
export namespace genesia::generation {
    struct Settings final {
        std::string positive;
        std::string negative;
        int width{defaults::width};
        int height{defaults::height};
        int steps{defaults::steps};
        float cfg{defaults::cfg};
        float denoise{1};
        bool operator==(const Settings&) const = default;
    };

} // namespace genesia::generation
