export module genesia.generation.configuration;

import genesia.sdxl;
import std;

export namespace genesia {
    struct Configuration final {
        std::filesystem::path checkpoint;
        std::filesystem::path output;
        std::filesystem::path cache;
        sdxl::Parameters parameters;
        std::vector<std::uint64_t> seeds;
        int warmup{};
    };
    Configuration read_configuration(const std::filesystem::path& path);
    void write_configuration(const Configuration& configuration, const std::filesystem::path& path);
}
