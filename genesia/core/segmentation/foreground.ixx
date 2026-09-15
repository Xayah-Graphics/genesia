export module genesia.segmentation.foreground;
export import genesia.data.datasets;
export import genesia.models.birefnet;
import std;

export namespace genesia::foreground {
    struct Result final {
        dataset::File image;
        std::string model_sha;
        std::filesystem::path path;
        bool cached{};
    };
    std::optional<Image> read_cached(const dataset::File& image);
    struct Pipeline final {
        std::unique_ptr<birefnet::Network> network;
        Result infer(const dataset::File& image, std::span<const std::uint8_t> rgb = {});
    };
} // namespace genesia::foreground
