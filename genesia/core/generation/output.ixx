export module genesia.generation.output;

import genesia.sdxl;
import genesia.prompt;
export import classifier.inference;
import std;

export namespace genesia {
    struct Record final {
        sdxl::Parameters parameters;
        std::uint64_t seed{};
        std::filesystem::path path;
        std::filesystem::path model;
        prompt::Pair prompt;
        std::shared_ptr<const prompt::Catalog> catalog;
        std::filesystem::path source;
        classifier::Results classification;
    };
    std::filesystem::path save_image(const sdxl::Output& output, const Record& record);
    Record read_record(const std::filesystem::path& path, std::shared_ptr<const prompt::Catalog> catalog);
} // namespace genesia
