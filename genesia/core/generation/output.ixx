export module genesia.generation.output;

import genesia.sdxl;
import genesia.generation.repaint;
import genesia.prompt;
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
        std::shared_ptr<const RepaintRecord> region;
    };
    struct ImageWriter final {
        std::filesystem::path directory;
        std::uint64_t next_index{1};

        explicit ImageWriter(std::filesystem::path directory);
        std::filesystem::path save(const sdxl::Output& output, const Record& record);
    };
    std::optional<Record> read_record(const std::filesystem::path& path, std::shared_ptr<const prompt::Catalog> catalog);
} // namespace genesia
