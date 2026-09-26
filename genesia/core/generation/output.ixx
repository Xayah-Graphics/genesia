export module genesia.generation.output;

import genesia.models.sdxl;
export import genesia.data.images;
import std;

export namespace genesia {
    std::filesystem::path save_image(const sdxl::Output& output, const Record& record);
} // namespace genesia
