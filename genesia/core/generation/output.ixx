export module genesia.generation.output;

import genesia.models.sdxl;
export import genesia.data.datasets;
import std;

export namespace genesia {
    dataset::File save_image(dataset::Index& index, const sdxl::Output& output, const Record& record);
} // namespace genesia
