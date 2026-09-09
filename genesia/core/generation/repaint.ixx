module;
#include "repaint-kernels.h"
#include <genesia/cuda.h>

export module genesia.generation.repaint;
import genesia.generation.images;
import genesia.sdxl;
import std;

export namespace genesia {
    struct RepaintSettings final {
        int context{96};
        int work_size{1024};
        int feather{8};
    };

    struct RepaintRecord final {
        RepaintSettings settings;
        std::array<int, 4> crop{};
        std::array<int, 2> resized{};
        std::array<int, 2> work{};
        // Original-resolution binary mask: [start pixel, run length], row-major.
        std::vector<std::array<std::uint32_t, 2>> mask_runs;
    };

    struct Repaint final {
        std::shared_ptr<const RepaintRecord> record;
        std::unique_ptr<sdxl::ImageInput> source;
        ::cuda::device_buffer<std::uint8_t> mask;

        Repaint(::cuda::stream_ref stream, const Image& original, std::span<const std::uint8_t> mask, RepaintSettings settings);
        const sdxl::Output& complete(const sdxl::Output* work = nullptr);

    private:
        ::cuda::stream_ref stream;
        repaint_kernels::Geometry geometry{};
        ::cuda::device_buffer<std::uint8_t> original;
        ::cuda::device_buffer<std::uint8_t> image;
        ::cuda::device_buffer<float> alpha;
        ::cuda::device_buffer<float> scratch;
        std::array<sdxl::Output, 2> outputs;
        std::size_t output_index{};
    };
} // namespace genesia
