module;
#include "repaint-kernels.h"
#include <genesia/cuda.h>

module genesia.generation.repaint;
import std;

namespace genesia {
    Repaint::Repaint(const ::cuda::stream_ref stream, const Image& input, const std::span<const std::uint8_t> allowed, const RepaintSettings settings)
        : mask{stream, ::cuda::device_default_memory_pool(stream.device())}, stream{stream}, original{stream, ::cuda::device_default_memory_pool(stream.device()), input.pixels.size(), ::cuda::no_init}, image{stream, ::cuda::device_default_memory_pool(stream.device()), input.pixels.size(), ::cuda::no_init}, alpha{stream, ::cuda::device_default_memory_pool(stream.device())}, scratch{stream, ::cuda::device_default_memory_pool(stream.device())}, outputs{sdxl::Output{stream, input.width, input.height}, sdxl::Output{stream, input.width, input.height}} {
        ::cuda::copy_bytes(stream, ::cuda::std::span<const std::uint8_t>{input.pixels.data(), input.pixels.size()}, original);
        auto saved = std::make_shared<RepaintRecord>();
        saved->settings = settings;
        int left = input.width, top = input.height, right = 0, bottom = 0;
        for (std::uint32_t i = 0; i < allowed.size();) {
            if (!allowed[i]) { ++i; continue; }
            const auto start = i;
            while (i < allowed.size() && allowed[i]) {
                left = std::min(left, int(i % input.width));
                right = std::max(right, int(i % input.width) + 1);
                top = std::min(top, int(i / input.width));
                bottom = std::max(bottom, int(i / input.width) + 1);
                ++i;
            }
            saved->mask_runs.push_back({start, i - start});
        }
        record = saved;
        if (saved->mask_runs.empty()) {
            stream.sync();
            return;
        }
        left = std::max(0, left - settings.context);
        top = std::max(0, top - settings.context);
        right = std::min(input.width, right + settings.context);
        bottom = std::min(input.height, bottom + settings.context);
        const int width = right - left, height = bottom - top;
        const double scale = settings.work_size ? double(settings.work_size) / std::max(width, height) : 1.0;
        const int resized_width = std::max(1, int(std::lround(width * scale)));
        const int resized_height = std::max(1, int(std::lround(height * scale)));
        const int work_width = (resized_width + 63) / 64 * 64;
        const int work_height = (resized_height + 63) / 64 * 64;
        geometry = {input.width, input.height, left, top, width, height, resized_width, resized_height, work_width, work_height};
        saved->crop = {left, top, width, height};
        saved->resized = {resized_width, resized_height};
        saved->work = {work_width, work_height};
        source = std::make_unique<sdxl::ImageInput>(stream, work_width, work_height);
        const auto pool = ::cuda::device_default_memory_pool(stream.device());
        mask = ::cuda::device_buffer<std::uint8_t>{stream, pool, std::size_t(work_width / 8) * (work_height / 8), ::cuda::no_init};
        alpha = ::cuda::device_buffer<float>{stream, pool, allowed.size(), ::cuda::no_init};
        scratch = ::cuda::device_buffer<float>{stream, pool, 3uz * std::max(std::size_t(resized_width) * height, std::size_t(width) * resized_height), ::cuda::no_init};
        ::cuda::device_buffer<std::uint8_t> binary{stream, pool, allowed.size(), ::cuda::no_init};
        ::cuda::copy_bytes(stream, ::cuda::std::span<const std::uint8_t>{allowed.data(), allowed.size()}, binary);
        repaint_kernels::prepare(stream, source->pixels.data(), mask.data(), alpha.data(), scratch.data(), original.data(), binary.data(), geometry, settings.feather);
        stream.sync();
        std::println(std::cerr, "REGION crop={},{},{},{} resized={}x{} work={}x{} feather={}", left, top, width, height, resized_width, resized_height, work_width, work_height, settings.feather);
    }

    const sdxl::Output& Repaint::complete(const sdxl::Output* work) {
        const auto started = std::chrono::steady_clock::now();
        auto& result = outputs[output_index++ % 2];
        result.device_pixels = image.data();
        result.sample_seconds = work ? work->sample_seconds : 0;
        result.decode_seconds = work ? work->decode_seconds : 0;
        if (work) repaint_kernels::composite(stream, image.data(), scratch.data(), work->device_pixels, original.data(), alpha.data(), geometry);
        else ::cuda::copy_bytes(stream, original, image);
        ::cuda::copy_bytes(stream, image, result.pixels);
        stream.sync();
        std::println(std::cerr, "COMPOSITE {:.3f}s", std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count());
        return result;
    }
} // namespace genesia
