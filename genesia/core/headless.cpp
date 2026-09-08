module;

#include "sdxl/control.h"
#include <genesia/cuda.h>

module genesia.headless;

import genesia.generation.output;
import genesia.sdxl;
import std;

namespace genesia::headless {
    void run(const prompt::Preset& preset, const std::shared_ptr<const prompt::Catalog>& catalog) {
        const sdxl::Parameters parameters{prompt::compose(*catalog, preset.prompt.positive), prompt::compose(*catalog, preset.prompt.negative)};
        const auto started = std::chrono::steady_clock::now();
        ::cuda::stream stream{::cuda::devices[0]};
        ::cuda::host_buffer<sdxl::Control> control{stream, ::cuda::pinned_default_memory_pool(), 1, ::cuda::no_init};
        std::construct_at(control.data());
        std::println("LOAD SDXL / RTX 5090 / CUDA 13.3");
        std::cout.flush();
        sdxl::Model model{stream, defaults::checkpoint, defaults::cache};
        const double load_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        sdxl::Inference inference{model, parameters, control.data()[0]};
        std::println("READY load={:.3f}s prepare={:.3f}s cache={}/{} memory={:.2f}GiB", load_seconds, inference.prepare_seconds, inference.cache_hits, inference.cache_misses, inference.resident_bytes / double(1ull << 30));
        std::cout.flush();
        for (int i = 0; i < defaults::warmup; ++i) inference.generate(defaults::seeds.front());
        ImageWriter images{defaults::output};
        std::future<void> pending_save;
        for (const auto seed : defaults::seeds) {
            const auto& result = inference.generate(seed);
            if (pending_save.valid()) pending_save.get();
            const Record record{parameters, seed, {}, std::filesystem::path{defaults::checkpoint}.filename(), preset.prompt, catalog};
            std::println("GENERATE seed={} sample={:.3f}s decode={:.3f}s", seed, result.sample_seconds, result.decode_seconds);
            std::cout.flush();
            pending_save = std::async(std::launch::async, [&images, &result, record] { images.save(result, record); });
        }
        if (pending_save.valid()) pending_save.get();
    }
} // namespace genesia::headless
