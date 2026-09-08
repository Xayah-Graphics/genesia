module;

#include <genesia/cuda.h>
#include "sdxl/control.h"

module genesia.headless;

import genesia.generation.output;
import genesia.sdxl;
import std;

namespace genesia::headless {
    void run(const Configuration& configuration) {
        const auto started = std::chrono::steady_clock::now();
        ::cuda::stream stream{::cuda::devices[0]};
        ::cuda::host_buffer<sdxl::Control> control{stream, ::cuda::pinned_default_memory_pool(), 1, ::cuda::no_init};
        std::construct_at(control.data());
        std::println("LOAD SDXL / RTX 5090 / CUDA 13.3");
        std::cout.flush();
        sdxl::Model model{stream, configuration.checkpoint, configuration.cache};
        const double load_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        sdxl::Inference inference{model, configuration.parameters, control.data()[0]};
        std::println("READY load={:.3f}s prepare={:.3f}s cache={}/{} memory={:.2f}GiB", load_seconds, inference.prepare_seconds, inference.cache_hits, inference.cache_misses, inference.resident_bytes / double(1ull << 30));
        std::cout.flush();
        for (int i = 0; i < configuration.warmup; ++i) inference.generate(configuration.seeds.front());
        ImageWriter images{configuration.output};
        std::future<void> pending_save;
        for (const auto seed : configuration.seeds) {
            const auto& result = inference.generate(seed);
            if (pending_save.valid()) pending_save.get();
            const Record record{configuration.parameters, seed, {}, configuration.checkpoint.filename(), configuration.prompt, configuration.catalog};
            std::println("GENERATE seed={} sample={:.3f}s decode={:.3f}s", seed, result.sample_seconds, result.decode_seconds);
            std::cout.flush();
            pending_save = std::async(std::launch::async, [&images, &result, record] { images.save(result, record); });
        }
        if (pending_save.valid()) pending_save.get();
    }
}
