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
        const auto directory = session_directory(configuration.output);
        std::array<std::future<void>, 2> saves;
        for (std::size_t i = 0; i < configuration.seeds.size(); ++i) {
            if (saves[i % 2].valid()) saves[i % 2].get();
            const auto generation_started = std::chrono::steady_clock::now();
            const auto& result = inference.generate(configuration.seeds[i]);
            const Record record{configuration.parameters, result.seed, directory / std::format("{:03}-{}", i, result.seed), load_seconds, inference.prepare_seconds,
                result.sample_seconds, result.decode_seconds, inference.resident_bytes, inference.cache_hits, inference.cache_misses, generation_started};
            std::println("GENERATE seed={} sample={:.3f}s decode={:.3f}s", result.seed, result.sample_seconds, result.decode_seconds);
            std::cout.flush();
            saves[i % 2] = std::async(std::launch::async, [&result, record] { save(result, record); });
        }
        for (auto& save : saves) if (save.valid()) save.get();
    }
}
