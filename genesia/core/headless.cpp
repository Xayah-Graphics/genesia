module;

#include "sdxl/control.h"
#include <genesia/cuda.h>
#include <nlohmann/json.hpp>

module genesia.headless;

import genesia.generation.output;
import genesia.sdxl;
import std;

namespace genesia::headless {
    void run(const prompt::Preset& preset, const std::shared_ptr<const prompt::Catalog>& catalog, const int count, const std::optional<std::uint64_t> first_seed) {
        const sdxl::Parameters parameters{prompt::compose(*catalog, preset.prompt.positive), prompt::compose(*catalog, preset.prompt.negative)};
        const auto started = std::chrono::steady_clock::now();
        ::cuda::stream stream{::cuda::devices[0]};
        ::cuda::host_buffer<sdxl::Control> control{stream, ::cuda::pinned_default_memory_pool(), 1, ::cuda::no_init};
        std::construct_at(control.data());
        std::println(std::cerr, "LOAD SDXL / RTX 5090 / CUDA 13.3");
        sdxl::Model model{stream, defaults::checkpoint, defaults::cache};
        const double load_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        sdxl::Inference inference{model, parameters, control.data()[0]};
        std::println(std::cerr, "READY load={:.3f}s prepare={:.3f}s cache={}/{} memory={:.2f}GiB", load_seconds, inference.prepare_seconds, inference.cache_hits, inference.cache_misses, inference.resident_bytes / double(1ull << 30));
        ImageWriter images{defaults::output};
        std::future<void> pending_save;
        std::random_device random;
        std::uniform_int_distribution<std::uint64_t> seeds;
        for (int i = 0; i < count; ++i) {
            const auto seed = first_seed ? *first_seed + std::uint64_t(i) : seeds(random);
            const auto& result = inference.generate(seed);
            // The two pinned outputs alternate; finish the previous save before
            // its storage can be reused by the next generate() call.
            if (pending_save.valid()) pending_save.get();
            const Record record{parameters, seed, {}, std::filesystem::path{defaults::checkpoint}.filename(), preset.prompt, catalog};
            std::println(std::cerr, "GENERATE {}/{} seed={} sample={:.3f}s decode={:.3f}s", i + 1, count, seed, result.sample_seconds, result.decode_seconds);
            pending_save = std::async(std::launch::async, [&images, &result, record, i, count] {
                const auto path = std::filesystem::absolute(images.save(result, record)).lexically_normal().generic_u8string();
                const nlohmann::json output{{"index", i + 1}, {"total", count}, {"seed", std::to_string(record.seed)}, {"path", std::string{path.begin(), path.end()}}};
                std::println("{}", output.dump());
                std::cout.flush();
            });
        }
        if (pending_save.valid()) pending_save.get();
    }
} // namespace genesia::headless
