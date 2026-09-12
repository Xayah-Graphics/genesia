module;

#include "sdxl/control.h"
#include <genesia/cuda.h>

#include <nlohmann/json.hpp>

module genesia.headless;

import genesia.generation.output;
import genesia.generation.images;
import genesia.sdxl;
import std;

namespace genesia::headless {
    void run(const std::optional<prompt::Preset>& preset, std::shared_ptr<const prompt::Catalog> catalog, const Options& options) {
        sdxl::Parameters parameters;
        prompt::Pair prompt;
        Image original;
        if (!options.source.empty()) {
            original         = read_image(options.source);
            const auto saved = read_record(options.source, catalog);
            if (saved) {
                parameters = saved->parameters;
                prompt     = saved->prompt;
                catalog    = saved->catalog;
            } else if (!preset) throw std::runtime_error{"Source PNG has no Genesia prompt metadata; specify --preset or --prompt-file"};
            parameters.width   = original.width;
            parameters.height  = original.height;
            parameters.denoise = *options.denoise;
        }
        if (preset) {
            prompt              = preset->prompt;
            parameters.positive = prompt::compose(*catalog, prompt.positive);
            parameters.negative = prompt::compose(*catalog, prompt.negative);
        }
        ::cuda::stream stream{::cuda::devices[0]};
        classifier::Pipeline classifiers;
        const classifier::Selection selection{options.classifiers, options.discard_failed};
        classifiers.prepare(classifier::discover(), selection, parameters.width, parameters.height);
        ::cuda::host_buffer<sdxl::Control> control{stream, ::cuda::pinned_default_memory_pool(), 1, ::cuda::no_init};
        std::construct_at(control.data());
        std::unique_ptr<sdxl::ImageInput> source;
        std::unique_ptr<sdxl::Model> model;
        std::unique_ptr<sdxl::Inference> inference;
        std::optional<sdxl::Output> unchanged;
        if (parameters.denoise > 0) {
            if (!options.source.empty()) {
                if (original.width % 64 || original.height % 64) throw std::invalid_argument{"Whole-image Repaint needs dimensions divisible by 64"};
                source = std::make_unique<sdxl::ImageInput>(stream, original.pixels, original.width, original.height);
            }
            const auto started = std::chrono::steady_clock::now();
            std::println(std::cerr, "LOAD SDXL / RTX 5090 / CUDA 13.3");
            model                     = std::make_unique<sdxl::Model>(stream, defaults::checkpoint, defaults::cache);
            const double load_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
            if (source) {
                const auto encoding = std::chrono::steady_clock::now();
                model->encode(*source);
                std::println(std::cerr, "ENCODE {:.3f}s", std::chrono::duration<double>(std::chrono::steady_clock::now() - encoding).count());
            }
            inference = std::make_unique<sdxl::Inference>(*model, parameters, control.data()[0], nullptr, source.get());
            std::println(std::cerr, "READY load={:.3f}s prepare={:.3f}s cache={}/{} memory={:.2f}GiB", load_seconds, inference->prepare_seconds, inference->cache_hits, inference->cache_misses, inference->resident_bytes / double(1ull << 30));
        } else {
            unchanged.emplace(stream, original.width, original.height);
            std::ranges::copy(original.pixels, unchanged->pixels.data());
            if (!selection.enabled.empty()) {
                source                   = std::make_unique<sdxl::ImageInput>(stream, original.pixels, original.width, original.height);
                unchanged->device_pixels = source->pixels.data();
            }
        }
        ImageWriter images{defaults::output};
        std::future<void> pending_save;
        std::random_device random;
        std::uniform_int_distribution<std::uint64_t> seeds;
        for (int i = 0; i < options.count; ++i) {
            const auto seed    = options.first_seed ? *options.first_seed + std::uint64_t(i) : seeds(random);
            const auto& result = inference ? inference->generate(seed) : *unchanged;
            // The two pinned outputs alternate; finish the previous save before
            // its storage can be reused by the next generate() call.
            if (pending_save.valid()) pending_save.get();
            Record record{parameters, seed, {}, std::filesystem::path{defaults::checkpoint}.filename(), prompt, catalog, options.source.empty() ? std::filesystem::path{} : std::filesystem::absolute(options.source).lexically_normal()};
            if (!selection.enabled.empty()) record.classification = classifiers.run({result.device_pixels, result.width, result.height, std::size_t(result.width) * 3}, result.stream.get());
            record.discarded = selection.discard_failed && record.classification.error.empty() && !record.classification.passed;
            std::println(std::cerr, "GENERATE {}/{} seed={} sample={:.3f}s decode={:.3f}s", i + 1, options.count, seed, result.sample_seconds, result.decode_seconds);
            pending_save = std::async(std::launch::async, [&images, &result, record, i, count = options.count] {
                nlohmann::json output{{"index", i + 1}, {"total", count}, {"seed", std::to_string(record.seed)}, {"path", nullptr}};
                if (!record.discarded) {
                    const auto path = std::filesystem::absolute(images.save(result, record)).lexically_normal().generic_u8string();
                    output["path"]  = std::string{path.begin(), path.end()};
                }
                if (!record.classification.classifiers.empty()) {
                    output["classification"] = record.classification;
                    output["saved"]          = !record.discarded;
                    output["discarded"]      = record.discarded;
                }
                std::println("{}", output.dump());
                std::cout.flush();
            });
            if (!record.classification.error.empty()) {
                pending_save.get();
                throw std::runtime_error{record.classification.error};
            }
        }
        if (pending_save.valid()) pending_save.get();
    }
} // namespace genesia::headless
