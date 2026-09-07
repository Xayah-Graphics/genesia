module;
#include <genesia/cuda.h>
#include <cudnn.h>
#include <nlohmann/json.hpp>
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb_image_resize2.h>
module genesia.generation.output;

import genesia.sdxl;
import genesia.sdxl.tokenizer;
import std;

namespace genesia {
    std::filesystem::path session_directory(const std::filesystem::path& root) {
        return root / std::format("{:%Y%m%d-%H%M%S}", std::chrono::floor<std::chrono::milliseconds>(std::chrono::system_clock::now()));
    }
    void save(const sdxl::Output& output, const Record& record) {
        const auto started = std::chrono::steady_clock::now();
        std::filesystem::create_directories(record.directory);
        const auto image = record.directory / "image.png";
        if (!stbi_write_png(image.string().c_str(), output.width, output.height, 3, output.pixels.data(), output.width * 3)) throw std::runtime_error{"PNG write failed"};
        const std::uint64_t bytes = output.latent.size() * sizeof(float);
        const nlohmann::json tensor{{"latent", {{"dtype", "F32"}, {"shape", {1, 4, output.height / 8, output.width / 8}}, {"data_offsets", {0, bytes}}}}};
        std::string header = tensor.dump();
        header.append((8 - header.size() % 8) % 8, ' ');
        const std::uint64_t size = header.size();
        std::ofstream latent{record.directory / "latent.safetensors", std::ios::binary};
        latent.exceptions(std::ios::badbit | std::ios::failbit);
        latent.write(reinterpret_cast<const char*>(&size), 8);
        latent.write(header.data(), header.size());
        latent.write(reinterpret_cast<const char*>(output.latent.data()), bytes);
        latent.close();
        const auto& p = record.parameters;
        const double save_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        const double generation_to_saved_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - record.generation_started).count();
        nlohmann::json run{
            {"backend", "CUDA"}, {"sampling_execution", "cuda_graph_while"}, {"device", "NVIDIA GeForce RTX 5090"},
            {"seed", record.seed}, {"width", p.width}, {"height", p.height}, {"steps", p.steps}, {"cfg", p.cfg}, {"positive", p.positive}, {"negative", p.negative},
            {"sampler", "euler"}, {"scheduler", "simple"}, {"denoise", 1.0}, {"latent_scale", 0.13025}, {"latent_layout", "NCHW"},
            {"rng", "philox4x32_10_box_muller_v1"}, {"tokenizer", sdxl::Tokenizer::implementation},
            {"precision", {{"clip", "float16"}, {"unet", "float16"}, {"vae", "bfloat16"}, {"sampling", "float32"}}},
            {"toolchain", {{"cuda_runtime", CUDART_VERSION}, {"cudnn", cudnnGetVersion()}, {"architecture", "sm_120a"}}},
            {"cache", {{"hits", record.cache_hits}, {"misses", record.cache_misses}}}, {"device_memory_used_bytes", record.resident_bytes},
            {"timing", {{"load_seconds", record.load_seconds}, {"prepare_seconds", record.prepare_seconds}, {"initialize_seconds", output.initialize_seconds},
                {"sample_seconds", output.sample_seconds}, {"decode_seconds", output.decode_seconds}, {"transfer_seconds", output.transfer_seconds},
                {"save_seconds", save_seconds}, {"generation_to_saved_seconds", generation_to_saved_seconds}}}};
        run["tag_catalog_sha256"] = prompt::Catalog::sha256;
        for (const auto& [name, side] : {std::pair{"positive", &record.prompt.positive}, std::pair{"negative", &record.prompt.negative}}) {
            auto& saved = run["prompt"][name];
            saved["fixed"] = side->fixed;
            saved["groups"] = nlohmann::json::array();
            for (const auto& group : side->groups) {
                nlohmann::json tags = nlohmann::json::array();
                for (const auto tag : group.tags) {
                    const auto& entry = record.catalog->tags[tag.id];
                    tags.push_back({{"name", entry.name}, {"text", entry.text}, {"weight", tag.weight}, {"source", entry.category == -1 ? "custom" : "danbooru"}});
                }
                saved["groups"].push_back({{"name", group.name}, {"enabled", group.enabled}, {"tags", std::move(tags)}});
            }
        }
        std::ofstream report{record.directory / "run.json"};
        report.exceptions(std::ios::badbit | std::ios::failbit);
        report << run.dump(2) << '\n';
        std::println("SAVE {} {:.3f}s", record.directory.string(), save_seconds);
        std::cout.flush();
    }
    Image read_image(const std::filesystem::path& path) {
        Image result;
        int channels;
        auto* pixels = stbi_load(path.string().c_str(), &result.width, &result.height, &channels, 3);
        if (!pixels) throw std::runtime_error{stbi_failure_reason()};
        result.pixels.assign(pixels, pixels + std::size_t(result.width) * result.height * 3);
        stbi_image_free(pixels);
        return result;
    }
    Image thumbnail(const sdxl::Output& output) {
        Image result{192, std::max(1, output.height * 192 / output.width)};
        result.pixels.resize(std::size_t(result.width) * result.height * 3);
        stbir_resize_uint8_srgb(output.pixels.data(), output.width, output.height, 0, result.pixels.data(), result.width, result.height, 0, STBIR_RGB);
        return result;
    }
}
