module;
#include <httplib.h>
#include <nlohmann/json.hpp>
module edit.processing;
import tools.files;
import std;
namespace edit {
    namespace {
        struct Stopped final {};
        nlohmann::json workflow(const std::string& image, const std::string& prompt, const std::uint64_t seed, const std::string& prefix) {
            nlohmann::json graph;
            graph["image"]  = {{"class_type", "LoadImage"}, {"inputs", {{"image", image}}}};
            graph["model"]  = {{"class_type", "UNETLoader"}, {"inputs", {{"unet_name", "qwen_image_2.1_int8_convrot.safetensors"}, {"weight_dtype", "default"}}}};
            graph["clip"]   = {{"class_type", "CLIPLoader"}, {"inputs", {{"clip_name", "qwen3vl_8b_int8_convrot.safetensors"}, {"type", "qwen_image"}, {"device", "default"}}}};
            graph["vae"]    = {{"class_type", "VAELoader"}, {"inputs", {{"vae_name", "qwen_image_2.1_vae_bf16.safetensors"}}}};
            graph["cache"]  = {{"class_type", "QwenImage21Cache"}, {"inputs", {{"model", {"model", 0}}, {"device", "gpu"}, {"dtype", "default"}}}};
            graph["encode"] = {{"class_type", "TextEncodeQwenImage21"}, {"inputs", {{"clip", {"clip", 0}}, {"vae", {"vae", 0}}, {"images.image_1", {"image", 0}}, {"prompt", prompt}, {"negative_prompt", ""}, {"resolution", 0}}}};
            graph["sample"] = {{"class_type", "KSampler"}, {"inputs", {{"model", {"cache", 0}}, {"positive", {"encode", 0}}, {"negative", {"encode", 1}}, {"latent_image", {"encode", 2}}, {"seed", seed}, {"steps", 25}, {"cfg", 1.0}, {"sampler_name", "euler"}, {"scheduler", "simple"}, {"denoise", 1.0}}}};
            graph["decode"] = {{"class_type", "VAEDecode"}, {"inputs", {{"samples", {"sample", 0}}, {"vae", {"vae", 0}}}}};
            graph["save"]   = {{"class_type", "SaveImageAdvanced"}, {"inputs", {{"images", {"decode", 0}}, {"filename_prefix", prefix}, {"format", "png"}, {"format.bit_depth", "8-bit"}, {"format.input_color_space", "sRGB"}}}};
            return graph;
        }
    } // namespace
    bool check_connection() {
        httplib::Client client{std::string{server_url}};
        client.set_connection_timeout(1);
        client.set_read_timeout(1);
        client.set_write_timeout(1);
        client.set_max_timeout(std::chrono::seconds{2});
        const auto response = client.Get("/system_stats");
        return response && response->status == 200;
    }
    Result process(const Request& options, const std::atomic_bool& interrupted, const std::function<void(const Progress&)>& progress) {
        Result result;
        httplib::Client client{std::string{server_url}};
        client.set_connection_timeout(5);
        client.set_read_timeout(10);
        client.set_write_timeout(10);
        const auto request = [&](const std::string& route, const std::optional<nlohmann::json>& body = {}) {
            const auto response = body ? client.Post(route, body->dump(), "application/json") : client.Get(route);
            if (!response) throw std::runtime_error{std::format("ComfyUI {}: {}", route, httplib::to_string(response.error()))};
            if (response->status != 200) throw std::runtime_error{std::format("ComfyUI {}: HTTP {}\n{}", route, response->status, response->body)};
            return response->body.empty() ? nlohmann::json{} : nlohmann::json::parse(response->body);
        };
        std::filesystem::path temporary_input, temporary_output, pending_output, current_file;
        std::string prompt_id;
        std::vector<std::string> submitted;
        bool output_created{};
        try {
            progress({Stage::preparing});
            if (interrupted) throw Stopped{};
            if (options.prompt.find_first_not_of(" \t\r\n") == std::string::npos) throw std::runtime_error{"Enter a positive prompt before editing images"};
            const auto input     = std::filesystem::canonical(options.input);
            const bool directory = std::filesystem::is_directory(input);
            result.output        = (directory ? input : input.parent_path()) / "fix";
            std::vector<std::filesystem::path> images;
            if (directory) {
                for (const auto& entry : std::filesystem::directory_iterator{input}) {
                    if (interrupted) throw Stopped{};
                    if (!entry.is_regular_file()) continue;
                    auto extension = tools::files::utf8(entry.path().extension());
                    std::ranges::transform(extension, extension.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
                    if (extension == ".png") images.push_back(entry.path());
                }
            } else {
                auto extension = tools::files::utf8(input.extension());
                std::ranges::transform(extension, extension.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
                if (extension != ".png") throw std::runtime_error{"Edit accepts PNG images"};
                images.push_back(input);
            }
            if (images.empty()) throw std::runtime_error{"No PNG images directly inside " + tools::files::utf8(input)};
            std::ranges::sort(images);
            result.total = images.size();
            progress({Stage::preparing, 0, result.total});
            const auto stats      = request("/system_stats");
            const auto arguments  = stats.at("system").at("argv").get<std::vector<std::string>>();
            const auto input_arg  = std::ranges::find(arguments, "--input-directory");
            const auto output_arg = std::ranges::find(arguments, "--output-directory");
            if (input_arg == arguments.end() || std::next(input_arg) == arguments.end() || output_arg == arguments.end() || std::next(output_arg) == arguments.end()) throw std::runtime_error{"Start local ComfyUI with --input-directory and --output-directory"};
            const auto input_directory  = tools::files::path(*std::next(input_arg));
            const auto output_directory = tools::files::path(*std::next(output_arg));
            if (interrupted) throw Stopped{};
            std::random_device random;
            const auto batch        = std::format("edit-{:016x}-{:016x}", std::chrono::system_clock::now().time_since_epoch().count(), std::uniform_int_distribution<std::uint64_t>{}(random));
            const auto batch_input  = input_directory / batch;
            const auto batch_output = output_directory / batch;
            if (!std::filesystem::create_directory(batch_input)) throw std::runtime_error{"Temporary directory already exists: " + tools::files::utf8(batch_input)};
            temporary_input = batch_input;
            if (!std::filesystem::create_directory(batch_output)) throw std::runtime_error{"Temporary directory already exists: " + tools::files::utf8(batch_output)};
            temporary_output = batch_output;
            for (std::size_t i = 0; i < images.size(); ++i) {
                if (interrupted) throw Stopped{};
                current_file = images[i];
                progress({Stage::editing, result.completed, result.total, current_file});
                const auto name   = std::format("{:06}.png", i);
                const auto source = temporary_input / name;
                std::filesystem::copy_file(current_file, source);
                const auto seed  = std::uniform_int_distribution<std::uint64_t>{}(random);
                const auto graph = workflow(batch + "/" + name, options.prompt, seed, batch + "/edited");
                const auto high  = std::uniform_int_distribution<std::uint64_t>{}(random);
                const auto low   = std::uniform_int_distribution<std::uint64_t>{}(random);
                if (interrupted) throw Stopped{};
                // Retain the ID before submission so a lost reply cannot leave an untracked job.
                prompt_id = std::format("{:08x}-{:04x}-{:04x}-{:04x}-{:012x}", high >> 32, (high >> 16) & 0xffff, (high & 0x0fff) | 0x4000, (low >> 48 & 0x3fff) | 0x8000, low & 0xffffffffffff);
                submitted.push_back(prompt_id);
                request("/prompt", nlohmann::json{{"prompt_id", prompt_id}, {"prompt", graph}});
                nlohmann::json completed;
                for (;;) {
                    if (interrupted) throw Stopped{};
                    const auto history = request("/history/" + prompt_id);
                    if (history.contains(prompt_id)) {
                        completed = history.at(prompt_id);
                        prompt_id.clear();
                        break;
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds{100});
                }
                if (completed.at("status").at("status_str") != "success") throw std::runtime_error{completed.at("status").at("messages").dump(2)};
                if (interrupted) throw Stopped{};
                const auto& saved    = completed.at("outputs").at("save").at("images").at(0);
                const auto generated = output_directory / tools::files::path(saved.at("subfolder").get<std::string>()) / tools::files::path(saved.at("filename").get<std::string>());
                output_created |= std::filesystem::create_directories(result.output);
                pending_output = result.output / tools::files::path(batch + ".part");
                std::filesystem::copy_file(generated, pending_output);
                tools::files::publish(pending_output, result.output / current_file.filename());
                pending_output.clear();
                ++result.completed;
                std::filesystem::remove(source);
                std::filesystem::remove(generated);
                progress({Stage::editing, result.completed, result.total, current_file});
            }
        } catch (const Stopped&) {
            result.stopped = true;
        } catch (const std::exception& failure) {
            result.error = current_file.empty() ? failure.what() : std::format("{}\n{}", tools::files::utf8(current_file), failure.what());
        }
        if (!prompt_id.empty()) {
            try {
                progress({Stage::stopping, result.completed, result.total, current_file});
                request("/api/jobs/" + prompt_id + "/cancel", nlohmann::json::object());
                const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{60};
                for (;;) {
                    const auto queue = request("/queue");
                    bool pending{};
                    for (const auto name : {"queue_running", "queue_pending"})
                        for (const auto& item : queue.at(name)) pending |= item.at(1) == prompt_id;
                    if (!pending) break;
                    if (std::chrono::steady_clock::now() >= deadline) throw std::runtime_error{"ComfyUI has not stopped task " + prompt_id};
                    std::this_thread::sleep_for(std::chrono::milliseconds{100});
                }
                prompt_id.clear();
            } catch (const std::exception& failure) {
                if (!result.error.empty()) result.error += '\n';
                result.error += std::format("{}\nTemporary input retained: {}\nTemporary output retained: {}", failure.what(), tools::files::utf8(temporary_input), tools::files::utf8(temporary_output));
            }
        }
        // Never remove files that a server job may still be reading or writing.
        for (const auto& path : {pending_output, prompt_id.empty() ? temporary_input : std::filesystem::path{}, prompt_id.empty() ? temporary_output : std::filesystem::path{}}) {
            if (path.empty()) continue;
            try {
                std::filesystem::remove_all(path);
            } catch (const std::exception& failure) {
                if (!result.error.empty()) result.error += '\n';
                result.error += failure.what();
            }
        }
        if (output_created && !result.completed) {
            try {
                std::filesystem::remove(result.output);
            } catch (const std::exception& failure) {
                if (!result.error.empty()) result.error += '\n';
                result.error += failure.what();
            }
        }
        if (prompt_id.empty() && !submitted.empty()) {
            try {
                request("/history", nlohmann::json{{"delete", submitted}});
            } catch (const std::exception& failure) {
                if (!result.error.empty()) result.error += '\n';
                result.error += failure.what();
            }
        }
        return result;
    }
} // namespace edit
