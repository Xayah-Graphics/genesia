module;
#include <httplib.h>
#include <nlohmann/json.hpp>
module genesia.generation.qwen_edit;
import genesia.io.files;
import genesia.io.safetensors;
import genesia.project;
import std;

namespace genesia::qwen {
    namespace {
        nlohmann::json workflow(const std::string& image, const std::string& prompt, const std::uint64_t seed, const std::string& prefix) {
            // Qwen Image 2.1 Edit: retain the original subgraph's node IDs and settings.
            nlohmann::json graph;
            graph["509"]     = {{"class_type", "LoadImage"}, {"inputs", {{"image", image}}}};
            graph["459:451"] = {{"class_type", "UNETLoader"}, {"inputs", {{"unet_name", "qwen_image_2.1_int8_convrot.safetensors"}, {"weight_dtype", "default"}}}};
            graph["459:453"] = {{"class_type", "CLIPLoader"}, {"inputs", {{"clip_name", "qwen3vl_8b_int8_convrot.safetensors"}, {"type", "qwen_image"}, {"device", "default"}}}};
            graph["459:454"] = {{"class_type", "VAELoader"}, {"inputs", {{"vae_name", "qwen_image_2.1_vae_bf16.safetensors"}}}};
            graph["459:456"] = {{"class_type", "EmptyLatentImage"}, {"inputs", {{"width", 1024}, {"height", 1536}, {"batch_size", 1}}}};
            graph["459:469"] = {{"class_type", "QwenImage21Cache"}, {"inputs", {{"model", {"459:451", 0}}, {"device", "gpu"}, {"dtype", "default"}}}};
            graph["459:474"] = {{"class_type", "TextEncodeQwenImage21"}, {"inputs", {{"clip", {"459:453", 0}}, {"vae", {"459:454", 0}}, {"images.image_1", {"509", 0}}, {"prompt", prompt}, {"negative_prompt", ""}, {"resolution", 0}}}};
            graph["459:468"] = {{"class_type", "ComfySwitchNode"}, {"inputs", {{"switch", false}, {"on_false", {"459:474", 2}}, {"on_true", {"459:456", 0}}}}};
            graph["459:458"] = {{"class_type", "KSampler"}, {"inputs", {{"model", {"459:469", 0}}, {"positive", {"459:474", 0}}, {"negative", {"459:474", 1}}, {"latent_image", {"459:468", 0}}, {"seed", seed}, {"steps", 25}, {"cfg", 1.0}, {"sampler_name", "euler"}, {"scheduler", "simple"}, {"denoise", 1.0}}}};
            graph["459:457"] = {{"class_type", "VAEDecode"}, {"inputs", {{"samples", {"459:458", 0}}, {"vae", {"459:454", 0}}}}};
            graph["461"]     = {{"class_type", "SaveImageAdvanced"}, {"inputs", {{"images", {"459:457", 0}}, {"filename_prefix", prefix}, {"format", "png"}, {"format.bit_depth", "8-bit"}, {"format.input_color_space", "sRGB"}}}};
            return graph;
        }
    } // namespace

    Profile load_profile(const models::Descriptor& model) {
        const files::SafeFile file{model.path};
        Profile result;
        result.classes  = nlohmann::json::parse(file.header.at("__metadata__").at("classes").get<std::string>()).get<std::vector<std::string>>();
        const auto path = project::directory / files::path(model.id) / ".genesia" / "fix-image.json";
        if (std::filesystem::exists(path)) result.prompts = files::read_json(path).at("prompts").get<std::map<std::string, std::string>>();
        for (const auto& label : result.classes) result.prompts.try_emplace(label);
        return result;
    }

    void save_profile(const std::string_view key, const Profile& profile) {
        files::write_json(project::directory / files::path(key) / ".genesia" / "fix-image.json", {{"prompts", profile.prompts}});
    }

    Result fix_images(dataset::Index& index, const std::string_view key, const std::filesystem::path& input, const std::map<std::string, std::string>& prompts, const std::atomic_bool& interrupted, const std::function<void(const runtime::Progress&)>& progress) {
        const auto root = std::filesystem::absolute(input).lexically_normal();
        Result result{.output = root / "fixed"};
        httplib::Client client{std::string{project::comfy_server}};
        client.set_connection_timeout(5);
        client.set_read_timeout(10);
        client.set_write_timeout(10);
        const auto request = [&](const std::string& route, const std::optional<nlohmann::json>& body = {}) {
            const auto response = body ? client.Post(route, body->dump(), "application/json") : client.Get(route);
            if (!response) throw std::runtime_error{std::format("ComfyUI {}: {}", route, httplib::to_string(response.error()))};
            if (response->status != 200) throw std::runtime_error{std::format("ComfyUI {}: HTTP {}\n{}", route, response->status, response->body)};
            return response->body.empty() ? nlohmann::json{} : nlohmann::json::parse(response->body);
        };
        std::filesystem::path temporary_input, temporary_output, pending_output;
        std::string prompt_id, current_file, current_label;
        bool connected{}, output_created{};
        std::vector<std::string> submitted;
        const auto cancel = [&] {
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
        };
        try {
            progress({runtime::BatchProgress{runtime::Stage::preparing}});
            const auto stats      = request("/system_stats");
            connected             = true;
            const auto arguments  = stats.at("system").at("argv").get<std::vector<std::string>>();
            const auto input_arg  = std::ranges::find(arguments, "--input-directory");
            const auto output_arg = std::ranges::find(arguments, "--output-directory");
            if (input_arg == arguments.end() || std::next(input_arg) == arguments.end() || output_arg == arguments.end() || std::next(output_arg) == arguments.end()) throw std::runtime_error{"Start local ComfyUI with --input-directory and --output-directory"};
            const auto input_directory  = files::path(*std::next(input_arg));
            const auto output_directory = files::path(*std::next(output_arg));
            request("/free", nlohmann::json{{"unload_models", true}, {"free_memory", true}});
            if (interrupted) throw runtime::Stopped{};
            std::vector<dataset::File> images;
            for (const auto& entry : std::filesystem::directory_iterator{root}) {
                if (interrupted) throw runtime::Stopped{};
                if (!entry.is_regular_file()) continue;
                auto extension = files::utf8(entry.path().extension());
                std::ranges::transform(extension, extension.begin(), [](const unsigned char c) { return static_cast<char>(std::tolower(c)); });
                if (extension == ".png") images.push_back(index.identify(entry.path()));
            }
            if (images.empty()) throw std::runtime_error{"No PNG images directly inside " + files::utf8(root)};
            std::ranges::sort(images, {}, &dataset::File::path);
            result.total     = images.size();
            const auto model = models::resolve(key, dataset::ConceptType::classifier);
            std::vector<std::string> labels;
            {
                classification::Predictions predictions;
                for (const auto& image : images) {
                    if (interrupted) throw runtime::Stopped{};
                    current_file = files::utf8(image.path);
                    current_label.clear();
                    const auto prediction = predictions.infer(model, image);
                    current_label         = prediction.label;
                    const auto found      = prompts.find(prediction.label);
                    if (found == prompts.end() || found->second.find_first_not_of(" \t\r\n") == std::string::npos) throw std::runtime_error{"Missing Edit prompt for label: " + prediction.label};
                    labels.push_back(prediction.label);
                    progress({runtime::BatchProgress{runtime::Stage::classifying, labels.size(), images.size(), current_file, current_label}});
                }
            }
            std::random_device random;
            const auto batch        = std::format("genesia-fix-{:016x}-{:016x}", std::chrono::system_clock::now().time_since_epoch().count(), std::uniform_int_distribution<std::uint64_t>{}(random));
            const auto batch_input  = input_directory / batch;
            const auto batch_output = output_directory / batch;
            if (!std::filesystem::create_directory(batch_input)) throw std::runtime_error{"Temporary directory already exists: " + files::utf8(batch_input)};
            temporary_input = batch_input;
            if (!std::filesystem::create_directory(batch_output)) throw std::runtime_error{"Temporary directory already exists: " + files::utf8(batch_output)};
            temporary_output = batch_output;
            for (std::size_t i = 0; i < images.size(); ++i) {
                if (interrupted) throw runtime::Stopped{};
                current_file  = files::utf8(images[i].path);
                current_label = labels[i];
                progress({runtime::BatchProgress{runtime::Stage::editing, result.completed, result.total, current_file, current_label}});
                const auto source = temporary_input / "input.png";
                std::filesystem::copy_file(images[i].path, source);
                const auto seed  = std::uniform_int_distribution<std::uint64_t>{}(random);
                const auto graph = workflow(batch + "/input.png", prompts.at(current_label), seed, batch + "/edited");
                const auto high  = std::uniform_int_distribution<std::uint64_t>{}(random);
                const auto low   = std::uniform_int_distribution<std::uint64_t>{}(random);
                // Keep the ID before submitting so a lost HTTP reply can still be cancelled.
                prompt_id = std::format("{:08x}-{:04x}-{:04x}-{:04x}-{:012x}", high >> 32, (high >> 16) & 0xffff, (high & 0x0fff) | 0x4000, (low >> 48 & 0x3fff) | 0x8000, low & 0xffffffffffff);
                submitted.push_back(prompt_id);
                request("/prompt", nlohmann::json{{"prompt_id", prompt_id}, {"prompt", graph}, {"extra_data", {{"extra_pnginfo", {{"genesia_fix", {{"source", current_file}, {"classifier", key}, {"model_sha", model.sha}, {"label", current_label}}}}}}}});
                nlohmann::json completed;
                for (;;) {
                    if (interrupted) throw runtime::Stopped{};
                    const auto history = request("/history/" + prompt_id);
                    if (history.contains(prompt_id)) {
                        completed = history.at(prompt_id);
                        prompt_id.clear();
                        break;
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds{100});
                }
                if (completed.at("status").at("status_str") != "success") throw std::runtime_error{completed.at("status").at("messages").dump(2)};
                if (interrupted) throw runtime::Stopped{};
                const auto& saved      = completed.at("outputs").at("461").at("images").at(0);
                const auto generated   = temporary_output / files::path(saved.at("filename").get<std::string>());
                const auto destination = result.output / images[i].path.filename();
                output_created |= std::filesystem::create_directories(result.output);
                pending_output = result.output / files::path(batch + ".part");
                std::filesystem::copy_file(generated, pending_output);
                files::publish(pending_output, destination);
                pending_output.clear();
                ++result.completed;
                std::filesystem::remove(source);
                std::filesystem::remove(generated);
                progress({runtime::BatchProgress{runtime::Stage::editing, result.completed, result.total, current_file, current_label}});
            }
        } catch (const runtime::Stopped&) {
            result.stopped = true;
        } catch (const std::exception& failure) {
            result.error = failure.what();
            if (!current_file.empty()) result.error = std::format("{}\nLabel: {}\n{}", current_file, current_label, result.error);
        }
        // A cancelled server task must finish before its input/output files can be removed.
        if (!prompt_id.empty()) {
            try {
                cancel();
            } catch (const std::exception& failure) {
                if (!result.error.empty()) result.error += '\n';
                result.error += std::format("{}\nTemporary input retained: {}\nTemporary output retained: {}", failure.what(), files::utf8(temporary_input), files::utf8(temporary_output));
            }
        }
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
        if (connected && prompt_id.empty()) {
            try {
                if (!submitted.empty()) request("/history", nlohmann::json{{"delete", submitted}});
                request("/free", nlohmann::json{{"unload_models", true}, {"free_memory", true}});
            } catch (const std::exception& failure) {
                if (!result.error.empty()) result.error += '\n';
                result.error += failure.what();
            }
        }
        return result;
    }
} // namespace genesia::qwen
