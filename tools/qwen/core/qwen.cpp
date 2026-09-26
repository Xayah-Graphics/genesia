module;
#include <httplib.h>
#include <nlohmann/json.hpp>
module qwen.processing;
import tools.files;
import std;

namespace qwen {
    namespace {
        std::string encode_base64(const std::span<const std::uint8_t> bytes) {
            constexpr std::string_view alphabet{"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"};
            std::string result;
            result.reserve((bytes.size() + 2) / 3 * 4);
            for (std::size_t i = 0; i < bytes.size(); i += 3) {
                const std::uint32_t value = (std::uint32_t(bytes[i]) << 16) | (i + 1 < bytes.size() ? std::uint32_t(bytes[i + 1]) << 8 : 0) | (i + 2 < bytes.size() ? bytes[i + 2] : 0);
                result.push_back(alphabet[(value >> 18) & 63]);
                result.push_back(alphabet[(value >> 12) & 63]);
                result.push_back(i + 1 < bytes.size() ? alphabet[(value >> 6) & 63] : '=');
                result.push_back(i + 2 < bytes.size() ? alphabet[value & 63] : '=');
            }
            return result;
        }
        std::string classify(const std::filesystem::path& file, const Request& request) {
            httplib::Client client{std::string{server_url}};
            client.set_connection_timeout(5);
            client.set_read_timeout(600);
            client.set_write_timeout(30);
            const auto bytes = tools::files::read_bytes(file);
            const auto image = "data:image/png;base64," + encode_base64(bytes);
            const nlohmann::json schema{{"type", "object"}, {"properties", {{"result", {{"type", "string"}, {"enum", request.choices}}}}}, {"required", {"result"}}, {"additionalProperties", false}};
            const nlohmann::json payload{
                {"model", model},
                {"messages", {{{"role", "user"}, {"content", {{{"type", "image_url"}, {"image_url", {{"url", image}}}}, {{"type", "text"}, {"text", request.prompt}}}}}}},
                {"temperature", 0},
                {"max_tokens", 1024},
                {"reasoning_effort", "none"},
                {"stream", false},
                {"response_format", {{"type", "json_schema"}, {"json_schema", {{"name", "image_result"}, {"strict", true}, {"schema", schema}}}}}
            };
            const auto response = client.Post("/v1/chat/completions", payload.dump(), "application/json");
            if (!response) throw std::runtime_error{std::format("Qwen request failed: {}", httplib::to_string(response.error()))};
            if (response->status != 200) throw std::runtime_error{std::format("Qwen request returned HTTP {}: {}", response->status, response->body)};
            const auto reply   = nlohmann::json::parse(response->body);
            const auto content = reply.at("choices").at(0).at("message").at("content").get<std::string>();
            const auto result   = nlohmann::json::parse(content).at("result").get<std::string>();
            if (std::ranges::find(request.choices, result) == request.choices.end()) throw std::runtime_error{"Qwen returned a result outside the candidate list"};
            return result;
        }
    } // namespace
    bool check_connection() {
        httplib::Client client{std::string{server_url}};
        client.set_connection_timeout(1);
        client.set_read_timeout(1);
        client.set_write_timeout(1);
        const auto response = client.Get("/v1/models");
        return response && response->status == 200;
    }
    Result process(const Request& request, std::atomic_bool& interrupted, const std::function<void(const Progress&)>& progress) {
        Result result;
        try {
            if (request.prompt.find_first_not_of(" \t\r\n") == std::string::npos) throw std::runtime_error{"Qwen prompt cannot be empty"};
            if (request.choices.size() < 2) throw std::runtime_error{"Qwen requires at least two candidate results"};
            for (std::size_t i = 0; i < request.choices.size(); ++i) {
                if (request.choices[i].find_first_not_of(" \t\r\n") == std::string::npos) throw std::runtime_error{"Qwen candidate results cannot be empty"};
                for (std::size_t j = 0; j < i; ++j)
                    if (request.choices[i] == request.choices[j]) throw std::runtime_error{"Qwen candidate results must be unique"};
            }
            const auto input = std::filesystem::canonical(request.input);
            std::vector<std::filesystem::path> files;
            if (std::filesystem::is_directory(input)) {
                progress({Stage::scanning});
                for (const auto& entry : std::filesystem::directory_iterator{input}) {
                    auto extension = entry.path().extension().string();
                    std::ranges::transform(extension, extension.begin(), [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
                    if (entry.is_regular_file() && extension == ".png") files.push_back(entry.path());
                }
            } else {
                auto extension = input.extension().string();
                std::ranges::transform(extension, extension.begin(), [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
                if (extension != ".png") throw std::runtime_error{"Qwen accepts PNG images"};
                files.push_back(input);
            }
            std::ranges::sort(files);
            result.total = files.size();
            if (files.empty()) throw std::runtime_error{"No PNG images in the input"};
            std::atomic_size_t next{};
            std::mutex mutex;
            std::string error;
            const auto worker = [&] {
                while (!interrupted) {
                    const auto index = next.fetch_add(1);
                    if (index >= files.size()) break;
                    try {
                        const auto label = classify(files[index], request);
                        std::size_t completed{};
                        {
                            const std::lock_guard lock{mutex};
                            result.items.push_back({files[index], label});
                            completed = result.items.size();
                        }
                        progress({Stage::classifying, completed, result.total, files[index], label});
                    } catch (const std::exception& failure) {
                        const std::lock_guard lock{mutex};
                        if (error.empty()) error = files[index].string() + ": " + failure.what();
                        interrupted = true;
                        break;
                    }
                }
            };
            const auto workers = std::min<std::size_t>(4, files.size());
            {
                std::vector<std::jthread> threads;
                for (std::size_t i = 0; i < workers; ++i) threads.emplace_back(worker);
            }
            result.error   = std::move(error);
            result.stopped = interrupted && result.items.size() != result.total;
        } catch (const std::exception& failure) {
            result.error = failure.what();
        }
        return result;
    }
} // namespace qwen
