module;
#include <GLFW/glfw3.h>
#include <genesia/cuda.h>
#include <httplib.h>
#include <iphlpapi.h>
#include <stb_image_write.h>
#include <nlohmann/json.hpp>
module genesia.editor.web;
import genesia.compute.device;
import genesia.io.files;
import std;

namespace genesia::editor {
    namespace {
#include "genesia-web-page.h"
    }

    Web::Web() {
        server.new_task_queue = [] { return new httplib::ThreadPool{4, 4, 16}; };
        server.set_read_timeout(3, 0);
        server.set_write_timeout(3, 0);
        server.set_keep_alive_max_count(1);
        server.set_payload_max_length(1024);
        server.Get("/", [](const httplib::Request&, httplib::Response& response) {
            response.set_header("Cache-Control", "no-store");
            response.set_content(web_page, "text/html; charset=utf-8");
        });
        server.Get("/api/state", [this](const httplib::Request& request, httplib::Response& response) {
            {
                const std::lock_guard lock{mutex};
                if (request.get_param_value("live") == "1") last_seen = std::chrono::steady_clock::now();
                response.set_header("Cache-Control", "no-store");
                response.set_content(nlohmann::json{{"available", accepting && !requested}, {"status", error.empty() ? status : error}, {"error", !error.empty()}, {"version", std::to_string(revision) + (requested ? "-pending" : "")}, {"image", bool(image) && !requested}, {"final", bool(image) && !requested && mime == "image/png"}, {"history_session", history_session}, {"history_count", history.size()}}.dump(), "application/json");
            }
            glfwPostEmptyEvent();
        });
        server.Post("/api/generate", [this](const httplib::Request& request, httplib::Response& response) {
            response.set_header("Cache-Control", "no-store");
            // A custom header keeps cross-origin HTML forms from triggering generation.
            if (request.get_header_value("X-Genesia") != "generate") {
                response.status = 403;
                response.set_content("请从 Genesia 页面提交", "text/plain; charset=utf-8");
                return;
            }
            {
                const std::lock_guard lock{mutex};
                if (!enabled || !accepting || requested) {
                    response.status = 409;
                    response.set_content(requested || busy ? "主机忙碌" : status, "text/plain; charset=utf-8");
                    return;
                }
                requested = true;
                accepting = false;
                error.clear();
                status = "正在提交";
            }
            glfwPostEmptyEvent();
            response.status = 202;
        });
        server.Get("/api/preview", [this](const httplib::Request& request, httplib::Response& response) {
            std::shared_ptr<const std::string> current;
            std::string type;
            {
                const std::lock_guard lock{mutex};
                response.set_header("Cache-Control", "no-store");
                if (!image || request.get_param_value("v") != std::to_string(revision)) {
                    response.status = 204;
                    return;
                }
                current = image;
                type    = mime;
            }
            response.set_content_provider(current->size(), type, [current](std::size_t offset, std::size_t length, httplib::DataSink& sink) { return sink.write(current->data() + offset, length); });
        });
        server.Get(R"(/api/history/([0-9]+)/([0-9]+))", [this](const httplib::Request& request, httplib::Response& response) {
            response.set_header("Cache-Control", "no-store");
            std::filesystem::path path;
            {
                const std::lock_guard lock{mutex};
                if (request.matches[1].str() != history_session) {
                    response.status = 410;
                    response.set_content("主机已重启，这次运行的历史记录已结束", "text/plain; charset=utf-8");
                    return;
                }
                const auto index = std::stoull(request.matches[2].str());
                if (index >= history.size()) {
                    response.status = 404;
                    response.set_content("历史图片不存在", "text/plain; charset=utf-8");
                    return;
                }
                path = history[index];
            }
            try {
                auto bytes = std::make_shared<const std::vector<std::uint8_t>>(files::read_bytes(path));
                response.set_content_provider(bytes->size(), "image/png", [bytes](std::size_t offset, std::size_t length, httplib::DataSink& sink) { return sink.write(reinterpret_cast<const char*>(bytes->data() + offset), length); });
            } catch (const std::exception& failure) {
                response.status = 500;
                response.set_content(failure.what(), "text/plain; charset=utf-8");
            }
        });
        encoder = std::jthread{[this] { encode(); }};
    }

    Web::~Web() {
        stop();
        {
            const std::lock_guard lock{mutex};
            closing = true;
        }
        condition.notify_one();
        encoder.join();
    }

    void Web::start() {
        if (listener.joinable()) listener.join();
        addresses.clear();
        ULONG bytes{};
        constexpr ULONG flags = GAA_FLAG_SKIP_ANYCAST | GAA_FLAG_SKIP_MULTICAST | GAA_FLAG_SKIP_DNS_SERVER;
        auto result           = GetAdaptersAddresses(AF_INET, flags, nullptr, nullptr, &bytes);
        if (result != ERROR_BUFFER_OVERFLOW) throw std::system_error{static_cast<int>(result), std::system_category(), "Read LAN addresses"};
        std::vector<std::byte> storage(bytes);
        auto* adapters = reinterpret_cast<IP_ADAPTER_ADDRESSES*>(storage.data());
        result         = GetAdaptersAddresses(AF_INET, flags, nullptr, adapters, &bytes);
        if (result != NO_ERROR) throw std::system_error{static_cast<int>(result), std::system_category(), "Read LAN addresses"};
        for (auto* adapter = adapters; adapter; adapter = adapter->Next) {
            if (adapter->OperStatus != IfOperStatusUp || adapter->IfType == IF_TYPE_SOFTWARE_LOOPBACK) continue;
            for (auto* address = adapter->FirstUnicastAddress; address; address = address->Next) {
                char text[INET_ADDRSTRLEN];
                const auto* ipv4 = reinterpret_cast<const sockaddr_in*>(address->Address.lpSockaddr);
                if (!InetNtopA(AF_INET, &ipv4->sin_addr, text, sizeof(text))) throw std::system_error{WSAGetLastError(), std::system_category(), "Format LAN address"};
                addresses.push_back(std::format("http://{}:7860", text));
            }
        }
        if (addresses.empty()) throw std::runtime_error{"No active LAN IPv4 address"};
        if (!server.bind_to_port("0.0.0.0", 7860)) throw std::runtime_error{"Cannot listen on port 7860"};
        {
            const std::lock_guard lock{mutex};
            enabled = true;
            error.clear();
            if (!final_image.empty()) pending = Frame{.task = task, .sequence = ++sequence, .file = final_image};
        }
        condition.notify_one();
        listener = std::jthread{[this] {
            const bool success = server.listen_after_bind();
            const std::lock_guard lock{mutex};
            if (!success && enabled) error = "局域网服务监听失败";
            enabled   = false;
            accepting = false;
            glfwPostEmptyEvent();
        }};
    }

    void Web::stop() {
        {
            const std::lock_guard lock{mutex};
            enabled   = false;
            requested = false;
            accepting = false;
            last_seen = {};
            image.reset();
            pending.reset();
            ++sequence;
            ++revision;
        }
        server.stop();
        if (listener.joinable()) listener.join();
    }

    bool Web::watching() {
        const std::lock_guard lock{mutex};
        return enabled && std::chrono::steady_clock::now() - last_seen < std::chrono::seconds{3};
    }

    void Web::update(const runtime::Snapshot& state, std::string unavailable, std::string failure) {
        const std::lock_guard lock{mutex};
        if (!failure.empty()) error = std::move(failure);
        busy = state.active.has_value();
        if (!state.error.empty()) unavailable = state.error;
        else if (state.finished) unavailable = "主机正在关闭";
        accepting = enabled && !busy && unavailable.empty();
        if (!unavailable.empty()) status = std::move(unavailable);
        else if (busy) {
            status = "主机忙碌";
            if (state.active->kind == runtime::Kind::generate && !std::get<runtime::Generate>(state.active->request->operation).source) {
                if (state.active->stopping) status = "正在停止";
                else if (state.active->state == runtime::State::saving) status = "正在保存";
                else if (state.generation.stage == runtime::GenerationStage::loading) status = "正在加载模型";
                else if (state.generation.stage == runtime::GenerationStage::sampling) status = std::format("生成中 · {} / {}", state.generation.completed, state.generation.steps);
                else status = "正在处理";
            }
        } else status = "就绪 · 使用主机当前配置";
    }

    void Web::receive(const runtime::Event& event) {
        const std::lock_guard lock{mutex};
        if (event.kind == runtime::EventKind::saved && event.record.source.empty()) history.push_back(event.record.path);
        if (event.id < task) return;
        if (event.kind == runtime::EventKind::task && event.task.kind == runtime::Kind::generate && event.task.request && !std::get<runtime::Generate>(event.task.request->operation).source) {
            if (task != event.id) {
                task = event.id;
                ++sequence;
                ++revision;
                image.reset();
                pending.reset();
                final_image.clear();
                error.clear();
                accepting = false;
                busy      = true;
            }
            if (event.task.state == runtime::State::failed) error = event.task.error;
        } else if (event.kind == runtime::EventKind::saved && event.record.source.empty()) {
            final_image = event.record.path;
            if (enabled) {
                pending = Frame{.task = event.id, .sequence = ++sequence, .file = final_image};
                condition.notify_one();
            }
        }
    }

    void Web::capture(const std::uint64_t id, const std::uint8_t* pixels, const int width, const int height, const ::cuda::stream_ref stream) {
        if (!watching()) return;
        {
            const std::lock_guard lock{mutex};
            if (id != task || !final_image.empty()) return;
        }
        Frame frame{.task = id, .width = width, .height = height, .pixels = std::vector<std::uint8_t>(std::size_t(width) * height * 3)};
        compute::check(cudaMemcpyAsync(frame.pixels.data(), pixels, frame.pixels.size(), cudaMemcpyDeviceToHost, stream.get()));
        stream.sync();
        {
            const std::lock_guard lock{mutex};
            if (!enabled || task != id || !final_image.empty()) return;
            frame.sequence = ++sequence;
            pending        = std::move(frame);
        }
        condition.notify_one();
    }

    void Web::encode() {
        for (;;) {
            Frame frame;
            {
                std::unique_lock lock{mutex};
                condition.wait(lock, [this] { return closing || pending.has_value(); });
                if (closing) return;
                frame = std::move(*pending);
                pending.reset();
            }
            try {
                auto encoded = std::make_shared<std::string>();
                if (!frame.file.empty()) {
                    const auto bytes = files::read_bytes(frame.file);
                    encoded->assign(reinterpret_cast<const char*>(bytes.data()), bytes.size());
                } else {
                    const auto write = [](void* context, void* data, const int size) { static_cast<std::string*>(context)->append(static_cast<const char*>(data), size); };
                    if (!stbi_write_jpg_to_func(write, encoded.get(), frame.width, frame.height, 3, frame.pixels.data(), 85)) throw std::runtime_error{"Cannot encode web preview"};
                }
                const std::lock_guard lock{mutex};
                if (!enabled || frame.task != task || frame.sequence != sequence) continue;
                image = std::move(encoded);
                mime  = frame.file.empty() ? "image/jpeg" : "image/png";
                ++revision;
            } catch (const std::exception& failure) {
                const std::lock_guard lock{mutex};
                if (enabled && frame.task == task && frame.sequence == sequence) error = failure.what();
            }
        }
    }
} // namespace genesia::editor
