module;
#include <genesia/cuda.h>
#include <httplib.h>
#include <iphlpapi.h>
#include <stb_image_write.h>
#include <nlohmann/json.hpp>
module genesia.lan;
import genesia.compute.device;
import genesia.io.files;
import std;

namespace genesia::lan {
    namespace {
#include "genesia-lan-page.h"

        constexpr std::string_view subnet{"192.168.1."};
    } // namespace

    Server::Server() = default;

    Server::~Server() {
        stop();
    }

    bool Server::start(const std::string& address, runtime::Session& session, void (*wake)()) {
        const auto host = std::format("{}:{}", address, port);
        const auto url  = "http://" + host;
        int bind_error{};
        http.set_address_family(AF_INET);
        http.set_socket_options([](const socket_t socket) {
            if (!httplib::set_socket_opt(socket, SOL_SOCKET, SO_EXCLUSIVEADDRUSE, 1)) throw std::system_error{WSAGetLastError(), std::system_category(), "Set exclusive LAN socket"};
        });
        http.set_read_timeout(2);
        http.set_write_timeout(2);
        http.set_keep_alive_max_count(1);
        http.set_default_headers({{"Cache-Control", "no-store"}});
        http.set_payload_max_length(1024);
        http.set_pre_routing_handler([host](const httplib::Request& request, httplib::Response& response) {
            if (request.remote_addr.starts_with(subnet) && request.get_header_value("Host") == host) return httplib::Server::HandlerResponse::Unhandled;
            response.status = 403;
            return httplib::Server::HandlerResponse::Handled;
        });
        http.Get("/", [](const httplib::Request&, httplib::Response& response) { response.set_content(page, sizeof(page) - 1, "text/html; charset=utf-8"); });
        http.Get("/api/history", [](const httplib::Request&, httplib::Response& response) {
            const auto paths = list_output_images();
            auto images      = nlohmann::json::array();
            for (const auto& path : paths | std::views::reverse)
                if (!std::filesystem::is_symlink(path)) images.push_back(files::utf8(path.filename()));
            response.set_content(images.dump(), "application/json");
        });
        http.Get("/api/history/image", [](const httplib::Request& request, httplib::Response& response) {
            const auto name  = request.get_param_value("name");
            const auto paths = list_output_images();
            const auto image = std::ranges::find(paths, name, [](const auto& path) { return files::utf8(path.filename()); });
            if (image == paths.end() || !std::filesystem::is_regular_file(std::filesystem::symlink_status(*image))) {
                response.status = 404;
                return;
            }
            const auto bytes = files::read_bytes(*image);
            response.set_content(reinterpret_cast<const char*>(bytes.data()), bytes.size(), "image/png");
        });
        http.Post("/api/generate", [this, &session, url, wake](const httplib::Request& request, httplib::Response& response) {
            if (request.get_header_value("Origin") != url) {
                response.status = 403;
                return;
            }
            const std::lock_guard lock{mutex};
            const auto state = session.snapshot();
            if (state.finished) {
                response.status = 503;
                return;
            }
            if (state.active || (job && !job->rejection && job->state < runtime::State::complete)) {
                response.status = 409;
                return;
            }
            job.emplace(Job{.token = files::random_id()});
            response.status = 202;
            response.set_content(job->token, "text/plain");
            requested = true;
            wake();
        });
        http.Get(R"(/api/jobs/([0-9a-f]{32}))", [this, &session](const httplib::Request& request, httplib::Response& response) {
            std::filesystem::path image;
            runtime::PreviewFrame preview;
            int steps{};
            {
                const std::lock_guard lock{mutex};
                if (!job || request.matches[1].str() != job->token) {
                    response.status = 404;
                    return;
                }
                if (job->rejection) {
                    response.status = job->rejection;
                    return;
                }
                if (job->state > runtime::State::complete || session.snapshot().finished) {
                    response.status = 500;
                    return;
                }
                if (job->state < runtime::State::complete) {
                    if (!job->preview.pixels || request.get_param_value("step") == std::to_string(job->preview.step)) {
                        response.status = 202;
                        return;
                    }
                    preview = job->preview;
                    steps   = job->steps;
                } else image = job->image;
            }
            if (preview.pixels) {
                std::string jpeg;
                const auto write = [](void* context, void* data, const int size) { static_cast<std::string*>(context)->append(static_cast<const char*>(data), size); };
                if (!stbi_write_jpg_to_func(write, &jpeg, preview.width, preview.height, 3, preview.pixels.get(), 85)) throw std::runtime_error{"Cannot encode LAN preview"};
                response.set_header("X-Genesia-Step", std::to_string(preview.step));
                response.set_header("X-Genesia-Steps", std::to_string(steps));
                response.set_content(std::move(jpeg), "image/jpeg");
                return;
            }
            const auto bytes = files::read_bytes(image);
            response.set_header("X-Genesia-Image", httplib::encode_uri_component(files::utf8(image.filename())));
            response.set_content(reinterpret_cast<const char*>(bytes.data()), bytes.size(), "image/png");
        });

        http.set_error_logger([&](const httplib::Error& error, const httplib::Request*) {
            if (error != httplib::Error::BindIPAddress) return;
            bind_error = WSAGetLastError();
        });
        const auto bound = http.bind_to_port(address, port, AI_NUMERICHOST);
        http.set_error_logger({});
        if (!bound) {
            if (bind_error == WSAEADDRINUSE || bind_error == WSAEACCES) return false;
            if (bind_error) throw std::system_error{bind_error, std::system_category(), "Cannot listen on " + url};
            throw std::runtime_error{"Cannot listen on " + url};
        }
        listening = true;
        listener  = std::jthread{[this] {
            http.listen_after_bind();
            listening = false;
        }};
        http.wait_until_ready();
        return true;
    }

    void Server::stop() {
        http.stop();
        if (listener.joinable()) listener.join();
    }

    void Server::complete_submission(std::expected<runtime::TaskStatus, int> submission) {
        const std::lock_guard lock{mutex};
        if (submission) {
            job->id    = submission->id;
            job->steps = submission->request->parameters.steps;
        } else job->rejection = submission.error();
        requested = false;
    }

    void Server::receive(const runtime::Delivery& delivery) {
        const std::lock_guard lock{mutex};
        if (!job || !job->id) return;
        for (const auto& event : delivery.events) {
            if (event.id != job->id) continue;
            if (event.kind == runtime::EventKind::saved) job->image = event.record.path;
            if (event.kind == runtime::EventKind::task) job->state = event.task.state;
        }
        if (job->state >= runtime::State::complete) {
            job->preview = {};
            return;
        }
        for (const auto& frame : delivery.previews | std::views::reverse) {
            if (frame.id != job->id || !frame.pixels) continue;
            job->preview = {.id = frame.id, .step = frame.step, .width = frame.width, .height = frame.height, .pixels = frame.pixels};
            break;
        }
    }

    std::shared_ptr<const std::uint8_t> capture_preview(const std::uint8_t* source, const int width, const int height, const std::shared_ptr<::cuda::stream>& stream) {
        // A preview can outlive the engine while the HTTP handler encodes it.
        const auto bytes = std::size_t(width) * height * 3;
        auto* data       = static_cast<std::uint8_t*>(::cuda::pinned_default_memory_pool().allocate(*stream, bytes));
        std::shared_ptr<const std::uint8_t> pixels{data, [stream, bytes](std::uint8_t* allocation) { ::cuda::pinned_default_memory_pool().deallocate(*stream, allocation, bytes); }};
        compute::check(cudaMemcpyAsync(data, source, bytes, cudaMemcpyDeviceToHost, stream->get()));
        return pixels;
    }

    std::string wifi_address() {
        constexpr ULONG flags = GAA_FLAG_SKIP_ANYCAST | GAA_FLAG_SKIP_MULTICAST | GAA_FLAG_SKIP_DNS_SERVER | GAA_FLAG_SKIP_FRIENDLY_NAME;
        ULONG size            = 15 * 1024;
        auto storage          = std::make_unique<std::byte[]>(size);
        auto* adapters        = reinterpret_cast<IP_ADAPTER_ADDRESSES*>(storage.get());
        auto result           = GetAdaptersAddresses(AF_INET, flags, nullptr, adapters, &size);
        while (result == ERROR_BUFFER_OVERFLOW) {
            storage  = std::make_unique<std::byte[]>(size);
            adapters = reinterpret_cast<IP_ADAPTER_ADDRESSES*>(storage.get());
            result   = GetAdaptersAddresses(AF_INET, flags, nullptr, adapters, &size);
        }
        if (result == ERROR_NO_DATA) return {};
        if (result != NO_ERROR) throw std::system_error{static_cast<int>(result), std::system_category(), "Read Wi-Fi addresses"};
        std::string address;
        for (const auto* adapter = adapters; adapter; adapter = adapter->Next) {
            if (adapter->IfType != IF_TYPE_IEEE80211 || adapter->OperStatus != IfOperStatusUp) continue;
            for (const auto* entry = adapter->FirstUnicastAddress; entry; entry = entry->Next) {
                if (entry->Address.lpSockaddr->sa_family != AF_INET || entry->OnLinkPrefixLength != 24 || entry->DadState != IpDadStatePreferred) continue;
                const auto& ipv4 = reinterpret_cast<const sockaddr_in*>(entry->Address.lpSockaddr)->sin_addr;
                std::array<char, INET_ADDRSTRLEN> text;
                if (!InetNtopA(AF_INET, &ipv4, text.data(), text.size())) throw std::system_error{WSAGetLastError(), std::system_category(), "Format Wi-Fi address"};
                if (!std::string_view{text.data()}.starts_with(subnet)) continue;
                if (!address.empty()) throw std::runtime_error{"More than one Wi-Fi address on 192.168.1.0/24"};
                address = text.data();
            }
        }
        return address;
    }
} // namespace genesia::lan
