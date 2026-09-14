module;
#include "sdxl/control.h"
#include <csignal>
#include <genesia/cuda.h>
#include <nlohmann/json.hpp>
module genesia.headless;
import genesia.files;
import std;
namespace genesia::headless {
    namespace {
        std::atomic_bool interrupted{};
        void interrupt(int signal) {
            interrupted.store(true);
            std::signal(signal, interrupt);
        }
    } // namespace
    int run(const std::optional<prompt::Preset>& preset, std::shared_ptr<const prompt::Catalog> catalog, const Options& options) {
        std::signal(SIGINT, interrupt);
        std::signal(SIGTERM, interrupt);
#ifdef SIGBREAK
        std::signal(SIGBREAK, interrupt);
#endif
        auto request = options.request;
        if (request.kind == work::Kind::train) {
            const auto assigned = dataset::read_concept(request.concept_key);
            if (assigned.type == dataset::ConceptType::none) throw std::runtime_error{"Assign a concept type before training"};
            if (assigned.type == dataset::ConceptType::lora) throw std::runtime_error{"LoRA training is not implemented"};
        }
        if (request.kind == work::Kind::generate) {
            if (!options.source.empty()) {
                const auto source  = std::filesystem::absolute(options.source).lexically_normal();
                const auto record  = read_record(source, catalog);
                request.parameters = record.parameters;
                request.prompt     = record.prompt;
                catalog            = record.catalog;
                dataset::Index index;
                const auto file            = index.identify(source);
                request.source             = work::RepaintSource{file.sha, source};
                request.parameters.denoise = *options.denoise;
            }
            if (preset) request.prompt = preset->prompt;
            request.catalog             = catalog;
            request.parameters.positive = prompt::compose(*catalog, request.prompt.positive);
            request.parameters.negative = prompt::compose(*catalog, request.prompt.negative);
            if (options.width) request.parameters.width = *options.width;
            if (options.height) request.parameters.height = *options.height;
            if (options.steps) request.parameters.steps = *options.steps;
            if (options.cfg) request.parameters.cfg = *options.cfg;
        }
        work::Session session;
        session.activated = options.activated;
        std::random_device random;
        std::uniform_int_distribution<std::uint64_t> seeds;
        int submitted{};
        std::uint64_t generation_task{};
        const auto submit = [&] {
            request.seed    = options.first_seed ? *options.first_seed + submitted : seeds(random);
            generation_task = session.enqueue(request);
            ++submitted;
        };
        submit();
        bool failed{}, shutting_down{};
        for (;;) {
            if (interrupted.load() && !shutting_down) {
                session.shutdown();
                shutting_down = true;
            }
            std::deque<work::Event> events;
            bool complete;
            {
                const std::lock_guard lock{session.mutex};
                events.swap(session.events);
                complete = !session.active && session.queue.empty() && std::ranges::all_of(session.jobs, [](const auto& entry) {
                    const auto& state = entry.second.at("state");
                    return state == "complete" || state == "failed" || state == "stopped";
                });
                if (!session.error.empty()) throw std::runtime_error{session.error};
            }
            bool next{};
            for (auto& event : events) {
                if (event.kind != work::EventKind::task) continue;
                const auto& state = event.data.at("state");
                failed |= state == "failed";
                if (request.kind == work::Kind::audit && state == "complete" && !request.category.empty()) {
                    const auto classes = event.data.at("result").at("classes").get<std::vector<std::string>>();
                    if (!std::ranges::contains(classes, request.category)) throw std::runtime_error{"Unknown audit category: " + request.category};
                    auto& rows = event.data.at("result").at("rows");
                    std::erase_if(rows.get_ref<nlohmann::json::array_t&>(), [&](const auto& row) { return row.at("label") != request.category; });
                }
                std::println("{}", event.data.dump());
                std::cout.flush();
                if (event.id == generation_task && state == "complete" && request.kind == work::Kind::generate && submitted < options.count) next = true;
            }
            if (next && !shutting_down) {
                submit();
                complete = false;
            }
            if (complete) break;
            std::this_thread::sleep_for(std::chrono::milliseconds{10});
        }
        return interrupted.load() ? 130 : failed ? 1 : 0;
    }
} // namespace genesia::headless
