module genesia.runtime.catalog;
import genesia.io.changes;
import std;
namespace genesia::runtime {
    Catalog::Catalog(std::function<void()> wake) : notify{std::move(wake)}, worker{[this](std::stop_token stop) { scan(stop); }} {}
    Catalog::~Catalog() {
        worker.request_stop();
        worker.join();
    }
    std::shared_ptr<const CatalogState> Catalog::poll() {
        const std::lock_guard lock{mutex};
        pending = false;
        return std::exchange(update, {});
    }
    void Catalog::refresh() {
        rescan = true;
    }
    void Catalog::scan(const std::stop_token stop) {
        std::uint64_t revision{};
        try {
            std::filesystem::create_directories(project::raw);
            files::Changes changes{project::directory};
            dataset::Index index;
            auto due = std::chrono::steady_clock::time_point::max();
            while (!stop.stop_requested()) {
                if (rescan.exchange(false) || std::chrono::steady_clock::now() >= due) {
                    auto next = std::make_shared<CatalogState>();
                    {
                        const files::Lock lock{"image-moves"};
                        index.scan();
                        next->roots = std::move(index.roots);
                        for (const auto& root : next->roots)
                            for (const auto& collection : root.concepts) {
                                try {
                                    const auto assigned            = dataset::read_concept(collection.key);
                                    next->concepts[collection.key] = assigned;
                                    if (assigned.type == dataset::ConceptType::classifier) next->classifiers[collection.key] = training::inspect(assigned, root);
                                } catch (const std::exception& failure) {
                                    next->concept_errors[collection.key] = failure.what();
                                }
                            }
                    }
                    next->revision = ++revision;
                    next->ready    = true;
                    {
                        const std::lock_guard lock{mutex};
                        update  = std::move(next);
                        pending = true;
                    }
                    if (notify) notify();
                    due = std::chrono::steady_clock::time_point::max();
                }
                if (changes.poll(std::chrono::milliseconds{100})) due = std::chrono::steady_clock::now() + std::chrono::milliseconds{120};
            }
        } catch (const std::exception& failure) {
            auto next      = std::make_shared<CatalogState>();
            next->revision = ++revision;
            next->error    = failure.what();
            {
                const std::lock_guard lock{mutex};
                update  = std::move(next);
                pending = true;
            }
            if (notify) notify();
        }
    }
} // namespace genesia::runtime
