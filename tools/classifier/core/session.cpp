module classifier.session;
import std;
namespace classifier::runtime {
    Session::Session(std::function<void()> callback) : notify{std::move(callback)}, worker{[this] { run(); }} {}
    Session::~Session() {
        {
            const std::lock_guard lock{mutex};
            closing     = true;
            interrupted = true;
        }
        condition.notify_one();
        worker.join();
    }
    void Session::submit(Request request) {
        {
            const std::lock_guard lock{mutex};
            if (state.busy) throw std::runtime_error{"Stop the current task before starting another"};
            interrupted    = false;
            state.kind     = Kind(request.operation.index());
            state.state    = State::running;
            state.busy     = true;
            state.stopping = false;
            state.progress = {};
            state.error.clear();
            state.result.reset();
            ++state.revision;
            events.clear();
            pending = std::move(request);
        }
        condition.notify_one();
        if (notify) notify();
    }
    void Session::cancel() {
        {
            const std::lock_guard lock{mutex};
            interrupted    = true;
            state.stopping = state.busy;
            ++state.revision;
        }
        if (notify) notify();
    }
    Delivery Session::drain() {
        const std::lock_guard lock{mutex};
        Delivery result{state};
        result.progress.swap(events);
        return result;
    }
    void Session::progress(const Progress& value) {
        {
            const std::lock_guard lock{mutex};
            state.progress = value;
            events.push_back(value);
            ++state.revision;
        }
        if (notify) notify();
    }
    void Session::open(const std::filesystem::path& root) {
        auto classifier = std::make_shared<models::Info>(models::inspect(root));
        if (predictions && predictions->descriptor != classifier->model) predictions.reset();
        {
            const std::lock_guard lock{mutex};
            state.classifier = std::move(classifier);
            state.dataset.reset();
            state.training.reset();
            state.audit.reset();
            ++state.revision;
        }
        if (notify) notify();
    }
    void Session::read_training(const std::filesystem::path& root) {
        auto info = std::make_shared<training::Info>(training::read_state(root), training::read_history(root));
        {
            const std::lock_guard lock{mutex};
            state.training = std::move(info);
            ++state.revision;
        }
        if (notify) notify();
    }
    std::shared_ptr<const training::TrainingData> Session::scan(const std::filesystem::path& root) {
        auto source = std::make_shared<training::TrainingData>(training::inspect(root, interrupted, [this](const Progress& value) { progress(value); }));
        {
            const std::lock_guard lock{mutex};
            state.dataset = source;
            ++state.revision;
        }
        if (notify) notify();
        return source;
    }
    void Session::train(const training::Options& options) {
        open(options.root);
        read_training(options.root);
        predictions.reset();
        {
            const std::lock_guard lock{mutex};
            state.kind = Kind::train;
            if (options.restart) state.training = std::make_shared<training::Info>();
            ++state.revision;
        }
        if (notify) notify();
        std::exception_ptr failure;
        try {
            const auto source = scan(options.root);
            training::train(options, *source, interrupted, [this](const Progress& value) { progress(value); });
        } catch (...) {
            failure = std::current_exception();
        }
        auto classifier = std::make_shared<models::Info>(models::inspect(options.root));
        {
            const std::lock_guard lock{mutex};
            state.classifier = std::move(classifier);
            ++state.revision;
        }
        read_training(options.root);
        if (failure) std::rethrow_exception(failure);
    }
    void Session::run() {
        for (;;) {
            Request request;
            {
                std::unique_lock lock{mutex};
                condition.wait(lock, [this] { return closing || pending.has_value(); });
                if (closing) break;
                request = std::move(*pending);
                pending.reset();
            }
            State completed = State::complete;
            std::string error;
            std::shared_ptr<const Result> result;
            try {
                result = std::make_shared<Result>(execute(request));
                if (interrupted) completed = State::stopped;
            } catch (const Stopped&) {
                completed = State::stopped;
            } catch (const std::exception& failure) {
                completed = State::failed;
                error     = failure.what();
            }
            {
                const std::lock_guard lock{mutex};
                state.state    = completed;
                state.busy     = false;
                state.stopping = false;
                state.error    = std::move(error);
                state.result   = std::move(result);
                ++state.revision;
            }
            if (notify) notify();
        }
        predictions.reset();
    }
    Result Session::execute(const Request& request) {
        return std::visit(
            [&]<typename T>(const T& operation) -> Result {
                const auto root   = std::filesystem::canonical([&]() {
                    if constexpr (std::same_as<T, Train>) return operation.options.root;
                    else return operation.root;
                }());
                const auto report = [this](const Progress& value) { progress(value); };
                if (interrupted) throw Stopped{};
                if constexpr (std::same_as<T, Load>) {
                    if (std::filesystem::exists(root / ".genesia")) open(root);
                    else train(training::Options{root});
                    return {};
                } else if constexpr (std::same_as<T, TrainingDetails>) {
                    read_training(root);
                    return {};
                } else if constexpr (std::same_as<T, Train>) {
                    auto options = operation.options;
                    options.root = root;
                    train(options);
                    return {};
                } else if constexpr (std::same_as<T, Fix> || std::same_as<T, Recycle>) {
                    std::shared_ptr<training::TrainingData> source;
                    std::shared_ptr<classification::Audit> audit;
                    {
                        const std::lock_guard lock{mutex};
                        if (state.audit && state.dataset->root == root) {
                            source = std::make_shared<training::TrainingData>(*state.dataset);
                            audit  = std::make_shared<classification::Audit>(*state.audit);
                        }
                    }
                    if (!source) source = std::make_shared<training::TrainingData>(*scan(root));
                    if (interrupted) throw Stopped{};
                    Result result;
                    std::exception_ptr failure;
                    try {
                        if constexpr (std::same_as<T, Fix>) result.value = classification::fix(*source, operation.sha, operation.category, audit.get());
                        else classification::recycle(*source, *audit, operation.sha, operation.file, interrupted);
                    } catch (...) {
                        failure = std::current_exception();
                    }
                    {
                        const std::lock_guard lock{mutex};
                        state.dataset = std::move(source);
                        state.audit   = std::move(audit);
                        ++state.revision;
                    }
                    if (notify) notify();
                    if (failure) std::rethrow_exception(failure);
                    return result;
                } else {
                    const auto model = models::resolve(root);
                    if (!predictions) predictions = std::make_unique<classification::Predictions>();
                    if constexpr (std::same_as<T, Audit>) {
                        {
                            const std::lock_guard lock{mutex};
                            state.audit.reset();
                            ++state.revision;
                        }
                        auto audit = std::make_shared<classification::Audit>(classification::audit(*scan(root), model, *predictions, operation.refresh, interrupted, report));
                        {
                            const std::lock_guard lock{mutex};
                            state.audit = std::move(audit);
                            ++state.revision;
                        }
                        if (notify) notify();
                        return {};
                    } else {
                        const auto input = std::filesystem::canonical(operation.input);
                        if constexpr (std::same_as<T, Infer>) {
                            const auto file = dataset::read_file(input);
                            if (interrupted) throw Stopped{};
                            return {Inferred{input, predictions->infer(model, file, operation.refresh, report)}};
                        } else return {classification::classify(model, input, *predictions, interrupted, report)};
                    }
                }
            },
            request.operation);
    }
} // namespace classifier::runtime
