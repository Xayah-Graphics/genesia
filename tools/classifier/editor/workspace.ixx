export module classifier.editor.workspace;
export import classifier.session;
import tools.editor.platform.window;
import std;
export namespace classifier::editor {
    struct Workspace final {
        tools::editor::WindowPlatform& window;
        const float& dpi;
        runtime::Session session;
        runtime::Snapshot state;
        std::string error;
        explicit Workspace(tools::editor::WindowPlatform& window, const float& dpi);
        void submit(runtime::Request request);
        void receive();
        void drop();
        void draw();

    private:
        enum class Mode { inference, training };
        enum class Section { train, audit };
        Mode mode{Mode::inference};
        Section section{Section::train};
        std::shared_ptr<const models::Info> classifier;
        std::shared_ptr<const training::Info> training;
        std::shared_ptr<const runtime::Result> inference;
        std::shared_ptr<const classification::Audit> report;
        training::Config config;
        int steps{400}, last_step{};
        std::vector<float> losses;
        std::optional<training::Metrics> metrics;
        std::string category;
        float threshold{80};
        std::vector<std::size_t> audit_rows;
        bool reset_popups{}, training_requested{}, audit_requested{}, metrics_expanded{};
        void filter_audit();
        float content_width() const;
        void draw_path(const std::filesystem::path& path, float reserved = 0);
        void draw_progress();
        void draw_training();
        void draw_audit();
        void draw_inference();
        void draw_settings();
        void draw_metrics();
    };
} // namespace classifier::editor
