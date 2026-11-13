export module genesia.runtime.diagnostics;
import std;

export namespace genesia::diagnostics {
    extern std::filesystem::path log_path;
    void initialize(bool editor);
    void record(std::string_view event, std::string_view detail = {}, std::source_location source = std::source_location::current());
} // namespace genesia::diagnostics
