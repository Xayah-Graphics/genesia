export module tools.editor.platform.files;
import std;
export namespace tools::editor::shell {
    void reveal(const std::filesystem::path& path);
    void recycle(const std::filesystem::path& path);
} // namespace tools::editor::shell
