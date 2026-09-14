module;
#if defined(_WIN32)
#include <Windows.h>
#endif
export module genesia.io.changes;
import std;
export namespace genesia::files {
    struct Changes final {
        explicit Changes(std::filesystem::path directory);
        ~Changes();
        Changes(const Changes&)            = delete;
        Changes& operator=(const Changes&) = delete;
        bool poll(std::chrono::milliseconds timeout);

    private:
        std::filesystem::path directory;
#if defined(_WIN32)
        HANDLE folder{}, event{};
        OVERLAPPED operation{};
        std::vector<std::byte> buffer;
        bool reading{};
        void watch();
#else
        int handle{-1};
        std::map<int, std::filesystem::path> watches;
#endif
    };
} // namespace genesia::files
