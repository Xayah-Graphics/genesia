module;
#include <nlohmann/json.hpp>
#if defined(_WIN32)
#include <Windows.h>
#else
#include <sys/stat.h>
#endif
module genesia.data.datasets;
import genesia.project;
import std;
import genesia.io.files;

namespace genesia::dataset {
    void Index::scan(const std::optional<std::string> root) {
        const files::Lock lock{"index"};
        const auto cache_path = project::state_directory / "hashes.json";
        cache.clear();
        if (std::filesystem::exists(cache_path)) {
            std::ifstream input{cache_path};
            input.exceptions(std::ios::badbit | std::ios::failbit);
            const auto stored = nlohmann::json::parse(input);
            if (stored.at("version") != 1) throw std::runtime_error{"Unsupported dataset hash cache version"};
            for (const auto& [entity, entry] : stored.at("files").get_ref<const nlohmann::json::object_t&>()) {
                Cached value{entry.at("modified"), entry.at("bytes"), entry.at("sha"), entry.at("width"), entry.at("height")};
                if (value.sha.size() != 64 || !std::ranges::all_of(value.sha, [](const char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); })) throw std::runtime_error{"Malformed dataset SHA cache entry: " + entity};
                cache.emplace(entity, std::move(value));
            }
        }
        std::filesystem::create_directories(project::raw);
        roots.clear();
        std::vector<std::filesystem::path> folders;
        if (root) folders.push_back(project::directory / files::path(*root));
        else
            for (const auto& entry : std::filesystem::directory_iterator{project::directory})
                if (entry.is_directory() && !files::utf8(entry.path().filename()).starts_with('.')) folders.push_back(entry.path());
        std::ranges::sort(folders, [](const auto& a, const auto& b) { return a.filename() == "raw" ? b.filename() != "raw" : b.filename() == "raw" ? false : a.filename() < b.filename(); });
        for (const auto& folder : folders) {
            auto& found   = roots.emplace_back();
            found.all.key = found.all.name = files::utf8(folder.filename());
            try {
                for (const auto& entry : std::filesystem::directory_iterator{folder})
                    if (entry.is_directory() && !files::utf8(entry.path().filename()).starts_with('.')) found.concepts.push_back({files::utf8(entry.path().lexically_relative(project::directory)), files::utf8(entry.path().filename())});
                std::ranges::sort(found.concepts, {}, &Collection::name);
                for (auto iterator = std::filesystem::recursive_directory_iterator{folder}; iterator != std::filesystem::recursive_directory_iterator{}; ++iterator) {
                    const auto& entry = *iterator;
                    if (entry.is_directory()) {
                        if (files::utf8(entry.path().filename()).starts_with('.')) iterator.disable_recursion_pending();
                        continue;
                    }
                    auto extension = entry.path().extension().string();
                    std::ranges::transform(extension, extension.begin(), [](const unsigned char c) { return static_cast<char>(std::tolower(c)); });
                    if (entry.is_regular_file() && extension == ".png") found.files.push_back(identify(entry.path()));
                }
                std::ranges::sort(found.files, [](const File& a, const File& b) { return std::tie(a.modified, a.path) < std::tie(b.modified, b.path); });
                std::map<std::string, std::vector<const File*>> resources;
                std::map<std::string, std::set<std::string>> members;
                for (const auto& file : found.files) {
                    auto& paths = resources[file.sha];
                    if (paths.empty()) found.all.images.push_back(file);
                    paths.push_back(&file);
                    const auto relative = file.path.lexically_relative(folder);
                    if (std::distance(relative.begin(), relative.end()) < 2) continue;
                    const auto concept_name = files::utf8(*relative.begin());
                    auto& collection        = *std::ranges::find(found.concepts, concept_name, &Collection::name);
                    if (members[concept_name].insert(file.sha).second) collection.images.push_back(file);
                }
                for (const auto& [sha, paths] : resources) {
                    if (std::ranges::all_of(paths, [&](const File* file) { return file->entity == paths.front()->entity; })) continue;
                    auto& conflict = found.conflicts.emplace_back();
                    for (const auto* file : paths) conflict.push_back(file->path);
                }
                found.ready = found.conflicts.empty();
            } catch (const std::exception& error) {
                found.error = std::format("{}: {}", files::utf8(folder), error.what());
            }
        }
        nlohmann::json files = nlohmann::json::object();
        for (const auto& [entity, entry] : cache) files[entity] = {{"modified", entry.modified}, {"bytes", entry.bytes}, {"sha", entry.sha}, {"width", entry.width}, {"height", entry.height}};
        files::write_json(cache_path, {{"version", 1}, {"files", std::move(files)}});
    }

    File Index::identify(const std::filesystem::path& path) {
        File result;
        result.path = path;
#if defined(_WIN32)
        const auto handle = CreateFileW(path.c_str(), FILE_READ_ATTRIBUTES, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (handle == INVALID_HANDLE_VALUE) throw std::system_error{static_cast<int>(GetLastError()), std::system_category(), files::utf8(path)};
        const std::unique_ptr<std::remove_pointer_t<HANDLE>, decltype(&CloseHandle)> file{handle, CloseHandle};
        FILE_ID_INFO identity;
        FILE_BASIC_INFO times;
        FILE_STANDARD_INFO size;
        if (!GetFileInformationByHandleEx(handle, FileIdInfo, &identity, sizeof(identity)) || !GetFileInformationByHandleEx(handle, FileBasicInfo, &times, sizeof(times)) || !GetFileInformationByHandleEx(handle, FileStandardInfo, &size, sizeof(size))) throw std::system_error{static_cast<int>(GetLastError()), std::system_category(), files::utf8(path)};
        result.entity = std::format("{:016x}:", identity.VolumeSerialNumber);
        for (const auto byte : identity.FileId.Identifier) result.entity += std::format("{:02x}", byte);
        result.modified = times.LastWriteTime.QuadPart;
        result.bytes    = size.EndOfFile.QuadPart;
#else
        struct stat info;
        if (stat(path.c_str(), &info) == -1) throw std::system_error{errno, std::generic_category(), files::utf8(path)};
        result.entity   = std::format("{}:{}", info.st_dev, info.st_ino);
        result.modified = info.st_mtim.tv_sec * 1'000'000'000LL + info.st_mtim.tv_nsec;
        result.bytes    = info.st_size;
#endif
        auto entry = cache.find(result.entity);
        if (entry == cache.end() || entry->second.modified != result.modified || entry->second.bytes != result.bytes) {
            auto sha       = files::digest(path);
            const auto png = read_record(path);
            entry          = cache.insert_or_assign(result.entity, Cached{result.modified, result.bytes, std::move(sha), png.parameters.width, png.parameters.height}).first;
        }
        result.sha    = entry->second.sha;
        result.width  = entry->second.width;
        result.height = entry->second.height;
        return result;
    }
} // namespace genesia::dataset
