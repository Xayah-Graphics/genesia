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
    std::string lora_issue(const Root& root, const std::string_view key) {
        const auto folder = project::directory / files::path(key);
        std::map<std::string, std::vector<std::filesystem::path>> resources;
        for (const auto& file : root.files) {
            const auto relative = file.path.lexically_relative(folder);
            if (!relative.empty() && *relative.begin() != "..") resources[file.sha].push_back(file.path);
        }
        std::size_t count{};
        std::string paths;
        for (const auto& [sha, members] : resources) {
            if (members.size() < 2) continue;
            ++count;
            paths += "\n\nSHA " + sha;
            for (const auto& path : members) paths += "\n" + files::utf8(path);
        }
        return count ? std::format("{} duplicate image groups in LoRA concept {}. Each SHA must have exactly one path, including hard links.{}", count, key, paths) : "";
    }
    Concept assign_type(const Root& root, const std::string_view key, const ConceptType type) {
        auto result = read_concept(key);
        if (result.locked) throw std::runtime_error{"Concept type is locked by its training history: " + result.key};
        if (type == ConceptType::lora) {
            const auto issue = lora_issue(root, result.key);
            if (!issue.empty()) throw std::runtime_error{issue};
        }
        result.type = type;
        files::write_json(result.path / ".genesia" / "concept.json", result);
        return result;
    }
    void Index::flush() {
        if (!dirty) return;
        nlohmann::json files = nlohmann::json::object();
        for (const auto& [entity, entry] : cache) files[entity] = {{"modified", entry.modified}, {"bytes", entry.bytes}, {"sha", entry.sha}, {"width", entry.width}, {"height", entry.height}};
        files::write_json(project::state_directory / "hashes.json", {{"version", 1}, {"files", std::move(files)}});
        dirty = false;
    }
    void Index::scan(const std::string_view name) {
        load_cache();
        const auto folder   = project::directory / files::path(name);
        const auto existing = std::ranges::find(roots, name, [](const Root& root) { return root.all.key; });
        auto& found         = existing == roots.end() ? roots.emplace_back() : *existing;
        found               = {};
        found.all.key = found.all.name = files::utf8(folder.filename());
        try {
            for (const auto& entry : std::filesystem::directory_iterator{folder})
                if (entry.is_directory() && !files::utf8(entry.path().filename()).starts_with('.')) found.concepts.push_back({files::utf8(entry.path().lexically_relative(project::directory)), files::utf8(entry.path().filename())});
            std::ranges::sort(found.concepts, {}, &Collection::name);
            for (auto iterator = std::filesystem::recursive_directory_iterator{folder}; iterator != std::filesystem::recursive_directory_iterator{}; ++iterator) {
                const auto& entry = *iterator;
                if (entry.is_directory()) {
                    if (files::utf8(entry.path().filename()).starts_with('.')) iterator.disable_recursion_pending();
                    else found.directories.push_back(entry.path());
                    continue;
                }
                auto extension = entry.path().extension().string();
                std::ranges::transform(extension, extension.begin(), [](const unsigned char c) { return static_cast<char>(std::tolower(c)); });
                if (entry.is_regular_file() && extension == ".png") found.files.push_back(identify(entry.path()));
            }
            rebuild(found);
            if (folder == project::raw)
                for (const auto& file : found.files) {
                    if (file.path.parent_path() != project::raw) continue;
                    const auto filename = files::utf8(file.path.filename());
                    if (!filename.starts_with("genesia_") || !filename.ends_with(".png")) continue;
                    const std::string_view digits{filename.data() + 8, filename.size() - 12};
                    std::uint64_t number{};
                    const auto parsed = std::from_chars(digits.data(), digits.data() + digits.size(), number);
                    if (parsed.ec == std::errc{} && parsed.ptr == digits.data() + digits.size()) next_output = std::max(next_output, number + 1);
                }
        } catch (const std::exception& error) {
            found.error = std::format("{}: {}", files::utf8(folder), error.what());
        }
    }
    File Index::identify(const std::filesystem::path& path, std::optional<std::array<int, 2>> dimensions) {
        load_cache();
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
            dirty    = true;
            auto sha = files::digest(path);
            if (!dimensions) {
                const auto image = read_image_info(path);
                dimensions       = std::array{image.width, image.height};
            }
            entry = cache.insert_or_assign(result.entity, Cached{result.modified, result.bytes, std::move(sha), (*dimensions)[0], (*dimensions)[1]}).first;
        }
        result.sha    = entry->second.sha;
        result.width  = entry->second.width;
        result.height = entry->second.height;
        return result;
    }
    void Index::insert(File file) {
        auto& raw         = *std::ranges::find(roots, std::string_view{"raw"}, [](const Root& root) { return root.all.key; });
        const auto before = [](const File& a, const File& b) { return std::tie(a.modified, a.path) < std::tie(b.modified, b.path); };
        if (!std::ranges::contains(raw.all.images, file.sha, &File::sha)) raw.all.images.insert(std::ranges::lower_bound(raw.all.images, file, before), file);
        raw.files.insert(std::ranges::lower_bound(raw.files, file, before), std::move(file));
    }
    std::vector<std::string> Index::apply(const std::span<const Move> moves) {
        std::map<std::filesystem::path, std::filesystem::path> destinations;
        for (const auto& move : moves) destinations.emplace(move.source, move.destination);
        std::vector<std::string> changed;
        for (auto& root : roots) {
            bool moved{};
            for (auto& file : root.files) {
                const auto found = destinations.find(file.path);
                if (found == destinations.end()) continue;
                file.path = found->second;
                moved     = true;
            }
            if (!moved) continue;
            rebuild(root);
            changed.push_back(root.all.key);
        }
        return changed;
    }
    void Index::rebuild(Root& root) {
        root.all.images.clear();
        root.conflicts.clear();
        for (auto& collection : root.concepts) collection.images.clear();
        const auto folder = project::directory / files::path(root.all.key);
        for (const auto& file : root.files) {
            for (auto parent = file.path.parent_path(); parent != folder; parent = parent.parent_path())
                if (!std::ranges::contains(root.directories, parent)) root.directories.push_back(parent);
            const auto relative = file.path.lexically_relative(folder);
            if (std::distance(relative.begin(), relative.end()) < 2) continue;
            const auto name = files::utf8(*relative.begin());
            if (!std::ranges::contains(root.concepts, name, &Collection::name)) root.concepts.push_back({root.all.key + "/" + name, name});
        }
        std::ranges::sort(root.concepts, {}, &Collection::name);
        std::ranges::sort(root.files, [](const File& a, const File& b) { return std::tie(a.modified, a.path) < std::tie(b.modified, b.path); });
        std::map<std::string, std::vector<const File*>> resources;
        std::map<std::string, std::set<std::string>> members;
        for (const auto& file : root.files) {
            auto& paths = resources[file.sha];
            if (paths.empty()) root.all.images.push_back(file);
            paths.push_back(&file);
            const auto relative = file.path.lexically_relative((project::directory / files::path(root.all.key)));
            if (std::distance(relative.begin(), relative.end()) < 2) continue;
            const auto concept_name = files::utf8(*relative.begin());
            auto& collection        = *std::ranges::find(root.concepts, concept_name, &Collection::name);
            if (members[concept_name].insert(file.sha).second) collection.images.push_back(file);
        }
        for (const auto& [sha, paths] : resources) {
            if (std::ranges::all_of(paths, [&](const File* file) { return file->entity == paths.front()->entity; })) continue;
            auto& conflict = root.conflicts.emplace_back();
            for (const auto* file : paths) conflict.push_back(file->path);
        }
        root.ready = root.conflicts.empty();
    }
    void Index::load_cache() {
        if (loaded) return;
        const auto cache_path = project::state_directory / "hashes.json";
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
        loaded = true;
    }
} // namespace genesia::dataset
