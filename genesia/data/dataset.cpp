module;
#include <nlohmann/json.hpp>
#if defined(_WIN32)
#include <Windows.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#endif
module genesia.dataset;
import std;
import genesia.hash;
import genesia.files;

namespace genesia::dataset {
    Lock::Lock(const std::string_view name, const bool wait, const std::filesystem::path& directory, const bool shared) {
        std::filesystem::create_directories(directory);
        const auto path = directory / std::format("{}.lock", name);
#if defined(_WIN32)
        const auto file = CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file == INVALID_HANDLE_VALUE) throw std::system_error{static_cast<int>(GetLastError()), std::system_category(), "Open dataset lock"};
        OVERLAPPED operation{};
        if (!LockFileEx(file, (shared ? 0 : LOCKFILE_EXCLUSIVE_LOCK) | (wait ? 0 : LOCKFILE_FAIL_IMMEDIATELY), 0, 1, 0, &operation)) {
            const auto error = GetLastError();
            CloseHandle(file);
            if (!wait && error == ERROR_LOCK_VIOLATION) return;
            throw std::system_error{static_cast<int>(error), std::system_category(), "Lock dataset"};
        }
        handle = reinterpret_cast<std::intptr_t>(file);
#else
        handle = open(path.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0666);
        if (handle == -1) throw std::system_error{errno, std::generic_category(), "Open dataset lock"};
        if (flock(static_cast<int>(handle), (shared ? LOCK_SH : LOCK_EX) | (wait ? 0 : LOCK_NB)) == -1) {
            const auto error = errno;
            close(static_cast<int>(handle));
            if (!wait && error == EWOULDBLOCK) return;
            throw std::system_error{error, std::generic_category(), "Lock dataset"};
        }
#endif
        acquired = true;
    }

    Lock::~Lock() {
        if (!acquired) return;
#if defined(_WIN32)
        CloseHandle(reinterpret_cast<HANDLE>(handle));
#else
        close(static_cast<int>(handle));
#endif
    }

    ConceptType parse_concept_type(const std::string_view name) {
        const auto found = std::ranges::find(concept_types, name);
        if (found == concept_types.end()) throw std::runtime_error{"Unknown concept type: " + std::string(name)};
        return static_cast<ConceptType>(found - concept_types.begin());
    }
    void to_json(nlohmann::json& json, const Concept& value) {
        json = {{"version", 2}, {"type", concept_types[static_cast<std::size_t>(value.type)]}, {"locked", value.locked}};
    }
    Concept read_concept(const std::string_view key) {
        const auto relative = files::path(key);
        if (relative.is_absolute() || std::distance(relative.begin(), relative.end()) != 2 || std::ranges::any_of(relative, [](const auto& part) { return files::utf8(part).starts_with('.'); })) throw std::runtime_error{"Expected ROOT/CONCEPT"};
        Concept result{files::utf8(relative), directory / relative};
        if (!std::filesystem::is_directory(result.path)) throw std::runtime_error{"Concept not found: " + result.key};
        const auto state    = result.path / ".genesia";
        const auto manifest = state / "concept.json";
        if (std::filesystem::exists(manifest)) {
            const auto value = files::read_json(manifest);
            if (value.at("version") != 2) throw std::runtime_error{"Old or unsupported concept state: " + result.key + ". Remove the old training artifacts before assigning a type."};
            result.type   = parse_concept_type(value.at("type").get_ref<const std::string&>());
            result.locked = value.at("locked").get<bool>();
            if (result.type == ConceptType::none && result.locked) throw std::runtime_error{"An unassigned concept cannot have a locked training type: " + result.key};
        } else if (std::filesystem::exists(state) && !std::filesystem::is_empty(state)) throw std::runtime_error{"Concept state has no type manifest: " + result.key};
        return result;
    }
    Concept assign_type(const std::string_view key, const ConceptType type) {
        const auto normalized = files::utf8(files::path(key));
        const auto identity   = sha256({reinterpret_cast<const unsigned char*>(normalized.data()), normalized.size()});
        const Lock pending{"concept-type-" + identity, false};
        if (!pending.acquired) throw std::runtime_error{"Concept has queued or running training: " + normalized};
        const Lock operation{"concept-" + identity, false};
        if (!operation.acquired) throw std::runtime_error{"Concept is in use: " + normalized};
        auto result = read_concept(normalized);
        if (result.locked) throw std::runtime_error{"Concept type is locked by its training history: " + result.key};
        result.type = type;
        files::write_json(result.path / ".genesia" / "concept.json", result);
        return result;
    }

    Png read_png(const std::filesystem::path& path) {
        std::ifstream file{path, std::ios::binary};
        file.exceptions(std::ios::badbit | std::ios::failbit);
        std::array<unsigned char, 8> signature;
        file.read(reinterpret_cast<char*>(signature.data()), signature.size());
        if (signature != std::array<unsigned char, 8>{137, 80, 78, 71, 13, 10, 26, 10}) throw std::runtime_error{std::format("Not a PNG: {}", path.string())};
        Png result;
        nlohmann::json metadata;
        for (;;) {
            std::uint32_t length;
            std::array<char, 4> type;
            file.read(reinterpret_cast<char*>(&length), 4);
            file.read(type.data(), 4);
            length = std::byteswap(length);
            const std::string_view chunk{type.data(), type.size()};
            if (chunk == "IEND") break;
            if (chunk == "IHDR") {
                std::array<std::uint32_t, 2> dimensions;
                file.read(reinterpret_cast<char*>(dimensions.data()), 8);
                result.width  = static_cast<int>(std::byteswap(dimensions[0]));
                result.height = static_cast<int>(std::byteswap(dimensions[1]));
                file.seekg(length - 8, std::ios::cur);
            } else if (chunk == "iTXt") {
                std::string text(length, '\0');
                file.read(text.data(), text.size());
                if (text.starts_with(std::string_view{"genesia\0", 8})) {
                    if (!text.starts_with(std::string_view{"genesia\0\0\0\0\0", 12})) throw std::runtime_error{"Unsupported Genesia PNG metadata encoding"};
                    metadata = nlohmann::json::parse(text.begin() + 12, text.end());
                }
            } else file.seekg(length, std::ios::cur);
            file.seekg(4, std::ios::cur);
        }
        if (metadata.is_null()) throw std::runtime_error{std::format("Missing Genesia PNG metadata: {}", path.string())};
        if (metadata.at("version") != 1) throw std::runtime_error{"Unsupported Genesia PNG metadata version"};
        result.model = metadata.at("model");
        result.seed  = metadata.at("seed");
        result.steps = metadata.at("steps");
        result.cfg   = metadata.at("cfg");
        for (const auto& [name, side] : {std::pair{"positive", &result.prompt[0]}, std::pair{"negative", &result.prompt[1]}}) {
            const auto& saved = metadata.at("prompt").at(name);
            side->text        = saved.at("text");
            side->fixed       = saved.at("fixed");
            for (const auto& group : saved.at("groups").get_ref<const nlohmann::json::array_t&>()) {
                auto& parsed = side->groups.emplace_back(group.at("enabled").get<bool>());
                for (const auto& tag : group.at("tags").get_ref<const nlohmann::json::array_t&>()) {
                    auto& item = parsed.tags.emplace_back(tag.at("name").get<std::string>(), std::nullopt, tag.at("weight").get<float>());
                    if (tag.contains("text")) item.text = tag.at("text").get<std::string>();
                }
            }
        }
        if (metadata.contains("repaint")) {
            result.source  = metadata.at("repaint").at("source");
            result.denoise = metadata.at("repaint").at("denoise");
        }
        return result;
    }

    void Index::scan(const std::optional<std::string> root) {
        const Lock lock{"index"};
        const auto cache_path = state_directory / "hashes.json";
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
        std::filesystem::create_directories(raw);
        roots.clear();
        std::vector<std::filesystem::path> folders;
        if (root) folders.push_back(directory / *root);
        else
            for (const auto& entry : std::filesystem::directory_iterator{directory})
                if (entry.is_directory() && !entry.path().filename().string().starts_with('.')) folders.push_back(entry.path());
        std::ranges::sort(folders, [](const auto& a, const auto& b) { return a.filename() == "raw" ? b.filename() != "raw" : b.filename() == "raw" ? false : a.filename() < b.filename(); });
        for (const auto& folder : folders) {
            auto& found   = roots.emplace_back();
            found.all.key = found.all.name = folder.filename().string();
            try {
                for (const auto& entry : std::filesystem::directory_iterator{folder})
                    if (entry.is_directory() && !entry.path().filename().string().starts_with('.')) found.concepts.push_back({(entry.path().lexically_relative(directory)).generic_string(), entry.path().filename().string()});
                std::ranges::sort(found.concepts, {}, &Collection::name);
                for (auto iterator = std::filesystem::recursive_directory_iterator{folder}; iterator != std::filesystem::recursive_directory_iterator{}; ++iterator) {
                    const auto& entry = *iterator;
                    if (entry.is_directory()) {
                        if (entry.path().filename().string().starts_with('.')) iterator.disable_recursion_pending();
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
                    const auto concept_name = relative.begin()->string();
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
                found.error = std::format("{}: {}", folder.string(), error.what());
            }
        }
        nlohmann::json files = nlohmann::json::object();
        for (const auto& [entity, entry] : cache) files[entity] = {{"modified", entry.modified}, {"bytes", entry.bytes}, {"sha", entry.sha}, {"width", entry.width}, {"height", entry.height}};
        const auto temporary = state_directory / "hashes.json.part";
        std::ofstream output{temporary, std::ios::binary | std::ios::trunc};
        output.exceptions(std::ios::badbit | std::ios::failbit);
        output << nlohmann::json{{"version", 1}, {"files", std::move(files)}}.dump();
        output.close();
#if defined(_WIN32)
        if (!MoveFileExW(temporary.c_str(), cache_path.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) throw std::system_error{static_cast<int>(GetLastError()), std::system_category(), "Publish dataset hash cache"};
#else
        std::filesystem::rename(temporary, cache_path);
#endif
    }

    File Index::identify(const std::filesystem::path& path) {
        File result;
        result.path = path;
#if defined(_WIN32)
        const auto handle = CreateFileW(path.c_str(), FILE_READ_ATTRIBUTES, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (handle == INVALID_HANDLE_VALUE) throw std::system_error{static_cast<int>(GetLastError()), std::system_category(), path.string()};
        const std::unique_ptr<std::remove_pointer_t<HANDLE>, decltype(&CloseHandle)> file{handle, CloseHandle};
        FILE_ID_INFO identity;
        FILE_BASIC_INFO times;
        FILE_STANDARD_INFO size;
        if (!GetFileInformationByHandleEx(handle, FileIdInfo, &identity, sizeof(identity)) || !GetFileInformationByHandleEx(handle, FileBasicInfo, &times, sizeof(times)) || !GetFileInformationByHandleEx(handle, FileStandardInfo, &size, sizeof(size))) throw std::system_error{static_cast<int>(GetLastError()), std::system_category(), path.string()};
        result.entity = std::format("{:016x}:", identity.VolumeSerialNumber);
        for (const auto byte : identity.FileId.Identifier) result.entity += std::format("{:02x}", byte);
        result.modified = times.LastWriteTime.QuadPart;
        result.bytes    = size.EndOfFile.QuadPart;
#else
        struct stat info;
        if (stat(path.c_str(), &info) == -1) throw std::system_error{errno, std::generic_category(), path.string()};
        result.entity   = std::format("{}:{}", info.st_dev, info.st_ino);
        result.modified = info.st_mtim.tv_sec * 1'000'000'000LL + info.st_mtim.tv_nsec;
        result.bytes    = info.st_size;
#endif
        auto entry = cache.find(result.entity);
        if (entry == cache.end() || entry->second.modified != result.modified || entry->second.bytes != result.bytes) {
            std::ifstream input{path, std::ios::binary};
            input.exceptions(std::ios::badbit);
            if (!input) throw std::runtime_error{"Cannot read image: " + path.string()};
            Sha256 hash;
            std::vector<unsigned char> buffer(1024 * 1024);
            while (input.read(reinterpret_cast<char*>(buffer.data()), buffer.size()) || input.gcount()) hash.update(std::span{buffer.data(), static_cast<std::size_t>(input.gcount())});
            if (!input.eof()) throw std::runtime_error{"Cannot read image: " + path.string()};
            std::string sha;
            for (const auto byte : hash.finish()) sha += std::format("{:02x}", byte);
            const auto png = read_png(path);
            entry          = cache.insert_or_assign(result.entity, Cached{result.modified, result.bytes, std::move(sha), png.width, png.height}).first;
        }
        result.sha    = entry->second.sha;
        result.width  = entry->second.width;
        result.height = entry->second.height;
        return result;
    }
} // namespace genesia::dataset
