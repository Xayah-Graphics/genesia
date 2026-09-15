module;
#include <nlohmann/json.hpp>
module genesia.data.captions;
import genesia.runtime.progress;
import std;

namespace genesia::caption {
    void to_json(nlohmann::json& json, const Result& result) {
        json = {{"folder", result.folder}, {"tags", result.tags}, {"effective", result.effective}, {"caption", compose(result.effective)}, {"bypass", result.bypass}, {"excluded", result.excluded}};
    }
    std::vector<std::string> parse(const std::string_view text) {
        std::vector<std::string> tags;
        for (const auto part : text | std::views::split(',')) {
            const std::string_view value{part.begin(), part.end()};
            const auto begin = value.find_first_not_of(" \t\r\n");
            if (begin == std::string_view::npos) continue;
            const std::string tag{value.substr(begin, value.find_last_not_of(" \t\r\n") - begin + 1)};
            if (tag.find_first_of("\r\n") != std::string::npos) throw std::runtime_error{"A caption tag must be a single line"};
            if (!std::ranges::contains(tags, tag)) tags.push_back(tag);
        }
        return tags;
    }
    std::vector<std::string> resolve(const Document& document, const std::string_view concept_key, const std::string_view folder) {
        std::vector<std::string> result{files::utf8(files::path(concept_key).filename())};
        auto path = files::path(folder);
        for (;;) {
            const auto found = document.folders.find(files::utf8(path));
            if (found != document.folders.end())
                for (const auto& tag : found->second)
                    if (!std::ranges::contains(result, tag)) result.push_back(tag);
            if (path == ".") break;
            path = path.has_parent_path() ? path.parent_path() : std::filesystem::path{"."};
        }
        return result;
    }
    std::string compose(const std::span<const std::string> tags) {
        std::string result;
        for (const auto& tag : tags) {
            if (!result.empty()) result += ", ";
            result += tag;
        }
        return result;
    }
    bool excluded(const Document& document, const std::string_view folder) {
        auto path = files::path(folder);
        for (;;) {
            if (document.bypass.contains(files::utf8(path))) return true;
            if (path == ".") return false;
            path = path.has_parent_path() ? path.parent_path() : std::filesystem::path{"."};
        }
    }
    Dataset inspect(const dataset::Concept& assigned, const dataset::Root& root) {
        Dataset result{.key = assigned.key, .issue = dataset::lora_issue(root, assigned.key)};
        const auto path = assigned.path / ".genesia" / "captions.json";
        if (std::filesystem::exists(path)) {
            const auto json = files::read_json(path);
            if (json.at("version") != 1) throw std::runtime_error{"Unsupported caption format: " + files::utf8(path)};
            json.at("folders").get_to(result.document.folders);
            if (const auto bypass = json.find("bypass"); bypass != json.end()) bypass->get_to(result.document.bypass);
        }
        std::map<std::string, std::pair<std::set<std::string>, std::set<std::string>>> folders;
        folders["."];
        for (const auto& directory : root.directories) {
            const auto relative = directory.lexically_relative(assigned.path);
            if (!relative.empty() && *relative.begin() != "..") folders[files::utf8(relative)];
        }
        for (const auto& file : root.files) {
            auto relative = file.path.parent_path().lexically_relative(assigned.path);
            if (relative.empty() || *relative.begin() == "..") continue;
            folders[files::utf8(relative)].first.insert(file.sha);
            for (;;) {
                folders[files::utf8(relative)].second.insert(file.sha);
                if (relative == ".") break;
                relative = relative.has_parent_path() ? relative.parent_path() : std::filesystem::path{"."};
            }
        }
        if (root.error.empty()) {
            const auto removed_tags   = std::erase_if(result.document.folders, [&](const auto& entry) { return !folders.contains(entry.first); });
            const auto removed_bypass = std::erase_if(result.document.bypass, [&](const auto& name) { return !folders.contains(name); });
            if (removed_tags || removed_bypass) files::write_json(path, {{"version", 1}, {"folders", result.document.folders}, {"bypass", result.document.bypass}});
        }
        std::set<std::string> export_images;
        for (const auto& [name, counts] : folders) {
            const bool bypassed = excluded(result.document, name);
            result.folders.push_back({name, counts.first.size(), counts.second.size(), bypassed});
            if (bypassed) continue;
            export_images.insert_range(counts.first);
            const auto tags = result.document.folders.find(name);
            if (tags == result.document.folders.end() || tags->second.empty()) result.missing_tags.insert(name);
        }
        result.export_images = export_images.size();
        return result;
    }
    Result edit(const Dataset& source, const std::string_view folder, const std::optional<std::vector<std::string>>& tags, const std::optional<bool> bypass) {
        if (!std::ranges::contains(source.folders, folder, &Folder::path)) throw std::runtime_error{"Caption folder does not exist: " + std::string{folder}};
        auto document = source.document;
        if (tags) {
            auto normalized = parse(compose(*tags));
            if (normalized.empty()) document.folders.erase(std::string{folder});
            else document.folders[std::string{folder}] = std::move(normalized);
        }
        if (bypass) {
            if (*bypass) document.bypass.emplace(folder);
            else document.bypass.erase(std::string{folder});
        }
        if (tags || bypass) files::write_json(project::directory / files::path(source.key) / ".genesia" / "captions.json", {{"version", 1}, {"folders", document.folders}, {"bypass", document.bypass}});
        const auto found = document.folders.find(std::string{folder});
        return {std::string{folder}, found == document.folders.end() ? std::vector<std::string>{} : found->second, resolve(document, source.key, folder), document.bypass.contains(std::string{folder}), excluded(document, folder)};
    }
    Exported export_dataset(const Dataset& source, const dataset::Root& root, const std::filesystem::path& destination, const std::atomic_bool& interrupted, const std::function<std::filesystem::path(const dataset::File&)>& mask, const std::function<void(std::size_t, std::size_t)>& progress) {
        if (!source.issue.empty()) throw std::runtime_error{source.issue};
        if (!root.ready) throw std::runtime_error{root.error.empty() ? "Dataset root has independent image copies" : root.error};
        if (!source.missing_tags.empty()) {
            std::string error = std::format("{} non-bypassed directories need their own tags, including empty directories. Inherited tags do not satisfy this requirement:", source.missing_tags.size());
            for (const auto& path : source.missing_tags) error += "\n" + source.key + (path == "." ? " (concept root)" : "/" + path);
            throw std::runtime_error{error};
        }
        const auto folder = project::directory / files::path(source.key);
        auto output       = std::filesystem::weakly_canonical(std::filesystem::absolute(destination));
        if (!output.has_filename()) output = output.parent_path();
        for (auto parent = output.parent_path(); parent != parent.root_path(); parent = parent.parent_path())
            if (std::filesystem::exists(parent) && std::filesystem::equivalent(parent, project::directory)) throw std::runtime_error{"Export outside the project's data directory"};
        if (std::filesystem::exists(output)) throw std::runtime_error{"Export requires a new directory: " + files::utf8(output)};
        std::vector<std::pair<const dataset::File*, std::string>> samples;
        for (const auto& file : root.files) {
            const auto path = file.path.lexically_relative(folder);
            if (path.empty() || *path.begin() == "..") continue;
            const auto parent = path.has_parent_path() ? files::utf8(path.parent_path()) : ".";
            if (excluded(source.document, parent)) continue;
            samples.emplace_back(&file, compose(resolve(source.document, source.key, parent)));
        }
        if (samples.empty()) throw std::runtime_error{"No images to export"};
        // Build beside the destination. An interrupted export never looks complete.
        const auto staging = std::filesystem::path{output.native() + std::filesystem::path{".genesia-export"}.native()};
        if (!std::filesystem::create_directories(staging)) throw std::runtime_error{"Unfinished export directory exists: " + files::utf8(staging)};
        std::size_t completed{};
        try {
            progress(0, samples.size());
            for (const auto& [file, text] : samples) {
                if (interrupted.load()) throw runtime::Stopped{};
                const auto mask_path = mask(*file);
                if (interrupted.load()) throw runtime::Stopped{};
                const auto target = staging / file->path.lexically_relative(folder);
                std::filesystem::create_directories(target.parent_path());
                std::filesystem::copy_file(file->path, target);
                const auto mask_target = target.parent_path() / (target.stem().native() + std::filesystem::path{"-masklabel.png"}.native());
                std::filesystem::copy_file(mask_path, mask_target);
                auto caption = target;
                caption.replace_extension(".txt");
                files::write_bytes(caption, {reinterpret_cast<const std::uint8_t*>(text.data()), text.size()});
                progress(++completed, samples.size());
            }
            if (interrupted.load()) throw runtime::Stopped{};
            files::move(staging, output);
        } catch (...) {
            std::error_code error;
            std::filesystem::remove_all(staging, error);
            if (error) throw std::runtime_error{"Export failed; cannot remove unfinished files at " + files::utf8(staging) + ": " + error.message()};
            throw;
        }
        return {output, completed};
    }
} // namespace genesia::caption
