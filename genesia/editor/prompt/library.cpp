module;
#include <nlohmann/json.hpp>
module genesia.editor.prompt.library;
import genesia.io.files;
import std;

namespace genesia::editor::prompts {
    namespace {
        nlohmann::json read_json(const std::filesystem::path& path) {
            try {
                std::ifstream file{path};
                file.exceptions(std::ios::badbit | std::ios::failbit);
                return nlohmann::json::parse(file);
            } catch (const std::exception& error) {
                throw std::runtime_error{std::format("{}: {}", files::utf8(path), error.what())};
            }
        }

        void read_options(const nlohmann::json& json, std::vector<std::string>& order, std::map<std::string, std::array<std::string, 2>>& options) {
            for (const auto& input : json) {
                const auto name = input.at("name").get<std::string>();
                order.push_back(name);
                options[name] = {input.at("positive"), input.at("negative")};
            }
        }
    } // namespace

    Library::Library(std::filesystem::path root) : directory{std::move(root)} {
        const auto manifest = read_json(directory / "library.json");
        character_order     = manifest.at("characters").get<std::vector<std::string>>();
        category_order      = manifest.at("categories").get<std::vector<std::string>>();
        for (const auto& name : category_order) {
            const auto json = read_json(directory / "categories" / files::path(name + ".json"));
            auto& category  = categories[name];
            read_options(json.at("options"), category.order, category.options);
        }
        for (const auto& name : character_order) {
            const auto root       = directory / "characters" / files::path(name);
            const auto json       = read_json(root / "character.json");
            auto& character       = characters[name];
            character.description = json.at("description");
            for (const auto& input : json.at("parts")) {
                auto& part   = character.parts.emplace_back();
                part.name    = input.at("name");
                part.initial = input.at("default");
                read_options(input.at("options"), part.order, part.options);
            }
        }
        for (const auto& input : read_json(directory / "rules.json")) {
            auto& rule   = rules.emplace_back();
            rule.reason  = input.at("reason");
            rule.disable = input.at("disable").get<std::vector<std::string>>();
            for (const auto& [key, choices] : input.at("when").items())
                for (const auto& choice : choices) rule.when[key].push_back(choice.is_null() ? std::nullopt : std::optional{choice.get<std::string>()});
        }
    }

    void select_character(const Library& library, Recipe& recipe, std::string character) {
        const auto& definition = library.characters.at(character);
        recipe.character       = std::move(character);
        recipe.parts.clear();
        recipe.categories.clear();
        for (const auto& part : definition.parts) recipe.parts.emplace(part.name, part.initial);
        for (const auto& name : library.category_order) recipe.categories.emplace(name, std::nullopt);
    }

    Composition compose(const Library& library, const Recipe& recipe, const prompt::Catalog& catalog) {
        Composition result;
        std::map<std::string, std::optional<std::string>> selections{{"character", recipe.character}};
        for (const auto& [name, option] : recipe.parts) selections.emplace("part." + name, option);
        for (const auto& name : library.category_order) selections.emplace("category." + name, recipe.categories.at(name));
        for (std::size_t i = 0; i < library.rules.size(); ++i) {
            const auto& rule   = library.rules[i];
            const bool matched = std::ranges::all_of(rule.when, [&](const auto& condition) {
                const auto found = selections.find(condition.first);
                return found != selections.end() && std::ranges::contains(condition.second, found->second);
            });
            if (matched)
                for (const auto& target : rule.disable) result.disabled[target].push_back(i);
        }
        const auto append = [&](const std::string& key, const std::array<std::string, 2>& text) {
            if (result.disabled.contains(key)) return;
            for (std::size_t side = 0; side < 2; ++side) {
                if (text[side].empty()) continue;
                if (!result.text[side].empty()) result.text[side] += ", ";
                result.text[side] += text[side];
            }
        };
        const auto& character = library.characters.at(recipe.character);
        for (const auto& part : character.parts) append("part." + part.name, part.options.at(recipe.parts.at(part.name)));
        for (const auto& name : library.category_order) {
            const auto& selected = recipe.categories.at(name);
            if (!selected) continue;
            append("category." + name, library.categories.at(name).options.at(*selected));
        }
        const std::array sides{&recipe.free.positive, &recipe.free.negative};
        for (std::size_t side = 0; side < 2; ++side) {
            const auto text = prompt::compose(catalog, *sides[side]);
            if (text.empty()) continue;
            if (!result.text[side].empty()) result.text[side] += ", ";
            result.text[side] += text;
        }
        return result;
    }

    Preset read_preset(const std::filesystem::path& directory, std::string name, const prompt::Catalog& catalog) {
        const auto json = read_json(directory / "presets" / files::path(name + ".json"));
        Preset result{std::move(name)};
        auto& recipe     = result.recipe;
        recipe.character = json.at("character");
        recipe.parts     = json.at("parts").get<std::map<std::string, std::string>>();
        for (const auto& [category, value] : json.at("categories").items()) recipe.categories.emplace(category, value.is_null() ? std::nullopt : std::optional{value.get<std::string>()});
        for (const auto& [key, side] : {std::pair{"positive", &recipe.free.positive}, std::pair{"negative", &recipe.free.negative}}) {
            const auto& input = json.at("free").at(key);
            side->fixed       = input.at("fixed");
            for (const auto& source : input.at("groups")) {
                auto& group = side->groups.emplace_back(std::vector<prompt::Tag>{}, source.at("enabled").get<bool>());
                for (const auto& tag : source.at("tags")) {
                    const auto value = prompt::parse_tag(catalog, tag.get<std::string>());
                    if (!value) throw std::invalid_argument{std::format("{}: {}", key, value.error().message)};
                    const auto id = catalog.resolve(value->name);
                    if (!id) throw std::invalid_argument{std::format("{}: {}", key, id.error())};
                    group.tags.push_back({*id, value->weight});
                }
            }
        }
        return result;
    }

    void write_preset(const std::filesystem::path& directory, const Preset& preset, const prompt::Catalog& catalog, const bool replace) {
        const auto& recipe = preset.recipe;
        nlohmann::json json{{"character", recipe.character}, {"parts", recipe.parts}, {"categories", nlohmann::json::object()}};
        for (const auto& [name, option] : recipe.categories) json["categories"][name] = option ? nlohmann::json(*option) : nlohmann::json{};
        for (const auto& [key, side] : {std::pair{"positive", &recipe.free.positive}, std::pair{"negative", &recipe.free.negative}}) {
            auto& output     = json["free"][key];
            output["fixed"]  = side->fixed;
            output["groups"] = nlohmann::json::array();
            for (const auto& group : side->groups) {
                nlohmann::json tags = nlohmann::json::array();
                for (const auto tag : group.tags) tags.push_back(prompt::serialize(catalog, std::span{&tag, 1}));
                output["groups"].push_back({{"enabled", group.enabled}, {"tags", std::move(tags)}});
            }
        }
        std::filesystem::create_directories(directory / "presets");
        std::ofstream file{directory / "presets" / files::path(preset.name + ".json"), std::ios::out | (replace ? std::ios::trunc : std::ios::noreplace)};
        file.exceptions(std::ios::badbit | std::ios::failbit);
        file << json.dump(2) << '\n';
        file.close();
    }

    std::vector<std::string> list_presets(const std::filesystem::path& directory) {
        std::vector<std::string> result;
        for (const auto& entry : std::filesystem::directory_iterator{directory / "presets"})
            if (entry.is_regular_file() && entry.path().extension() == ".json") result.push_back(files::utf8(entry.path().stem()));
        std::ranges::sort(result);
        return result;
    }

    void export_prompt(const std::filesystem::path& directory, const std::string_view name, const Composition& composition) {
        std::filesystem::create_directories(directory / "exports");
        std::ofstream file{directory / "exports" / files::path(std::string{name} + ".json")};
        file.exceptions(std::ios::badbit | std::ios::failbit);
        file << nlohmann::json{{"positive", composition.text[0]}, {"negative", composition.text[1]}}.dump(2) << '\n';
        file.close();
    }
} // namespace genesia::editor::prompts
