module;
#include <nlohmann/json.hpp>
module genesia.prompt.library;
import genesia.io.files;
import std;

namespace genesia::prompts {
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

        void read_options(const nlohmann::json& json, std::vector<std::string>& order, std::map<std::string, std::array<std::string, 2>>& options, const prompt::Catalog& catalog, const std::string& source) {
            for (const auto& input : json) {
                const auto name = input.at("name").get<std::string>();
                order.push_back(name);
                options[name] = {input.at("positive"), input.at("negative")};
                for (std::size_t side = 0; side < 2; ++side) prompt::check(catalog, options.at(name)[side], std::format("{} / {} / {}", source, name, side ? "negative" : "positive"));
            }
        }
    } // namespace

    Library::Library(std::filesystem::path root, const prompt::Catalog& catalog) : directory{std::move(root)} {
        for (const auto& entry : std::filesystem::directory_iterator{directory / "characters"}) {
            if (!entry.is_directory()) continue;
            const auto json       = read_json(entry.path() / "character.json");
            auto& character       = characters[files::utf8(entry.path().filename())];
            character.description = json.at("description");
            character.hidden      = json.at("hidden").get<std::vector<std::string>>();
            for (const auto& input : json.at("parts")) {
                auto& part   = character.parts.emplace_back();
                part.name    = input.at("name");
                part.initial = input.at("default").get<std::string>();
                read_options(input.at("options"), part.order, part.options, catalog, std::format("{} / parts / {}", files::utf8(entry.path() / "character.json"), part.name));
            }
        }
        if (!std::filesystem::exists(directory / "scenes")) return;
        for (const auto& category : std::filesystem::directory_iterator{directory / "scenes"}) {
            if (!category.is_directory()) {
                if (category.path().extension() == ".json") throw std::runtime_error{std::format("Place scene definitions inside a category folder: {}", files::utf8(category.path()))};
                continue;
            }
            for (const auto& entry : std::filesystem::directory_iterator{category.path()}) {
                if (!entry.is_regular_file() || entry.path().extension() != ".json") continue;
                const auto [position, inserted] = scenes.try_emplace(files::utf8(entry.path().stem()));
                if (!inserted) throw std::runtime_error{std::format("Scene names must be unique across categories: '{}' in {} and {}", position->first, position->second.category, files::utf8(category.path().filename()))};
                const auto json   = read_json(entry.path());
                auto& scene       = position->second;
                scene.category    = files::utf8(category.path().filename());
                scene.description = json.at("description");
                scene.text        = {json.at("positive"), json.at("negative")};
                for (std::size_t side = 0; side < 2; ++side) prompt::check(catalog, scene.text[side], std::format("{} / {}", files::utf8(entry.path()), side ? "negative" : "positive"));
                for (const auto& input : json.at("variations")) {
                    auto& variation   = scene.variations.emplace_back();
                    variation.name    = input.at("name");
                    variation.initial = input.at("default").get<std::string>();
                    read_options(input.at("options"), variation.order, variation.options, catalog, std::format("{} / variations / {}", files::utf8(entry.path()), variation.name));
                }
                for (const auto& input : json.at("rules")) {
                    auto& rule   = scene.rules.emplace_back();
                    rule.reason  = input.at("reason");
                    rule.disable = input.at("disable").get<std::vector<std::string>>();
                    rule.require = input.at("require").get<std::vector<std::string>>();
                    rule.when    = input.at("when").get<std::map<std::string, std::vector<std::string>>>();
                }
            }
        }
    }

    void select_character(const Library& library, Recipe& recipe, std::string character) {
        const auto& definition = library.characters.at(character);
        recipe.character       = std::move(character);
        recipe.parts.clear();
        for (const auto& part : definition.parts) recipe.parts.emplace(part.name, part.initial);
    }

    void select_scene(const Library& library, Recipe& recipe, std::optional<std::string> scene) {
        recipe.scene = std::move(scene);
        recipe.variations.clear();
        if (recipe.scene)
            for (const auto& variation : library.scenes.at(*recipe.scene).variations) recipe.variations.emplace(variation.name, variation.initial);
    }

    Composition compose(const Library& library, const Recipe& recipe, const prompt::Catalog& catalog) {
        Composition result;
        const auto& character = library.characters.at(recipe.character);
        std::map<std::string, std::string> selections;
        for (const auto& part : character.parts) {
            selections.emplace("part." + part.name, recipe.parts.at(part.name));
            result.parts[part.name].hidden = std::ranges::contains(character.hidden, part.name);
        }
        const Scene* scene = recipe.scene ? &library.scenes.at(*recipe.scene) : nullptr;
        if (scene) {
            for (const auto& variation : scene->variations) selections.emplace("variation." + variation.name, recipe.variations.at(variation.name));
            for (std::size_t i = 0; i < scene->rules.size(); ++i) {
                const auto& rule = scene->rules[i];
                if (!std::ranges::all_of(rule.when, [&](const auto& condition) {
                        const auto found = selections.find(condition.first);
                        return found != selections.end() && std::ranges::contains(condition.second, found->second);
                    }))
                    continue;
                for (const auto& target : rule.disable)
                    if (const auto part = result.parts.find(target); part != result.parts.end()) part->second.disable.push_back(i);
                for (const auto& target : rule.require) {
                    const auto part = result.parts.find(target);
                    if (part != result.parts.end()) part->second.require.push_back(i);
                    else result.error += std::format("{} requires missing part '{}'.\n  Require: {}\n", *recipe.scene, target, rule.reason);
                }
            }
            for (const auto& part : character.parts) {
                const auto& effect = result.parts.at(part.name);
                if (effect.require.empty() || effect.disable.empty()) continue;
                result.error += std::format("{}: part '{}' is both required and disabled.\n", *recipe.scene, part.name);
                for (std::size_t i = 0; i < scene->rules.size(); ++i) {
                    if (std::ranges::contains(effect.require, i)) result.error += "  Require: " + scene->rules[i].reason + '\n';
                    if (std::ranges::contains(effect.disable, i)) result.error += "  Disable: " + scene->rules[i].reason + '\n';
                }
            }
        }
        if (!result.error.empty()) return result;
        result.text           = card_prompt(library, recipe, result, false);
        const auto scene_text = card_prompt(library, recipe, result, true);
        const std::array sides{&recipe.free.positive, &recipe.free.negative};
        for (std::size_t side = 0; side < 2; ++side) {
            if (!scene_text[side].empty()) {
                if (!result.text[side].empty()) result.text[side] += ", ";
                result.text[side] += scene_text[side];
            }
            const auto text = prompt::compose(catalog, *sides[side]);
            if (text.empty()) continue;
            if (!result.text[side].empty()) result.text[side] += ", ";
            result.text[side] += text;
        }
        return result;
    }

    std::array<std::string, 2> card_prompt(const Library& library, const Recipe& recipe, const Composition& composition, const bool scene) {
        std::array<std::string, 2> result;
        const auto append = [&](const std::array<std::string, 2>& text) {
            for (std::size_t side = 0; side < 2; ++side) {
                if (text[side].empty()) continue;
                if (!result[side].empty()) result[side] += ", ";
                result[side] += text[side];
            }
        };
        if (scene) {
            if (!recipe.scene) return result;
            const auto& definition = library.scenes.at(*recipe.scene);
            append(definition.text);
            for (const auto& variation : definition.variations) append(variation.options.at(recipe.variations.at(variation.name)));
        } else {
            for (const auto& part : library.characters.at(recipe.character).parts) {
                const auto& effect = composition.parts.at(part.name);
                if (!effect.disable.empty() || (effect.hidden && effect.require.empty())) continue;
                append(part.options.at(recipe.parts.at(part.name)));
            }
        }
        return result;
    }

    Preset read_preset(const Library& library, std::string name, const prompt::Catalog& catalog) {
        const auto path = library.directory / "presets" / files::path(name + ".json");
        const auto json = read_json(path);
        Preset result{std::move(name)};
        auto& recipe     = result.recipe;
        recipe.character = json.at("character");
        recipe.parts     = json.at("parts").get<std::map<std::string, std::string>>();
        for (const auto& part : library.characters.at(recipe.character).parts) recipe.parts.try_emplace(part.name, part.initial);
        if (!json.at("scene").is_null()) recipe.scene = json.at("scene").get<std::string>();
        recipe.variations = json.at("variations").get<std::map<std::string, std::string>>();
        if (recipe.scene)
            for (const auto& variation : library.scenes.at(*recipe.scene).variations) recipe.variations.try_emplace(variation.name, variation.initial);
        for (const auto& [key, side] : {std::pair{"positive", &recipe.free.positive}, std::pair{"negative", &recipe.free.negative}}) {
            const auto& input = json.at("free").at(key);
            side->fixed       = input.at("fixed");
            for (const auto& source : input.at("groups")) {
                auto& group = side->groups.emplace_back(std::vector<prompt::Tag>{}, source.at("enabled").get<bool>());
                for (const auto& tag : source.at("tags")) {
                    const auto value = prompt::parse_tag(catalog, tag.get<std::string>());
                    if (!value) throw std::invalid_argument{std::format("{} / free / {} / groups / {} / tags / {}: {}", files::utf8(path), key, side->groups.size() - 1, group.tags.size(), value.error().message)};
                    const auto id = catalog.resolve(value->name);
                    if (!id) throw std::invalid_argument{std::format("{} / free / {} / groups / {} / tags / {}: {}", files::utf8(path), key, side->groups.size() - 1, group.tags.size(), id.error())};
                    group.tags.push_back({*id, value->weight});
                }
            }
        }
        return result;
    }

    void write_preset(const Library& library, const Preset& preset, const prompt::Catalog& catalog, const bool replace) {
        const auto& recipe = preset.recipe;
        nlohmann::json json{{"character", recipe.character}, {"parts", nlohmann::json::object()}, {"scene", recipe.scene ? nlohmann::json(*recipe.scene) : nlohmann::json{}}, {"variations", nlohmann::json::object()}};
        for (const auto& part : library.characters.at(recipe.character).parts) {
            const auto& option = recipe.parts.at(part.name);
            if (option != part.initial) json["parts"][part.name] = option;
        }
        if (recipe.scene)
            for (const auto& variation : library.scenes.at(*recipe.scene).variations) {
                const auto& option = recipe.variations.at(variation.name);
                if (option != variation.initial) json["variations"][variation.name] = option;
            }
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
        std::filesystem::create_directories(library.directory / "presets");
        std::ofstream file{library.directory / "presets" / files::path(preset.name + ".json"), std::ios::out | (replace ? std::ios::trunc : std::ios::noreplace)};
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
        if (!composition.error.empty()) throw std::runtime_error{composition.error};
        std::filesystem::create_directories(directory / "exports");
        std::ofstream file{directory / "exports" / files::path(std::string{name} + ".json")};
        file.exceptions(std::ios::badbit | std::ios::failbit);
        file << nlohmann::json{{"positive", composition.text[0]}, {"negative", composition.text[1]}}.dump(2) << '\n';
        file.close();
    }
} // namespace genesia::prompts
