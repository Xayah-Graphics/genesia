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

        auto& read_reference(auto& values, const std::string& name, const std::string& source) {
            const auto found = values.find(name);
            if (found != values.end()) return *found;
            const auto available = values | std::views::keys | std::ranges::to<std::vector<std::string>>();
            throw std::runtime_error{std::format("{}:\n  Unknown name '{}'.\n  Available names: {}", source, name, nlohmann::json(available).dump())};
        }

        std::array<std::string, 2> read_text(const nlohmann::json& json, const prompt::Catalog& catalog, const std::string& source) {
            std::array<std::string, 2> text;
            for (std::size_t side = 0; side < 2; ++side) {
                const auto key     = side ? "negative" : "positive";
                const auto context = std::format("{} / {}", source, key);
                try {
                    text[side] = json.at(key).get<std::string>();
                } catch (const nlohmann::json::exception& error) {
                    throw std::runtime_error{std::format("{}: {}", context, error.what())};
                }
                prompt::check(catalog, text[side], context);
            }
            return text;
        }

        Choices read_choices(const nlohmann::json& json, const prompt::Catalog& catalog, const std::string& source) {
            auto context = source + " / name";
            try {
                Choices result;
                result.name           = json.at("name").get<std::string>();
                const auto group_path = std::format("{} / {}", source, result.name);
                context               = group_path + " / default";
                result.initial        = json.at("default").get<std::string>();
                context               = group_path + " / options";
                for (const auto& input : json.at("options")) {
                    context         = std::format("{} / options / {} / name", group_path, result.order.size());
                    const auto name = input.at("name").get<std::string>();
                    result.order.push_back(name);
                    auto& option    = result.options[name];
                    const auto path = std::format("{} / options / {}", group_path, name);
                    option.text     = read_text(input, catalog, path);
                    context         = path + " / suboptions";
                    if (const auto groups = input.find("suboptions"); groups != input.end())
                        for (const auto& group : *groups) {
                            context               = std::format("{} / suboptions / {} / name", path, option.suboptions.size());
                            auto& suboptions      = option.suboptions.emplace_back();
                            suboptions.name       = group.at("name").get<std::string>();
                            const auto child_path = std::format("{} / suboptions / {}", path, suboptions.name);
                            context               = child_path + " / default";
                            suboptions.initial    = group.at("default").get<std::string>();
                            context               = child_path + " / options";
                            for (const auto& item : group.at("options")) {
                                context          = std::format("{} / options / {} / name", child_path, suboptions.order.size());
                                const auto value = item.at("name").get<std::string>();
                                context          = std::format("{} / options / {}", child_path, value);
                                if (item.contains("suboptions")) throw std::runtime_error{context + ": Suboptions support only one level"};
                                suboptions.order.push_back(value);
                                suboptions.options[value] = read_text(item, catalog, context);
                            }
                        }
                }
                return result;
            } catch (const nlohmann::json::exception& error) {
                throw std::runtime_error{std::format("{}: {}", context, error.what())};
            }
        }

        void read_selections(const nlohmann::json& json, const std::vector<Choices>& groups, std::map<std::string, Selection>& selections, const std::string& source, const std::string& definition, const std::string_view key) {
            auto context = std::format("{} / {}", source, key);
            try {
                const auto& options   = json.at(key);
                context               = std::format("{} / suboptions / {}", source, key);
                const auto suboptions = json.value("suboptions", nlohmann::json::object()).value(key, nlohmann::json::object());
                for (const auto& group : groups) {
                    context             = std::format("{} / {} / {}", source, key, group.name);
                    const auto selected = options.value(group.name, group.initial);
                    if (!options.contains(group.name)) context = std::format("{} / {} / {} / default", definition, key, group.name);
                    const auto& [name, option] = read_reference(group.options, selected, context);
                    auto& selection            = selections[group.name];
                    selection.option           = name;
                    for (const auto& child : option.suboptions) {
                        const auto field = std::format("{} / {} / {} / options / {} / suboptions / {} / default", definition, key, group.name, name, child.name);
                        selection.suboptions.emplace(child.name, read_reference(child.options, child.initial, field).first);
                    }
                    if (const auto values = suboptions.find(group.name); values != suboptions.end())
                        for (const auto& [child, value] : values->items()) {
                            context              = std::format("{} / suboptions / {} / {} / {} (option '{}')", source, key, group.name, child, name);
                            auto& selected_child = read_reference(selection.suboptions, child, context).second;
                            const auto& choices  = *std::ranges::find(option.suboptions, child, &Suboptions::name);
                            selected_child       = read_reference(choices.options, value.get<std::string>(), context).first;
                        }
                }
            } catch (const nlohmann::json::exception& error) {
                throw std::runtime_error{std::format("{}: {}", context, error.what())};
            }
        }

        void write_selections(nlohmann::json& options, nlohmann::json& suboptions, const std::vector<Choices>& groups, const std::map<std::string, Selection>& selections) {
            for (const auto& group : groups) {
                const auto& selected = selections.at(group.name);
                if (selected.option != group.initial) options[group.name] = selected.option;
                for (const auto& child : group.options.at(selected.option).suboptions) {
                    const auto& value = selected.suboptions.at(child.name);
                    if (value != child.initial) suboptions[group.name][child.name] = value;
                }
            }
        }
    } // namespace

    Library::Library(std::filesystem::path root, const prompt::Catalog& catalog) : directory{std::move(root)} {
        std::string context;
        try {
            for (const auto& entry : std::filesystem::directory_iterator{directory / "characters"}) {
                if (!entry.is_directory()) continue;
                const auto source     = files::utf8(entry.path() / "character.json");
                const auto json       = read_json(entry.path() / "character.json");
                auto& character       = characters[files::utf8(entry.path().filename())];
                context               = source + " / description";
                character.description = json.at("description");
                context               = source + " / hidden";
                character.hidden      = json.at("hidden").get<std::vector<std::string>>();
                context               = source + " / parts";
                for (const auto& input : json.at("parts")) character.parts.push_back(read_choices(input, catalog, context));
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
                    const auto source = files::utf8(entry.path());
                    auto& scene       = position->second;
                    scene.category    = files::utf8(category.path().filename());
                    context           = source + " / description";
                    scene.description = json.at("description");
                    scene.text        = read_text(json, catalog, source);
                    context           = source + " / variations";
                    for (const auto& input : json.at("variations")) scene.variations.push_back(read_choices(input, catalog, context));
                    context = source + " / rules";
                    for (const auto& input : json.at("rules")) {
                        const auto rule_path = std::format("{} / rules / {}", source, scene.rules.size());
                        auto& rule           = scene.rules.emplace_back();
                        context              = rule_path + " / reason";
                        rule.reason          = input.at("reason");
                        context              = rule_path + " / disable";
                        rule.disable         = input.at("disable").get<std::vector<std::string>>();
                        context              = rule_path + " / require";
                        rule.require         = input.at("require").get<std::vector<std::string>>();
                        context              = rule_path + " / when";
                        rule.when            = input.at("when").get<std::map<std::string, std::vector<std::string>>>();
                    }
                }
            }
        } catch (const nlohmann::json::exception& error) {
            throw std::runtime_error{std::format("{}: {}", context, error.what())};
        }
    }

    void select_option(const Choices& choices, Selection& selection, std::string option) {
        const auto& definition = choices.options.at(option);
        selection.option       = std::move(option);
        selection.suboptions.clear();
        for (const auto& group : definition.suboptions) selection.suboptions.emplace(group.name, group.initial);
    }

    void select_character(const Library& library, Recipe& recipe, std::string character) {
        const auto& definition = library.characters.at(character);
        recipe.character       = std::move(character);
        recipe.parts.clear();
        for (const auto& part : definition.parts) select_option(part, recipe.parts[part.name], part.initial);
    }

    void select_scene(const Library& library, Recipe& recipe, std::optional<std::string> scene) {
        recipe.scene = std::move(scene);
        recipe.variations.clear();
        if (recipe.scene)
            for (const auto& variation : library.scenes.at(*recipe.scene).variations) select_option(variation, recipe.variations[variation.name], variation.initial);
    }

    Composition compose(const Library& library, const Recipe& recipe, const prompt::Catalog& catalog) {
        Composition result;
        const auto& character = library.characters.at(recipe.character);
        std::map<std::string, std::string> selections;
        for (const auto& part : character.parts) {
            const auto& selected = recipe.parts.at(part.name);
            selections.emplace("part." + part.name, selected.option);
            for (const auto& group : part.options.at(selected.option).suboptions) selections.emplace("part." + part.name + "." + group.name, selected.suboptions.at(group.name));
            result.parts[part.name].hidden = std::ranges::contains(character.hidden, part.name);
        }
        const Scene* scene = recipe.scene ? &library.scenes.at(*recipe.scene) : nullptr;
        if (scene) {
            for (const auto& variation : scene->variations) {
                const auto& selected = recipe.variations.at(variation.name);
                selections.emplace("variation." + variation.name, selected.option);
                for (const auto& group : variation.options.at(selected.option).suboptions) selections.emplace("variation." + variation.name + "." + group.name, selected.suboptions.at(group.name));
            }
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

    std::array<std::string, 2> option_prompt(const Choices& choices, const Selection& selection) {
        const auto& option = choices.options.at(selection.option);
        auto result        = option.text;
        for (const auto& group : option.suboptions) {
            const auto& text = group.options.at(selection.suboptions.at(group.name));
            for (std::size_t side = 0; side < 2; ++side) {
                if (text[side].empty()) continue;
                if (!result[side].empty()) result[side] += ", ";
                result[side] += text[side];
            }
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
            for (const auto& variation : definition.variations) append(option_prompt(variation, recipe.variations.at(variation.name)));
        } else {
            for (const auto& part : library.characters.at(recipe.character).parts) {
                const auto& effect = composition.parts.at(part.name);
                if (!effect.disable.empty() || (effect.hidden && effect.require.empty())) continue;
                append(option_prompt(part, recipe.parts.at(part.name)));
            }
        }
        return result;
    }

    Preset read_preset(const Library& library, std::string name, const prompt::Catalog& catalog) {
        const auto path   = library.directory / "presets" / files::path(name + ".json");
        const auto json   = read_json(path);
        const auto source = files::utf8(path);
        auto context      = source + " / character";
        try {
            Preset result{std::move(name)};
            auto& recipe          = result.recipe;
            recipe.character      = json.at("character");
            const auto& character = read_reference(library.characters, recipe.character, context).second;
            read_selections(json, character.parts, recipe.parts, source, files::utf8(library.directory / "characters" / files::path(recipe.character) / "character.json"), "parts");
            context = source + " / scene";
            if (!json.at("scene").is_null()) recipe.scene = json.at("scene").get<std::string>();
            if (recipe.scene) {
                const auto& scene = read_reference(library.scenes, *recipe.scene, context).second;
                read_selections(json, scene.variations, recipe.variations, source, files::utf8(library.directory / "scenes" / files::path(scene.category) / files::path(*recipe.scene + ".json")), "variations");
            }
            for (const auto& [key, side] : {std::pair{"positive", &recipe.free.positive}, std::pair{"negative", &recipe.free.negative}}) {
                const auto side_path = std::format("{} / free / {}", source, key);
                context              = side_path;
                const auto& input    = json.at("free").at(key);
                context              = side_path + " / fixed";
                side->fixed          = input.at("fixed");
                context              = side_path + " / groups";
                for (const auto& group_json : input.at("groups")) {
                    const auto group_path = std::format("{} / groups / {}", side_path, side->groups.size());
                    context               = group_path + " / enabled";
                    auto& group           = side->groups.emplace_back(std::vector<prompt::Tag>{}, group_json.at("enabled").get<bool>());
                    context               = group_path + " / tags";
                    for (const auto& tag : group_json.at("tags")) {
                        context          = std::format("{} / tags / {}", group_path, group.tags.size());
                        const auto value = prompt::parse_tag(catalog, tag.get<std::string>());
                        if (!value) throw std::invalid_argument{std::format("{}: {}", context, value.error().message)};
                        const auto id = catalog.resolve(value->name);
                        if (!id) throw std::invalid_argument{std::format("{}: {}", context, id.error())};
                        group.tags.push_back({*id, value->weight});
                    }
                }
            }
            return result;
        } catch (const nlohmann::json::exception& error) {
            throw std::runtime_error{std::format("{}: {}", context, error.what())};
        }
    }

    void write_preset(const Library& library, const Preset& preset, const prompt::Catalog& catalog, const bool replace) {
        const auto& recipe = preset.recipe;
        nlohmann::json json{{"character", recipe.character}, {"parts", nlohmann::json::object()}, {"scene", recipe.scene ? nlohmann::json(*recipe.scene) : nlohmann::json{}}, {"variations", nlohmann::json::object()}};
        nlohmann::json part_suboptions = nlohmann::json::object(), variation_suboptions = nlohmann::json::object();
        write_selections(json["parts"], part_suboptions, library.characters.at(recipe.character).parts, recipe.parts);
        if (recipe.scene) write_selections(json["variations"], variation_suboptions, library.scenes.at(*recipe.scene).variations, recipe.variations);
        if (!part_suboptions.empty()) json["suboptions"]["parts"] = std::move(part_suboptions);
        if (!variation_suboptions.empty()) json["suboptions"]["variations"] = std::move(variation_suboptions);
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
