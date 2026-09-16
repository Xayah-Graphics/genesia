module;
#include <Windows.h>
#include <imgui.h>
#include <imgui_internal.h>
#include <shellapi.h>
module genesia.editor.panels.prompts;
import genesia.io.files;
import std;

namespace genesia::editor {
    PromptPanel::PromptPanel(const prompts::Library& source, Renderer& display, WindowPlatform& platform) : library{source}, images{display}, window{platform} {}

    void PromptPanel::update(const prompts::Recipe& recipe, const prompt::Catalog& catalog) {
        if (!evaluated || *evaluated != recipe) {
            evaluated   = recipe;
            composition = {};
            location.reset();
            scene_preview.clear();
            try {
                location.emplace(library.directory / "characters" / files::path(recipe.character) / "images", recipe.parts);
                if (recipe.scene) scene_preview = location->scene(*recipe.scene, recipe.variations);
                composition = prompts::compose(library, recipe, catalog);
            } catch (const std::exception& failure) {
                composition.error = std::format("Prompt configuration for {}: {}", recipe.character, failure.what());
            }
        }
        std::set<std::filesystem::path> wanted;
        if (location) {
            wanted.insert(location->directory / "profile.png");
            if (!scene_preview.empty()) wanted.insert(scene_preview);
        }
        const auto targets = wanted;
        // Menu candidates stay cached while hovered, but do not affect generation readiness.
        if (hover_frame < ImGui::GetFrameCount() - 1) hovered_preview.clear();
        if (!hovered_preview.empty()) wanted.insert(hovered_preview);
        incoming.reset();
        const auto& dropped = window.dropped.empty() ? window.dragged : window.dropped;
        if (dropped.size() == 1) {
            auto extension = dropped.front().extension().native();
            std::ranges::transform(extension, extension.begin(), [](wchar_t c) { return std::towlower(c); });
            if (extension == L".png") {
                incoming = previews::long_path(dropped.front());
                wanted.insert(*incoming);
            }
        }
        images.update(wanted);
        error = composition.error;
        ready = error.empty() && location.has_value();
        for (const auto& path : targets) {
            const auto& texture = images.textures.at(path);
            if (texture.loading) ready = false;
            if (!texture.error.empty()) {
                if (!error.empty()) error += '\n';
                error += texture.error;
                ready = false;
            }
        }
    }

    void PromptPanel::status() {
        if (!images.error.empty()) ImGui::TextWrapped("%s", images.error.c_str());
        if (images.saving) ImGui::TextDisabled("Saving preview...");
        else if (!images.status.empty()) ImGui::TextDisabled("%s", images.status.c_str());
    }

    void PromptPanel::draw(prompts::Recipe& recipe, const prompt::Catalog& catalog, const float scale) {
        auto next             = recipe;
        const auto& character = library.characters.at(recipe.character);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {14 * scale, 14 * scale});
        ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 10 * scale);
        ImGui::PushStyleVar(ImGuiStyleVar_ChildBorderSize, scale);
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, {6 * scale, 3 * scale});
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {8 * scale, 8 * scale});
        ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, {0, 2 * scale});
        ImGui::PushStyleColor(ImGuiCol_ChildBg, {0.075F, 0.080F, 0.098F, 1});
        ImGui::PushStyleColor(ImGuiCol_Border, {0.42F, 0.46F, 0.55F, 0.16F});
        const auto choose = [&](const char* id, const auto& content, const bool searchable = false) {
            if (ImGui::GetCurrentWindow()->SkipItems) return false;
            const auto origin     = ImGui::GetCursorScreenPos();
            const float available = std::max(1.0F, ImGui::GetContentRegionAvail().x);
            const float inset     = 6 * scale;
            auto* draw            = ImGui::GetWindowDrawList();
            ImDrawListSplitter layers;
            layers.Split(draw, 2);
            layers.SetCurrentChannel(draw, 1);
            ImGui::SetCursorScreenPos({origin.x + inset, origin.y});
            ImGui::BeginGroup();
            content(std::max(1.0F, available - 30 * scale));
            ImGui::EndGroup();
            const auto text = ImGui::GetItemRectSize();
            const ImVec2 size{std::min(available, std::max(124 * scale, text.x + 30 * scale)), std::max(ImGui::GetFrameHeight(), text.y + 3 * scale)};
            const ImRect bounds{origin, {origin.x + size.x, origin.y + size.y}};
            ImGui::SetCursorScreenPos(origin);
            const bool pressed = ImGui::InvisibleButton(id, size, ImGuiButtonFlags_EnableNav);
            const auto key     = ImGui::GetItemID();
            bool open          = ImGui::IsPopupOpen(key, ImGuiPopupFlags_None);
            if (pressed && !open) {
                ImGui::OpenPopupEx(key, ImGuiPopupFlags_None);
                open = true;
            }
            const bool highlighted = open || ImGui::IsItemHovered() || (ImGui::IsItemFocused() && ImGui::GetCurrentContext()->NavCursorVisible);
            layers.SetCurrentChannel(draw, 0);
            if (highlighted) {
                draw->AddRectFilled(bounds.Min, bounds.Max, ImGui::GetColorU32(ImVec4{0.114F, 0.129F, 0.161F, 1}), 7 * scale);
                draw->AddRect(bounds.Min, bounds.Max, ImGui::GetColorU32(open ? ImVec4{0.286F, 0.325F, 0.384F, 1} : ImVec4{0.188F, 0.212F, 0.255F, 1}), 7 * scale, 0, scale);
            }
            const ImVec2 arrow{bounds.Max.x - 11 * scale, origin.y + ImGui::GetFrameHeight() / 2};
            const float direction = open ? -1.0F : 1.0F;
            const auto ink        = ImGui::GetColorU32(ImVec4{0.60F, 0.624F, 0.667F, 1});
            draw->AddLine({arrow.x - 3 * scale, arrow.y - direction * 1.5F * scale}, {arrow.x, arrow.y + direction * 1.5F * scale}, ink, scale);
            draw->AddLine({arrow.x, arrow.y + direction * 1.5F * scale}, {arrow.x + 3 * scale, arrow.y - direction * 1.5F * scale}, ink, scale);
            layers.Merge(draw);
            if (!open) return false;
            ImGui::PushStyleVar(ImGuiStyleVar_PopupRounding, 10 * scale);
            ImGui::PushStyleVar(ImGuiStyleVar_PopupBorderSize, scale);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {6 * scale, 6 * scale});
            ImGui::PushStyleColor(ImGuiCol_PopupBg, {0.114F, 0.129F, 0.161F, 1});
            ImGui::PushStyleColor(ImGuiCol_Border, {0.227F, 0.259F, 0.310F, 1});
            if (searchable) {
                const float width = std::min(std::max(bounds.GetWidth(), 360 * scale), ImGui::GetWindowViewport()->WorkSize.x - 40 * scale);
                ImGui::SetNextWindowSizeConstraints({width, 0}, {width, std::numeric_limits<float>::max()});
            }
            const bool visible = ImGui::BeginComboPopup(key, {bounds.Min, {bounds.Max.x, bounds.Max.y + 6 * scale}}, searchable ? ImGuiComboFlags_HeightLargest : ImGuiComboFlags_None);
            ImGui::PopStyleColor(2);
            ImGui::PopStyleVar(3);
            return visible;
        };
        const auto option = [&](const std::string& name, const bool selected, const bool compact = false) {
            ImGui::PushID(name.c_str());
            const float width = compact ? ImGui::GetContentRegionAvail().x : std::min(std::max(ImGui::GetContentRegionAvail().x, ImGui::CalcTextSize(name.c_str()).x + 38 * scale), ImGui::GetWindowViewport()->WorkSize.x - 40 * scale);
            const auto text   = ImGui::CalcTextSize(name.c_str(), nullptr, false, compact ? -1.0F : std::max(1.0F, width - 38 * scale));
            const auto origin = ImGui::GetCursorScreenPos();
            const ImVec2 end{origin.x + width, origin.y + text.y + 12 * scale};
            ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {0, 3 * scale});
            const bool pressed = ImGui::InvisibleButton("##Option", {width, text.y + 12 * scale}, ImGuiButtonFlags_EnableNav);
            ImGui::PopStyleVar();
            if (selected) ImGui::SetItemDefaultFocus();
            auto* draw = ImGui::GetWindowDrawList();
            if (selected || ImGui::IsItemHovered()) draw->AddRectFilled(origin, end, ImGui::GetColorU32(selected ? ImVec4{0.161F, 0.192F, 0.243F, 1} : ImVec4{0.145F, 0.169F, 0.208F, 1}), 6 * scale);
            const auto ink = ImGui::GetColorU32(selected ? ImVec4{0.753F, 0.820F, 0.898F, 1} : ImGui::GetStyleColorVec4(ImGuiCol_Text));
            const ImVec4 clip{origin.x + 9 * scale, origin.y, end.x - 28 * scale, end.y};
            draw->AddText(ImGui::GetFont(), ImGui::GetFontSize(), {clip.x, origin.y + 6 * scale}, ink, name.c_str(), nullptr, compact ? 0.0F : std::max(1.0F, width - 38 * scale), &clip);
            if (selected) ImGui::RenderCheckMark(draw, {end.x - 18 * scale, origin.y + 6 * scale + (text.y - 10 * scale) / 2}, ink, 10 * scale);
            ImGui::RenderNavCursor({origin, end}, ImGui::GetItemID());
            if (pressed) ImGui::CloseCurrentPopup();
            ImGui::PopID();
            return pressed;
        };
        constexpr auto card_flags   = ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_AlwaysUseWindowPadding | ImGuiChildFlags_Borders;
        constexpr auto window_flags = ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse;
        if (!error.empty() && error != composition.error) ImGui::TextWrapped("%s", error.c_str());
        const auto& style           = ImGui::GetStyle();
        const float padding_height  = 2 * (style.WindowPadding.y + style.CellPadding.y);
        const float profile_height  = 360 * scale;
        const prompts::Scene* scene = recipe.scene ? &library.scenes.at(*recipe.scene) : nullptr;
        for (const bool scene_card : {false, true}) {
            const bool populated = !scene_card || scene;
            ImGui::PushID(scene_card ? "Scene" : "Character");
            if (populated) ImGui::SetNextWindowSizeConstraints({0, profile_height}, {std::numeric_limits<float>::max(), std::numeric_limits<float>::max()});
            if (ImGui::BeginChild("##Card", {0, 0}, card_flags, window_flags)) {
                const float portrait_height = profile_height - padding_height;
                const float portrait        = std::min(portrait_height / 1.5F, ImGui::GetContentRegionAvail().x * 0.44F);
                // Keep each table ID tied to a fixed column layout across scene changes.
                if (ImGui::BeginTable(populated ? "##Profile" : "##EmptyScene", populated ? 3 : 1, ImGuiTableFlags_SizingStretchProp)) {
                    if (populated) {
                        ImGui::TableSetupColumn("Preview", ImGuiTableColumnFlags_WidthFixed, portrait);
                        ImGui::TableSetupColumn("Gap", ImGuiTableColumnFlags_WidthFixed, 16 * scale);
                    }
                    ImGui::TableSetupColumn("Details");
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0);
                    if (populated) {
                        if (scene_card && !scene_preview.empty()) picture(scene_preview, recipe.character + " / " + *recipe.scene, portrait, portrait_height, scale, true);
                        else if (!scene_card && location) picture(location->directory / "profile.png", recipe.character, portrait, portrait_height, scale, false);
                        ImGui::TableSetColumnIndex(2);
                    }
                    ImGui::PushFont(nullptr, 20);
                    const auto title = [&](const float width) {
                        ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + width);
                        ImGui::TextUnformatted(scene_card ? (recipe.scene ? recipe.scene->c_str() : "Scene: None") : recipe.character.c_str());
                        ImGui::PopTextWrapPos();
                    };
                    if (choose("##Selection", title, true)) {
                        ImGui::PushFont(nullptr, 13);
                        if (ImGui::IsWindowAppearing()) ImGui::SetKeyboardFocusHere();
                        ImGui::SetNextItemWidth(-1);
                        auto& search        = selection_search[scene_card];
                        const bool searched = ImGui::InputTextWithHint("##Search", scene_card ? "Search scenes or categories..." : "Search characters...", search.data(), search.size());
                        struct SelectionRow {
                            std::string_view label;
                            const std::string* name;
                        };
                        std::vector<SelectionRow> rows;
                        if (scene_card) {
                            if (option("None", !recipe.scene, true)) next.scene.reset();
                            std::map<std::string_view, std::vector<const std::string*>> matches;
                            for (const auto& [name, definition] : library.scenes)
                                if (!search[0] || ImStristr(name.c_str(), nullptr, search.data(), nullptr) || ImStristr(definition.category.c_str(), nullptr, search.data(), nullptr) || ImStristr(definition.description.c_str(), nullptr, search.data(), nullptr)) matches[definition.category].push_back(&name);
                            for (const auto& [category, names] : matches) {
                                rows.push_back({category, nullptr});
                                for (const auto* name : names) rows.push_back({*name, name});
                            }
                        } else {
                            for (const auto& [name, definition] : library.characters)
                                if (!search[0] || ImStristr(name.c_str(), nullptr, search.data(), nullptr) || ImStristr(definition.description.c_str(), nullptr, search.data(), nullptr)) rows.push_back({name, &name});
                        }
                        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {0, 3 * scale});
                        if (ImGui::BeginChild("##Selections", {0, 260 * scale})) {
                            if (searched) ImGui::SetScrollY(0);
                            ImGuiListClipper clipper;
                            clipper.Begin(static_cast<int>(rows.size()), ImGui::GetTextLineHeight() + 15 * scale);
                            if (ImGui::IsWindowAppearing())
                                for (std::size_t i = 0; i < rows.size(); ++i)
                                    if (rows[i].name && (scene_card ? recipe.scene == *rows[i].name : recipe.character == *rows[i].name)) clipper.IncludeItemByIndex(static_cast<int>(i));
                            while (clipper.Step())
                                for (int i = clipper.DisplayStart; i < clipper.DisplayEnd; ++i) {
                                    const auto& row = rows[i];
                                    if (!row.name) {
                                        const auto origin = ImGui::GetCursorScreenPos();
                                        const float width = ImGui::GetContentRegionAvail().x;
                                        ImGui::Dummy({width, ImGui::GetTextLineHeight() + 12 * scale});
                                        auto* draw = ImGui::GetWindowDrawList();
                                        const ImVec4 clip{origin.x + 9 * scale, origin.y, origin.x + width - 9 * scale, ImGui::GetItemRectMax().y};
                                        draw->AddText(ImGui::GetFont(), ImGui::GetFontSize(), {clip.x, origin.y + 6 * scale}, ImGui::GetColorU32(ImGuiCol_TextDisabled), row.label.data(), row.label.data() + row.label.size(), 0, &clip);
                                        const float line = clip.x + ImGui::CalcTextSize(row.label.data(), row.label.data() + row.label.size()).x + 10 * scale;
                                        const float y    = origin.y + 6 * scale + ImGui::GetTextLineHeight() / 2;
                                        if (line < clip.z) draw->AddLine({line, y}, {clip.z, y}, ImGui::GetColorU32(ImGuiCol_Border), scale);
                                        continue;
                                    }
                                    const auto& name = *row.name;
                                    if (option(name, scene_card ? recipe.scene == name : recipe.character == name, true)) {
                                        if (scene_card) next.scene = name;
                                        else next.character = name;
                                    }
                                    if (ImGui::IsItemHovered()) selection_preview(name, scene_card, recipe, scale);
                                }
                            if (rows.empty()) ImGui::TextDisabled(scene_card ? "No matching scenes" : "No matching characters");
                        }
                        ImGui::EndChild();
                        ImGui::PopStyleVar();
                        ImGui::PopFont();
                        ImGui::EndCombo();
                    }
                    ImGui::PopFont();
                    if (populated) {
                        const auto& description = scene_card ? scene->description : character.description;
                        if (!description.empty()) {
                            const auto heading = ImGui::GetItemRectSize();
                            ImGui::PushFont(nullptr, 13);
                            const float remaining = ImGui::GetContentRegionAvail().x - heading.x - 8 * scale;
                            if (remaining >= std::min(ImGui::CalcTextSize(description.c_str()).x, 96 * scale)) {
                                ImGui::SameLine(0, 8 * scale);
                                ImGui::SetCursorPosY(ImGui::GetCursorPosY() + (heading.y - ImGui::GetTextLineHeight()) / 2);
                            }
                            ImGui::PushTextWrapPos(0);
                            ImGui::TextDisabled("%s", description.c_str());
                            ImGui::PopTextWrapPos();
                            ImGui::PopFont();
                        }
                    }
                    if (scene_card && scene) {
                        ImGui::TextDisabled("Base prompt");
                        ImGui::TextWrapped("%s", scene->text[0].empty() ? "None" : scene->text[0].c_str());
                        if (!scene->text[1].empty()) {
                            ImGui::TextDisabled("Negative prompt");
                            ImGui::TextWrapped("%s", scene->text[1].c_str());
                        }
                    }
                    if (populated) {
                        ImGui::Dummy({0, 10 * scale});
                        if (ImGui::BeginTable("##Choices", 2, ImGuiTableFlags_SizingStretchProp)) {
                            ImGui::TableSetupColumn("Group", ImGuiTableColumnFlags_WidthFixed, std::min(108 * scale, ImGui::GetContentRegionAvail().x * 0.36F));
                            ImGui::TableSetupColumn("Option");
                            for (const auto& group : scene_card ? scene->variations : character.parts) {
                                ImGui::PushID(group.name.c_str());
                                const auto& selected = scene_card ? recipe.variations.at(group.name) : recipe.parts.at(group.name);
                                const auto effect    = composition.parts.find(group.name);
                                const auto row       = [&](const prompts::Suboptions* child) {
                                    const auto& label = child ? child->name : group.name;
                                    const auto& value = child ? selected.suboptions.at(child->name) : selected.option;
                                    const auto& text  = child ? child->options.at(value) : group.options.at(value).text;
                                    const auto& order = child ? child->order : group.order;
                                    ImGui::PushID(child ? 1 : 0);
                                    ImGui::PushID(label.c_str());
                                    ImGui::TableNextRow();
                                    ImGui::TableSetColumnIndex(0);
                                    ImGui::AlignTextToFramePadding();
                                    if (child) ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 12 * scale);
                                    ImGui::PushTextWrapPos(0);
                                    ImGui::TextDisabled("%s", label.c_str());
                                    ImGui::PopTextWrapPos();
                                    ImGui::TableSetColumnIndex(1);
                                    ImGui::AlignTextToFramePadding();
                                    const auto content = [&](const float width) { option_text(text, value, scale, width, !scene_card && effect != composition.parts.end() ? &effect->second : nullptr); };
                                    if (order.size() == 1) {
                                        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 6 * scale);
                                        content(ImGui::GetContentRegionAvail().x);
                                    } else if (choose("##Option", content)) {
                                        for (const auto& name : order) {
                                            if (option(name, value == name) && value != name) {
                                                auto& target = (scene_card ? next.variations : next.parts).at(group.name);
                                                if (child) target.suboptions.at(child->name) = name;
                                                else prompts::select_option(group, target, name);
                                            }
                                            if (ImGui::IsItemHovered()) selection_preview(name, scene_card, recipe, scale, &group, child ? &child->name : nullptr);
                                        }
                                        ImGui::EndCombo();
                                    }
                                    ImGui::PopID();
                                    ImGui::PopID();
                                };
                                row(nullptr);
                                for (const auto& child : group.options.at(selected.option).suboptions) row(&child);
                                ImGui::PopID();
                            }
                            ImGui::EndTable();
                        }
                    } else ImGui::TextDisabled("Choose a scene to configure its variations.");
                    if (scene_card) {
                        bool heading{};
                        for (const auto& part : character.parts) {
                            const auto found = composition.parts.find(part.name);
                            if (found == composition.parts.end() || found->second.require.empty()) continue;
                            if (!heading) {
                                ImGui::Dummy({0, 10 * scale});
                                ImGui::TextDisabled("CHARACTER PARTS");
                                heading = true;
                            }
                            const auto& effect = found->second;
                            ImGui::TextDisabled(effect.disable.empty() ? "Required" : "Conflict");
                            ImGui::SameLine(0, 8 * scale);
                            option_text(prompts::option_prompt(part, recipe.parts.at(part.name)), part.name, scale, ImGui::GetContentRegionAvail().x, &effect);
                        }
                        if (!composition.error.empty()) {
                            ImGui::Spacing();
                            ImGui::PushStyleColor(ImGuiCol_Text, {0.91F, 0.55F, 0.51F, 1});
                            ImGui::TextWrapped("%s", composition.error.c_str());
                            ImGui::PopStyleColor();
                        }
                    }
                    ImGui::EndTable();
                }
            }
            ImGui::EndChild();
            ImGui::PopID();
        }
        if (next.character != recipe.character) prompts::select_character(library, next, next.character);
        if (next.scene != recipe.scene) prompts::select_scene(library, next, next.scene);
        if (next != recipe) {
            recipe = std::move(next);
            update(recipe, catalog);
        }
        ImGui::PopStyleColor(2);
        ImGui::PopStyleVar(6);
    }

    void PromptPanel::selection_preview(const std::string& name, const bool scene, const prompts::Recipe& recipe, const float scale, const prompts::Choices* group, const std::string* suboption) {
        const auto& character = !scene && !group ? name : recipe.character;
        auto parts            = recipe.parts;
        if (character != recipe.character) {
            parts.clear();
            for (const auto& part : library.characters.at(character).parts) prompts::select_option(part, parts[part.name], part.initial);
        } else if (group && !scene) {
            auto& selected = parts.at(group->name);
            if (suboption) selected.suboptions.at(*suboption) = name;
            else if (selected.option != name) prompts::select_option(*group, selected, name);
        }
        const previews::Location target{library.directory / "characters" / files::path(character) / "images", parts};
        if (scene) {
            const auto& scene_name = group ? *recipe.scene : name;
            auto variations        = recipe.variations;
            if (recipe.scene != scene_name) {
                variations.clear();
                for (const auto& variation : library.scenes.at(scene_name).variations) prompts::select_option(variation, variations[variation.name], variation.initial);
            } else if (group) {
                auto& selected = variations.at(group->name);
                if (suboption) selected.suboptions.at(*suboption) = name;
                else if (selected.option != name) prompts::select_option(*group, selected, name);
            }
            hovered_preview = target.scene(scene_name, variations);
        } else hovered_preview = target.directory / "profile.png";
        hover_frame         = ImGui::GetFrameCount();
        const auto& texture = images.request(hovered_preview);
        ImGui::BeginTooltip();
        ImGui::PushFont(nullptr, 13);
        const auto origin = ImGui::GetCursorScreenPos();
        const ImVec2 size{112 * scale, 168 * scale};
        ImGui::Dummy(size);
        auto* draw = ImGui::GetWindowDrawList();
        draw->AddRectFilled(origin, {origin.x + size.x, origin.y + size.y}, ImGui::GetColorU32(ImVec4{0.055F, 0.061F, 0.076F, 1}), 7 * scale);
        if (texture.id) {
            const float factor = std::min(size.x / texture.width, size.y / texture.height);
            const ImVec2 minimum{origin.x + (size.x - texture.width * factor) / 2, origin.y + (size.y - texture.height * factor) / 2};
            draw->AddImageRounded(texture.id, minimum, {minimum.x + texture.width * factor, minimum.y + texture.height * factor}, {0, 0}, {1, 1}, ImGui::GetColorU32(ImVec4{1, 1, 1, 1}), 7 * scale);
        } else {
            const char* label = texture.loading ? "Loading..." : texture.error.empty() ? "No preview" : "Preview error";
            const auto text   = ImGui::CalcTextSize(label);
            draw->AddText({origin.x + (size.x - text.x) / 2, origin.y + (size.y - text.y) / 2}, ImGui::GetColorU32(ImGuiCol_TextDisabled), label);
        }
        ImGui::SameLine(0, 12 * scale);
        ImGui::BeginGroup();
        ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + 240 * scale);
        ImGui::TextWrapped("%s", name.c_str());
        const auto description = suboption ? group->name + " / " + *suboption : group ? group->name : scene ? library.scenes.at(name).description : library.characters.at(name).description;
        if (!description.empty()) {
            ImGui::Spacing();
            ImGui::TextWrapped("%s", description.c_str());
        }
        if (group) {
            const auto& selected = (scene ? recipe.variations : recipe.parts).at(group->name);
            const auto& parent   = group->options.at(suboption ? selected.option : name);
            const auto& text     = suboption ? std::ranges::find(parent.suboptions, *suboption, &prompts::Suboptions::name)->options.at(name) : parent.text;
            if (text[0].empty() && text[1].empty()) {
                ImGui::Spacing();
                ImGui::TextDisabled("No prompt");
            }
            for (std::size_t side = 0; side < 2; ++side) {
                if (text[side].empty()) continue;
                ImGui::Spacing();
                ImGui::TextDisabled(side ? "Negative" : "Positive");
                ImGui::TextWrapped("%s", text[side].c_str());
            }
        }
        if (!texture.error.empty()) {
            ImGui::Spacing();
            ImGui::PushStyleColor(ImGuiCol_Text, {0.91F, 0.55F, 0.51F, 1});
            ImGui::TextWrapped("%s", texture.error.c_str());
            ImGui::PopStyleColor();
        }
        ImGui::PopTextWrapPos();
        ImGui::EndGroup();
        ImGui::PopFont();
        ImGui::EndTooltip();
    }

    void PromptPanel::picture(const std::filesystem::path& path, const std::string& name, const float width, const float height, const float scale, const bool scene) {
        const auto origin = ImGui::GetCursorScreenPos();
        const ImVec2 end{origin.x + width, origin.y + height};
        ImGui::PushID(files::utf8(path).c_str());
        ImGui::InvisibleButton("##Preview", {width, height}, ImGuiButtonFlags_MouseButtonRight);
        if (!ImGui::IsItemVisible()) {
            ImGui::PopID();
            return;
        }
        const bool hovered  = ImGui::IsItemHovered();
        const auto& texture = images.textures.at(path);
        ImRect bounds{origin, end};
        bounds.ClipWith(ImGui::GetCurrentWindow()->ClipRect);
        const ImVec2 pointer{window.drop_position[0], window.drop_position[1]};
        const bool dragging                    = incoming.has_value() && !ImGui::IsPopupOpen(nullptr, ImGuiPopupFlags_AnyPopupId | ImGuiPopupFlags_AnyPopupLevel);
        const bool over                        = dragging && bounds.Contains(pointer);
        const previews::Images::Texture* shown = &texture;
        if (over) {
            const auto source = images.textures.find(*incoming);
            if (source != images.textures.end() && source->second.id) shown = &source->second;
        }
        auto* draw = ImGui::GetWindowDrawList();
        draw->AddRectFilled(origin, end, ImGui::GetColorU32(ImVec4{0.055F, 0.061F, 0.076F, 1}), 7 * scale);
        if (shown->id) {
            const float factor = std::min(width / shown->width, height / shown->height);
            const ImVec2 minimum{origin.x + (width - shown->width * factor) / 2, origin.y + (height - shown->height * factor) / 2};
            draw->AddImageRounded(shown->id, minimum, {minimum.x + shown->width * factor, minimum.y + shown->height * factor}, {0, 0}, {1, 1}, ImGui::GetColorU32(ImVec4{1, 1, 1, over ? 0.7F : 1.0F}), 7 * scale);
        } else {
            const bool failed = !texture.error.empty();
            const char* label = texture.loading ? "Loading..." : failed ? "Preview error" : "Drop PNG here";
            const auto ink    = ImGui::GetColorU32(failed ? ImVec4{0.83F, 0.48F, 0.46F, 1} : ImVec4{0.40F, 0.44F, 0.52F, 1});
            const ImVec2 center{origin.x + width / 2, origin.y + height / 2 - (width >= 120 * scale ? 12 : 0) * scale};
            draw->AddRect({center.x - 10 * scale, center.y - 8 * scale}, {center.x + 10 * scale, center.y + 8 * scale}, ink, 2 * scale);
            draw->AddCircleFilled({center.x + 4 * scale, center.y - 3 * scale}, 1.5F * scale, ink);
            draw->AddLine({center.x - 7 * scale, center.y + 4 * scale}, {center.x - 2 * scale, center.y - scale}, ink, scale);
            draw->AddLine({center.x - 2 * scale, center.y - scale}, {center.x + 6 * scale, center.y + 5 * scale}, ink, scale);
            if (width >= 120 * scale) {
                const auto text = ImGui::CalcTextSize(label);
                draw->AddText({center.x - text.x / 2, center.y + 16 * scale}, ink, label);
            }
        }
        if (dragging) draw->AddRect(origin, end, ImGui::GetColorU32(over ? ImVec4{0.48F, 0.77F, 0.91F, 1} : ImVec4{0.40F, 0.61F, 0.74F, 0.55F}), 7 * scale, 0, (over ? 2 : 1) * scale);
        if (over) {
            ImGui::SetNextWindowPos({pointer.x + 16 * scale, pointer.y + 20 * scale});
            ImGui::BeginTooltip();
            ImGui::TextUnformatted(images.saving ? "Saving preview..." : texture.exists ? "Replace preview" : "Set preview");
            ImGui::TextUnformatted(name.c_str());
            ImGui::EndTooltip();
            if (!window.dropped.empty()) {
                if (images.saving) images.error = "Wait for the current preview change to finish.";
                else images.edit(path, location->root, *incoming);
                window.dropped.clear();
            }
        } else if (hovered) {
            ImGui::BeginTooltip();
            ImGui::PushTextWrapPos(ImGui::GetFontSize() * 28);
            ImGui::TextUnformatted(name.c_str());
            if (!composition.error.empty()) ImGui::TextWrapped("%s", composition.error.c_str());
            else {
                const auto text = prompts::card_prompt(library, *evaluated, composition, scene);
                for (std::size_t side = 0; side < 2; ++side) {
                    ImGui::Spacing();
                    ImGui::TextDisabled(side ? "Negative" : "Positive");
                    if (text[side].empty()) ImGui::TextDisabled("None");
                    else ImGui::TextWrapped("%s", text[side].c_str());
                }
            }
            ImGui::PopTextWrapPos();
            ImGui::EndTooltip();
        }
        if (ImGui::BeginPopupContextItem("##PreviewActions")) {
            if (ImGui::MenuItem("Clear preview", nullptr, false, texture.exists && !images.saving)) images.edit(path, location->root, std::nullopt);
            if (ImGui::MenuItem("Open containing folder")) {
                try {
                    auto folder = path.parent_path();
                    while (!std::filesystem::exists(folder)) folder = folder.parent_path();
                    const auto result = reinterpret_cast<std::intptr_t>(ShellExecuteW(window.native_window, L"open", folder.c_str(), nullptr, nullptr, SW_SHOWNORMAL));
                    if (result <= 32) throw std::runtime_error{std::format("Cannot open preview folder ({}): {}", result, files::utf8(folder))};
                } catch (const std::exception& failure) {
                    images.error = failure.what();
                }
            }
            ImGui::EndPopup();
        }
        ImGui::PopID();
    }

    void PromptPanel::option_text(const std::array<std::string, 2>& value, const std::string& name, const float scale, const float width, const prompts::Composition::Part* effect) const {
        const bool required = effect && !effect->require.empty();
        const bool disabled = effect && (!effect->disable.empty() || (effect->hidden && !required));
        const bool conflict = required && disabled;
        ImGui::BeginGroup();
        if (disabled) ImGui::PushStyleColor(ImGuiCol_Text, conflict ? ImVec4{0.91F, 0.55F, 0.51F, 1} : ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {0, 0});
        const auto origin = ImGui::GetCursorScreenPos();
        const float inset = required ? 10 * scale : 0;
        if (required) ImGui::GetWindowDrawList()->AddCircleFilled({origin.x + 3 * scale, origin.y + ImGui::GetTextLineHeight() / 2}, 2 * scale, ImGui::GetColorU32(conflict ? ImVec4{0.91F, 0.55F, 0.51F, 1} : ImVec4{0.48F, 0.73F, 0.65F, 1}));
        const char* cursor = name.data();
        const char* end    = cursor + name.size();
        while (cursor < end) {
            ImGui::SetCursorScreenPos({origin.x + inset, ImGui::GetCursorScreenPos().y});
            const char* line = ImGui::GetFont()->CalcWordWrapPosition(ImGui::GetFontSize(), cursor, end, std::max(1.0F, width - inset));
            if (line == cursor) {
                ++line;
                while (line < end && (static_cast<unsigned char>(*line) & 0xc0) == 0x80) ++line;
            }
            ImGui::TextUnformatted(cursor, line);
            if (disabled) {
                const auto minimum = ImGui::GetItemRectMin();
                const auto maximum = ImGui::GetItemRectMax();
                const float y      = (minimum.y + maximum.y) / 2;
                ImGui::GetWindowDrawList()->AddLine({minimum.x, y}, {maximum.x, y}, ImGui::GetColorU32(ImGuiCol_Text), scale);
            }
            cursor = line;
            while (cursor < end && (*cursor == ' ' || *cursor == '\t')) ++cursor;
            if (cursor < end && *cursor == '\n') ++cursor;
        }
        ImGui::PopStyleVar();
        if (disabled) ImGui::PopStyleColor();
        ImGui::EndGroup();
        if (ImGui::IsItemHovered()) {
            ImGui::BeginTooltip();
            ImGui::PushTextWrapPos(ImGui::GetFontSize() * 28);
            if (value[0].empty() && value[1].empty()) ImGui::TextDisabled("No prompt");
            for (std::size_t side = 0; side < 2; ++side) {
                if (value[side].empty()) continue;
                ImGui::TextDisabled(side ? "Negative" : "Positive");
                ImGui::TextWrapped("%s", value[side].c_str());
            }
            if (effect && (effect->hidden || !effect->require.empty() || !effect->disable.empty())) {
                ImGui::Separator();
                if (effect->hidden) ImGui::TextDisabled("Hidden by the character definition by default.");
                if (evaluated->scene) {
                    const auto& rules = library.scenes.at(*evaluated->scene).rules;
                    for (std::size_t i = 0; i < rules.size(); ++i) {
                        if (std::ranges::contains(effect->require, i)) ImGui::TextWrapped("Require: %s", rules[i].reason.c_str());
                        if (std::ranges::contains(effect->disable, i)) ImGui::TextWrapped("Disable: %s", rules[i].reason.c_str());
                    }
                }
            }
            ImGui::PopTextWrapPos();
            ImGui::EndTooltip();
        }
    }

} // namespace genesia::editor
