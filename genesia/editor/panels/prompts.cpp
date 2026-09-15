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
            evaluated = recipe;
            composition.reset();
            location.reset();
            error.clear();
            try {
                composition = prompts::compose(library, recipe, catalog);
                location.emplace(library.directory / "characters" / files::path(recipe.character) / "images", recipe.parts);
            } catch (const std::exception& failure) {
                error = std::format("Prompt configuration for {}: {}", recipe.character, failure.what());
            }
        }
        std::set<std::filesystem::path> wanted;
        if (location) {
            wanted.insert(location->directory / "profile.png");
            for (const auto& [name, option] : recipe.categories)
                if (option) wanted.insert(location->directory / files::path(name) / files::path(*option + ".png"));
        }
        const auto targets = wanted;
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
        ready = composition.has_value() && location.has_value();
        if (ready) error.clear();
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
        if (images.undo) {
            if (images.saving || !images.status.empty()) ImGui::SameLine();
            ImGui::BeginDisabled(images.saving);
            if (ImGui::SmallButton("Undo preview")) images.restore();
            ImGui::EndDisabled();
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Undo the last preview change\n%s", files::utf8(images.undo->path).c_str());
        }
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
        const auto choose = [&](const char* id, const auto& content) {
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
            const bool visible = ImGui::BeginComboPopup(key, {bounds.Min, {bounds.Max.x, bounds.Max.y + 6 * scale}}, ImGuiComboFlags_None);
            ImGui::PopStyleColor(2);
            ImGui::PopStyleVar(3);
            return visible;
        };
        const auto option = [&](const std::string& name, const bool selected) {
            ImGui::PushID(name.c_str());
            const float width = std::min(std::max(ImGui::GetContentRegionAvail().x, ImGui::CalcTextSize(name.c_str()).x + 38 * scale), ImGui::GetWindowViewport()->WorkSize.x - 40 * scale);
            const auto text   = ImGui::CalcTextSize(name.c_str(), nullptr, false, std::max(1.0F, width - 38 * scale));
            const auto origin = ImGui::GetCursorScreenPos();
            const ImVec2 end{origin.x + width, origin.y + text.y + 12 * scale};
            ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {0, 3 * scale});
            const bool pressed = ImGui::InvisibleButton("##Option", {width, text.y + 12 * scale}, ImGuiButtonFlags_EnableNav);
            ImGui::PopStyleVar();
            if (selected) ImGui::SetItemDefaultFocus();
            auto* draw = ImGui::GetWindowDrawList();
            if (selected || ImGui::IsItemHovered()) draw->AddRectFilled(origin, end, ImGui::GetColorU32(selected ? ImVec4{0.161F, 0.192F, 0.243F, 1} : ImVec4{0.145F, 0.169F, 0.208F, 1}), 6 * scale);
            const auto ink = ImGui::GetColorU32(selected ? ImVec4{0.753F, 0.820F, 0.898F, 1} : ImGui::GetStyleColorVec4(ImGuiCol_Text));
            draw->AddText(ImGui::GetFont(), ImGui::GetFontSize(), {origin.x + 9 * scale, origin.y + 6 * scale}, ink, name.c_str(), nullptr, std::max(1.0F, width - 38 * scale));
            if (selected) ImGui::RenderCheckMark(draw, {end.x - 18 * scale, origin.y + 6 * scale + (text.y - 10 * scale) / 2}, ink, 10 * scale);
            ImGui::RenderNavCursor({origin, end}, ImGui::GetItemID());
            if (pressed) ImGui::CloseCurrentPopup();
            ImGui::PopID();
            return pressed;
        };
        constexpr auto card_flags   = ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_AlwaysUseWindowPadding | ImGuiChildFlags_Borders;
        constexpr auto window_flags = ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse;
        if (!error.empty()) ImGui::TextWrapped("%s", error.c_str());
        const auto& style          = ImGui::GetStyle();
        const float padding_height = 2 * (style.WindowPadding.y + style.CellPadding.y);
        const float profile_height = 480 * scale;
        ImGui::SetNextWindowSizeConstraints({0, profile_height}, {std::numeric_limits<float>::max(), std::numeric_limits<float>::max()});
        if (ImGui::BeginChild("##CharacterCard", {0, 0}, card_flags, window_flags)) {
            const float portrait_height = profile_height - padding_height;
            const float portrait        = std::min(portrait_height / 1.5F, ImGui::GetContentRegionAvail().x * 0.44F);
            if (ImGui::BeginTable("##CharacterProfile", 3, ImGuiTableFlags_SizingStretchProp)) {
                ImGui::TableSetupColumn("Portrait", ImGuiTableColumnFlags_WidthFixed, portrait);
                ImGui::TableSetupColumn("Gap", ImGuiTableColumnFlags_WidthFixed, 16 * scale);
                ImGui::TableSetupColumn("Character");
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                if (location) picture(location->directory / "profile.png", recipe.character, portrait, portrait_height, scale);
                ImGui::TableSetColumnIndex(2);
                ImGui::PushFont(nullptr, 20);
                if (library.character_order.size() == 1) ImGui::TextWrapped("%s", recipe.character.c_str());
                else if (choose("##Character", [&](const float width) {
                             ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + width);
                             ImGui::TextUnformatted(recipe.character.c_str());
                             ImGui::PopTextWrapPos();
                         })) {
                    for (const auto& name : library.character_order)
                        if (option(name, recipe.character == name)) next.character = name;
                    ImGui::EndCombo();
                }
                ImGui::PopFont();
                if (!character.description.empty()) {
                    ImGui::Dummy({0, 4 * scale});
                    const auto origin = ImGui::GetCursorScreenPos();
                    ImGui::BeginGroup();
                    ImGui::Indent(12 * scale);
                    ImGui::PushFont(nullptr, 11);
                    ImGui::TextDisabled("ABOUT");
                    ImGui::PopFont();
                    ImGui::PushStyleColor(ImGuiCol_Text, {0.69F, 0.73F, 0.79F, 1});
                    ImGui::TextWrapped("%s", character.description.c_str());
                    ImGui::PopStyleColor();
                    ImGui::Unindent(12 * scale);
                    ImGui::EndGroup();
                    ImGui::GetWindowDrawList()->AddLine({origin.x, origin.y + 2 * scale}, {origin.x, ImGui::GetItemRectMax().y - 2 * scale}, ImGui::GetColorU32(ImVec4{0.40F, 0.51F, 0.63F, 0.6F}), 2 * scale);
                }
                ImGui::Dummy({0, 10 * scale});
                if (ImGui::BeginTable("##Parts", 2, ImGuiTableFlags_SizingStretchProp)) {
                    ImGui::TableSetupColumn("Part", ImGuiTableColumnFlags_WidthFixed, std::min(108 * scale, ImGui::GetContentRegionAvail().x * 0.36F));
                    ImGui::TableSetupColumn("Option");
                    for (const auto& part : character.parts) {
                        ImGui::PushID(part.name.c_str());
                        const auto& selected = recipe.parts.at(part.name);
                        const auto& text     = part.options.at(selected);
                        ImGui::TableNextRow();
                        ImGui::TableSetColumnIndex(0);
                        ImGui::AlignTextToFramePadding();
                        ImGui::PushTextWrapPos(0);
                        ImGui::TextDisabled("%s", part.name.c_str());
                        ImGui::PopTextWrapPos();
                        ImGui::TableSetColumnIndex(1);
                        ImGui::AlignTextToFramePadding();
                        const auto content = [&](const float width) { option_text(text, "part." + part.name, selected, scale, width, true); };
                        if (part.options.size() == 1) {
                            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 6 * scale);
                            content(ImGui::GetContentRegionAvail().x);
                        } else if (choose("##Part", content)) {
                            for (const auto& name : part.order)
                                if (option(name, selected == name)) next.parts.at(part.name) = name;
                            ImGui::EndCombo();
                        }
                        ImGui::PopID();
                    }
                    ImGui::EndTable();
                }
                ImGui::EndTable();
            }
        }
        ImGui::EndChild();
        for (const auto& name : library.category_order) {
            const auto& category = library.categories.at(name);
            const auto& selected = recipe.categories.at(name);
            ImGui::PushID(name.c_str());
            if (ImGui::BeginChild("##CategoryCard", {0, 0}, card_flags, window_flags)) {
                if (ImGui::BeginTable("##Category", selected ? 3 : 1, ImGuiTableFlags_SizingStretchProp)) {
                    if (selected) {
                        ImGui::TableSetupColumn("Preview", ImGuiTableColumnFlags_WidthFixed, 64 * scale);
                        ImGui::TableSetupColumn("Gap", ImGuiTableColumnFlags_WidthFixed, 14 * scale);
                    }
                    ImGui::TableSetupColumn("Prompt");
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0);
                    if (selected) {
                        if (location) picture(location->directory / files::path(name) / files::path(*selected + ".png"), recipe.character + " · " + name + " / " + *selected, 64 * scale, 64 * scale, scale);
                        ImGui::TableSetColumnIndex(2);
                    }
                    ImGui::AlignTextToFramePadding();
                    ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + ImGui::GetContentRegionAvail().x * 0.4F);
                    ImGui::TextDisabled("%s", name.c_str());
                    ImGui::PopTextWrapPos();
                    ImGui::SameLine(0, 12 * scale);
                    if (choose("##Option", [&](const float width) {
                            ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + width);
                            ImGui::TextUnformatted(selected ? selected->c_str() : "None");
                            ImGui::PopTextWrapPos();
                        })) {
                        if (option("None", !selected)) next.categories.at(name).reset();
                        for (const auto& choice : category.order)
                            if (option(choice, selected == choice)) next.categories.at(name) = choice;
                        ImGui::EndCombo();
                    }
                    if (selected) option_text(category.options.at(*selected), "category." + name, *selected, scale, ImGui::GetContentRegionAvail().x, false);
                    ImGui::EndTable();
                }
            }
            ImGui::EndChild();
            ImGui::PopID();
        }
        if (next.character != recipe.character) prompts::select_character(library, next, next.character);
        if (next != recipe) {
            recipe = std::move(next);
            update(recipe, catalog);
        }
        ImGui::PopStyleColor(2);
        ImGui::PopStyleVar(6);
    }

    void PromptPanel::picture(const std::filesystem::path& path, const std::string& name, const float width, const float height, const float scale) {
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
            ImGui::TextUnformatted("Drop a PNG from File Explorer\nRight-click for preview actions");
            ImGui::TextWrapped("%s", texture.error.empty() ? files::utf8(path).c_str() : texture.error.c_str());
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

    void PromptPanel::option_text(const std::array<std::string, 2>& value, const std::string& key, const std::string& name, const float scale, const float width, const bool name_only) const {
        const std::vector<std::size_t>* reasons{};
        if (composition) {
            const auto found = composition->disabled.find(key);
            if (found != composition->disabled.end()) reasons = &found->second;
        }
        ImGui::BeginGroup();
        if (reasons) ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {0, 0});
        if (!name_only && value[0].empty() && value[1].empty()) {
            ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + width);
            ImGui::TextDisabled("%s", name.c_str());
            ImGui::PopTextWrapPos();
        }
        for (std::size_t side = 0; side < (name_only ? 1U : value.size()); ++side) {
            const auto& text = name_only ? name : value[side];
            if (text.empty()) continue;
            if (side) {
                ImGui::TextDisabled("Negative:");
                ImGui::SameLine(0, 4 * scale);
            }
            const char* cursor    = text.data();
            const char* end       = cursor + text.size();
            const float available = std::max(1.0F, width - (side ? ImGui::CalcTextSize("Negative:").x + 4 * scale : 0));
            while (cursor < end) {
                const char* line = ImGui::GetFont()->CalcWordWrapPosition(ImGui::GetFontSize(), cursor, end, available);
                if (line == cursor) {
                    ++line;
                    while (line < end && (static_cast<unsigned char>(*line) & 0xc0) == 0x80) ++line;
                }
                ImGui::TextUnformatted(cursor, line);
                if (reasons) {
                    const auto minimum = ImGui::GetItemRectMin();
                    const auto maximum = ImGui::GetItemRectMax();
                    const float y      = (minimum.y + maximum.y) / 2;
                    ImGui::GetWindowDrawList()->AddLine({minimum.x, y}, {maximum.x, y}, ImGui::GetColorU32(ImGuiCol_Text), scale);
                }
                cursor = line;
                while (cursor < end && (*cursor == ' ' || *cursor == '\t')) ++cursor;
                if (cursor < end && *cursor == '\n') ++cursor;
            }
        }
        ImGui::PopStyleVar();
        if (reasons) ImGui::PopStyleColor();
        ImGui::EndGroup();
        if (ImGui::IsItemHovered()) {
            ImGui::BeginTooltip();
            ImGui::PushTextWrapPos(ImGui::GetFontSize() * 28);
            if (name_only) {
                if (value[0].empty() && value[1].empty()) ImGui::TextDisabled("No prompt");
                for (std::size_t side = 0; side < 2; ++side) {
                    if (value[side].empty()) continue;
                    ImGui::TextDisabled(side ? "Negative" : "Positive");
                    ImGui::TextWrapped("%s", value[side].c_str());
                }
            } else ImGui::TextUnformatted(name.c_str());
            if (reasons) {
                ImGui::Separator();
                for (const auto index : *reasons) ImGui::TextWrapped("%s", library.rules[index].reason.c_str());
            }
            ImGui::PopTextWrapPos();
            ImGui::EndTooltip();
        }
    }

} // namespace genesia::editor
