module;
#include <imgui.h>
#include <imgui_internal.h>
#include <misc/cpp/imgui_stdlib.h>
module genesia.editor.ui.tag_editor;
import std;

namespace genesia::editor {
    TagSearch::TagSearch(const prompt::Catalog& source) : catalog{source}, keys{source.names} {
        keys.append_range(source.alias_names);
        std::vector<std::pair<std::uint32_t, std::uint32_t>> pairs;
        std::vector<std::uint32_t> grams;
        for (std::uint32_t id = 0; id < keys.size(); ++id) {
            grams.clear();
            const auto key = keys[id].name;
            for (std::size_t i = 0; i < key.size(); ++i) {
                std::uint32_t gram{};
                for (std::size_t n = 0; n < 3 && i + n < key.size(); ++n) {
                    gram |= std::uint32_t{static_cast<unsigned char>(key[i + n])} << (8 * n);
                    grams.push_back(gram);
                }
            }
            std::ranges::sort(grams);
            grams.erase(std::unique(grams.begin(), grams.end()), grams.end());
            for (const auto gram : grams) pairs.emplace_back(gram, id);
        }
        std::ranges::sort(pairs);
        postings.reserve(pairs.size());
        for (const auto [gram, key] : pairs) {
            if (index.empty() || index.back().gram != gram) index.push_back({gram, static_cast<std::uint32_t>(postings.size()), 0});
            postings.push_back(key);
            ++index.back().count;
        }
    }

    std::vector<TagSuggestion> TagSearch::search(const std::string_view text) const {
        const auto query = prompt::normalize(text);
        std::vector<TagSuggestion> result;
        if (query.empty()) return result;
        const std::size_t length = std::min(3uz, query.size());
        const Posting* rarest{};
        for (std::size_t i = 0; i + length <= query.size(); ++i) {
            std::uint32_t gram{};
            for (std::size_t n = 0; n < length; ++n) gram |= std::uint32_t{static_cast<unsigned char>(query[i + n])} << (8 * n);
            const auto found = std::ranges::lower_bound(index, gram, {}, &Posting::gram);
            if (found == index.end() || found->gram != gram) return result;
            if (!rarest || found->count < rarest->count) rarest = &*found;
        }
        const auto better = [this](const TagSuggestion a, const TagSuggestion b) {
            if (a.rank != b.rank) return a.rank < b.rank;
            const auto& x = catalog.tags[a.tag];
            const auto& y = catalog.tags[b.tag];
            return x.count != y.count ? x.count > y.count : x.name < y.name;
        };
        for (const auto id : std::span{postings}.subspan(rarest->begin, rarest->count)) {
            const auto key = keys[id];
            const auto position = key.name.find(query);
            if (position == std::string_view::npos) continue;
            const bool alias = key.name != catalog.tags[key.tag].name;
            const TagSuggestion candidate{key.tag, key.name, (key.name == query ? 0 : position == 0 ? 2 : 4) + int(alias)};
            const auto duplicate = std::ranges::find(result, key.tag, &TagSuggestion::tag);
            if (duplicate != result.end()) {
                if (!better(candidate, *duplicate)) continue;
                result.erase(duplicate);
            }
            result.insert(std::ranges::lower_bound(result, candidate, better), candidate);
            if (result.size() > 8) result.pop_back();
        }
        return result;
    }

    void TagEditor::replace(prompt::Group& group, std::vector<prompt::Tag> tags) {
        if (group.tags == tags) return;
        group.tags = std::move(tags);
        selection.clear();
        anchor = 0;
        if (input_active) focus_input = true;
    }

    bool TagEditor::commit(prompt::Group& group, const prompt::Catalog& catalog, const std::optional<std::uint32_t> candidate) {
        if (input.empty() && !editing && !candidate) return true;
        auto parsed = prompt::parse(catalog, input);
        if (candidate) {
            const auto value = prompt::parse_tag(catalog, std::string_view{input}.substr(completion_begin, completion_end - completion_begin), true);
            if (!value) {
                error = value.error();
                valid = false;
                select_error = focus_input = true;
                return false;
            }
            const std::array tag{prompt::Tag{*candidate, value->weight}};
            input.replace(completion_begin, completion_end - completion_begin, prompt::serialize(catalog, tag));
            parsed = prompt::parse(catalog, input);
            ++revision;
        }
        if (!parsed || parsed->empty()) {
            error = parsed ? prompt::Error{"Enter a tag", 0, input.size()} : parsed.error();
            valid = false;
            select_error = focus_input = true;
            return false;
        }
        auto next = group.tags;
        const auto position = editing.value_or(next.size());
        if (editing) next.erase(next.begin() + position);
        next.insert(next.begin() + position, parsed->begin(), parsed->end());
        replace(group, std::move(next));
        input.clear();
        editing.reset();
        error.reset();
        query.clear();
        suggestions.clear();
        menu_open = false;
        valid = focus_input = true;
        ++revision;
        return true;
    }

    float TagEditor::measure(const prompt::Group& group, const prompt::Catalog& catalog, const float width, const float scale) const {
        const float gap = 6 * scale;
        float x{}, rows = 1;
        for (std::size_t i = 0; i <= group.tags.size(); ++i) {
            if (i == group.tags.size() && editing) break;
            float item = 220 * scale;
            if (i < group.tags.size() && editing != i) {
                const auto tag = group.tags[i];
                const auto text = catalog.tags[tag.id].text;
                item = ImGui::CalcTextSize(text.data(), text.data() + text.size()).x + 38 * scale;
                if (tag.weight != 1) item += ImGui::CalcTextSize(std::format("{:g}\xC3\x97", tag.weight).c_str()).x + 12 * scale;
            }
            item = std::min(item, width);
            if (x && x + item > width) { ++rows; x = 0; }
            x += item + gap;
        }
        return rows * 34 * scale + (error ? 24 * scale : 0);
    }

    bool TagEditor::draw(const char* payload_type, const std::size_t group_index, prompt::Group& group, const TagSearch& search, const float width, const float scale, std::optional<TagMove>& move) {
        const auto& catalog = search.catalog;
        bool changed{};
        ImGui::PushID(static_cast<int>(id));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {0, 0});
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {6 * scale, 6 * scale});
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, {8 * scale, 4 * scale});
        const bool visible = ImGui::BeginChild("##Tags", {width, measure(group, catalog, width, scale) - (error ? 24 * scale : 0)}, ImGuiChildFlags_None, ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
        if (!visible) {
            menu_open = input_active = false;
            ImGui::EndChild();
            ImGui::PopStyleVar(3);
            if (error) ImGui::Dummy({0, 24 * scale});
            ImGui::PopID();
            return false;
        }
        const auto origin = ImGui::GetCursorScreenPos();
        const float content_width = ImGui::GetContentRegionAvail().x;
        float x{}, y{};
        std::optional<std::size_t> remove;
        ImVec2 input_min{}, input_max{};
        bool enter{}, paste{}, separator{}, tag_hovered{};
        for (std::size_t i = 0; i <= group.tags.size(); ++i) {
            if (i == group.tags.size() && editing) break;
            const bool editor = i == group.tags.size() || editing == i;
            const auto tag = i < group.tags.size() ? group.tags[i] : prompt::Tag{};
            const auto text = editor ? std::string_view{} : catalog.tags[tag.id].text;
            const std::string weight = !editor && tag.weight != 1 ? std::format("{:g}\xC3\x97", tag.weight) : "";
            const float weight_width = weight.empty() ? 0 : ImGui::CalcTextSize(weight.c_str()).x + 12 * scale;
            float item_width = editor ? 220 * scale : ImGui::CalcTextSize(text.data(), text.data() + text.size()).x + weight_width + 38 * scale;
            item_width = std::min(item_width, content_width);
            if (x && x + item_width > content_width) { x = 0; y += 34 * scale; }
            ImGui::SetCursorScreenPos({origin.x + x, origin.y + y});
            ImGui::PushID(static_cast<int>(i));
            if (editor) {
                ImGui::PushID(static_cast<int>(revision));
                if (std::exchange(focus_input, false)) ImGui::SetKeyboardFocusHere();
                ImGui::PushStyleColor(ImGuiCol_FrameBg, input_active ? ImVec4{0.12F, 0.125F, 0.155F, 1} : ImVec4{0, 0, 0, 0});
                const bool edited = ImGui::InputTextMultiline("##Add", &input, {item_width, 28 * scale}, ImGuiInputTextFlags_CallbackCompletion | ImGuiInputTextFlags_CallbackAlways, input_callback, this);
                ImGui::PopStyleColor();
                input_active = ImGui::IsItemActive();
                changed |= edited;
                input_min = ImGui::GetItemRectMin();
                input_max = ImGui::GetItemRectMax();
                if (input.empty() && !input_active) ImGui::GetWindowDrawList()->AddText({input_min.x + 8 * scale, input_min.y + 5 * scale}, ImGui::GetColorU32(ImGuiCol_TextDisabled), "Add tag...");
                enter = std::exchange(completion_requested, false);
                paste = input_active && ImGui::GetIO().KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_V);
                separator = edited && !input.empty() && input.back() == ',';
                if (input != analyzed_input || cursor != analyzed_cursor) {
                    analyzed_input = input;
                    analyzed_cursor = cursor;
                    const auto parsed = prompt::parse(catalog, input);
                    valid = input.empty() ? !editing : parsed.has_value() && !parsed->empty();
                    const auto position = std::min(cursor, input.size());
                    const auto previous = std::string_view{input}.substr(0, position).find_last_of(",\r\n");
                    completion_begin = previous == std::string_view::npos ? 0 : previous + 1;
                    completion_end = std::min(input.find_first_of(",\r\n", position), input.size());
                    const auto value = prompt::parse_tag(catalog, std::string_view{input}.substr(completion_begin, completion_end - completion_begin), true);
                    std::string next_query = value ? prompt::normalize(value->name) : "";
                    if (query != next_query) {
                        query = std::move(next_query);
                        suggestions = search.search(query);
                        highlighted = 0;
                        menu_open = !suggestions.empty();
                    }
                    if (edited) error.reset();
                }
                if (input_active && menu_open) {
                    if (ImGui::IsKeyPressed(ImGuiKey_DownArrow)) highlighted = (highlighted + 1) % static_cast<int>(suggestions.size());
                    if (ImGui::IsKeyPressed(ImGuiKey_UpArrow)) highlighted = (highlighted + static_cast<int>(suggestions.size()) - 1) % static_cast<int>(suggestions.size());
                }
                ImGui::PopID();
            } else {
                ImGui::BeginDisabled(editing.has_value());
                const auto p = ImGui::GetCursorScreenPos();
                const bool clicked = ImGui::InvisibleButton("##Tag", {item_width, 28 * scale});
                const bool hovered = ImGui::IsItemHovered();
                tag_hovered |= ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenBlockedByActiveItem);
                const bool selected = selection.contains(i);
                auto* draw = ImGui::GetWindowDrawList();
                auto* storage = ImGui::GetStateStorage();
                const auto key = ImGui::GetItemID();
                const float alpha = std::lerp(storage->GetFloat(key), selected || hovered ? 1.0F : 0.0F, std::min(1.0F, ImGui::GetIO().DeltaTime / 0.12F));
                storage->SetFloat(key, alpha);
                draw->AddRectFilled(p, {p.x + item_width, p.y + 28 * scale}, ImGui::GetColorU32(selected ? ImVec4{0.23F, 0.21F, 0.32F, alpha * 0.70F} : ImVec4{0.20F, 0.205F, 0.25F, alpha * 0.55F}), 5 * scale);
                draw->PushClipRect(p, {p.x + item_width - weight_width - 24 * scale, p.y + 28 * scale}, true);
                draw->AddText({p.x + 10 * scale, p.y + (28 * scale - ImGui::GetFontSize()) / 2}, ImGui::GetColorU32(selected ? ImVec4{0.79F, 0.75F, 0.95F, 1} : ImVec4{0.71F + alpha * 0.15F, 0.71F + alpha * 0.15F, 0.76F + alpha * 0.15F, 1}), text.data(), text.data() + text.size());
                draw->PopClipRect();
                if (!weight.empty()) draw->AddText({p.x + item_width - weight_width - 22 * scale, p.y + (28 * scale - ImGui::GetFontSize()) / 2}, IM_COL32(177, 168, 218, 255), weight.c_str());
                if (hovered) {
                    const ImVec2 c{p.x + item_width - 13 * scale, p.y + 14 * scale};
                    draw->AddLine({c.x - 3 * scale, c.y - 3 * scale}, {c.x + 3 * scale, c.y + 3 * scale}, IM_COL32(165, 166, 180, 255), scale);
                    draw->AddLine({c.x - 3 * scale, c.y + 3 * scale}, {c.x + 3 * scale, c.y - 3 * scale}, IM_COL32(165, 166, 180, 255), scale);
                }
                if (clicked) {
                    if (ImGui::GetIO().MousePos.x >= p.x + item_width - 24 * scale) remove = i;
                    else if (!weight.empty() && ImGui::GetIO().MousePos.x >= p.x + item_width - 24 * scale - weight_width) ImGui::OpenPopup("Weight");
                    else if (ImGui::GetIO().KeyShift) {
                        selection.clear();
                        for (std::size_t k = std::min(anchor, i); k <= std::max(anchor, i); ++k) selection.insert(k);
                    } else {
                        const bool deselect = selected && (ImGui::GetIO().KeyCtrl || selection.size() == 1);
                        if (!ImGui::GetIO().KeyCtrl) selection.clear();
                        if (deselect) selection.erase(i);
                        else selection.insert(i);
                        anchor = i;
                    }
                }
                if (hovered && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
                    editing = i;
                    input = prompt::serialize(catalog, std::span{group.tags}.subspan(i, 1));
                    cursor = input.size();
                    focus_input = true;
                    error.reset();
                    ++revision;
                }
                if (ImGui::BeginDragDropSource()) {
                    const std::array location{group_index, i};
                    ImGui::SetDragDropPayload(payload_type, location.data(), sizeof(location));
                    ImGui::TextUnformatted(text.data(), text.data() + text.size());
                    ImGui::EndDragDropSource();
                }
                if (!editing && ImGui::BeginDragDropTarget()) {
                    if (const auto* payload = ImGui::AcceptDragDropPayload(payload_type)) {
                        const auto* source = static_cast<const std::size_t*>(payload->Data);
                        move = TagMove{source[0], source[1], group_index, i};
                    }
                    ImGui::EndDragDropTarget();
                }
                bool open_weight{};
                if (ImGui::BeginPopupContextItem("Actions")) {
                    if (ImGui::MenuItem("Edit tag")) {
                        editing = i;
                        input = prompt::serialize(catalog, std::span{group.tags}.subspan(i, 1));
                        cursor = input.size();
                        focus_input = true;
                        ++revision;
                    }
                    if (ImGui::MenuItem("Weight...")) open_weight = true;
                    if (ImGui::MenuItem("Remove")) remove = i;
                    ImGui::EndPopup();
                }
                if (open_weight) ImGui::OpenPopup("Weight");
                if (ImGui::BeginPopup("Weight")) {
                    float value = group.tags[i].weight;
                    if (ImGui::InputFloat("Weight", &value, 0.05F, 0.1F, "%.2f", ImGuiInputTextFlags_EnterReturnsTrue) && std::isfinite(value)) {
                        auto next = group.tags;
                        next[i].weight = value;
                        replace(group, std::move(next));
                        changed = true;
                    }
                    ImGui::EndPopup();
                }
                if (hovered) ImGui::SetTooltip("%.*s\nDouble-click to edit. Ctrl+Up/Down adjusts weight.", static_cast<int>(catalog.tags[tag.id].name.size()), catalog.tags[tag.id].name.data());
                ImGui::EndDisabled();
            }
            ImGui::PopID();
            x += item_width + 6 * scale;
        }
        if (ImGui::IsMouseClicked(ImGuiMouseButton_Left) && !tag_hovered && !ImGui::IsPopupOpen(nullptr, ImGuiPopupFlags_AnyPopupId | ImGuiPopupFlags_AnyPopupLevel)) selection.clear();
        const bool focused = ImGui::IsWindowFocused(ImGuiFocusedFlags_ChildWindows);
        if (focused && !editing && !ImGui::GetDragDropPayload() && (!input_active || input.empty()) && (!ImGui::GetIO().WantTextInput || input_active)) {
            const ImGuiInputFlags routing = input_active ? ImGuiInputFlags_RouteGlobal | ImGuiInputFlags_RouteOverActive : ImGuiInputFlags_RouteFocused;
            if (ImGui::Shortcut(ImGuiMod_Ctrl | ImGuiKey_A, routing)) {
                for (std::size_t i = 0; i < group.tags.size(); ++i) selection.insert(i);
            }
            if (!group.tags.empty() && !ImGui::GetIO().KeyCtrl && (ImGui::IsKeyPressed(ImGuiKey_LeftArrow) || ImGui::IsKeyPressed(ImGuiKey_RightArrow))) {
                const int direction = ImGui::IsKeyPressed(ImGuiKey_LeftArrow) ? -1 : 1;
                anchor = static_cast<std::size_t>(std::clamp(static_cast<int>(selection.empty() ? 0 : *selection.rbegin()) + direction, 0, static_cast<int>(group.tags.size()) - 1));
                if (!ImGui::GetIO().KeyShift) selection.clear();
                selection.insert(anchor);
            }
            const bool cut = ImGui::Shortcut(ImGuiMod_Ctrl | ImGuiKey_X, routing);
            const bool copy = ImGui::Shortcut(ImGuiMod_Ctrl | ImGuiKey_C, routing);
            if ((cut || copy) && !selection.empty()) {
                std::vector<prompt::Tag> copied;
                for (const auto i : selection) copied.push_back(group.tags[i]);
                ImGui::SetClipboardText(prompt::serialize(catalog, copied).c_str());
            }
            const bool erase = ImGui::Shortcut(ImGuiKey_Delete, routing);
            const bool backspace = ImGui::Shortcut(ImGuiKey_Backspace, routing);
            if (backspace && selection.empty() && !group.tags.empty()) {
                anchor = group.tags.size() - 1;
                selection.insert(anchor);
            } else if ((cut || erase || backspace) && !selection.empty()) {
                auto next = group.tags;
                for (const auto i : std::views::reverse(selection)) next.erase(next.begin() + i);
                replace(group, std::move(next));
                changed = true;
            }
            if (!input_active && ImGui::Shortcut(ImGuiMod_Ctrl | ImGuiKey_V, routing)) {
                input = ImGui::GetClipboardText();
                changed |= commit(group, catalog);
                focus_input = true;
                ++revision;
            }
            const bool increase = ImGui::Shortcut(ImGuiMod_Ctrl | ImGuiKey_UpArrow, routing);
            const bool decrease = ImGui::Shortcut(ImGuiMod_Ctrl | ImGuiKey_DownArrow, routing);
            if ((increase || decrease) && !selection.empty()) {
                auto next = group.tags;
                for (const auto i : selection) next[i].weight = static_cast<float>(std::round((double(next[i].weight) + (increase ? 0.05 : -0.05)) * 100) / 100);
                const auto selected = selection;
                replace(group, std::move(next));
                selection = selected;
                changed = true;
            }
        }
        if (remove) {
            auto next = group.tags;
            next.erase(next.begin() + *remove);
            replace(group, std::move(next));
            changed = true;
        }
        ImGui::EndChild();
        ImGui::PopStyleVar(3);
        if (enter || paste || separator) {
            const auto candidate = enter && menu_open && !suggestions.empty() ? std::optional{suggestions[highlighted].tag} : std::nullopt;
            changed |= commit(group, catalog, candidate);
        }
        if ((input_active || menu_open || !ImGui::GetIO().WantTextInput) && ImGui::IsKeyPressed(ImGuiKey_Escape, false)) {
            if (menu_open) menu_open = false;
            else if (editing) {
                editing.reset();
                input.clear();
                error.reset();
                valid = true;
                ++revision;
            }
        }
        if (menu_open && !suggestions.empty()) {
            const float popup_width = std::min(420 * scale, ImGui::GetIO().DisplaySize.x - 24 * scale);
            const float popup_height = (suggestions.size() * 44 + 12) * scale;
            const float below = ImGui::GetIO().DisplaySize.y - input_max.y - 12 * scale;
            const float top = below >= popup_height ? input_max.y + 6 * scale : std::max(6 * scale, input_min.y - popup_height - 6 * scale);
            ImGui::SetNextWindowPos({std::clamp(input_min.x, 12 * scale, ImGui::GetIO().DisplaySize.x - popup_width - 12 * scale), top});
            ImGui::SetNextWindowSize({popup_width, popup_height});
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {6 * scale, 6 * scale});
            ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {0, 0});
            ImGui::Begin(std::format("##Suggestions{}", id).c_str(), nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNavFocus);
            // Keep suggestions above the composer without taking its text input focus.
            ImGui::BringWindowToDisplayFront(ImGui::GetCurrentWindow());
            std::optional<std::uint32_t> chosen;
            for (int i = 0; i < static_cast<int>(suggestions.size()); ++i) {
                const auto suggestion = suggestions[i];
                const auto& entry = catalog.tags[suggestion.tag];
                const auto p = ImGui::GetCursorScreenPos();
                ImGui::PushID(i);
                if (ImGui::Selectable("##Suggestion", highlighted == i, ImGuiSelectableFlags_None, {0, 44 * scale})) chosen = suggestion.tag;
                if (ImGui::IsItemHovered() && (ImGui::GetIO().MouseDelta.x || ImGui::GetIO().MouseDelta.y)) highlighted = i;
                auto* draw = ImGui::GetWindowDrawList();
                const std::array<ImU32, 6> colors{IM_COL32(153, 171, 218, 255), IM_COL32(218, 151, 166, 255), 0, IM_COL32(182, 158, 221, 255), IM_COL32(136, 194, 157, 255), IM_COL32(206, 191, 141, 255)};
                draw->AddCircleFilled({p.x + 10 * scale, p.y + 15 * scale}, 3 * scale, entry.category < 0 ? IM_COL32(169, 157, 243, 255) : colors[entry.category]);
                const auto label = entry.text;
                const auto match = prompt::normalize(label).find(query);
                draw->PushClipRect({p.x + 22 * scale, p.y}, {p.x + popup_width - 80 * scale, p.y + 44 * scale}, true);
                draw->AddText({p.x + 22 * scale, p.y + 5 * scale}, ImGui::GetColorU32(ImGuiCol_Text), label.data(), label.data() + label.size());
                if (match != std::string::npos && match + query.size() <= label.size()) draw->AddText({p.x + 22 * scale + ImGui::CalcTextSize(label.data(), label.data() + match).x, p.y + 5 * scale}, IM_COL32(188, 179, 255, 255), label.data() + match, label.data() + match + query.size());
                ImGui::PushFont(nullptr, 12);
                const auto detail = entry.category < 0 ? std::string_view{"Custom tag"} : suggestion.match != entry.name ? suggestion.match : entry.name;
                draw->AddText({p.x + 22 * scale, p.y + 25 * scale}, ImGui::GetColorU32(ImGuiCol_TextDisabled), detail.data(), detail.data() + detail.size());
                ImGui::PopFont();
                draw->PopClipRect();
                if (entry.count) {
                    const auto count = entry.count >= 1000000 ? std::format("{:.1f}m", entry.count / 1000000.0) : entry.count >= 1000 ? std::format("{:.0f}k", entry.count / 1000.0) : std::to_string(entry.count);
                    draw->AddText({p.x + popup_width - 20 * scale - ImGui::CalcTextSize(count.c_str()).x, p.y + 6 * scale}, ImGui::GetColorU32(ImGuiCol_TextDisabled), count.c_str());
                }
                ImGui::PopID();
            }
            const bool hovered = ImGui::IsWindowHovered(ImGuiHoveredFlags_AllowWhenBlockedByActiveItem);
            ImGui::End();
            ImGui::PopStyleVar(2);
            if (chosen) changed |= commit(group, catalog, chosen);
            if (ImGui::IsMouseClicked(ImGuiMouseButton_Left) && !hovered && !ImGui::IsMouseHoveringRect(input_min, input_max)) menu_open = false;
        }
        if (error) {
            ImGui::PushStyleColor(ImGuiCol_Text, {0.91F, 0.58F, 0.58F, 1});
            ImGui::TextUnformatted(error->message.c_str());
            ImGui::PopStyleColor();
        }
        ImGui::PopID();
        return changed;
    }

    int TagEditor::input_callback(ImGuiInputTextCallbackData* data) {
        auto& editor = *static_cast<TagEditor*>(data->UserData);
        if (data->EventFlag == ImGuiInputTextFlags_CallbackCompletion) editor.completion_requested = true;
        if (data->EventFlag == ImGuiInputTextFlags_CallbackAlways) {
            if (ImGui::IsKeyPressed(ImGuiKey_Enter, false)) {
                if (data->CursorPos > 0 && data->Buf[data->CursorPos - 1] == '\n') data->DeleteChars(data->CursorPos - 1, 1);
                editor.completion_requested = true;
            }
            if (std::exchange(editor.select_error, false) && editor.error) {
                data->SelectionStart = static_cast<int>(editor.error->offset);
                data->SelectionEnd = std::min(data->BufTextLen, static_cast<int>(editor.error->offset + editor.error->length));
                data->CursorPos = data->SelectionEnd;
            }
        }
        editor.cursor = static_cast<std::size_t>(data->CursorPos);
        return 0;
    }

    void PromptEditor::reset(const prompt::Pair& prompt, const bool clear_history) {
        const std::array sides{&prompt.positive, &prompt.negative};
        for (std::size_t s = 0; s < sides.size(); ++s) {
            groups[s].clear();
            groups[s].resize(sides[s]->groups.size());
            for (auto& editor : groups[s]) editor.id = ++next_id;
        }
        if (clear_history) {
            undo.clear();
            redo.clear();
            negative = false;
        }
        valid = true;
        escape_owned = focus_input = false;
    }

    void PromptEditor::suspend() {
        for (auto& side : groups) {
            for (auto& editor : side) {
                editor.menu_open = editor.input_active = editor.focus_input = false;
                editor.selection.clear();
            }
        }
        escape_owned = focus_input = false;
    }

    bool PromptEditor::remember(prompt::Pair before, const prompt::Pair& after) {
        if (before == after) return false;
        undo.push_back(std::move(before));
        if (undo.size() > 128) undo.erase(undo.begin());
        redo.clear();
        return true;
    }

    bool PromptEditor::commit(prompt::Pair& prompt, const prompt::Catalog& catalog) {
        auto before = prompt;
        const std::array sides{&prompt.positive, &prompt.negative};
        for (std::size_t s = 0; s < sides.size(); ++s) {
            for (std::size_t i = 0; i < groups[s].size(); ++i) {
                if (!groups[s][i].commit(sides[s]->groups[i], catalog)) {
                    negative = s == 1;
                    groups[s][i].collapsed = false;
                    valid = false;
                    remember(std::move(before), prompt);
                    return false;
                }
            }
        }
        valid = true;
        remember(std::move(before), prompt);
        return true;
    }

    float PromptEditor::measure(const prompt::Pair& prompt, const prompt::Catalog& catalog, const float width, const float scale) const {
        const auto& side = negative ? prompt.negative : prompt.positive;
        const auto& editors = groups[negative];
        float height = 110 * scale;
        for (std::size_t i = 0; i < side.groups.size(); ++i) {
            height += (editors[i].collapsed ? 42 : 46) * scale;
            if (!editors[i].collapsed) height += editors[i].measure(side.groups[i], catalog, width - 24 * scale, scale);
        }
        return height;
    }

    bool PromptEditor::draw(prompt::Pair& prompt, const TagSearch& search, const float scale) {
        auto before = prompt;
        bool changed{};
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {8 * scale, 6 * scale});
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, {8 * scale, 5 * scale});
        ImGui::PushStyleColor(ImGuiCol_Button, {0, 0, 0, 0});
        for (int s = 0; s < 2; ++s) {
            if (s) ImGui::SameLine();
            const bool selected = negative == bool(s);
            ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(selected ? ImGuiCol_Text : ImGuiCol_TextDisabled));
            if (ImGui::Button(s ? "Negative" : "Positive", {88 * scale, 30 * scale}) && !selected) {
                suspend();
                negative = bool(s);
            }
            ImGui::PopStyleColor();
            if (selected) {
                const auto p = ImGui::GetItemRectMin();
                ImGui::GetWindowDrawList()->AddLine({p.x + 8 * scale, p.y + 29 * scale}, {p.x + 80 * scale, p.y + 29 * scale},
                    ImGui::GetColorU32(ImVec4{0.65F, 0.60F, 0.88F, 0.8F}), scale);
            }
        }
        ImGui::Spacing();
        auto& side = negative ? prompt.negative : prompt.positive;
        auto& editors = groups[negative];
        const char* tag_payload = negative ? "GENESIA_NEGATIVE_TAG" : "GENESIA_POSITIVE_TAG";
        const char* group_payload = negative ? "GENESIA_NEGATIVE_GROUP" : "GENESIA_POSITIVE_GROUP";
        std::optional<TagMove> tag_move;
        std::optional<std::pair<std::size_t, std::size_t>> group_move;
        std::optional<std::size_t> remove;
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {0, 0});
        ImGui::BeginChild(negative ? "##NegativeGroups" : "##PositiveGroups", {0, -36 * scale}, ImGuiChildFlags_None, ImGuiWindowFlags_NoBackground);
        const float width = ImGui::GetContentRegionAvail().x - 8 * scale;
        for (std::size_t i = 0; i < side.groups.size(); ++i) {
            auto& group = side.groups[i];
            auto& editor = editors[i];
            ImGui::PushID(static_cast<int>(editor.id));
            ImGui::BeginGroup();
            const auto origin = ImGui::GetCursorScreenPos();
            auto* draw = ImGui::GetWindowDrawList();
            if (ImGui::InvisibleButton("##Heading", {width - 64 * scale, 30 * scale})) {
                editor.collapsed = !editor.collapsed;
                editor.menu_open = editor.input_active = editor.focus_input = false;
                editor.selection.clear();
            }
            const bool hovered = ImGui::IsItemHovered();
            const ImU32 label_color = ImGui::GetColorU32(group.enabled && hovered ? ImGuiCol_Text : ImGuiCol_TextDisabled);
            const ImVec2 arrow{origin.x + 8 * scale, origin.y + 15 * scale};
            if (editor.collapsed) draw->AddTriangleFilled({arrow.x - 2 * scale, arrow.y - 4 * scale}, {arrow.x + 3 * scale, arrow.y}, {arrow.x - 2 * scale, arrow.y + 4 * scale}, label_color);
            else draw->AddTriangleFilled({arrow.x - 4 * scale, arrow.y - 2 * scale}, {arrow.x + 4 * scale, arrow.y - 2 * scale}, {arrow.x, arrow.y + 3 * scale}, label_color);
            draw->PushClipRect({origin.x + 22 * scale, origin.y}, {origin.x + width - 70 * scale, origin.y + 30 * scale}, true);
            draw->AddText({origin.x + 22 * scale, origin.y + (30 * scale - ImGui::GetFontSize()) / 2}, label_color, group.name.c_str());
            const auto count = std::format("{}{}", group.tags.size(), group.enabled ? "" : "  Off");
            draw->AddText({origin.x + 34 * scale + ImGui::CalcTextSize(group.name.c_str()).x, origin.y + (30 * scale - ImGui::GetFontSize()) / 2},
                ImGui::GetColorU32(ImVec4{0.43F, 0.44F, 0.50F, 1}), count.c_str());
            draw->PopClipRect();
            if (hovered) ImGui::SetTooltip("Click: collapse or expand\nDrag: reorder group\nRight-click: group actions");
            if (ImGui::BeginDragDropSource()) {
                ImGui::SetDragDropPayload(group_payload, &i, sizeof(i));
                ImGui::TextUnformatted(group.name.c_str());
                ImGui::EndDragDropSource();
            }
            if (hovered && ImGui::IsMouseReleased(ImGuiMouseButton_Right)) ImGui::OpenPopup("Group actions");
            ImGui::SetCursorScreenPos({origin.x + width - 60 * scale, origin.y});
            if (ImGui::InvisibleButton("##Enabled", {28 * scale, 30 * scale})) group.enabled = !group.enabled;
            const ImVec2 indicator{origin.x + width - 46 * scale, origin.y + 15 * scale};
            if (group.enabled) draw->AddCircleFilled(indicator, 3.5F * scale, ImGui::GetColorU32(ImVec4{0.65F, 0.60F, 0.88F, 0.85F}));
            else draw->AddCircle(indicator, 3.5F * scale, ImGui::GetColorU32(ImGuiCol_TextDisabled), 0, scale);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip(group.enabled ? "Disable group\nKeep tags, exclude from generation" : "Enable group");
            ImGui::SetCursorScreenPos({origin.x + width - 28 * scale, origin.y});
            if (ImGui::InvisibleButton("##Actions", {28 * scale, 30 * scale})) ImGui::OpenPopup("Group actions");
            for (int dot = -1; dot <= 1; ++dot) draw->AddCircleFilled({origin.x + width - (14 - dot * 4) * scale, origin.y + 15 * scale}, scale, ImGui::GetColorU32(ImGuiCol_TextDisabled));
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Group actions");
            bool rename{};
            if (ImGui::BeginPopup("Group actions")) {
                if (ImGui::MenuItem("Rename...")) rename = true;
                if (ImGui::MenuItem(group.enabled ? "Disable group" : "Enable group")) group.enabled = !group.enabled;
                ImGui::Separator();
                if (ImGui::MenuItem("Move up", nullptr, false, i > 0)) group_move = std::pair{i, i - 1};
                if (ImGui::MenuItem("Move down", nullptr, false, i + 1 < side.groups.size())) group_move = std::pair{i, i + 1};
                if (ImGui::MenuItem("Delete group", "Ctrl+Z to undo")) remove = i;
                ImGui::EndPopup();
            }
            if (rename) {
                editor.rename = group.name;
                editor.rename_focus = true;
                ImGui::OpenPopup("Rename group");
            }
            if (ImGui::BeginPopup("Rename group")) {
                if (std::exchange(editor.rename_focus, false)) ImGui::SetKeyboardFocusHere();
                ImGui::SetNextItemWidth(220 * scale);
                const bool entered = ImGui::InputText("##Name", &editor.rename, ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_AutoSelectAll);
                changed |= ImGui::IsItemEdited();
                ImGui::SameLine();
                if (ImGui::Button("Rename") || entered) {
                    group.name = editor.rename;
                    ImGui::CloseCurrentPopup();
                }
                ImGui::EndPopup();
            }
            if (!editor.collapsed) {
                ImGui::SetCursorScreenPos({origin.x + 16 * scale, origin.y + 34 * scale});
                if (!group.enabled) ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.55F);
                if (editor.focus_input) ImGui::SetScrollHereY();
                changed |= editor.draw(tag_payload, i, group, search, width - 16 * scale, scale, tag_move);
                if (!group.enabled) ImGui::PopStyleVar();
            }
            ImGui::EndGroup();
            if (ImGui::BeginDragDropTargetCustom({ImGui::GetItemRectMin(), ImGui::GetItemRectMax()}, ImGui::GetID("##GroupDrop"))) {
                if (!editor.editing) {
                    if (const auto* payload = ImGui::AcceptDragDropPayload(tag_payload)) {
                        const auto* source = static_cast<const std::size_t*>(payload->Data);
                        tag_move = TagMove{source[0], source[1], i, group.tags.size()};
                    }
                }
                if (const auto* payload = ImGui::AcceptDragDropPayload(group_payload)) group_move = std::pair{*static_cast<const std::size_t*>(payload->Data), i};
                ImGui::EndDragDropTarget();
            }
            ImGui::Spacing();
            ImGui::PopID();
        }
        ImGui::EndChild();
        ImGui::PopStyleVar();
        if (ImGui::Button("+ Add group", {0, 30 * scale})) {
            side.groups.push_back({std::format("Group {}", side.groups.size() + 1)});
            editors.emplace_back().id = ++next_id;
            editors.back().focus_input = true;
        }
        ImGui::SameLine();
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + ImGui::GetContentRegionAvail().x - ImGui::CalcTextSize("Base prompt").x - 16 * scale);
        ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
        if (ImGui::Button("Base prompt", {0, 30 * scale})) ImGui::OpenPopup("Base prompt");
        ImGui::PopStyleColor();
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Read only. Always appended after enabled groups.");
        ImGui::SetNextWindowSize({std::min(560 * scale, ImGui::GetIO().DisplaySize.x - 24 * scale), 240 * scale});
        if (ImGui::BeginPopup("Base prompt")) {
            ImGui::TextDisabled("Edit in the configuration file.");
            ImGui::PushID(negative);
            ImGui::InputTextMultiline("##Fixed", &side.fixed, {0, -1}, ImGuiInputTextFlags_ReadOnly | ImGuiInputTextFlags_WordWrap);
            ImGui::PopID();
            ImGui::EndPopup();
        }
        ImGui::PopStyleColor();
        ImGui::PopStyleVar(2);

        // Apply moves after drawing so no row keeps references into a modified vector.
        if (tag_move) {
            const auto [from, index, to, insertion] = *tag_move;
            auto& source = side.groups[from].tags;
            const auto tag = source[index];
            source.erase(source.begin() + index);
            auto& target = side.groups[to].tags;
            target.insert(target.begin() + insertion - (from == to && index < insertion ? 1 : 0), tag);
            editors[from].selection.clear();
            editors[to].selection.clear();
            editors[from].anchor = editors[to].anchor = 0;
        }
        if (group_move) {
            const auto [from, to] = *group_move;
            auto group = std::move(side.groups[from]);
            auto editor = std::move(editors[from]);
            side.groups.erase(side.groups.begin() + from);
            editors.erase(editors.begin() + from);
            side.groups.insert(side.groups.begin() + to, std::move(group));
            editors.insert(editors.begin() + to, std::move(editor));
        }
        if (remove) {
            side.groups.erase(side.groups.begin() + *remove);
            editors.erase(editors.begin() + *remove);
        }
        changed |= remember(std::move(before), prompt);
        const bool pending = std::ranges::any_of(groups | std::views::join, [](const TagEditor& editor) { return !editor.input.empty() || editor.editing; });
        const bool empty_input_active = std::ranges::any_of(editors, [](const TagEditor& editor) { return editor.input_active && editor.input.empty(); });
        if (ImGui::IsWindowFocused(ImGuiFocusedFlags_ChildWindows) && !pending && !ImGui::GetDragDropPayload() && (!ImGui::GetIO().WantTextInput || empty_input_active)) {
            const ImGuiInputFlags routing = empty_input_active ? ImGuiInputFlags_RouteGlobal | ImGuiInputFlags_RouteOverActive : ImGuiInputFlags_RouteFocused;
            const bool backwards = ImGui::Shortcut(ImGuiMod_Ctrl | ImGuiKey_Z, routing);
            const bool forwards = ImGui::Shortcut(ImGuiMod_Ctrl | ImGuiKey_Y, routing) || ImGui::Shortcut(ImGuiMod_Ctrl | ImGuiMod_Shift | ImGuiKey_Z, routing);
            if ((backwards && !undo.empty()) || (forwards && !redo.empty())) {
                auto& source = backwards ? undo : redo;
                auto& target = backwards ? redo : undo;
                target.push_back(std::move(prompt));
                prompt = std::move(source.back());
                source.pop_back();
                ImGui::ClearActiveID();
                reset(prompt, false);
                changed = true;
            }
        }
        valid = std::ranges::all_of(groups | std::views::join, &TagEditor::valid);
        escape_owned = std::ranges::any_of(groups[negative], [](const TagEditor& editor) { return !editor.collapsed && (editor.menu_open || editor.editing); });
        focus_input = std::ranges::any_of(groups[negative], &TagEditor::focus_input);
        return changed;
    }

}
