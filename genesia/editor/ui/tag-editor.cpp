module;
#include <imgui.h>
#include <imgui_internal.h>
#include <misc/cpp/imgui_stdlib.h>
module genesia.editor.ui.tag_editor;
import std;

namespace genesia::editor {
    namespace {
        constexpr float tag_height  = 26;
        constexpr float tag_gap     = 4;
        constexpr float tag_padding = 8;
        constexpr float weight_gap  = 6;

        void group_frame(const ImVec2 minimum, const ImVec2 maximum, const float scale, const bool enabled, const bool hovered, const bool dashed = false) {
            auto* draw        = ImGui::GetWindowDrawList();
            const auto border = ImGui::GetColorU32(ImVec4{0.55F, 0.55F, 0.62F, hovered ? 0.45F : 0.22F});
            if (!dashed) {
                draw->AddRectFilled(minimum, maximum, ImGui::GetColorU32(ImVec4{0.071F, 0.075F, 0.086F, 1}), 12 * scale);
                draw->AddRect(minimum, maximum, border, 12 * scale, 0, scale);
            } else {
                draw->PathRect(minimum, maximum, 12 * scale);
                const std::vector<ImVec2> path{draw->_Path.begin(), draw->_Path.end()};
                draw->PathClear();
                float offset{};
                for (std::size_t i = 0; i < path.size(); ++i) {
                    const auto a = path[i], b = path[(i + 1) % path.size()];
                    const float length = std::hypot(b.x - a.x, b.y - a.y);
                    for (float start = -std::fmod(offset, 8 * scale); start < length; start += 8 * scale) {
                        const float from = std::max(0.0F, start), to = std::min(length, start + 4 * scale);
                        if (to <= from) continue;
                        draw->AddLine({std::lerp(a.x, b.x, from / length), std::lerp(a.y, b.y, from / length)}, {std::lerp(a.x, b.x, to / length), std::lerp(a.y, b.y, to / length)}, border, scale);
                    }
                    offset += length;
                }
            }
            if (!enabled) {
                ImGui::PushFont(nullptr, 12);
                const auto text = ImGui::CalcTextSize("Off");
                const ImVec2 p{maximum.x - text.x - 12 * scale, minimum.y - text.y / 2};
                draw->AddRectFilled({p.x - 4 * scale, p.y}, {p.x + text.x + 4 * scale, p.y + text.y}, ImGui::GetColorU32(ImVec4{0.063F, 0.067F, 0.078F, 1}));
                draw->AddText(p, ImGui::GetColorU32(ImGuiCol_TextDisabled), "Off");
                ImGui::PopFont();
            }
        }

        void fixed_text(const prompt::Pair& prompt) {
            ImGui::Spacing();
            if (!ImGui::TreeNodeEx("Fixed text", ImGuiTreeNodeFlags_NoTreePushOnOpen)) return;
            ImGui::PushTextWrapPos();
            ImGui::TextDisabled("Read only. Appended after enabled groups.");
            if (!prompt.positive.fixed.empty()) {
                ImGui::TextDisabled("Positive");
                ImGui::TextUnformatted(prompt.positive.fixed.c_str());
            }
            if (!prompt.negative.fixed.empty()) {
                ImGui::TextDisabled("Negative");
                ImGui::TextUnformatted(prompt.negative.fixed.c_str());
            }
            ImGui::PopTextWrapPos();
        }
    } // namespace

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
            const auto key      = keys[id];
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

    TagLayout::TagLayout(const prompt::Group& group, const prompt::Catalog& catalog, const float width, const float scale, const TagEditor& editor) : width{width} {
        input_expanded          = editor.editing || editor.input_active || editor.focus_input || !editor.input.empty();
        const auto editing      = editor.editing;
        const std::size_t count = group.tags.size() + std::size_t(!editing);
        items.reserve(count);
        float x{}, y{};
        for (std::size_t i = 0; i < count; ++i) {
            const bool input         = i == group.tags.size() || editing == i;
            const auto tag           = input ? prompt::Tag{} : group.tags[i];
            const auto text          = input ? std::string_view{} : catalog.tags[tag.id].text;
            std::string weight       = !input && tag.weight != 1 ? std::format("{:g}\xC3\x97", tag.weight) : "";
            const float weight_width = weight.empty() ? 0 : ImGui::CalcTextSize(weight.c_str()).x + weight_gap * scale;
            const bool added         = !input && editor.change && !editor.change->tags[i].original;
            const float item_width   = std::min(width, input ? (input_expanded ? 220 : tag_height) * scale : ImGui::CalcTextSize(text.data(), text.data() + text.size()).x + weight_width + (2 * tag_padding + (added ? 10 : 0)) * scale);
            if (x && x + item_width > width) {
                x = 0;
                y += (tag_height + tag_gap) * scale;
            }
            items.push_back({text, std::move(weight), {x, y}, item_width, weight_width, input, added});
            x += item_width + tag_gap * scale;
        }
        height = y + tag_height * scale;
    }

    void TagEditor::replace(prompt::Group& group, std::vector<prompt::Tag> tags) {
        if (group.tags == tags) return;
        group.tags = std::move(tags);
        selection.clear();
        anchor = 0;
        if (input_active) focus_input = true;
    }

    void TagEditor::erase(prompt::Group& group, const std::size_t index) {
        if (change && change->tags[index].original) change->tags[index].removed = !change->tags[index].removed;
        else {
            if (change) change->tags.erase(change->tags.begin() + index);
            group.tags.erase(group.tags.begin() + index);
        }
        selection.clear();
        anchor = 0;
    }

    bool TagEditor::commit(prompt::Group& group, const prompt::Catalog& catalog, const std::optional<std::uint32_t> candidate) {
        if (input.empty() && !editing && !candidate) return true;
        if (candidate) {
            const auto value = prompt::parse_tag(catalog, std::string_view{input}.substr(completion_begin, completion_end - completion_begin), true);
            if (!value) {
                error        = value.error();
                valid        = false;
                select_error = focus_input = true;
                return false;
            }
            const std::array tag{prompt::Tag{*candidate, value->weight}};
            input.replace(completion_begin, completion_end - completion_begin, prompt::serialize(catalog, tag));
            ++revision;
        }
        const auto parsed = prompt::parse(catalog, input);
        if (!parsed || parsed->empty()) {
            error        = parsed ? prompt::Error{"Enter a tag", 0, input.size()} : parsed.error();
            valid        = false;
            select_error = focus_input = true;
            return false;
        }
        auto next           = group.tags;
        const auto position = editing.value_or(next.size());
        if (editing) next.erase(next.begin() + position);
        next.insert(next.begin() + position, parsed->begin(), parsed->end());
        if (change) {
            TagChange original;
            if (editing) {
                original = change->tags[position];
                change->tags.erase(change->tags.begin() + position);
            }
            change->tags.insert(change->tags.begin() + position, parsed->size(), TagChange{});
            change->tags[position] = original;
        }
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

    void TagEditor::draw(const char* payload_type, const std::size_t group_index, prompt::Group& group, const TagSearch& search, const prompt::Catalog& catalog, const TagLayout& layout, const float scale, std::optional<TagMove>& move) {
        const float width = layout.width;
        ImGui::PushID(static_cast<int>(id));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {0, 0});
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {tag_gap * scale, tag_gap * scale});
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, {tag_padding * scale, 3 * scale});
        const bool visible = ImGui::BeginChild("##Tags", {width, layout.height}, ImGuiChildFlags_None, ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
        if (!visible) {
            menu_open = input_active = background_hovered = false;
            ImGui::EndChild();
            ImGui::PopStyleVar(3);
            if (error) ImGui::Dummy({width, ImGui::CalcTextSize(error->message.c_str(), nullptr, false, width).y});
            ImGui::PopID();
            return;
        }
        const auto origin         = ImGui::GetCursorScreenPos();
        const bool input_expanded = layout.input_expanded;
        std::optional<std::size_t> remove;
        ImVec2 input_min{}, input_max{};
        bool enter{}, paste{}, separator{}, tag_hovered{};
        for (std::size_t i = 0; i < layout.items.size(); ++i) {
            if (i == group.tags.size() && editing) break;
            const auto& item         = layout.items[i];
            const bool editor        = item.input;
            const auto tag           = editor ? prompt::Tag{} : group.tags[i];
            const auto text          = item.text;
            const auto& weight       = item.weight;
            const float item_width   = item.width;
            const float weight_width = item.weight_width;
            ImGui::SetCursorScreenPos({origin.x + item.position.x, origin.y + item.position.y});
            ImGui::PushID(static_cast<int>(i));
            if (editor && !input_expanded) {
                if (ImGui::InvisibleButton("##Add", {item_width, tag_height * scale}, ImGuiButtonFlags_EnableNav)) focus_input = true;
                input_min          = ImGui::GetItemRectMin();
                input_max          = ImGui::GetItemRectMax();
                const bool hovered = ImGui::IsItemHovered();
                const ImVec2 center{(input_min.x + input_max.x) / 2, (input_min.y + input_max.y) / 2};
                const auto ink = ImGui::GetColorU32(hovered ? ImGuiCol_Text : ImGuiCol_TextDisabled);
                auto* draw     = ImGui::GetWindowDrawList();
                draw->AddLine({center.x - 4 * scale, center.y}, {center.x + 4 * scale, center.y}, ink, scale);
                draw->AddLine({center.x, center.y - 4 * scale}, {center.x, center.y + 4 * scale}, ink, scale);
                if (hovered) ImGui::SetTooltip("Add tag");
            } else if (editor) {
                ImGui::PushID(static_cast<int>(revision));
                ImGui::PushStyleColor(ImGuiCol_FrameBg, input_active ? ImVec4{0.12F, 0.125F, 0.155F, 1} : ImVec4{0, 0, 0, 0});
                const bool edited = ImGui::InputTextMultiline("##Add", &input, {item_width, tag_height * scale}, ImGuiInputTextFlags_CallbackCompletion | ImGuiInputTextFlags_CallbackAlways, input_callback, this);
                ImGui::PopStyleColor();
                input_active = ImGui::IsItemActive();
                // Focus is applied next frame; keep the input expanded until activation.
                if (input_active) focus_input = false;
                else if (focus_input) ImGui::SetKeyboardFocusHere(-1);
                input_min = ImGui::GetItemRectMin();
                input_max = ImGui::GetItemRectMax();
                enter     = std::exchange(completion_requested, false);
                paste     = input_active && ImGui::GetIO().KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_V);
                separator = edited && !input.empty() && input.back() == ',';
                if (input != analyzed_input || cursor != analyzed_cursor) {
                    analyzed_input         = input;
                    analyzed_cursor        = cursor;
                    const auto parsed      = prompt::parse(catalog, input);
                    valid                  = input.empty() ? !editing : parsed.has_value() && !parsed->empty();
                    const auto position    = std::min(cursor, input.size());
                    const auto previous    = std::string_view{input}.substr(0, position).find_last_of(",\r\n");
                    completion_begin       = previous == std::string_view::npos ? 0 : previous + 1;
                    completion_end         = std::min(input.find_first_of(",\r\n", position), input.size());
                    const auto value       = prompt::parse_tag(catalog, std::string_view{input}.substr(completion_begin, completion_end - completion_begin), true);
                    std::string next_query = value ? prompt::normalize(value->name) : "";
                    if (query != next_query) {
                        query       = std::move(next_query);
                        suggestions = search.search(query);
                        highlighted = 0;
                        menu_open   = !suggestions.empty();
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
                const auto p            = ImGui::GetCursorScreenPos();
                const bool clicked      = ImGui::InvisibleButton("##Tag", {item_width, tag_height * scale});
                const bool hovered      = ImGui::IsItemHovered();
                const bool pointer_over = ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenBlockedByActiveItem);
                tag_hovered |= pointer_over;
                const bool selected = selection.contains(i);
                const auto* change  = this->change ? &this->change->tags[i] : nullptr;
                const bool removed  = change && change->removed;
                const bool modified = change && change->original && (*change->original != tag || change->group != group_index || change->index != i);
                const ImVec4 ink    = removed ? ImVec4{0.53F, 0.53F, 0.58F, 0.55F} : item.added ? ImVec4{0.58F, 0.80F, 0.67F, 1} : modified ? ImVec4{0.77F, 0.70F, 0.97F, 1} : ImVec4{0.76F, 0.76F, 0.81F, 1};
                auto* draw          = ImGui::GetWindowDrawList();
                auto* storage       = ImGui::GetStateStorage();
                const auto key      = ImGui::GetItemID();
                const float alpha   = std::lerp(storage->GetFloat(key), selected || hovered ? 1.0F : 0.0F, std::min(1.0F, ImGui::GetIO().DeltaTime / 0.12F));
                storage->SetFloat(key, alpha);
                draw->AddRectFilled(p, {p.x + item_width, p.y + tag_height * scale}, ImGui::GetColorU32(selected ? ImVec4{0.23F, 0.21F, 0.32F, alpha * 0.70F} : ImVec4{0.20F, 0.205F, 0.25F, alpha * 0.55F}), 5 * scale);
                if ((item.added || modified) && !removed) draw->AddRect(p, {p.x + item_width, p.y + tag_height * scale}, ImGui::GetColorU32(ImVec4{ink.x, ink.y, ink.z, 0.28F}), 5 * scale);
                if (item.added) {
                    const ImVec2 plus{p.x + (tag_padding + 3) * scale, p.y + tag_height * scale / 2};
                    draw->AddLine({plus.x - 2.5F * scale, plus.y}, {plus.x + 2.5F * scale, plus.y}, ImGui::GetColorU32(ink), scale);
                    draw->AddLine({plus.x, plus.y - 2.5F * scale}, {plus.x, plus.y + 2.5F * scale}, ImGui::GetColorU32(ink), scale);
                }
                draw->PushClipRect(p, {p.x + item_width - weight_width - tag_padding * scale, p.y + tag_height * scale}, true);
                draw->AddText({p.x + (tag_padding + (item.added ? 10 : 0)) * scale, p.y + (tag_height * scale - ImGui::GetFontSize()) / 2}, ImGui::GetColorU32(ink), text.data(), text.data() + text.size());
                draw->PopClipRect();
                if (!weight.empty()) draw->AddText({p.x + item_width - weight_width + (weight_gap - tag_padding) * scale, p.y + (tag_height * scale - ImGui::GetFontSize()) / 2}, ImGui::GetColorU32(ink), weight.c_str());
                if (removed) draw->AddLine({p.x + tag_padding * scale, p.y + tag_height * scale / 2}, {p.x + item_width - tag_padding * scale, p.y + tag_height * scale / 2}, ImGui::GetColorU32(ink), scale);
                if (pointer_over && !ImGui::GetDragDropPayload() && ImGui::IsMouseClicked(ImGuiMouseButton_Middle)) {
                    ImGui::FocusWindow(ImGui::GetCurrentWindow());
                    remove = i;
                }
                if (clicked && !removed) {
                    if (!weight.empty() && ImGui::GetIO().MousePos.x >= p.x + item_width - weight_width - tag_padding * scale) ImGui::OpenPopup("Weight");
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
                if (!removed && hovered && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
                    editing     = i;
                    input       = prompt::serialize(catalog, std::span{group.tags}.subspan(i, 1));
                    cursor      = input.size();
                    focus_input = true;
                    error.reset();
                    ++revision;
                }
                if (!removed && ImGui::BeginDragDropSource()) {
                    const std::array location{group_index, i};
                    ImGui::SetDragDropPayload(payload_type, location.data(), sizeof(location));
                    ImGui::TextUnformatted(text.data(), text.data() + text.size());
                    ImGui::EndDragDropSource();
                }
                if (!editing && ImGui::BeginDragDropTarget()) {
                    if (const auto* payload = ImGui::AcceptDragDropPayload(payload_type, ImGuiDragDropFlags_AcceptPeekOnly)) {
                        const bool after = ImGui::GetIO().MousePos.x >= p.x + item_width / 2;
                        const float line = p.x + (after ? item_width + tag_gap * scale / 2 : -tag_gap * scale / 2);
                        draw->AddLine({line, p.y + 3 * scale}, {line, p.y + (tag_height - 3) * scale}, ImGui::GetColorU32(ImVec4{0.72F, 0.67F, 0.95F, 1}), 2 * scale);
                        if (payload->IsDelivery()) {
                            const auto* source = static_cast<const std::size_t*>(payload->Data);
                            move               = TagMove{source[0], source[1], group_index, i + std::size_t(after)};
                        }
                    }
                    ImGui::EndDragDropTarget();
                }
                if (ImGui::BeginPopup("Weight")) {
                    float value = group.tags[i].weight;
                    if (ImGui::InputFloat("Weight", &value, 0.05F, 0.1F, "%.2f", ImGuiInputTextFlags_EnterReturnsTrue) && std::isfinite(value)) {
                        auto next      = group.tags;
                        next[i].weight = value;
                        replace(group, std::move(next));
                    }
                    ImGui::EndPopup();
                }
                if (hovered) {
                    ImGui::BeginTooltip();
                    ImGui::TextUnformatted(catalog.tags[tag.id].name.data(), catalog.tags[tag.id].name.data() + catalog.tags[tag.id].name.size());
                    if (modified) ImGui::Text("Original: %s", prompt::serialize(catalog, std::span{&*change->original, 1}).c_str());
                    ImGui::TextUnformatted(removed ? "Excluded from Repaint. Middle-click to restore." : "Middle-click to remove. Double-click to edit. Ctrl+Up/Down adjusts weight.");
                    ImGui::EndTooltip();
                }
                ImGui::EndDisabled();
            }
            ImGui::PopID();
        }
        background_hovered = ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows | ImGuiHoveredFlags_AllowWhenBlockedByActiveItem) && !tag_hovered && !ImGui::IsMouseHoveringRect(input_min, input_max);
        if (ImGui::IsMouseClicked(ImGuiMouseButton_Left) && !tag_hovered && !ImGui::IsPopupOpen(nullptr, ImGuiPopupFlags_AnyPopupId | ImGuiPopupFlags_AnyPopupLevel)) selection.clear();
        const bool focused = ImGui::IsWindowFocused(ImGuiFocusedFlags_ChildWindows);
        if (focused && !editing && !remove && !ImGui::GetDragDropPayload() && (!input_active || input.empty()) && (!ImGui::GetIO().WantTextInput || input_active)) {
            const ImGuiInputFlags routing = input_active ? ImGuiInputFlags_RouteGlobal | ImGuiInputFlags_RouteOverActive : ImGuiInputFlags_RouteFocused;
            if (ImGui::Shortcut(ImGuiMod_Ctrl | ImGuiKey_A, routing)) {
                for (std::size_t i = 0; i < group.tags.size(); ++i) selection.insert(i);
            }
            if (!group.tags.empty() && !ImGui::GetIO().KeyCtrl && (ImGui::IsKeyPressed(ImGuiKey_LeftArrow) || ImGui::IsKeyPressed(ImGuiKey_RightArrow))) {
                const int direction = ImGui::IsKeyPressed(ImGuiKey_LeftArrow) ? -1 : 1;
                anchor              = static_cast<std::size_t>(std::clamp(static_cast<int>(selection.empty() ? 0 : *selection.rbegin()) + direction, 0, static_cast<int>(group.tags.size()) - 1));
                if (!ImGui::GetIO().KeyShift) selection.clear();
                selection.insert(anchor);
            }
            const bool cut  = ImGui::Shortcut(ImGuiMod_Ctrl | ImGuiKey_X, routing);
            const bool copy = ImGui::Shortcut(ImGuiMod_Ctrl | ImGuiKey_C, routing);
            if ((cut || copy) && !selection.empty()) {
                std::vector<prompt::Tag> copied;
                for (const auto i : selection)
                    if (!change || !change->tags[i].removed) copied.push_back(group.tags[i]);
                ImGui::SetClipboardText(prompt::serialize(catalog, copied).c_str());
            }
            const bool erase     = ImGui::Shortcut(ImGuiKey_Delete, routing);
            const bool backspace = ImGui::Shortcut(ImGuiKey_Backspace, routing);
            if (backspace && selection.empty() && !group.tags.empty()) {
                anchor = group.tags.size() - 1;
                selection.insert(anchor);
            } else if ((cut || erase || backspace) && !selection.empty()) {
                const auto selected = selection;
                for (const auto i : std::views::reverse(selected))
                    if (!change || !change->tags[i].removed) this->erase(group, i);
            }
            if (!input_active && ImGui::Shortcut(ImGuiMod_Ctrl | ImGuiKey_V, routing)) {
                input = ImGui::GetClipboardText();
                commit(group, catalog);
                focus_input = true;
                ++revision;
            }
            const bool increase = ImGui::Shortcut(ImGuiMod_Ctrl | ImGuiKey_UpArrow, routing);
            const bool decrease = ImGui::Shortcut(ImGuiMod_Ctrl | ImGuiKey_DownArrow, routing);
            if ((increase || decrease) && !selection.empty()) {
                auto next = group.tags;
                for (const auto i : selection)
                    if (!change || !change->tags[i].removed) next[i].weight = static_cast<float>(std::round((double(next[i].weight) + (increase ? 0.05 : -0.05)) * 100) / 100);
                const auto selected = selection;
                replace(group, std::move(next));
                selection = selected;
            }
        }
        if (remove) erase(group, *remove);
        ImGui::EndChild();
        ImGui::PopStyleVar(3);
        if (enter || paste || separator) {
            const auto candidate = enter && menu_open && !suggestions.empty() ? std::optional{suggestions[highlighted].tag} : std::nullopt;
            commit(group, catalog, candidate);
        }
        if ((input_active || menu_open || !ImGui::GetIO().WantTextInput) && ImGui::IsKeyPressed(ImGuiKey_Escape, false)) {
            if (menu_open) {
                menu_open   = false;
                focus_input = true;
            } else if (editing) {
                editing.reset();
                input.clear();
                error.reset();
                valid = true;
                ++revision;
            }
        }
        bool suggestions_hovered{};
        if (menu_open && !suggestions.empty()) {
            const float popup_width  = std::min(420 * scale, ImGui::GetIO().DisplaySize.x - 24 * scale);
            const float popup_height = (suggestions.size() * 44 + 12) * scale;
            const float below        = ImGui::GetIO().DisplaySize.y - input_max.y - 12 * scale;
            const float top          = below >= popup_height ? input_max.y + 6 * scale : std::max(6 * scale, input_min.y - popup_height - 6 * scale);
            ImGui::SetNextWindowPos({std::clamp(input_min.x, 12 * scale, ImGui::GetIO().DisplaySize.x - popup_width - 12 * scale), top});
            ImGui::SetNextWindowSize({popup_width, popup_height});
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {6 * scale, 6 * scale});
            ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {0, 0});
            ImGui::Begin(std::format("##Suggestions{}", id).c_str(), nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNavFocus);
            // Keep suggestions above the tag column without taking its text input focus.
            ImGui::BringWindowToDisplayFront(ImGui::GetCurrentWindow());
            std::optional<std::uint32_t> chosen;
            for (int i = 0; i < static_cast<int>(suggestions.size()); ++i) {
                const auto suggestion = suggestions[i];
                const auto& entry     = catalog.tags[suggestion.tag];
                const auto p          = ImGui::GetCursorScreenPos();
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
            suggestions_hovered = ImGui::IsWindowHovered(ImGuiHoveredFlags_AllowWhenBlockedByActiveItem);
            ImGui::End();
            ImGui::PopStyleVar(2);
            if (chosen) commit(group, catalog, chosen);
            if (ImGui::IsMouseClicked(ImGuiMouseButton_Left) && !suggestions_hovered && !ImGui::IsMouseHoveringRect(input_min, input_max)) menu_open = false;
        }
        if (editing && !input_active && !focus_input && !suggestions_hovered && commit(group, catalog)) focus_input = false;
        if (error) {
            ImGui::PushStyleColor(ImGuiCol_Text, {0.91F, 0.58F, 0.58F, 1});
            ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + width);
            ImGui::TextUnformatted(error->message.c_str());
            ImGui::PopTextWrapPos();
            ImGui::PopStyleColor();
        }
        ImGui::PopID();
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
                data->SelectionEnd   = std::min(data->BufTextLen, static_cast<int>(editor.error->offset + editor.error->length));
                data->CursorPos      = data->SelectionEnd;
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
            for (std::size_t g = 0; g < groups[s].size(); ++g) {
                auto& editor = groups[s][g];
                editor.id    = ++next_id;
                if (tracking) {
                    editor.change.emplace(GroupChange{g, sides[s]->groups[g].enabled});
                    for (std::size_t i = 0; i < sides[s]->groups[g].tags.size(); ++i) editor.change->tags.push_back({sides[s]->groups[g].tags[i], g, i});
                }
            }
            additions[s] = TagEditor{.change = tracking ? std::optional{GroupChange{}} : std::nullopt, .id = ++next_id};
        }
        adding = {};
        if (clear_history) {
            undo.clear();
            redo.clear();
            negative_open = false;
        }
        valid        = true;
        escape_owned = focus_input = false;
    }

    void PromptEditor::suspend() {
        for (std::size_t s = 0; s < groups.size(); ++s) {
            for (auto& editor : groups[s]) {
                editor.menu_open = editor.input_active = editor.focus_input = false;
                editor.selection.clear();
            }
            additions[s].menu_open = additions[s].input_active = additions[s].focus_input = false;
        }
        escape_owned = focus_input = false;
    }

    PromptEditor::Snapshot PromptEditor::snapshot(const prompt::Pair& prompt) const {
        Snapshot result{prompt};
        for (std::size_t s = 0; s < groups.size(); ++s)
            for (const auto& group : groups[s]) result.changes[s].push_back(group.change);
        return result;
    }

    prompt::Pair PromptEditor::materialize(const prompt::Pair& prompt) const {
        auto result = prompt;
        const std::array sides{&result.positive, &result.negative};
        for (std::size_t s = 0; s < sides.size(); ++s) {
            for (std::size_t g = 0; g < groups[s].size(); ++g) {
                if (!groups[s][g].change) continue;
                auto& tags = sides[s]->groups[g].tags;
                for (std::size_t i = tags.size(); i-- > 0;)
                    if (groups[s][g].change->tags[i].removed) tags.erase(tags.begin() + i);
            }
            std::erase_if(sides[s]->groups, [](const prompt::Group& group) { return group.tags.empty(); });
        }
        return result;
    }

    void PromptEditor::remember(Snapshot before, const prompt::Pair& after) {
        if (before == snapshot(after)) return;
        undo.push_back(std::move(before));
        if (undo.size() > 128) undo.erase(undo.begin());
        redo.clear();
    }

    bool PromptEditor::commit(prompt::Pair& prompt, const prompt::Catalog& catalog) {
        auto before = snapshot(prompt);
        const std::array sides{&prompt.positive, &prompt.negative};
        for (std::size_t s = 0; s < sides.size(); ++s) {
            for (std::size_t i = 0; i < groups[s].size(); ++i) {
                if (!groups[s][i].commit(sides[s]->groups[i], catalog)) {
                    if (s) negative_open = true;
                    valid = false;
                    remember(std::move(before), prompt);
                    return false;
                }
            }
            prompt::Group created;
            if (!additions[s].commit(created, catalog)) {
                if (s) negative_open = true;
                valid = false;
                remember(std::move(before), prompt);
                return false;
            }
            if (!created.tags.empty()) {
                sides[s]->groups.push_back(std::move(created));
                groups[s].push_back(std::move(additions[s]));
                additions[s] = TagEditor{.change = tracking ? std::optional{GroupChange{}} : std::nullopt, .id = ++next_id};
                adding[s]    = false;
            }
        }
        valid = true;
        remember(std::move(before), prompt);
        return true;
    }

    void PromptEditor::draw_groups(prompt::Side& side, const std::size_t side_index, const TagSearch& search, const prompt::Catalog& catalog, const float scale) {
        ImGui::PushID(static_cast<int>(side_index));
        auto& editors             = groups[side_index];
        auto& addition            = additions[side_index];
        const char* tag_payload   = side_index ? "GENESIA_NEGATIVE_TAG" : "GENESIA_POSITIVE_TAG";
        const char* group_payload = side_index ? "GENESIA_NEGATIVE_GROUP" : "GENESIA_POSITIVE_GROUP";
        std::optional<TagMove> tag_move;
        std::optional<std::pair<std::size_t, std::size_t>> group_move;
        const float width         = ImGui::GetContentRegionAvail().x;
        const float content_width = width - 24 * scale;
        const auto accent         = ImGui::GetColorU32(ImVec4{0.72F, 0.67F, 0.95F, 1});
        auto* draw                = ImGui::GetWindowDrawList();
        for (std::size_t i = 0; i < side.groups.size(); ++i) {
            auto& group  = side.groups[i];
            auto& editor = editors[i];
            ImGui::PushID(static_cast<int>(editor.id));
            const auto origin        = ImGui::GetCursorScreenPos();
            const float error_height = editor.error ? ImGui::CalcTextSize(editor.error->message.c_str(), nullptr, false, content_width).y + 6 * scale : 0;
            const TagLayout layout{group, catalog, content_width, scale, editor};
            const ImRect bounds{origin, {origin.x + width, origin.y + layout.height + 24 * scale + error_height}};
            const bool hovered = ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows | ImGuiHoveredFlags_AllowWhenBlockedByActiveItem) && ImGui::IsMouseHoveringRect(bounds.Min, bounds.Max);
            group_frame(bounds.Min, bounds.Max, scale, group.enabled, hovered);
            if (editor.change && editor.change->original && (editor.change->enabled != group.enabled || *editor.change->original != i)) draw->AddRect(bounds.Min, bounds.Max, ImGui::GetColorU32(ImVec4{0.72F, 0.65F, 0.94F, 0.45F}), 12 * scale, 0, scale);
            ImGui::SetCursorScreenPos({origin.x + 12 * scale, origin.y + 12 * scale});
            if (!group.enabled) ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.45F);
            if (editor.focus_input) ImGui::SetScrollHereY();
            editor.draw(tag_payload, i, group, search, catalog, layout, scale, tag_move);
            if (!group.enabled) ImGui::PopStyleVar();
            ImGui::SetCursorScreenPos(origin);
            ImGui::Dummy(bounds.GetSize());
            const auto key = ImGui::GetID("##Group");
            ImGui::ItemAdd(bounds, key, nullptr, ImGuiItemFlags_NoNav);
            const bool background = editor.background_hovered || (hovered && ImGui::IsWindowHovered(ImGuiHoveredFlags_AllowWhenBlockedByActiveItem));
            if (background || ImGui::GetActiveID() == key) {
                bool held{}, hit{};
                ImGui::ButtonBehavior(bounds, key, &hit, &held, ImGuiButtonFlags_FlattenChildren | ImGuiButtonFlags_NoNavFocus);
                if (background) {
                    ImGui::SetTooltip("Drag: reorder group\nMiddle-click: %s group", group.enabled ? "disable" : "enable");
                    if (!ImGui::GetDragDropPayload() && ImGui::IsMouseClicked(ImGuiMouseButton_Middle)) {
                        ImGui::FocusWindow(ImGui::GetCurrentWindow());
                        group.enabled = !group.enabled;
                    }
                }
            }
            if (ImGui::BeginDragDropSource()) {
                ImGui::SetDragDropPayload(group_payload, &i, sizeof(i));
                ImGui::TextWrapped("%s", prompt::serialize(catalog, group.tags).c_str());
                ImGui::EndDragDropSource();
            }
            if (ImGui::BeginDragDropTargetCustom(bounds, ImGui::GetID("##GroupDrop"))) {
                if (!editor.editing) {
                    if (const auto* payload = ImGui::AcceptDragDropPayload(tag_payload, ImGuiDragDropFlags_AcceptPeekOnly)) {
                        draw->AddRect(bounds.Min, bounds.Max, accent, 12 * scale, 0, scale);
                        if (payload->IsDelivery()) {
                            const auto* source = static_cast<const std::size_t*>(payload->Data);
                            tag_move           = TagMove{source[0], source[1], i, group.tags.size()};
                        }
                    }
                }
                if (const auto* payload = ImGui::AcceptDragDropPayload(group_payload, ImGuiDragDropFlags_AcceptPeekOnly)) {
                    const bool after = ImGui::GetIO().MousePos.y >= bounds.GetCenter().y;
                    const float y    = after ? bounds.Max.y + 6 * scale : bounds.Min.y - 6 * scale;
                    draw->AddLine({bounds.Min.x + 4 * scale, y}, {bounds.Max.x - 4 * scale, y}, accent, 2 * scale);
                    if (payload->IsDelivery()) group_move = std::pair{*static_cast<const std::size_t*>(payload->Data), i + std::size_t(after)};
                }
                ImGui::EndDragDropTarget();
            }
            ImGui::Dummy({0, 6 * scale});
            ImGui::PopID();
        }

        prompt::Group created;
        if (adding[side_index]) {
            ImGui::PushID(static_cast<int>(addition.id));
            const auto origin        = ImGui::GetCursorScreenPos();
            const float error_height = addition.error ? ImGui::CalcTextSize(addition.error->message.c_str(), nullptr, false, content_width).y + 6 * scale : 0;
            const TagLayout layout{created, catalog, content_width, scale, addition};
            const ImVec2 size{width, layout.height + 24 * scale + error_height};
            group_frame(origin, {origin.x + size.x, origin.y + size.y}, scale, true, true, true);
            ImGui::SetCursorScreenPos({origin.x + 12 * scale, origin.y + 12 * scale});
            const bool cancel = !addition.menu_open && (addition.input_active || addition.focus_input) && ImGui::IsKeyPressed(ImGuiKey_Escape, false);
            addition.draw(tag_payload, side.groups.size(), created, search, catalog, layout, scale, tag_move);
            ImGui::SetCursorScreenPos(origin);
            ImGui::Dummy(size);
            ImGui::PopID();
            if (cancel || (addition.input.empty() && !addition.input_active && !addition.focus_input && created.tags.empty())) {
                if (cancel) ImGui::ClearActiveID();
                addition           = TagEditor{.change = tracking ? std::optional{GroupChange{}} : std::nullopt, .id = ++next_id};
                adding[side_index] = false;
            }
        } else {
            const auto origin = ImGui::GetCursorScreenPos();
            const ImVec2 size{width, 48 * scale};
            if (ImGui::InvisibleButton("##NewGroup", size, ImGuiButtonFlags_EnableNav)) {
                adding[side_index]   = true;
                addition.focus_input = true;
            }
            const bool hovered = ImGui::IsItemHovered();
            group_frame(origin, {origin.x + size.x, origin.y + size.y}, scale, true, hovered, true);
            const ImVec2 center{origin.x + size.x / 2, origin.y + size.y / 2};
            const auto ink = ImGui::GetColorU32(hovered ? ImVec4{0.77F, 0.73F, 0.91F, 1} : ImVec4{0.55F, 0.54F, 0.62F, 1});
            draw->AddLine({center.x - 5 * scale, center.y}, {center.x + 5 * scale, center.y}, ink, scale);
            draw->AddLine({center.x, center.y - 5 * scale}, {center.x, center.y + 5 * scale}, ink, scale);
            if (hovered) ImGui::SetTooltip("Drop a tag to create a group\nClick to enter the first tag");
            if (ImGui::BeginDragDropTarget()) {
                if (const auto* payload = ImGui::AcceptDragDropPayload(tag_payload, ImGuiDragDropFlags_AcceptPeekOnly)) {
                    draw->AddRect(origin, {origin.x + size.x, origin.y + size.y}, accent, 12 * scale, 0, scale);
                    if (payload->IsDelivery()) {
                        const auto* source = static_cast<const std::size_t*>(payload->Data);
                        tag_move           = TagMove{source[0], source[1], side.groups.size(), 0};
                    }
                }
                if (const auto* payload = ImGui::AcceptDragDropPayload(group_payload, ImGuiDragDropFlags_AcceptPeekOnly)) {
                    draw->AddLine({origin.x + 4 * scale, origin.y - 6 * scale}, {origin.x + size.x - 4 * scale, origin.y - 6 * scale}, accent, 2 * scale);
                    if (payload->IsDelivery()) group_move = std::pair{*static_cast<const std::size_t*>(payload->Data), side.groups.size()};
                }
                ImGui::EndDragDropTarget();
            }
        }

        // Commit moves together, then remove empty groups so drag indices stay stable.
        if (tag_move) {
            const auto [from, index, to, insertion] = *tag_move;
            const auto tag                          = side.groups[from].tags[index];
            const bool enabled                      = side.groups[from].enabled;
            auto& source                            = side.groups[from].tags;
            std::optional<TagChange> change;
            if (tracking) {
                change = editors[from].change->tags[index];
                editors[from].change->tags.erase(editors[from].change->tags.begin() + index);
            }
            source.erase(source.begin() + index);
            editors[from].selection.clear();
            editors[from].anchor = 0;
            if (to == side.groups.size()) {
                side.groups.push_back({{tag}, enabled});
                auto& added = editors.emplace_back();
                added.id    = ++next_id;
                if (tracking) added.change.emplace(GroupChange{std::nullopt, enabled, {*change}});
            } else {
                auto& target        = side.groups[to].tags;
                const auto position = insertion - (from == to && index < insertion ? 1 : 0);
                target.insert(target.begin() + position, tag);
                if (tracking) editors[to].change->tags.insert(editors[to].change->tags.begin() + position, *change);
                editors[to].selection.clear();
                editors[to].anchor = 0;
            }
        }
        if (group_move) {
            const auto [from, insertion] = *group_move;
            const auto to                = insertion - std::size_t(from < insertion);
            auto group                   = std::move(side.groups[from]);
            auto editor                  = std::move(editors[from]);
            side.groups.erase(side.groups.begin() + from);
            editors.erase(editors.begin() + from);
            side.groups.insert(side.groups.begin() + to, std::move(group));
            editors.insert(editors.begin() + to, std::move(editor));
        }
        if (!created.tags.empty()) {
            side.groups.push_back(std::move(created));
            editors.push_back(std::move(addition));
            addition           = TagEditor{.change = tracking ? std::optional{GroupChange{}} : std::nullopt, .id = ++next_id};
            adding[side_index] = false;
        }
        for (std::size_t i = side.groups.size(); i-- > 0;) {
            // An unfinished tag remains editable until it is committed or cleared.
            if (!side.groups[i].tags.empty() || !editors[i].input.empty() || editors[i].editing) continue;
            side.groups.erase(side.groups.begin() + i);
            editors.erase(editors.begin() + i);
        }
        ImGui::PopID();
    }

    void PromptEditor::draw(prompt::Pair& prompt, const TagSearch& search, const prompt::Catalog& catalog, const float scale) {
        auto before = snapshot(prompt);
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {6 * scale, 6 * scale});
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, {8 * scale, 5 * scale});
        ImGui::PushStyleColor(ImGuiCol_Button, {0, 0, 0, 0});
        draw_groups(prompt.positive, 0, search, catalog, scale);
        ImGui::Dummy({0, 10 * scale});
        ImGui::SetNextItemOpen(negative_open, ImGuiCond_Always);
        const bool show_negative = ImGui::TreeNodeEx("Negative", ImGuiTreeNodeFlags_NoTreePushOnOpen);
        if (negative_open && !show_negative) {
            for (auto& editor : groups[1]) {
                editor.menu_open = editor.input_active = editor.focus_input = false;
                editor.selection.clear();
            }
            additions[1].menu_open = additions[1].input_active = additions[1].focus_input = false;
        }
        negative_open = show_negative;
        if (negative_open) draw_groups(prompt.negative, 1, search, catalog, scale);
        fixed_text(prompt);
        ImGui::PopStyleColor();
        ImGui::PopStyleVar(2);
        remember(std::move(before), prompt);
        const auto unfinished         = [](const TagEditor& editor) { return !editor.input.empty() || editor.editing; };
        const auto empty_active       = [](const TagEditor& editor) { return editor.input_active && editor.input.empty(); };
        const bool pending            = std::ranges::any_of(groups | std::views::join, unfinished) || std::ranges::any_of(additions, unfinished);
        const bool empty_input_active = std::ranges::any_of(groups | std::views::join, empty_active) || std::ranges::any_of(additions, empty_active);
        if (ImGui::IsWindowFocused(ImGuiFocusedFlags_ChildWindows) && !pending && !ImGui::GetDragDropPayload() && (!ImGui::GetIO().WantTextInput || empty_input_active)) {
            const ImGuiInputFlags routing = empty_input_active ? ImGuiInputFlags_RouteGlobal | ImGuiInputFlags_RouteOverActive : ImGuiInputFlags_RouteFocused;
            const bool backwards          = ImGui::Shortcut(ImGuiMod_Ctrl | ImGuiKey_Z, routing);
            const bool forwards           = ImGui::Shortcut(ImGuiMod_Ctrl | ImGuiKey_Y, routing) || ImGui::Shortcut(ImGuiMod_Ctrl | ImGuiMod_Shift | ImGuiKey_Z, routing);
            if ((backwards && !undo.empty()) || (forwards && !redo.empty())) {
                auto& source = backwards ? undo : redo;
                auto& target = backwards ? redo : undo;
                target.push_back(snapshot(prompt));
                auto restored = std::move(source.back());
                source.pop_back();
                prompt = std::move(restored.prompt);
                ImGui::ClearActiveID();
                reset(prompt, false);
                for (std::size_t s = 0; s < groups.size(); ++s)
                    for (std::size_t g = 0; g < groups[s].size(); ++g) groups[s][g].change = std::move(restored.changes[s][g]);
            }
        }
        valid        = std::ranges::all_of(groups | std::views::join, &TagEditor::valid) && std::ranges::all_of(additions, &TagEditor::valid);
        escape_owned = focus_input = false;
        for (std::size_t s = 0; s < groups.size(); ++s) {
            if (s && !negative_open) continue;
            for (const auto& editor : groups[s]) {
                escape_owned |= editor.menu_open || editor.editing.has_value();
                focus_input |= editor.focus_input;
            }
            escape_owned |= adding[s];
            focus_input |= additions[s].focus_input;
        }
    }

} // namespace genesia::editor
