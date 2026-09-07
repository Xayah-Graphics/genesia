module;
#include <Windows.h>

#include <GLFW/glfw3.h>

#include "../../core/sdxl/control.h"
#include <genesia/cuda.h>
#include <imgui.h>
#include <imgui_internal.h>
#include <shellapi.h>
module genesia.editor.ui;
import genesia.generation.configuration;
import genesia.generation.output;
import genesia.sdxl;
import genesia.editor.platform.window;
import genesia.editor.ui.renderer;
import genesia.editor.platform.interop;
import genesia.editor.session;
import std;

namespace genesia::editor {
    namespace {
        constexpr ImGuiWindowFlags overlay = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar;
        constexpr float top_strip_height = 48;
        constexpr float control_height = 40;
        constexpr float bottom_margin = 24;
        constexpr float panel_gap = 12;

        void control_shade(const ImVec2 minimum, const ImVec2 maximum, const float scale) {
            auto* draw = ImGui::GetWindowDrawList();
            const float x = (minimum.x + maximum.x) / 2;
            const float y = (minimum.y + maximum.y) / 2;
            const float feather = 12 * scale;
            const ImVec2 first{minimum.x - feather, minimum.y - feather};
            const ImVec2 last{maximum.x + feather, maximum.y + feather};
            const ImU32 clear = IM_COL32(16, 17, 20, 0);
            const ImU32 shade = ImGui::GetColorU32(ImVec4{0.063F, 0.067F, 0.078F, 0.70F});
            draw->PushClipRect(first, last, false);
            draw->AddRectFilledMultiColor(first, {x, y}, clear, clear, shade, clear);
            draw->AddRectFilledMultiColor({x, first.y}, {last.x, y}, clear, clear, clear, shade);
            draw->AddRectFilledMultiColor({first.x, y}, {x, last.y}, clear, shade, clear, clear);
            draw->AddRectFilledMultiColor({x, y}, last, shade, clear, clear, clear);
            draw->PopClipRect();
        }

        bool text_button(const char* id, const char* label, const float scale, const float width = 0, const bool selected = false, const bool primary = false) {
            const auto text = ImGui::CalcTextSize(label);
            const auto origin = ImGui::GetCursorScreenPos();
            const ImVec2 size{width ? width : text.x + 24 * scale, (primary ? top_strip_height : control_height) * scale};
            ImGui::PushStyleColor(ImGuiCol_NavCursor, {0, 0, 0, 0});
            const bool clicked = ImGui::InvisibleButton(id, size, ImGuiButtonFlags_EnableNav);
            ImGui::PopStyleColor();
            const bool focused = ImGui::IsItemFocused() && ImGui::GetCurrentContext()->NavCursorVisible;
            const bool hovered = ImGui::IsItemHovered() || focused;
            auto* storage = ImGui::GetStateStorage();
            const auto key = ImGui::GetItemID();
            const float alpha = std::lerp(storage->GetFloat(key), hovered || selected ? 1.0F : 0.0F, std::min(1.0F, ImGui::GetIO().DeltaTime / 0.12F));
            storage->SetFloat(key, alpha);
            auto* draw = ImGui::GetWindowDrawList();
            if (primary && alpha > 0) {
                const auto clear = ImGui::GetColorU32(ImVec4{0.55F, 0.47F, 0.90F, 0});
                const auto tint = ImGui::GetColorU32(ImVec4{0.55F, 0.47F, 0.90F, alpha * 0.16F});
                draw->AddRectFilledMultiColor(origin, {origin.x + size.x, origin.y + size.y}, clear, tint, tint, clear);
            }
            const ImVec2 position{origin.x + (size.x - text.x) / 2, origin.y + (size.y - text.y) / 2};
            draw->AddText({position.x, position.y + scale}, ImGui::GetColorU32(ImVec4{0, 0, 0, 0.7F}), label);
            const ImVec4 ink = primary ? ImVec4{0.70F + 0.12F * alpha, 0.65F + 0.12F * alpha, 0.97F, 1} : ImVec4{0.57F + 0.31F * alpha, 0.58F + 0.30F * alpha, 0.64F + 0.29F * alpha, 1};
            draw->AddText(position, ImGui::GetColorU32(ink), label);
            if (focused) draw->AddLine({position.x, position.y + text.y + 3 * scale}, {position.x + text.x, position.y + text.y + 3 * scale}, ImGui::GetColorU32(ink), scale);
            return clicked;
        }

        bool number_field(const char* id, const char* label, const ImGuiDataType type, void* value, const ImVec2 size, const void* step, const char* format, const float scale, UserInterface::ParameterEdit& edit) {
            const auto origin = ImGui::GetCursorScreenPos();
            const ImVec2 end{origin.x + size.x, origin.y + size.y};
            const auto key = ImGui::GetID(id);
            const bool active = ImGui::GetActiveID() == key;
            const bool hovered = ImGui::IsWindowHovered(ImGuiHoveredFlags_AllowWhenBlockedByActiveItem) && ImGui::IsMouseHoveringRect(origin, end);
            auto* storage = ImGui::GetStateStorage();
            const float alpha = std::lerp(storage->GetFloat(key), active || hovered ? 1.0F : 0.0F, std::min(1.0F, ImGui::GetIO().DeltaTime / 0.12F));
            storage->SetFloat(key, alpha);
            auto* draw = ImGui::GetWindowDrawList();
            const auto text = ImGui::CalcTextSize(label);
            const float inset = *label ? size.x - 56 * scale : 0;
            if (*label) draw->AddText({origin.x + 4 * scale, origin.y + (size.y - text.y) / 2}, ImGui::GetColorU32(ImVec4{0.57F + 0.22F * alpha, 0.58F + 0.22F * alpha, 0.64F + 0.22F * alpha, 1}), label);
            const bool clicked = hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left);
            if (clicked && ImGui::GetIO().MousePos.x < origin.x + inset) ImGui::SetKeyboardFocusHere();
            bool stepped{};
            if (active && step) {
                ImGui::SetKeyOwner(ImGuiKey_UpArrow, key);
                ImGui::SetKeyOwner(ImGuiKey_DownArrow, key);
                const bool up = ImGui::IsKeyPressed(ImGuiKey_UpArrow, ImGuiInputFlags_Repeat, key);
                const bool down = ImGui::IsKeyPressed(ImGuiKey_DownArrow, ImGuiInputFlags_Repeat, key);
                if (up || down) {
                    auto* input = ImGui::GetInputTextState(key);
                    ImGui::DataTypeApplyFromText(input->TextA.Data, type, value, format);
                    ImGui::DataTypeApplyOp(type, up ? '+' : '-', value, value, step);
                    input->ReloadUserBufAndSelectAll();
                    stepped = true;
                }
            }
            ImGui::SetCursorScreenPos({origin.x + inset, origin.y});
            ImGui::SetNextItemWidth(size.x - inset);
            ImGui::PushStyleColor(ImGuiCol_FrameBg, {0, 0, 0, 0});
            ImGui::PushStyleColor(ImGuiCol_Text, {0.65F + 0.25F * alpha, 0.65F + 0.25F * alpha, 0.71F + 0.23F * alpha, 1});
            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, {4 * scale, (size.y - ImGui::GetFontSize()) / 2});
            const bool changed = ImGui::InputScalar(id, type, value, nullptr, nullptr, format);
            ImGui::PopStyleVar();
            ImGui::PopStyleColor(2);
            if (ImGui::IsItemActive()) edit = {key, type, value};
            else if (edit.id == key) edit = {};
            if (ImGui::IsItemActive()) draw->AddLine({origin.x + inset + 4 * scale, end.y - 6 * scale}, {end.x - 4 * scale, end.y - 6 * scale}, ImGui::GetColorU32(ImVec4{0.65F, 0.60F, 0.88F, 0.8F}), scale);
            if (stepped) ImGui::MarkItemEdited(key);
            return changed || stepped;
        }

    } // namespace

    UserInterface::UserInterface(Configuration settings, std::filesystem::path path, WindowPlatform& platform, Renderer& display, Interop& bridge, Session& generation) : configuration{std::move(settings)}, configuration_path{std::move(path)}, window{platform}, renderer{display}, interop{bridge}, session{generation}, draft{configuration.parameters}, prompt{configuration.prompt}, tag_search{*configuration.catalog}, seed{configuration.seeds.front()} {
        prompt_editor.reset(prompt);
        ImGui::StyleColorsDark();
        auto& style          = ImGui::GetStyle();
        style.WindowRounding = 16;
        style.ChildRounding = style.FrameRounding = style.GrabRounding = 8;
        style.PopupRounding                                            = 12;
        style.WindowBorderSize                                         = 0;
        style.PopupBorderSize                                          = 0;
        style.FrameBorderSize                                          = 0;
        style.WindowPadding                                            = {20, 16};
        style.FramePadding                                             = {12, 9};
        style.ItemSpacing                                              = {8, 10};
        style.ScrollbarSize                                            = 8;
        style.ScrollbarRounding                                        = 8;
        style.Colors[ImGuiCol_Text]                                    = {0.93F, 0.93F, 0.96F, 1};
        style.Colors[ImGuiCol_TextDisabled]                            = {0.57F, 0.58F, 0.64F, 1};
        style.Colors[ImGuiCol_WindowBg]                                = {0.095F, 0.10F, 0.125F, 0.985F};
        style.Colors[ImGuiCol_PopupBg]                                 = {0.12F, 0.125F, 0.15F, 1};
        style.Colors[ImGuiCol_Border]                                  = {0.70F, 0.72F, 0.85F, 0.10F};
        style.Colors[ImGuiCol_FrameBg]                                 = {0.07F, 0.075F, 0.095F, 1};
        style.Colors[ImGuiCol_FrameBgHovered]                          = {0.14F, 0.145F, 0.18F, 1};
        style.Colors[ImGuiCol_FrameBgActive]                           = {0.16F, 0.16F, 0.21F, 1};
        style.Colors[ImGuiCol_Button]                                  = {0.17F, 0.175F, 0.215F, 1};
        style.Colors[ImGuiCol_ButtonHovered]                           = {0.23F, 0.23F, 0.29F, 1};
        style.Colors[ImGuiCol_ButtonActive]                            = {0.30F, 0.29F, 0.38F, 1};
        style.Colors[ImGuiCol_Header]                                  = {0.35F, 0.34F, 0.55F, 0.35F};
        style.Colors[ImGuiCol_HeaderHovered]                           = {0.45F, 0.44F, 0.67F, 0.35F};
        style.Colors[ImGuiCol_CheckMark] = style.Colors[ImGuiCol_SliderGrab] = {0.63F, 0.62F, 1, 1};
        style.Colors[ImGuiCol_NavCursor]                                     = {0.63F, 0.62F, 1, 0.8F};
    }

    void UserInterface::receive() {
        if (transition_texture && glfwGetTime() - transition_started >= 0.12) renderer.retire(std::exchange(transition_texture, 0));
        std::deque<Event> events;
        std::deque<PreviewFrame> previews;
        std::optional<Request> active;
        {
            const std::lock_guard lock{session.mutex};
            events.swap(session.events);
            previews.swap(session.previews);
            active = session.active;
        }
        if (active && observed_task != active->id) {
            observed_task = active->id;
            preview_step  = 0;
        }
        for (std::size_t i = 0; i < previews.size(); ++i) {
            const auto& frame      = previews[i];
            auto& source           = session.preview_interop.slots[frame.slot];
            const bool final_ready = std::ranges::any_of(events, [&](const Event& event) { return event.kind == EventKind::generated && event.id == frame.id; });
            if (i + 1 != previews.size() || !following_latest || !configuration.preview.enabled || !renderer.visible || frame.id != observed_task || frame.step <= preview_step || final_ready || (!history.empty() && frame.id <= history.back().id)) {
                renderer.discard(*source.timeline, frame.ready);
                continue;
            }
            const auto texture = renderer.texture({std::uint32_t(frame.width), std::uint32_t(frame.height)});
            renderer.copy(texture, source.buffer, *source.timeline, frame.ready);
            show_image(texture, frame.width, frame.height, !image_live || selected != frame.id);
            selected     = frame.id;
            image_live   = true;
            preview_step = frame.step;
        }
        for (auto& event : events) {
            if (event.kind == EventKind::generated) {
                auto& source = interop.slots[event.slot];
                history.push_back({event.id, event.record, renderer.upload(event.image)});
                if (following_latest) {
                    const auto texture = renderer.texture({std::uint32_t(event.record.parameters.width), std::uint32_t(event.record.parameters.height)});
                    renderer.copy(texture, source.buffer, *source.timeline, event.ready);
                    show_image(texture, event.record.parameters.width, event.record.parameters.height, true);
                    image_live = false;
                    selected   = event.id;
                } else renderer.discard(*source.timeline, event.ready);
                if (!edited_since_submit && !ImGui::GetIO().WantTextInput) composer_open = false;
                animate_until = glfwGetTime() + 0.2;
                display_times.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - event.generated_at).count());
            } else if (event.kind == EventKind::saved) {
                for (auto& item : history)
                    if (item.id == event.id) item.saved = true;
            } else if (event.id == selected && !image_live) {
                show_image(renderer.upload(event.image), event.image.width, event.image.height, true);
                fit_image = true;
                pan       = {};
            }
        }
    }

    void UserInterface::show_image(const std::uint64_t texture, const int width, const int height, const bool transition) {
        if (transition_texture) renderer.retire(std::exchange(transition_texture, 0));
        if (transition) {
            transition_texture = image_texture;
            transition_width   = image_width;
            transition_height  = image_height;
            transition_started = glfwGetTime();
            animate_until      = transition_started + 0.12;
        } else if (image_texture) renderer.retire(image_texture);
        if (image_width != width || image_height != height) {
            fit_image = true;
            pan       = {};
        }
        image_texture = texture;
        image_width   = width;
        image_height  = height;
    }

    bool UserInterface::prepare_prompt() {
        if (!prompt_editor.commit(prompt, *configuration.catalog)) { composer_open = true; return false; }
        draft.positive = prompt::compose(*configuration.catalog, prompt.positive);
        draft.negative = prompt::compose(*configuration.catalog, prompt.negative);
        return true;
    }

    void UserInterface::commit_parameters() {
        if (!parameter_edit.id) return;
        auto* input = ImGui::GetInputTextState(parameter_edit.id);
        if (ImGui::DataTypeApplyFromText(input->TextA.Data, parameter_edit.type, parameter_edit.value, ImGui::DataTypeGetInfo(parameter_edit.type)->ScanFmt)) edited_since_submit = true;
        // The value is committed; do not replay the text after a parameter action.
        input->ID = 0;
        if (ImGui::GetCurrentContext()->InputTextDeactivatedState.ID == parameter_edit.id) ImGui::GetCurrentContext()->InputTextDeactivatedState.ID = 0;
        ImGui::ClearActiveID();
        parameter_edit = {};
    }

    void UserInterface::save_settings() {
        commit_parameters();
        if (!prepare_prompt()) return;
        configuration.parameters = draft;
        configuration.prompt = prompt;
        configuration.seeds = {seed};
        write_configuration(configuration, configuration_path);
    }

    void UserInterface::submit() {
        commit_parameters();
        if (!prepare_prompt()) return;
        if (random_seed) {
            std::random_device source;
            seed = (std::uint64_t(source()) << 32) | source();
        }
        session.enqueue(draft, seed, prompt);
        edited_since_submit = false;
        animate_until       = glfwGetTime() + 0.2;
    }

    UserInterface::ControlLayout UserInterface::control_layout(const float scale, const ImVec2 size) const {
        ControlLayout layout{};
        layout.left_width = ImGui::CalcTextSize("Prompt").x + 24 * scale;
        if (!following_latest) layout.latest_width = ImGui::CalcTextSize("View live").x + 36 * scale;
        layout.different = image_texture && (image_width != draft.width || image_height != draft.height);
        const float dimensions_width = 112 * scale + (layout.different ? ImGui::CalcTextSize("Next").x + 12 * scale : 0);
        if (layout.different) layout.image_label_width = ImGui::CalcTextSize(std::format("{} {} \xC3\x97 {} \xE2\x86\x92", image_live ? "Preview" : "Image", image_width, image_height).c_str()).x + 12 * scale;
        layout.right_width = layout.latest_width + layout.image_label_width + dimensions_width + (image_texture ? 116 * scale : 0);
        const float available = size.x - (2 * bottom_margin + history_amount * 300) * scale;
        layout.image_above = layout.right_width > available;
        if (layout.image_above) layout.right_width -= layout.image_label_width;
        layout.stacked = layout.image_above || layout.left_width + layout.right_width + panel_gap * scale > available;
        layout.height = (control_height + (layout.stacked ? control_height + 8 : 0)) * scale;
        return layout;
    }

    void UserInterface::canvas(const float scale, const ImVec2 size, const float composer_height, const float controls_height) {
        ImGui::SetNextWindowPos({0, 0});
        ImGui::SetNextWindowSize(size);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {0, 0});
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0);
        ImGui::Begin("##Canvas", nullptr, overlay | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNav);
        ImGui::InvisibleButton("##ImageArea", size);
        const bool hovered = ImGui::IsItemHovered();
        const float left   = history_amount * 300 * scale;
        const float bottom = size.y - composer_amount * (composer_height + controls_height + (bottom_margin + panel_gap) * scale);
        const ImVec2 center{(left + size.x) / 2, bottom / 2};
        const ImVec2 available{size.x - left, std::max(64 * scale, bottom)};
        auto* draw = ImGui::GetWindowDrawList();
        if (image_texture) {
            const float fitted = std::min(available.x / image_width, available.y / image_height);
            if (fit_image) zoom = fitted;
            if (hovered && ImGui::GetIO().MouseWheel != 0) {
                const float next = std::clamp(zoom * std::pow(1.15F, ImGui::GetIO().MouseWheel), 0.05F, 16.0F);
                const auto mouse = ImGui::GetIO().MousePos;
                pan              = {mouse.x - center.x - (mouse.x - center.x - pan.x) * next / zoom, mouse.y - center.y - (mouse.y - center.y - pan.y) * next / zoom};
                zoom             = next;
                fit_image        = false;
            }
            if (hovered && ImGui::IsKeyDown(ImGuiKey_Space) && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
                pan.x += ImGui::GetIO().MouseDelta.x;
                pan.y += ImGui::GetIO().MouseDelta.y;
                fit_image = false;
                ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeAll);
            }
            if (hovered && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
                fit_image = true;
                pan       = {};
            }
            const ImVec2 minimum{center.x + pan.x - image_width * zoom / 2, center.y + pan.y - image_height * zoom / 2};
            const ImVec2 maximum{minimum.x + image_width * zoom, minimum.y + image_height * zoom};
            draw->PushClipRect({left, 0}, {size.x, std::max(0.0F, bottom)}, true);
            const float fade = std::clamp(float((glfwGetTime() - transition_started) / 0.12), 0.0F, 1.0F);
            if (transition_texture) {
                const float previous_zoom = fit_image ? std::min(available.x / transition_width, available.y / transition_height) : zoom;
                const ImVec2 origin{center.x + pan.x - transition_width * previous_zoom / 2, center.y + pan.y - transition_height * previous_zoom / 2};
                draw->AddImage(transition_texture, origin, {origin.x + transition_width * previous_zoom, origin.y + transition_height * previous_zoom});
            }
            draw->AddImage(image_texture, minimum, maximum, {0, 0}, {1, 1}, ImGui::GetColorU32(ImVec4{1, 1, 1, transition_texture ? fade : 1.0F}));
            draw->PopClipRect();
        } else {
            const char* title = "Imagine something new.";
            ImGui::PushFont(nullptr, 30);
            const auto text = ImGui::CalcTextSize(title);
            draw->AddText(ImGui::GetFont(), ImGui::GetFontSize(), {center.x - text.x / 2, center.y - 30 * scale}, IM_COL32(218, 219, 230, 255), title);
            ImGui::PopFont();
            const char* subtitle = "A few words. A world of possibilities.";
            const auto hint      = ImGui::CalcTextSize(subtitle);
            draw->AddText({center.x - hint.x / 2, center.y + 20 * scale}, IM_COL32(119, 121, 137, 255), subtitle);
            draw->AddCircle({center.x, center.y - 88 * scale}, 14 * scale, IM_COL32(145, 142, 225, 180), 32, 1.5F * scale);
            draw->AddCircleFilled({center.x + 14 * scale, center.y - 100 * scale}, 3 * scale, IM_COL32(184, 182, 250, 255));
        }
        ImGui::End();
        ImGui::PopStyleVar(2);
    }

    void UserInterface::generation_settings(const float scale, const ImVec2 size) {
        const float row_height = 32 * scale;
        const float seed_label_width = ImGui::CalcTextSize("Seed").x + 12 * scale;
        const float mode_width = ImGui::CalcTextSize("Random").x + 16 * scale;
        const float panel_width = std::max(336 * scale, seed_label_width + mode_width + ImGui::CalcTextSize("18446744073709551615").x + 48 * scale);
        ImGui::SetNextWindowPos({size.x - 12 * scale, (top_strip_height + 6) * scale}, ImGuiCond_Always, {1, 0});
        ImGui::SetNextWindowSize({panel_width, 0});
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {16 * scale, 12 * scale});
        ImGui::PushStyleVar(ImGuiStyleVar_PopupRounding, 12 * scale);
        ImGui::PushStyleVar(ImGuiStyleVar_PopupBorderSize, scale);
        ImGui::PushStyleColor(ImGuiCol_PopupBg, {0.075F, 0.078F, 0.09F, 0.94F});
        ImGui::PushStyleColor(ImGuiCol_Border, {1, 1, 1, 0.055F});
        const bool open = ImGui::BeginPopup("Generation settings", ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar);
        ImGui::PopStyleColor(2);
        ImGui::PopStyleVar(3);
        if (!open) return;
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {24 * scale, 16 * scale});
        const float width = ImGui::GetContentRegionAvail().x;
        const ImVec2 field_size{(width - 24 * scale) / 2, row_height};
        constexpr int steps_step = 1;
        if (number_field("##Steps", "Steps", ImGuiDataType_S32, &draft.steps, field_size, &steps_step, "%d", scale, parameter_edit)) edited_since_submit = true;
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Sampling steps for the next image\nUp / Down: 1\nEnter to confirm, Esc to undo");
        ImGui::SameLine(0, 24 * scale);
        constexpr float cfg_step = 0.1F;
        if (number_field("##CFG", "CFG", ImGuiDataType_Float, &draft.cfg, field_size, &cfg_step, "%.1f", scale, parameter_edit)) edited_since_submit = true;
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Guidance scale for the next image\nUp / Down: 0.1\nEnter to confirm, Esc to undo");
        const auto origin = ImGui::GetCursorScreenPos();
        auto* draw = ImGui::GetWindowDrawList();
        draw->AddLine({origin.x, origin.y - 8 * scale}, {origin.x + width, origin.y - 8 * scale}, ImGui::GetColorU32(ImVec4{1, 1, 1, 0.06F}), scale);
        draw->AddText({origin.x + 4 * scale, origin.y + (row_height - ImGui::GetFontSize()) / 2}, ImGui::GetColorU32(ImGuiCol_TextDisabled), "Seed");
        if (!random_seed) {
            ImGui::SetCursorScreenPos({origin.x + seed_label_width, origin.y});
            if (number_field("##Seed", "", ImGuiDataType_U64, &seed, {width - seed_label_width - mode_width - 8 * scale, row_height}, nullptr, "%llu", scale, parameter_edit)) edited_since_submit = true;
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Seed for the next image\nEnter to confirm, Esc to undo");
        }
        ImGui::SetCursorScreenPos({origin.x + width - mode_width, origin.y});
        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 6 * scale);
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, {8 * scale, (row_height - ImGui::GetFontSize()) / 2});
        ImGui::PushStyleColor(ImGuiCol_Button, {1, 1, 1, random_seed ? 0.0F : 0.025F});
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, {1, 1, 1, 0.06F});
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, {0.55F, 0.48F, 0.92F, 0.14F});
        ImGui::PushStyleColor(ImGuiCol_Text, random_seed ? ImVec4{0.70F, 0.71F, 0.77F, 1} : ImVec4{0.74F, 0.70F, 0.94F, 1});
        const bool mode_clicked = ImGui::Button(random_seed ? "Random###SeedMode" : "Fixed###SeedMode", {mode_width, row_height});
        ImGui::PopStyleColor(4);
        ImGui::PopStyleVar(2);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s\nClick: Random / Fixed", random_seed ? "A new seed is chosen when queued." : "Reuse this seed for each image.");
        const bool row_hovered = ImGui::IsWindowHovered(ImGuiHoveredFlags_AllowWhenBlockedByActiveItem) && ImGui::IsMouseHoveringRect(origin, {origin.x + width, origin.y + row_height});
        if (mode_clicked || (row_hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Middle))) {
            commit_parameters();
            random_seed = !random_seed;
            edited_since_submit = true;
        }
        if (ImGui::IsKeyPressed(ImGuiKey_Escape, false)) ImGui::CloseCurrentPopup();
        ImGui::PopStyleVar();
        ImGui::EndPopup();
    }

    void UserInterface::top_strip(const float scale, const ImVec2 size) {
        bool active, paused, loaded, failed, unavailable;
        std::size_t queued;
        std::uint64_t active_id{};
        int steps{};
        double elapsed{};
        {
            const std::lock_guard lock{session.mutex};
            active = session.active.has_value();
            paused = session.paused;
            loaded = session.model_ready;
            failed = !session.error.empty();
            unavailable = session.worker_done;
            queued = session.queue.size();
            if (active) {
                active_id = session.active->id;
                steps     = session.active->parameters.steps;
                elapsed   = std::chrono::duration<double>(std::chrono::steady_clock::now() - session.started).count();
            }
        }
        const auto stage         = static_cast<sdxl::Stage>(::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>{session.control.data()[0].stage}.load());
        const auto completed     = ::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>{session.control.data()[0].completed}.load();
        const bool stopping      = active && ::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>{session.control.data()[0].cancel}.load();
        const bool working       = (active || !loaded) && !failed;
        const bool indeterminate = working && (!active || stage != sdxl::Stage::sampling || stopping);
        const double now         = glfwGetTime();
        if (working) {
            progress_alpha = 1;
            if (stopping) progress_label = "Stopping";
            else if (!loaded) progress_label = "Loading model";
            else if (stage == sdxl::Stage::sampling) progress_label = std::format("{} / {}", completed, steps);
            else if (stage == sdxl::Stage::decoding || stage == sdxl::Stage::transferring || stage == sdxl::Stage::complete) progress_label = "Finishing image";
            else progress_label = "Preparing";
            progress_time = active ? std::format("{:.1f}s", elapsed) : "";
        } else progress_alpha = paused || failed ? 0 : std::max(0.0F, progress_alpha - ImGui::GetIO().DeltaTime / 0.15F);
        if (working) refresh_at = now + (indeterminate ? 1.0 / 30 : 0.1);
        else if (progress_alpha > 0) refresh_at = now + 1.0 / 60;

        ImGui::SetNextWindowPos({0, 0});
        ImGui::SetNextWindowSize({size.x, top_strip_height * scale});
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {0, 0});
        ImGui::Begin("##ApplicationStrip", nullptr, overlay | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoNavFocus);
        ImGui::PopStyleVar();
        const auto minimum = ImGui::GetWindowPos();
        const ImVec2 maximum{minimum.x + ImGui::GetWindowWidth(), minimum.y + ImGui::GetWindowHeight()};
        auto* draw = ImGui::GetWindowDrawList();
        if (progress_alpha > 0) {
            draw->PushClipRect(minimum, maximum, false);
            if (indeterminate) {
                const float width         = maximum.x - minimum.x;
                const float segment_width = std::min(width * 0.18F, 180 * scale);
                const float x             = minimum.x - segment_width + float(std::fmod(now * 180 * scale, double(width + segment_width)));
                draw->AddRectFilled({x, minimum.y}, {x + segment_width, minimum.y + 2 * scale}, ImGui::GetColorU32(ImVec4{0.59F, 0.57F, 0.95F, 0.60F}));
            } else {
                const float fraction = working ? std::min(1.0F, float(completed) / steps) : 1.0F;
                const float x        = std::lerp(minimum.x, maximum.x, fraction);
                draw->AddRectFilled(minimum, {x, minimum.y + 2 * scale}, ImGui::GetColorU32(ImVec4{0.59F, 0.57F, 0.95F, 0.65F * progress_alpha}));
            }
            draw->PopClipRect();
        }
        if (image_texture) control_shade({12 * scale, 4 * scale}, {132 * scale, 44 * scale}, scale);
        ImGui::SetCursorPos({12 * scale, 4 * scale});
        if (text_button("##History", "", scale, 36 * scale, history_open)) history_open = !history_open;
        const auto history_origin = ImGui::GetItemRectMin();
        const ImVec2 center{history_origin.x + 18 * scale, history_origin.y + 20 * scale};
        const auto ink = ImGui::GetColorU32(history_open || ImGui::IsItemHovered() ? ImVec4{0.80F, 0.78F, 0.94F, 1} : ImVec4{0.57F, 0.58F, 0.64F, 1});
        draw->AddRect({center.x - 6 * scale, center.y - 6 * scale}, {center.x + 6 * scale, center.y + 6 * scale}, ink, scale, 0, scale);
        draw->AddLine({center.x - 2 * scale, center.y - 6 * scale}, {center.x - 2 * scale, center.y + 6 * scale}, ink, scale);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Session history");
        ImGui::SameLine(0, 0);
        if (text_button("##Application", "GENESIA", scale)) ImGui::OpenPopup("Application");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Application menu\nF11 to toggle fullscreen\nEsc to exit");
        const float left_end = ImGui::GetItemRectMax().x + 4 * scale;
        if (ImGui::BeginPopup("Application")) {
            ImGui::MenuItem("Live preview", nullptr, &configuration.preview.enabled);
            ImGui::Separator();
            if (ImGui::MenuItem("Save configuration", "Ctrl+S")) save_settings();
            if (ImGui::MenuItem("Open output folder")) ShellExecuteW(window.native_window, L"open", std::filesystem::absolute(configuration.output).c_str(), nullptr, nullptr, SW_SHOWNORMAL);
            ImGui::EndPopup();
        }
        const bool queue_visible      = queued != 0 || paused || ImGui::IsPopupOpen("Queue");
        const std::string queue_label = queue_visible ? std::format("Queue  {}", queued) : "";
        const char* stop_label        = active ? "Stop" : paused && !failed ? "Resume" : "";
        float controls_width = 150 * scale;
        for (const char* label : {queue_label.c_str(), stop_label})
            if (*label) controls_width += ImGui::CalcTextSize(label).x + 28 * scale;
        const bool queue_paused = paused && !active && !failed;
        const char* label = queue_paused ? "Queue paused" : progress_alpha > 0 ? progress_label.c_str() : "";
        const float alpha = queue_paused ? 1 : progress_alpha;
        const float right_start = maximum.x - controls_width;
        const float label_width = *label ? std::max(ImGui::CalcTextSize("Finishing image").x, ImGui::CalcTextSize(label).x) : 0;
        const auto time_text = ImGui::CalcTextSize(progress_time.c_str());
        const float time_width = std::max(ImGui::CalcTextSize("0000.0s").x, time_text.x);
        const bool show_time = *label && !queue_paused && !progress_time.empty() && label_width + 16 * scale + time_width <= right_start - left_end - 32 * scale;
        const float status_width = label_width + (show_time ? 16 * scale + time_width : 0);
        const float group_left = right_start - status_width - (*label ? 12 * scale : 0) - 8 * scale;
        if (image_texture) control_shade({group_left, minimum.y + 4 * scale}, maximum, scale);
        if (*label) {
            ImGui::SetCursorScreenPos({group_left + 8 * scale, minimum.y + 4 * scale});
            ImGui::Dummy({status_width, control_height * scale});
            const auto origin = ImGui::GetItemRectMin();
            const auto text = ImGui::CalcTextSize(label);
            const float y = origin.y + (control_height * scale - text.y) / 2;
            draw->AddText({origin.x, y}, ImGui::GetColorU32(ImVec4{0.89F, 0.89F, 0.94F, alpha}), label);
            if (show_time) {
                draw->AddCircleFilled({origin.x + label_width + 8 * scale, origin.y + control_height * scale / 2}, 1.25F * scale, ImGui::GetColorU32(ImVec4{0.57F, 0.58F, 0.64F, alpha}));
                draw->AddText({origin.x + status_width - time_text.x, y}, ImGui::GetColorU32(ImVec4{0.67F, 0.68F, 0.74F, alpha}), progress_time.c_str());
            }
            const bool preview_visible = image_live && selected == active_id;
            const bool hidden_time = !show_time && !queue_paused && !progress_time.empty();
            if (ImGui::IsItemHovered() && (hidden_time || preview_visible)) {
                ImGui::BeginTooltip();
                if (hidden_time) ImGui::Text("Elapsed: %s", progress_time.c_str());
                if (preview_visible) ImGui::Text("Displayed preview: step %u", preview_step);
                ImGui::EndTooltip();
            }
        }
        float x = right_start;
        if (queue_visible) {
            ImGui::SetCursorScreenPos({x, minimum.y + 4 * scale});
            if (text_button("##Queue", queue_label.c_str(), scale)) ImGui::OpenPopup("Queue");
            x = ImGui::GetItemRectMax().x + 4 * scale;
        }
        if (ImGui::BeginPopup("Queue")) {
            ImGui::TextDisabled("UP NEXT");
            {
                const std::lock_guard lock{session.mutex};
                if (session.queue.empty()) ImGui::TextUnformatted("No pending images");
                for (std::size_t i = 0; i < session.queue.size();) {
                    const auto& request = session.queue[i];
                    ImGui::PushID(static_cast<int>(request.id));
                    ImGui::Text("#%llu   %d x %d", static_cast<unsigned long long>(request.id + 1), request.parameters.width, request.parameters.height);
                    ImGui::SameLine();
                    const bool remove = ImGui::SmallButton("Remove");
                    ImGui::PopID();
                    if (remove) session.queue.erase(session.queue.begin() + i);
                    else ++i;
                }
            }
            ImGui::EndPopup();
        }
        if (*stop_label) {
            ImGui::SetCursorScreenPos({x, minimum.y + 4 * scale});
            if (text_button("##Playback", stop_label, scale)) {
                if (active) session.stop();
                else session.resume();
            }
            x = ImGui::GetItemRectMax().x + 4 * scale;
        }
        // The owner button remains a single-click action while its settings are open.
        if (ImGui::IsPopupOpen("Generation settings") && ImGui::IsWindowHovered(ImGuiHoveredFlags_AllowWhenBlockedByPopup | ImGuiHoveredFlags_AllowWhenBlockedByActiveItem)
            && ImGui::IsMouseHoveringRect({x, minimum.y}, maximum) && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            commit_parameters();
            ImGui::ClosePopupsOverWindow(ImGui::GetCurrentWindow(), false);
        }
        ImGui::SetCursorScreenPos({x, minimum.y});
        ImGui::PushFont(nullptr, 18);
        ImGui::BeginDisabled(unavailable || !prompt_editor.valid);
        if (text_button("##Generate", "Generate", scale, 150 * scale, false, true)) submit();
        ImGui::EndDisabled();
        const bool generate_hovered = ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled | ImGuiHoveredFlags_AllowWhenBlockedByActiveItem);
        const bool settings_requested = (generate_hovered && ImGui::IsMouseReleased(ImGuiMouseButton_Right)) || (ImGui::IsItemFocused() && ImGui::Shortcut(ImGuiMod_Shift | ImGuiKey_F10));
        if (settings_requested) {
            commit_parameters();
            ImGui::OpenPopup("Generation settings");
        }
        ImGui::PopFont();
        if (generate_hovered && !ImGui::IsPopupOpen("Generation settings")) {
            if (!prompt_editor.valid) ImGui::SetTooltip("Finish the tag input before generating.\nRight-click: generation settings");
            else ImGui::SetTooltip("%s\nCtrl+Shift+\x60\nRight-click: generation settings", active ? "Add image to queue" : "Generate image");
        }
        generation_settings(scale, size);
        window.drag_region = {};
        if (!ImGui::IsPopupOpen(nullptr, ImGuiPopupFlags_AnyPopupId | ImGuiPopupFlags_AnyPopupLevel)) window.drag_region = {left_end, 0, group_left - 4 * scale, maximum.y};
        ImGui::End();
    }

    void UserInterface::composer(const float scale, const ImVec2 size, const float height, const float controls_height) {
        if (!composer_open) prompt_editor.suspend();
        if (composer_amount < 0.03F) return;
        const float width = std::min(840 * scale, size.x - (48 + 300 * history_amount) * scale);
        const float center = (size.x + history_amount * 300 * scale) / 2;
        const float y = size.y - (bottom_margin + panel_gap) * scale - controls_height - height + (1 - composer_amount) * 12 * scale;
        ImGui::SetNextWindowPos({center - width / 2, y});
        ImGui::SetNextWindowSize({width, height});
        ImGui::PushStyleVar(ImGuiStyleVar_Alpha, composer_amount);
        ImGui::Begin("##Composer", nullptr, overlay | (composer_open ? ImGuiWindowFlags_None : ImGuiWindowFlags_NoInputs));
        if (prompt_editor.draw(prompt, tag_search, scale)) edited_since_submit = true;
        ImGui::End();
        ImGui::PopStyleVar();
    }

    void UserInterface::bottom_controls(const float scale, const ImVec2 size, const ControlLayout& layout) {
        const float left = (bottom_margin + history_amount * 300) * scale;
        const float y = size.y - (bottom_margin + control_height) * scale;
        const float left_y = y - (layout.stacked ? control_height + 8 : 0) * scale;
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {0, 0});
        ImGui::SetNextWindowPos({left, left_y});
        ImGui::SetNextWindowSize({layout.left_width, control_height * scale});
        ImGui::Begin("##PromptControls", nullptr, overlay | ImGuiWindowFlags_NoBackground);
        ImGui::PopStyleVar();
        if (image_texture) control_shade({left, left_y}, {left + layout.left_width, left_y + control_height * scale}, scale);
        if (text_button("##Prompt", "Prompt", scale, 0, composer_open)) composer_open = !composer_open;
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Edit prompt\nTab");
        ImGui::End();

        const float right = size.x - bottom_margin * scale;
        const float x = right - layout.right_width;
        ImGui::SetNextWindowPos({x, y});
        ImGui::SetNextWindowSize({layout.right_width, control_height * scale});
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {0, 0});
        ImGui::Begin("##ImageControls", nullptr, overlay | ImGuiWindowFlags_NoBackground);
        ImGui::PopStyleVar();
        if (image_texture) control_shade({x, y}, {right, y + control_height * scale}, scale);
        if (layout.latest_width) {
            bool active;
            {
                const std::lock_guard lock{session.mutex};
                active = session.active.has_value();
            }
            if (text_button("##Latest", active ? "View live" : "Latest", scale, layout.latest_width - 12 * scale)) {
                following_latest = true;
                preview_step = 0;
                if (!active && !history.empty()) {
                    selected = history.back().id;
                    image_live = false;
                    session.load(selected, history.back().record.directory / "image.png");
                }
            }
        }
        float dimensions_x = x + layout.latest_width;
        auto* draw = ImGui::GetWindowDrawList();
        if (layout.different) {
            const float info_x = layout.image_above ? right - layout.image_label_width : dimensions_x;
            const float info_y = layout.image_above ? y - (control_height + 8) * scale : y;
            if (layout.image_above) control_shade({info_x, info_y}, {right, info_y + control_height * scale}, scale);
            draw->PushClipRect({0, 0}, size, false);
            draw->AddText({info_x, info_y + (control_height * scale - ImGui::GetFontSize()) / 2}, ImGui::GetColorU32(ImVec4{0.54F, 0.55F, 0.61F, 1}), std::format("{} {} \xC3\x97 {} \xE2\x86\x92", image_live ? "Preview" : "Image", image_width, image_height).c_str());
            draw->PopClipRect();
            if (!layout.image_above) dimensions_x += layout.image_label_width;
        }
        const ImVec2 dimensions_origin{dimensions_x, y};
        if (layout.different) {
            draw->AddText({dimensions_x + 4 * scale, y + (control_height * scale - ImGui::GetFontSize()) / 2}, ImGui::GetColorU32(ImGuiCol_TextDisabled), "Next");
            dimensions_x += ImGui::CalcTextSize("Next").x + 12 * scale;
        }
        ImGui::SetCursorScreenPos({dimensions_x, y});
        constexpr int dimension_step = 64;
        if (number_field("##Width", "", ImGuiDataType_S32, &draft.width, {48 * scale, control_height * scale}, &dimension_step, "%d", scale, parameter_edit)) edited_since_submit = true;
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Next image width\nUp / Down: 64 px\nMiddle-click: swap dimensions\nRight-click: presets");
        draw->AddText({dimensions_x + 51 * scale, y + (control_height * scale - ImGui::GetFontSize()) / 2}, ImGui::GetColorU32(ImGuiCol_TextDisabled), "\xC3\x97");
        ImGui::SetCursorScreenPos({dimensions_x + 64 * scale, y});
        if (number_field("##Height", "", ImGuiDataType_S32, &draft.height, {48 * scale, control_height * scale}, &dimension_step, "%d", scale, parameter_edit)) edited_since_submit = true;
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Next image height\nUp / Down: 64 px\nMiddle-click: swap dimensions\nRight-click: presets");
        const bool dimensions_hovered = ImGui::IsWindowHovered(ImGuiHoveredFlags_AllowWhenBlockedByActiveItem) && ImGui::IsMouseHoveringRect(dimensions_origin, ImGui::GetItemRectMax());
        if (dimensions_hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Middle)) {
            commit_parameters();
            std::swap(draft.width, draft.height);
            edited_since_submit = true;
        }
        if (dimensions_hovered && ImGui::IsMouseReleased(ImGuiMouseButton_Right)) {
            commit_parameters();
            ImGui::OpenPopup("Resolution presets");
        }
        if (ImGui::BeginPopup("Resolution presets")) {
            for (const auto& [label, width, height] : std::array{std::tuple{"Portrait  1024 x 1536", 1024, 1536}, std::tuple{"Square  1024 x 1024", 1024, 1024}, std::tuple{"Landscape  1536 x 1024", 1536, 1024}}) {
                if (ImGui::MenuItem(label, nullptr, draft.width == width && draft.height == height)) {
                    draft.width = width;
                    draft.height = height;
                    edited_since_submit = true;
                }
            }
            if (image_texture) {
                ImGui::Separator();
                if (ImGui::MenuItem("Use image size")) {
                    draft.width = image_width;
                    draft.height = image_height;
                    edited_since_submit = true;
                }
            }
            ImGui::EndPopup();
        }
        if (image_texture) {
            ImGui::SetCursorScreenPos({right - 104 * scale, y});
            if (text_button("##View", std::format("{}{:.0f}%", fit_image ? "Fit \xC2\xB7 " : "", zoom * 100).c_str(), scale, 104 * scale)) {
                fit_image = false;
                zoom = 1;
                pan = {};
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Left-click: view at 100%%\nMiddle-click: fit image");
                if (ImGui::IsMouseClicked(ImGuiMouseButton_Middle)) {
                    fit_image = true;
                    pan = {};
                }
            }
        }
        ImGui::End();
    }

    void UserInterface::history_drawer(const float scale, const ImVec2 size) {
        if (history_amount < 0.03F) return;
        ImGui::SetNextWindowPos({16 * scale - (1 - history_amount) * 300 * scale, 52 * scale});
        ImGui::SetNextWindowSize({280 * scale, size.y - 76 * scale});
        ImGui::PushStyleVar(ImGuiStyleVar_Alpha, history_amount);
        ImGui::Begin("##HistoryDrawer", nullptr, overlay);
        ImGui::TextDisabled("SESSION HISTORY");
        ImGui::Spacing();
        if (history.empty()) {
            ImGui::TextWrapped("Your images will appear here.");
            ImGui::TextDisabled("Only this session is shown.");
        }
        ImGui::BeginChild("##Thumbnails", {0, 0});
        for (std::size_t reverse = 0; reverse < history.size(); ++reverse) {
            auto& item = history[history.size() - 1 - reverse];
            ImGui::PushID(static_cast<int>(item.id));
            const float card_width = 112 * scale;
            const ImVec2 p         = ImGui::GetCursorScreenPos();
            const ImVec2 card{card_width, 164 * scale};
            if (ImGui::InvisibleButton("Image", card) && item.saved) {
                following_latest = false;
                selected         = item.id;
                image_live       = false;
                session.load(item.id, item.record.directory / "image.png");
            }
            const bool hovered = ImGui::IsItemHovered();
            const bool chosen  = item.id == selected;
            auto* draw         = ImGui::GetWindowDrawList();
            if (hovered) draw->AddRectFilled(p, {p.x + card.x, p.y + card.y}, IM_COL32(30, 31, 38, 255), 6 * scale);
            const float fit = std::min((card_width - 12 * scale) / item.record.parameters.width, 134 * scale / item.record.parameters.height);
            const float w   = fit * item.record.parameters.width;
            const float h   = fit * item.record.parameters.height;
            const ImVec2 minimum{p.x + (card_width - w) / 2, p.y + (142 * scale - h) / 2};
            draw->AddImage(item.texture, minimum, {minimum.x + w, minimum.y + h});
            if (chosen) draw->AddLine({p.x + 8 * scale, p.y + card.y - scale}, {p.x + card.x - 8 * scale, p.y + card.y - scale}, IM_COL32(151, 145, 241, 210), 2 * scale);
            ImGui::PushFont(nullptr, 12);
            draw->AddText(ImGui::GetFont(), ImGui::GetFontSize(), {p.x + 8 * scale, p.y + 144 * scale}, IM_COL32(173, 175, 190, 255), std::format("#{:02}   {:.1f}s", item.id + 1, item.record.sample_seconds + item.record.decode_seconds).c_str());
            ImGui::PopFont();
            if (hovered) ImGui::SetTooltip("%d x %d\nSeed: %llu%s", item.record.parameters.width, item.record.parameters.height, static_cast<unsigned long long>(item.record.seed), item.saved ? "" : "\nSaving...");
            if (ImGui::BeginPopupContextItem()) {
                if (ImGui::MenuItem("Reuse settings")) {
                    commit_parameters();
                    draft               = item.record.parameters;
                    prompt              = item.record.prompt;
                    prompt_editor.reset(prompt);
                    seed                = item.record.seed;
                    random_seed         = false;
                    composer_open       = true;
                    edited_since_submit = true;
                }
                if (ImGui::MenuItem("Open image folder", nullptr, false, item.saved)) ShellExecuteW(window.native_window, L"open", std::filesystem::absolute(item.record.directory).c_str(), nullptr, nullptr, SW_SHOWNORMAL);
                ImGui::EndPopup();
            }
            ImGui::PopID();
            if (reverse % 2 == 0 && reverse + 1 < history.size()) ImGui::SameLine(0, 12 * scale);
        }
        ImGui::EndChild();
        ImGui::End();
        ImGui::PopStyleVar();
    }

    void UserInterface::draw() {
        // A dismissed popup no longer submits its fields; commit before another input takes focus.
        if (parameter_edit.id && ImGui::GetActiveID() != parameter_edit.id) commit_parameters();
        refresh_at                = std::numeric_limits<double>::infinity();
        const float scale         = renderer.dpi;
        const auto size           = ImGui::GetIO().DisplaySize;
        const float interpolation = std::min(1.0F, ImGui::GetIO().DeltaTime / 0.045F);
        composer_amount           = std::lerp(composer_amount, composer_open ? 1.0F : 0.0F, interpolation);
        history_amount            = std::lerp(history_amount, history_open ? 1.0F : 0.0F, interpolation);
        if (std::abs(composer_amount - float(composer_open)) < 0.01F) composer_amount = float(composer_open);
        if (std::abs(history_amount - float(history_open)) < 0.01F) history_amount = float(history_open);
        std::string error;
        {
            const std::lock_guard lock{session.mutex};
            session.preview_enabled = configuration.preview.enabled;
            session.preview_visible = renderer.visible && following_latest;
            error                   = session.error;
        }
        const auto controls = control_layout(scale, size);
        const float tag_width = std::min(840 * scale, size.x - (48 + 300 * history_amount) * scale) - 40 * scale;
        const float composer_height = std::min({prompt_editor.measure(prompt, *configuration.catalog, tag_width, scale),
            size.y * 0.4F, size.y - controls.height - (bottom_margin + panel_gap + 56) * scale});
        const bool dismissing = escape_owned || parameter_edit.id || (composer_open && prompt_editor.escape_owned) || ImGui::IsPopupOpen(nullptr, ImGuiPopupFlags_AnyPopupId | ImGuiPopupFlags_AnyPopupLevel);
        if (renderer.visible) {
            canvas(scale, size, composer_height, controls.height);
            composer(scale, size, composer_height, controls.height);
            bottom_controls(scale, size, controls);
            history_drawer(scale, size);
            top_strip(scale, size);
            if (!error.empty() && error != shown_error) {
                shown_error = error;
                ImGui::OpenPopup("Generation failed");
            }
            ImGui::SetNextWindowSize({540 * scale, 0});
            if (ImGui::BeginPopupModal("Generation failed", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
                ImGui::TextWrapped("%s", shown_error.c_str());
                ImGui::Spacing();
                if (ImGui::Button("Close")) ImGui::CloseCurrentPopup();
                ImGui::EndPopup();
            }
            if (ImGui::Shortcut(ImGuiMod_Ctrl | ImGuiMod_Shift | ImGuiKey_GraveAccent, ImGuiInputFlags_RouteGlobal | ImGuiInputFlags_RouteOverActive)) submit();
            if (ImGui::Shortcut(ImGuiMod_Ctrl | ImGuiKey_S, ImGuiInputFlags_RouteGlobal)) save_settings();
            if (ImGui::Shortcut(ImGuiKey_F11, ImGuiInputFlags_RouteGlobal | ImGuiInputFlags_RouteOverActive)) window.toggle_fullscreen();
            if (!dismissing && !ImGui::IsPopupOpen(nullptr, ImGuiPopupFlags_AnyPopupId | ImGuiPopupFlags_AnyPopupLevel) && ImGui::Shortcut(ImGuiKey_Escape, ImGuiInputFlags_RouteGlobal | ImGuiInputFlags_RouteOverActive)) window.request_close();
            escape_owned = parameter_edit.id || (composer_open && prompt_editor.escape_owned) || ImGui::IsPopupOpen(nullptr, ImGuiPopupFlags_AnyPopupId | ImGuiPopupFlags_AnyPopupLevel);
            if (!ImGui::GetIO().WantTextInput && !prompt_editor.focus_input && ImGui::Shortcut(ImGuiKey_Tab, ImGuiInputFlags_RouteGlobal)) composer_open = !composer_open;
            const auto& io = ImGui::GetIO();
            if (io.MouseDelta.x || io.MouseDelta.y || io.MouseWheel || io.InputQueueCharacters.Size || ImGui::IsAnyItemActive()) animate_until = glfwGetTime() + 0.16;
        }
    }
} // namespace genesia::editor
