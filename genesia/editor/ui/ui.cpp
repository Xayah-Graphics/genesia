module;
#include <Windows.h>
#include <GLFW/glfw3.h>
#include "../../core/sdxl/control.h"
#include <genesia/cuda.h>
#include <imgui.h>
#include <imgui_internal.h>
#include <shellapi.h>
module genesia.editor.ui;
import genesia.generation.defaults;
import genesia.prompt.preset;
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
        constexpr float top_strip_height   = 48;
        constexpr float control_height     = 40;
        constexpr float bottom_margin      = 24;

        void control_shade(const ImVec2 minimum, const ImVec2 maximum, const float scale) {
            auto* draw          = ImGui::GetWindowDrawList();
            const float x       = (minimum.x + maximum.x) / 2;
            const float y       = (minimum.y + maximum.y) / 2;
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

        void control_text(const std::string_view text, const ImVec2 position, const float right, const ImVec4 ink, const float scale) {
            auto* draw = ImGui::GetWindowDrawList();
            ImGui::PushStyleColor(ImGuiCol_Text, {0, 0, 0, 0.7F});
            ImGui::RenderTextEllipsis(draw, {position.x, position.y + scale}, {right, position.y + ImGui::GetFontSize() + scale}, right, text.data(), text.data() + text.size(), nullptr);
            ImGui::PopStyleColor();
            ImGui::PushStyleColor(ImGuiCol_Text, ink);
            ImGui::RenderTextEllipsis(draw, position, {right, position.y + ImGui::GetFontSize()}, right, text.data(), text.data() + text.size(), nullptr);
            ImGui::PopStyleColor();
        }

        bool text_button(const char* id, const char* label, const float scale, const float width = 0, const bool selected = false, const ImVec4 accent = {}) {
            const bool primary = accent.w > 0;
            const auto text    = ImGui::CalcTextSize(label);
            const auto origin  = ImGui::GetCursorScreenPos();
            const ImVec2 size{width ? width : text.x + 24 * scale, (primary ? top_strip_height : control_height) * scale};
            ImGui::PushStyleColor(ImGuiCol_NavCursor, {0, 0, 0, 0});
            const bool clicked = ImGui::InvisibleButton(id, size, ImGuiButtonFlags_EnableNav);
            ImGui::PopStyleColor();
            const bool disabled = ImGui::GetItemFlags() & ImGuiItemFlags_Disabled;
            const bool focused  = !disabled && ImGui::IsItemFocused() && ImGui::GetCurrentContext()->NavCursorVisible;
            const bool hovered  = ImGui::IsItemHovered() || focused;
            auto* storage       = ImGui::GetStateStorage();
            const auto key      = ImGui::GetItemID();
            const float alpha   = disabled ? 0 : std::lerp(storage->GetFloat(key), hovered || selected ? 1.0F : 0.0F, std::min(1.0F, ImGui::GetIO().DeltaTime / 0.12F));
            storage->SetFloat(key, alpha);
            auto* draw = ImGui::GetWindowDrawList();
            if (primary && alpha > 0) {
                const auto clear = ImGui::GetColorU32(ImVec4{accent.x, accent.y, accent.z, 0});
                const auto tint  = ImGui::GetColorU32(ImVec4{accent.x, accent.y, accent.z, alpha * 0.16F});
                draw->AddRectFilledMultiColor(origin, {origin.x + size.x, origin.y + size.y}, clear, tint, tint, clear);
            }
            const float visible = std::min(text.x, std::max(0.0F, size.x - 24 * scale));
            const ImVec2 position{origin.x + (size.x - visible) / 2, origin.y + (size.y - text.y) / 2};
            const ImVec4 ink = disabled ? ImVec4{0.62F, 0.62F, 0.62F, 1} : primary ? ImLerp(accent, ImVec4{1, 1, 1, 1}, alpha * 0.35F) : ImVec4{0.57F + 0.31F * alpha, 0.58F + 0.30F * alpha, 0.64F + 0.29F * alpha, 1};
            control_text(label, position, position.x + visible, ink, scale);
            if (focused) draw->AddLine({position.x, position.y + text.y + 3 * scale}, {position.x + visible, position.y + text.y + 3 * scale}, ImGui::GetColorU32(ink), scale);
            return clicked;
        }

        void number_field(const char* id, const char* label, const ImGuiDataType type, void* value, const ImVec2 size, const void* step, const char* format, const float scale, UserInterface::ParameterEdit& edit) {
            const auto origin = ImGui::GetCursorScreenPos();
            const ImVec2 end{origin.x + size.x, origin.y + size.y};
            const auto key     = ImGui::GetID(id);
            const bool active  = ImGui::GetActiveID() == key;
            const bool hovered = ImGui::IsWindowHovered(ImGuiHoveredFlags_AllowWhenBlockedByActiveItem) && ImGui::IsMouseHoveringRect(origin, end);
            auto* storage      = ImGui::GetStateStorage();
            const float alpha  = std::lerp(storage->GetFloat(key), active || hovered ? 1.0F : 0.0F, std::min(1.0F, ImGui::GetIO().DeltaTime / 0.12F));
            storage->SetFloat(key, alpha);
            auto* draw        = ImGui::GetWindowDrawList();
            const auto text   = ImGui::CalcTextSize(label);
            const float inset = *label ? size.x - 56 * scale : 0;
            if (*label) draw->AddText({origin.x + 4 * scale, origin.y + (size.y - text.y) / 2}, ImGui::GetColorU32(ImVec4{0.57F + 0.22F * alpha, 0.58F + 0.22F * alpha, 0.64F + 0.22F * alpha, 1}), label);
            const bool clicked = hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left);
            if (clicked && ImGui::GetIO().MousePos.x < origin.x + inset) ImGui::SetKeyboardFocusHere();
            bool stepped{};
            if (active && step) {
                ImGui::SetKeyOwner(ImGuiKey_UpArrow, key);
                ImGui::SetKeyOwner(ImGuiKey_DownArrow, key);
                const bool up   = ImGui::IsKeyPressed(ImGuiKey_UpArrow, ImGuiInputFlags_Repeat, key);
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
            ImGui::InputScalar(id, type, value, nullptr, nullptr, format);
            ImGui::PopStyleVar();
            ImGui::PopStyleColor(2);
            if (ImGui::IsItemActive()) edit = {key, type, value};
            else if (edit.id == key) edit = {};
            if (ImGui::IsItemActive()) draw->AddLine({origin.x + inset + 4 * scale, end.y - 6 * scale}, {end.x - 4 * scale, end.y - 6 * scale}, ImGui::GetColorU32(ImVec4{0.65F, 0.60F, 0.88F, 0.8F}), scale);
            if (stepped) ImGui::MarkItemEdited(key);
        }

    } // namespace

    void UserInterface::ImageView::scale_to(const float ratio, const ImVec2 position, const ImVec2 image, const bool fitting, const double now) {
        fit          = fitting;
        dragging     = false;
        initial_zoom = zoom;
        target_zoom  = ratio;
        anchor       = {center.x + position.x / (image.x * zoom), center.y + position.y / (image.y * zoom)};
        pivot        = position;
        started      = now;
    }

    void UserInterface::ImageView::update(const ImVec2 available, const ImVec2 image, const double now) {
        const float fitted = std::min(available.x / image.x, available.y / image.y);
        if (fit || target_zoom <= fitted) {
            fit         = true;
            target_zoom = fitted;
        }
        if (started >= 0) {
            const float progress = std::clamp(float((now - started) / 0.12), 0.0F, 1.0F);
            const float eased    = 1 - (1 - progress) * (1 - progress) * (1 - progress);
            zoom                 = std::max(fitted, std::exp(std::lerp(std::log(initial_zoom), std::log(target_zoom), eased)));
            center               = {anchor.x - pivot.x / (image.x * zoom), anchor.y - pivot.y / (image.y * zoom)};
            if (progress == 1) {
                zoom    = target_zoom;
                started = -1;
            }
        } else zoom = target_zoom;
        constrain(available, image);
    }

    void UserInterface::ImageView::constrain(const ImVec2 available, const ImVec2 image) {
        const float horizontal = std::min(0.5F, available.x / (2 * image.x * zoom));
        const float vertical   = std::min(0.5F, available.y / (2 * image.y * zoom));
        center.x               = std::clamp(center.x, horizontal, 1 - horizontal);
        center.y               = std::clamp(center.y, vertical, 1 - vertical);
    }

    UserInterface::RepaintDraft::RepaintDraft(const Record& source) : catalog{source.catalog}, prompt{source.prompt} {
        editor.tracking = true;
        editor.reset(prompt);
    }

    UserInterface::UserInterface(prompt::Preset preset, std::shared_ptr<const prompt::Catalog> catalog, WindowPlatform& platform, Renderer& display, std::string dataset) : catalog{std::move(catalog)}, preset{std::move(preset)}, window{platform}, renderer{display}, library{this->catalog, display}, prompt{this->preset.prompt}, tag_search{*this->catalog} {
        prompt_editor.reset(prompt);
        for (const auto& model : classifiers) classification.enabled.push_back(model.id);
        if (!dataset.empty()) {
            page           = Page::dataset;
            collection_key = std::move(dataset);
        }
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

    UserInterface::~UserInterface() {
        if (generation.texture) renderer.retire(generation.texture);
        if (repaint && repaint->result.texture) renderer.retire(repaint->result.texture);
    }

    void UserInterface::receive() {
        const auto revision = library.revision;
        library.receive();
        if (revision != library.revision) {
            for (auto& entry : library.roots) {
                std::vector<dataset::Collection*> collections{&entry.all};
                for (auto& candidate : entry.concepts) collections.push_back(&candidate);
                for (const auto* candidate : collections) {
                    const auto remembered = positions.find(candidate->key);
                    if (remembered == positions.end() || candidate->images.empty()) continue;
                    auto& position      = remembered->second;
                    const auto selected = std::ranges::find(candidate->images, position.selected, &dataset::File::sha);
                    const auto index    = selected == candidate->images.end() ? std::min(position.index, candidate->images.size() - 1) : static_cast<std::size_t>(selected - candidate->images.begin());
                    position.scroll += static_cast<float>(index) - position.index;
                    position.index    = index;
                    position.selected = candidate->images[index].sha;
                }
            }
        }
        synchronize_collection();
        if (runtime) {
            auto& session = runtime->session;
            std::deque<Event> events;
            std::deque<PreviewFrame> previews;
            {
                const std::lock_guard lock{session.mutex};
                events.swap(session.events);
                previews.swap(session.previews);
            }
            for (const auto& frame : previews) {
                auto& source           = runtime->preview.slots[frame.slot];
                Output* output         = !frame.from_image && (!generation.task || frame.id >= *generation.task) ? &generation : repaint && repaint->result.task == frame.id ? &repaint->result : nullptr;
                const bool final_ready = std::ranges::any_of(events, [&](const Event& event) { return event.kind == EventKind::generated && event.id == frame.id; });
                if (!output || !preview_enabled || !renderer.visible || final_ready || (output->task == frame.id && (output->record || frame.step <= output->step))) {
                    renderer.discard(*source.timeline, frame.ready);
                    continue;
                }
                if (output->task != frame.id) begin_output(*output, frame.id, frame.width, frame.height);
                if (output->texture) renderer.retire(output->texture);
                output->texture = renderer.texture({static_cast<std::uint32_t>(frame.width), static_cast<std::uint32_t>(frame.height)});
                renderer.copy(output->texture, source.buffer, *source.timeline, frame.ready);
                output->width   = frame.width;
                output->height  = frame.height;
                output->preview = true;
                output->step    = frame.step;
            }
            for (auto& event : events) {
                Output* output = event.record.source.empty() && (!generation.task || event.id >= *generation.task) ? &generation : repaint && repaint->result.task == event.id ? &repaint->result : nullptr;
                if (output && output->task != event.id) begin_output(*output, event.id, event.record.parameters.width, event.record.parameters.height);
                if (event.kind == EventKind::generated) {
                    auto& source = runtime->interop.slots[event.slot];
                    if (output) {
                        if (output->texture) renderer.retire(output->texture);
                        output->width   = event.record.parameters.width;
                        output->height  = event.record.parameters.height;
                        output->texture = renderer.texture({static_cast<std::uint32_t>(output->width), static_cast<std::uint32_t>(output->height)});
                        renderer.copy(output->texture, source.buffer, *source.timeline, event.ready);
                        output->record  = std::move(event.record);
                        output->preview = false;
                    } else renderer.discard(*source.timeline, event.ready);
                } else {
                    if (output) output->record = std::move(event.record);
                    library.refresh();
                }
            }
        }
        for (auto* output : {&generation, repaint ? &repaint->result : nullptr}) {
            if (!output || !output->record || output->record->path.empty()) continue;
            if (!output->saved) {
                const auto raw = std::ranges::find_if(library.roots, [](const auto& value) { return value.all.key == "raw"; });
                if (raw != library.roots.end() && raw->ready) {
                    const auto file = std::ranges::find(raw->files, output->record->path, &dataset::File::path);
                    if (file != raw->files.end()) output->saved = *file;
                }
            }
            if (output->saved && output->texture) {
                const auto cached = library.textures.find(output->saved->sha);
                if (cached != library.textures.end() && cached->second.texture) renderer.retire(std::exchange(output->texture, 0));
            }
        }
    }

    void UserInterface::synchronize_collection() {
        root       = nullptr;
        collection = nullptr;
        for (auto& entry : library.roots) {
            if (entry.all.key == collection_key) {
                root       = &entry;
                collection = &entry.all;
            }
            for (auto& candidate : entry.concepts)
                if (candidate.key == collection_key) {
                    root       = &entry;
                    collection = &candidate;
                }
        }
        if (library.ready && page == Page::dataset && !collection) action_error = "Dataset does not exist: " + collection_key;
        if (collection && root->ready && !collection->images.empty()) {
            if (!positions.contains(collection_key) || positions.at(collection_key).selected.empty()) {
                const auto index          = collection->images.size() - 1;
                positions[collection_key] = {collection->images[index].sha, index, static_cast<float>(index)};
            }
            if (locate) {
                const auto member = std::ranges::find(root->files, *locate, &dataset::File::path);
                if (member != root->files.end()) {
                    const auto found = std::ranges::find(collection->images, member->sha, &dataset::File::sha);
                    if (found != collection->images.end()) {
                        const auto index          = static_cast<std::size_t>(found - collection->images.begin());
                        positions[collection_key] = {found->sha, index, static_cast<float>(index)};
                        locate.reset();
                    }
                }
            }
        }
    }

    void UserInterface::select_collection(std::string key, std::optional<std::filesystem::path> target) {
        commit_parameters();
        if (!leave_repaint()) return;
        if (page == Page::generation && !prompt_editor.commit(prompt, *catalog)) {
            prompt_sidebar.open = true;
            return;
        }
        prompt_editor.suspend();
        if (page == Page::generation) generation_view = view;
        page           = Page::dataset;
        viewing        = View::browse;
        collection_key = std::move(key);
        locate         = std::move(target);
        synchronize_collection();
        view          = {};
        window.redraw = true;
    }

    void UserInterface::center_image(const std::size_t index) {
        locate.reset();
        auto& position    = positions[collection_key];
        position.index    = index;
        position.selected = collection->images[index].sha;
        animate_until     = glfwGetTime() + 0.3;
    }

    void UserInterface::start_repaint(const dataset::File& file) {
        const auto source = file;
        if (repaint && repaint->source.sha == source.sha) {
            leave_repaint();
            return;
        }
        const auto cached = library.textures.find(source.sha);
        if (cached == library.textures.end() || !cached->second.texture) return;
        const Page return_page  = repaint ? repaint->return_page : page;
        View return_view        = repaint ? repaint->return_view : viewing;
        ImageView return_camera = repaint ? repaint->return_camera : view;
        if (!leave_repaint()) return;
        if (page == Page::generation) {
            select_collection("raw", source.path);
            if (page != Page::dataset) return;
        }
        auto& edits = repaints[source.sha];
        if (!edits) edits = std::make_unique<RepaintDraft>(cached->second.record);
        repaint.emplace(source, return_page, return_view, return_camera);
        viewing             = View::repaint;
        view                = {};
        prompt_sidebar.open = true;
    }

    bool UserInterface::leave_repaint() {
        if (!repaint) return true;
        auto& edits = *repaints.at(repaint->source.sha);
        if (!edits.editor.commit(edits.prompt, *edits.catalog)) {
            prompt_sidebar.open = true;
            return false;
        }
        edits.editor.suspend();
        page    = repaint->return_page;
        viewing = repaint->return_view;
        view    = repaint->return_camera;
        if (repaint->result.texture) renderer.retire(repaint->result.texture);
        repaint.reset();
        return true;
    }

    void UserInterface::back() {
        commit_parameters();
        if (viewing == View::source || viewing == View::result) {
            viewing = View::comparison;
            return;
        }
        if (repaint) {
            leave_repaint();
            return;
        }
        if (viewing == View::inspect) {
            viewing = View::browse;
            view    = {};
        } else {
            page = Page::generation;
            view = generation_view;
        }
    }

    UserInterface::Picture UserInterface::resolve_image(const dataset::File& file, const Role role) const {
        Picture picture{0, file.width, file.height, nullptr, file, false, role};
        const auto cached = library.textures.find(file.sha);
        if (cached != library.textures.end() && cached->second.error.empty()) {
            picture.texture = cached->second.texture;
            picture.record  = &cached->second.record;
        }
        return picture;
    }

    UserInterface::Picture UserInterface::resolve_output(const Output& output, const Role role) const {
        if (output.saved) {
            auto picture = resolve_image(*output.saved, role);
            if (picture.texture) return picture;
        }
        return {output.texture, output.width, output.height, output.record ? &*output.record : nullptr, output.saved, output.preview, role};
    }
    void UserInterface::commit_parameters() {
        if (!parameter_edit.id) return;
        auto* input = ImGui::GetInputTextState(parameter_edit.id);
        ImGui::DataTypeApplyFromText(input->TextA.Data, parameter_edit.type, parameter_edit.value, ImGui::DataTypeGetInfo(parameter_edit.type)->ScanFmt);
        // The value is committed; do not replay the text after a parameter action.
        input->ID = 0;
        if (ImGui::GetCurrentContext()->InputTextDeactivatedState.ID == parameter_edit.id) ImGui::GetCurrentContext()->InputTextDeactivatedState.ID = 0;
        ImGui::ClearActiveID();
        parameter_edit = {};
    }

    bool UserInterface::save_prompt() {
        if (!prompt_editor.commit(prompt, *catalog)) return false;
        try {
            prompt::write_preset({preset.name, prompt}, *catalog);
            preset.prompt = prompt;
            preset_error.clear();
            return true;
        } catch (const std::exception& failure) {
            preset_error = failure.what();
            return false;
        }
    }

    void UserInterface::switch_preset() {
        try {
            auto next = prompt::read_preset(pending_preset, *catalog);
            preset    = std::move(next);
            prompt    = preset.prompt;
            prompt_editor.reset(prompt);
            preset_error.clear();
        } catch (const std::exception& failure) {
            preset_error = failure.what();
        }
        pending_preset.clear();
    }

    void UserInterface::preset_dialogs(const float scale) {
        if (!pending_preset.empty() && !ImGui::IsPopupOpen("Unsaved prompt")) {
            if (!prompt_editor.commit(prompt, *catalog)) pending_preset.clear();
            else if (prompt == preset.prompt) switch_preset();
            else ImGui::OpenPopup("Unsaved prompt");
        }
        ImGui::SetNextWindowSize({420 * scale, 0});
        if (ImGui::BeginPopupModal("Unsaved prompt", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::TextWrapped("Save changes to %s before switching?", preset.name.c_str());
            if (!preset_error.empty()) ImGui::TextWrapped("%s", preset_error.c_str());
            ImGui::Spacing();
            if (ImGui::Button("Save", {108 * scale, 0}) && save_prompt()) {
                switch_preset();
                ImGui::CloseCurrentPopup();
            }
            ImGui::SameLine();
            if (ImGui::Button("Discard", {108 * scale, 0})) {
                switch_preset();
                ImGui::CloseCurrentPopup();
            }
            ImGui::SameLine();
            if (ImGui::Button("Cancel", {108 * scale, 0}) || ImGui::IsKeyPressed(ImGuiKey_Escape, false)) {
                pending_preset.clear();
                preset_error.clear();
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
        }
        if (std::exchange(save_as_requested, false) && prompt_editor.commit(prompt, *catalog)) {
            preset_error.clear();
            new_preset_name.fill(0);
            ImGui::OpenPopup("Save prompt as");
        }
        ImGui::SetNextWindowSize({420 * scale, 0});
        if (ImGui::BeginPopupModal("Save prompt as", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::TextDisabled("Preset name");
            if (ImGui::IsWindowAppearing()) ImGui::SetKeyboardFocusHere();
            ImGui::SetNextItemWidth(-1);
            const bool enter = ImGui::InputText("##PresetName", new_preset_name.data(), new_preset_name.size(), ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_CallbackCharFilter, [](ImGuiInputTextCallbackData* data) {
                const auto c = data->EventChar;
                return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-' || c == '_' ? 0 : 1;
            });
            ImGui::TextDisabled("Letters, numbers, hyphens and underscores");
            if (!preset_error.empty()) ImGui::TextWrapped("%s", preset_error.c_str());
            ImGui::Spacing();
            ImGui::BeginDisabled(new_preset_name[0] == 0);
            if ((ImGui::Button("Save", {108 * scale, 0}) || enter) && new_preset_name[0]) {
                try {
                    prompt::Preset next{new_preset_name.data(), prompt};
                    prompt::write_preset(next, *catalog, false);
                    preset = std::move(next);
                    preset_error.clear();
                    ImGui::CloseCurrentPopup();
                } catch (const std::exception& failure) {
                    preset_error = failure.what();
                }
            }
            ImGui::EndDisabled();
            ImGui::SameLine();
            if (ImGui::Button("Cancel", {108 * scale, 0}) || ImGui::IsKeyPressed(ImGuiKey_Escape, false)) {
                preset_error.clear();
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
        }
        if (!preset_error.empty() && !ImGui::IsPopupOpen(nullptr, ImGuiPopupFlags_AnyPopupId | ImGuiPopupFlags_AnyPopupLevel)) ImGui::OpenPopup("Prompt preset error");
        ImGui::SetNextWindowSize({420 * scale, 0});
        if (ImGui::BeginPopupModal("Prompt preset error", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::TextWrapped("%s", preset_error.c_str());
            ImGui::Spacing();
            if (ImGui::Button("Close") || ImGui::IsKeyPressed(ImGuiKey_Escape, false)) {
                preset_error.clear();
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
        }
    }

    void UserInterface::begin_output(Output& output, const std::uint64_t task, const int width, const int height) {
        if (output.texture) renderer.retire(output.texture);
        output = {.task = task, .width = width, .height = height};
        if (&output == &generation) {
            if (page == Page::generation) view = {};
            else generation_view = {};
        }
    }

    void UserInterface::submit() {
        commit_parameters();
        auto parameters     = draft;
        auto prompt_catalog = catalog;
        prompt::Pair submitted;
        std::optional<RepaintSource> source;
        if (repaint) {
            auto& edits = *repaints.at(repaint->source.sha);
            if (!edits.editor.commit(edits.prompt, *edits.catalog)) {
                prompt_sidebar.open = true;
                return;
            }
            parameters.width   = repaint->source.width;
            parameters.height  = repaint->source.height;
            parameters.denoise = denoise;
            submitted          = edits.editor.materialize(edits.prompt);
            prompt_catalog     = edits.catalog;
            source.emplace(repaint->source.sha, repaint->source.path);
        } else {
            if (page != Page::generation) return;
            if (!prompt_editor.commit(prompt, *catalog)) {
                prompt_sidebar.open = true;
                return;
            }
            submitted          = prompt;
            parameters.denoise = 1;
        }
        parameters.positive = prompt::compose(*prompt_catalog, submitted.positive);
        parameters.negative = prompt::compose(*prompt_catalog, submitted.negative);
        if (random_seed) {
            std::random_device random;
            seed = std::uniform_int_distribution<std::uint64_t>{}(random);
        }
        if (!runtime) runtime = std::make_unique<GenerationRuntime>(renderer.device, classifiers, classification);
        const auto task = runtime->session.enqueue(parameters, seed, std::move(submitted), std::move(prompt_catalog), std::move(source));
        if (repaint) {
            begin_output(repaint->result, task, parameters.width, parameters.height);
            viewing = View::comparison;
            view    = {};
        }
        animate_until = glfwGetTime() + 0.2;
    }

    UserInterface::ControlLayout UserInterface::control_layout(const float scale, const ImVec2 size, const Picture& image) const {
        ControlLayout layout{};
        layout.different             = page == Page::generation && image.texture && (image.width != draft.width || image.height != draft.height);
        const float dimensions_width = page == Page::generation ? 112 * scale + (layout.different ? ImGui::CalcTextSize("Next").x + 12 * scale : 0) : image.texture ? ImGui::CalcTextSize(std::format("{} \xC3\x97 {}", image.width, image.height).c_str()).x : 0;
        if (layout.different) layout.image_label_width = ImGui::CalcTextSize(std::format("{} {} \xC3\x97 {} \xE2\x86\x92", image.preview ? "Preview" : "Image", image.width, image.height).c_str()).x + 12 * scale;
        layout.right_width    = layout.image_label_width + dimensions_width + (image.texture ? 116 * scale : 0);
        const float available = size.x - dataset_sidebar.width * dataset_sidebar.amount - 2 * bottom_margin * scale;
        layout.image_above    = layout.right_width > available;
        if (layout.image_above) layout.right_width -= layout.image_label_width;
        if (page == Page::generation || repaint)
            for (const auto& model : classifiers) layout.classifiers.push_back({.id = model.id, .available = true});
        if (image.record) {
            for (const auto& result : image.record->classification.classifiers) {
                const auto row = std::ranges::find(layout.classifiers, result.id, &ControlLayout::Classifier::id);
                if (row != layout.classifiers.end()) row->result = &result;
                else layout.classifiers.push_back({.id = result.id, .result = &result});
            }
        }
        if (layout.classifiers.size() > classifiers.size()) std::ranges::sort(layout.classifiers, {}, &ControlLayout::Classifier::id);
        for (auto& row : layout.classifiers) {
            row.enabled = std::ranges::contains(classification.enabled, row.id);
            row.verdict = image.preview && row.enabled ? "\xE2\x80\xA6" : "\xE2\x80\x94";
        }
        float width = 0;
        for (auto& row : layout.classifiers) {
            row.ink = ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled);
            if (row.result) {
                row.verdict = !row.result->error.empty() ? "ERROR" : row.result->accepted ? "PASS" : "FAIL \xC2\xB7 " + row.result->label;
                row.ink     = !row.result->error.empty() ? ImVec4{1, .67F, .2F, 1} : row.result->accepted ? ImVec4{.3F, .88F, .5F, 1} : ImVec4{1, .36F, .4F, 1};
            }
            width = std::max(width, ImGui::CalcTextSize(row.id.data(), row.id.data() + row.id.size()).x + ImGui::CalcTextSize(row.verdict.c_str()).x + ImGui::CalcTextSize("\xC2\xB7 ").x + 36 * scale);
        }
        const float minimum_width = std::min(width, 160 * scale);
        layout.classifier_bottom  = bottom_margin * scale;
        float classifier_space    = available - layout.right_width - bottom_margin * scale;
        if (classifier_space < minimum_width) {
            layout.classifier_bottom += (control_height + 16 + (layout.image_above ? control_height + 8 : 0)) * scale;
            classifier_space = available;
        }
        const float rows         = float(std::min(std::size_t{6}, layout.classifiers.size()));
        layout.classifier_height = std::min(rows * control_height * scale, std::max(0.0F, size.y - layout.classifier_bottom - (top_strip_height + 24) * scale));
        if (layout.classifiers.size() * control_height * scale > layout.classifier_height) width += 3 * scale;
        const float tag_space   = std::max(available - prompt_sidebar.width * prompt_sidebar.amount, std::min(available, minimum_width));
        layout.classifier_width = std::min(width, std::max(0.0F, std::min(classifier_space, tag_space)));
        return layout;
    }

    UserInterface::ImageAction UserInterface::image_panel(const char* id, const Picture& image, const ImVec2 origin, const ImVec2 size, const float scale, const bool interactive, const float brightness) {
        auto* draw = ImGui::GetWindowDrawList();
        ImGui::SetCursorScreenPos(origin);
        ImGui::InvisibleButton(id, size, ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonMiddle);
        const bool hovered = ImGui::IsItemHovered();
        const auto& io     = ImGui::GetIO();
        const float inset  = image.role == Role::image ? 0 : 8 * scale;
        const ImVec2 available{(size.x - 2 * inset) * io.DisplayFramebufferScale.x, (size.y - 2 * inset) * io.DisplayFramebufferScale.y};
        const ImVec2 dimensions{float(repaint && interactive ? repaint->source.width : image.width), float(repaint && interactive ? repaint->source.height : image.height)};
        if (!image.width || !image.height) return ImageAction::none;
        const float fitted = std::min(available.x / dimensions.x, available.y / dimensions.y);
        const double now   = frame_time;
        if (interactive) {
            view_available = available;
            view.update(available, dimensions, now);
        }
        const float zoom    = interactive ? view.zoom : fitted;
        const ImVec2 center = interactive ? view.center : ImVec2{0.5F, 0.5F};
        const ImVec2 extent{dimensions.x * zoom / io.DisplayFramebufferScale.x, dimensions.y * zoom / io.DisplayFramebufferScale.y};
        ImVec2 minimum{origin.x + size.x / 2 - center.x * extent.x, origin.y + size.y / 2 - center.y * extent.y};
        const bool over_image = hovered && ImGui::IsMouseHoveringRect(minimum, {minimum.x + extent.x, minimum.y + extent.y});
        ImageAction action    = ImageAction::none;
        if (image.texture && over_image && ImGui::IsMouseClicked(ImGuiMouseButton_Middle) && image.file) action = ImageAction::repaint;
        if (interactive && image.texture) {
            const bool movable = extent.x > size.x - 2 * inset + 1 || extent.y > size.y - 2 * inset + 1;
            if (over_image && ImGui::IsMouseClicked(ImGuiMouseButton_Left) && movable && !ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
                view.dragging = true;
                view.started  = -1;
            }
            if (view.dragging && ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
                view.center.x -= io.MouseDelta.x * io.DisplayFramebufferScale.x / (dimensions.x * view.zoom);
                view.center.y -= io.MouseDelta.y * io.DisplayFramebufferScale.y / (dimensions.y * view.zoom);
                view.constrain(available, dimensions);
            }
            if (!ImGui::IsMouseDown(ImGuiMouseButton_Left)) view.dragging = false;
            if (hovered && io.MouseWheel && !view.dragging) {
                ImGui::SetKeyOwner(ImGuiKey_MouseWheelY, ImGui::GetItemID());
                const float next = std::clamp(view.target_zoom * std::pow(1.15F, io.MouseWheel), fitted, std::max(16.0F, fitted));
                const ImVec2 pivot{(io.MousePos.x - origin.x - size.x / 2) * io.DisplayFramebufferScale.x, (io.MousePos.y - origin.y - size.y / 2) * io.DisplayFramebufferScale.y};
                view.scale_to(next, pivot, dimensions, next <= fitted, now);
            }
            if (over_image && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) view.scale_to(view.fit ? std::max(1.0F, fitted) : fitted, {}, dimensions, !view.fit || fitted >= 1, now);
            if (over_image && movable) renderer.hand_cursor = view.dragging ? 1 : 0;
            minimum = {origin.x + size.x / 2 - view.center.x * extent.x, origin.y + size.y / 2 - view.center.y * extent.y};
        }
        if (image.texture && over_image && ImGui::IsMouseReleased(ImGuiMouseButton_Left) && !ImGui::IsMouseDragPastThreshold(ImGuiMouseButton_Left) && io.MouseClickedLastCount[ImGuiMouseButton_Left] == 1) action = ImageAction::click;
        if (zoom == 1) {
            minimum.x = std::round(minimum.x * io.DisplayFramebufferScale.x) / io.DisplayFramebufferScale.x;
            minimum.y = std::round(minimum.y * io.DisplayFramebufferScale.y) / io.DisplayFramebufferScale.y;
        }
        const ImVec2 maximum{minimum.x + extent.x, minimum.y + extent.y};
        draw->PushClipRect(origin, {origin.x + size.x, origin.y + size.y}, true);
        if (image.texture) draw->AddImage(image.texture, minimum, maximum, {0, 0}, {1, 1}, ImGui::GetColorU32(ImVec4{brightness, brightness, brightness, 1}));
        else {
            const char* label = image.role == Role::result ? "Waiting for image" : "Loading image";
            if (image.file) {
                const auto cached = library.textures.find(image.file->sha);
                if (cached != library.textures.end() && !cached->second.error.empty()) label = "Image read error";
            }
            const auto text = ImGui::CalcTextSize(label);
            draw->AddText({origin.x + (size.x - text.x) / 2, origin.y + (size.y - text.y) / 2}, ImGui::GetColorU32(ImGuiCol_TextDisabled), label);
        }
        if (image.role != Role::image) {
            const ImVec4 first  = image.role == Role::source ? ImVec4{0.25F, 0.86F, 0.57F, 1} : ImVec4{0.65F, 0.43F, 0.97F, 1};
            const ImVec4 second = image.role == Role::source ? ImVec4{0.36F, 0.72F, 0.70F, 1} : ImVec4{0.32F, 0.58F, 0.97F, 1};
            for (int layer = 4; layer >= 0; --layer) {
                const float offset    = (1 + layer * 1.3F) * scale;
                const float thickness = (layer ? 1.5F : 1.0F) * scale;
                const ImVec2 a{minimum.x - offset, minimum.y - offset}, b{maximum.x + offset, maximum.y + offset};
                const float alpha = layer ? 0.035F : 0.85F;
                const auto c1     = ImGui::GetColorU32(ImVec4{first.x, first.y, first.z, alpha});
                const auto c2     = ImGui::GetColorU32(ImVec4{second.x, second.y, second.z, alpha});
                draw->AddRectFilledMultiColor(a, {b.x, a.y + thickness}, c1, c2, c2, c1);
                draw->AddRectFilledMultiColor({a.x, b.y - thickness}, b, c2, c1, c1, c2);
                draw->AddRectFilledMultiColor(a, {a.x + thickness, b.y}, c1, c1, c2, c2);
                draw->AddRectFilledMultiColor({b.x - thickness, a.y}, b, c2, c2, c1, c1);
            }
        }
        draw->PopClipRect();
        return action;
    }

    void UserInterface::canvas(const float scale, const ImVec2 size) {
        ImGui::SetNextWindowPos({0, 0});
        ImGui::SetNextWindowSize(size);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {0, 0});
        ImGui::Begin("##Canvas", nullptr, overlay | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoBringToFrontOnFocus);
        ImGui::PopStyleVar();
        const auto origin    = canvas_origin;
        const auto available = canvas_size;
        // Navigate on press so releasing a popup-dismissal click cannot also change the view.
        const bool navigating = ImGui::IsWindowHovered() && ImGui::IsMouseHoveringRect(origin, {origin.x + available.x, origin.y + available.y}) && !ImGui::IsPopupOpen(nullptr, ImGuiPopupFlags_AnyPopupId | ImGuiPopupFlags_AnyPopupLevel) && ImGui::IsMouseClicked(ImGuiMouseButton_Right);
        ImGui::PushClipRect(origin, {origin.x + available.x, origin.y + available.y}, true);
        std::vector<dataset::File> wanted;
        std::optional<dataset::File> repaint_source;
        if (page == Page::generation) {
            if (generation.saved) wanted.push_back(*generation.saved);
            const auto image = resolve_output(generation);
            if (image.texture) {
                if (image_panel("##GeneratedImage", image, origin, available, scale, true) == ImageAction::repaint) repaint_source = image.file;
            } else {
                auto* draw = ImGui::GetWindowDrawList();
                const ImVec2 center{origin.x + available.x / 2, origin.y + available.y / 2};
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
        } else if (!library.error.empty() || !collection || !root->ready) {
            ImGui::SetCursorScreenPos({origin.x + 16 * scale, (top_strip_height + 24) * scale});
            if (!library.error.empty()) ImGui::TextWrapped("%s", library.error.c_str());
            else if (!library.ready) ImGui::TextDisabled("Indexing datasets...");
            else if (!collection) ImGui::TextWrapped("Dataset does not exist: %s", collection_key.c_str());
            else {
                ImGui::TextColored({0.95F, 0.49F, 0.42F, 1}, "Dataset needs attention");
                if (!root->error.empty()) ImGui::TextWrapped("%s", root->error.c_str());
                if (!root->conflicts.empty()) ImGui::TextWrapped("Identical images are stored as independent files. Open the dataset sidebar to see all conflict paths.");
            }
        } else if (repaint) {
            wanted.push_back(repaint->source);
            if (repaint->result.saved) wanted.push_back(*repaint->result.saved);
            const auto source = resolve_image(repaint->source, Role::source);
            const auto result = resolve_output(repaint->result, Role::result);
            if (viewing == View::comparison) {
                const bool horizontal = repaint->source.width <= repaint->source.height;
                const float gap       = 20 * scale;
                const ImVec2 pane{horizontal ? (available.x - gap) / 2 : available.x, horizontal ? available.y : (available.y - gap) / 2};
                const ImVec2 next{origin.x + (horizontal ? pane.x + gap : 0), origin.y + (horizontal ? 0 : pane.y + gap)};
                ImageAction first, second;
                // Process the hovered pane first so both draws use the same camera this frame.
                if (ImGui::GetActiveID() == ImGui::GetID("##Result") || (ImGui::GetActiveID() != ImGui::GetID("##Source") && ImGui::IsMouseHoveringRect(next, {next.x + pane.x, next.y + pane.y}))) {
                    second = image_panel("##Result", result, next, pane, scale, true);
                    first  = image_panel("##Source", source, origin, pane, scale, true);
                } else {
                    first  = image_panel("##Source", source, origin, pane, scale, true);
                    second = image_panel("##Result", result, next, pane, scale, true);
                }
                if (first == ImageAction::repaint) repaint_source = source.file;
                else if (second == ImageAction::repaint) repaint_source = result.file;
                else if (first == ImageAction::click) viewing = View::source;
                else if (second == ImageAction::click) viewing = View::result;
            } else {
                const auto image = viewing == View::result ? result : source;
                if (image_panel("##RepaintImage", image, origin, available, scale, true) == ImageAction::repaint) repaint_source = image.file;
            }
        } else if (collection->images.empty()) {
            ImGui::SetCursorScreenPos({origin.x + 16 * scale, (top_strip_height + 24) * scale});
            ImGui::TextDisabled("This dataset is empty.");
        } else {
            auto& position = positions.at(collection_key);
            if (viewing == View::inspect) {
                const auto& file = collection->images[position.index];
                wanted.push_back(file);
                if (image_panel("##InspectedImage", resolve_image(file), origin, available, scale, true) == ImageAction::repaint) repaint_source = file;
            } else {
                wanted.push_back(collection->images[position.index]);
                const auto& io = ImGui::GetIO();
                if (ImGui::IsWindowHovered() && ImGui::IsMouseHoveringRect(origin, {origin.x + available.x, origin.y + available.y}) && io.MouseWheel) {
                    const auto index = std::clamp<std::ptrdiff_t>(static_cast<std::ptrdiff_t>(position.index) - (io.MouseWheel > 0 ? 1 : -1), 0, collection->images.size() - 1);
                    center_image(static_cast<std::size_t>(index));
                    ImGui::SetKeyOwner(ImGuiKey_MouseWheelY, ImGui::GetID("##DatasetScroll"));
                }
                position.scroll = std::lerp(position.scroll, static_cast<float>(position.index), std::min(1.0F, io.DeltaTime / 0.055F));
                if (std::abs(position.scroll - position.index) < 0.001F) position.scroll = static_cast<float>(position.index);
                else animate_until = glfwGetTime() + 0.1;
                const auto anchor = static_cast<std::size_t>(std::floor(position.scroll));
                const auto first  = anchor > 3 ? anchor - 3 : 0;
                const auto last   = std::min(collection->images.size(), anchor + 5);
                const float gap   = 24 * scale;
                const auto width  = [&](const std::size_t index) {
                    const auto& file = collection->images[index];
                    return std::min(available.x * 0.68F, available.y * file.width / file.height);
                };
                const float advance = anchor + 1 < collection->images.size() ? (width(anchor) + width(anchor + 1)) / 2 + gap : 0;
                float x             = origin.x + available.x / 2 - (position.scroll - anchor) * advance;
                for (auto i = anchor; i > first; --i) x -= (width(i) + width(i - 1)) / 2 + gap;
                for (auto i = first; i < last; ++i) {
                    const auto& file = collection->images[i];
                    wanted.push_back(file);
                    const float image_width = width(i);
                    const float brightness  = std::lerp(1.0F, 0.60F, std::min(1.0F, std::abs(float(i) - position.scroll)));
                    ImGui::PushID(static_cast<int>(i));
                    const auto action = image_panel("##DatasetImage", resolve_image(file), {x - image_width / 2, origin.y}, {image_width, available.y}, scale, false, brightness);
                    ImGui::PopID();
                    if (action == ImageAction::repaint) repaint_source = file;
                    else if (action == ImageAction::click) {
                        if (i == position.index && position.scroll == float(position.index)) {
                            viewing = View::inspect;
                            view    = {};
                        } else center_image(i);
                    }
                    if (i + 1 < last) x += (image_width + width(i + 1)) / 2 + gap;
                }
            }
        }
        for (const auto& file : wanted) {
            const auto cached = library.textures.find(file.sha);
            if (cached != library.textures.end() && !cached->second.error.empty()) action_error = cached->second.error;
        }
        library.request(std::move(wanted));
        ImGui::PopClipRect();
        ImGui::End();
        if (navigating) {
            if (page == Page::generation) select_collection("raw", generation.record && !generation.record->path.empty() ? std::optional{generation.record->path} : std::nullopt);
            else back();
        } else if (repaint_source) start_repaint(*repaint_source);
    }
    void UserInterface::generation_settings(const float scale, const ImVec2 size) {
        const float row_height       = 32 * scale;
        const float seed_label_width = ImGui::CalcTextSize("Seed").x + 12 * scale;
        const float mode_width       = ImGui::CalcTextSize("Random").x + 16 * scale;
        const float panel_width      = std::max(336 * scale, seed_label_width + mode_width + ImGui::CalcTextSize("18446744073709551615").x + 48 * scale);
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
        number_field("##Steps", "Steps", ImGuiDataType_S32, &draft.steps, field_size, &steps_step, "%d", scale, parameter_edit);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Sampling steps for the next image\nUp / Down: 1\nEnter to confirm, Esc to undo");
        ImGui::SameLine(0, 24 * scale);
        constexpr float cfg_step = 0.1F;
        number_field("##CFG", "CFG", ImGuiDataType_Float, &draft.cfg, field_size, &cfg_step, "%.1f", scale, parameter_edit);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Guidance scale for the next image\nUp / Down: 0.1\nEnter to confirm, Esc to undo");
        const auto origin = ImGui::GetCursorScreenPos();
        auto* draw        = ImGui::GetWindowDrawList();
        draw->AddLine({origin.x, origin.y - 8 * scale}, {origin.x + width, origin.y - 8 * scale}, ImGui::GetColorU32(ImVec4{1, 1, 1, 0.06F}), scale);
        draw->AddText({origin.x + 4 * scale, origin.y + (row_height - ImGui::GetFontSize()) / 2}, ImGui::GetColorU32(ImGuiCol_TextDisabled), "Seed");
        if (!random_seed) {
            ImGui::SetCursorScreenPos({origin.x + seed_label_width, origin.y});
            number_field("##Seed", "", ImGuiDataType_U64, &seed, {width - seed_label_width - mode_width - 8 * scale, row_height}, nullptr, "%llu", scale, parameter_edit);
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
        }
        if (repaint.has_value()) {
            ImGui::TextDisabled("Denoise");
            ImGui::SetNextItemWidth(-1);
            ImGui::SliderFloat("##Denoise", &denoise, 0, 1, "%.2f", ImGuiSliderFlags_AlwaysClamp);
        }
        if (ImGui::IsKeyPressed(ImGuiKey_Escape, false)) ImGui::CloseCurrentPopup();
        ImGui::PopStyleVar();
        ImGui::EndPopup();
    }

    void UserInterface::top_strip(const float scale, const ImVec2 size) {
        bool active{}, paused{}, loaded{true}, failed{}, unavailable{};
        std::size_t queued{};
        int steps{};
        double elapsed{};
        if (runtime) {
            const auto& session = runtime->session;
            const std::lock_guard lock{runtime->session.mutex};
            active      = session.active.has_value();
            paused      = session.paused;
            loaded      = session.model_ready;
            failed      = !session.error.empty();
            unavailable = session.worker_done;
            queued      = session.queue.size();
            if (active) {
                steps   = session.active->parameters.steps;
                elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - session.started).count();
            }
        }
        const auto stage         = runtime ? static_cast<sdxl::Stage>(::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>{runtime->session.control.data()[0].stage}.load()) : sdxl::Stage::idle;
        const auto completed     = runtime ? ::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>{runtime->session.control.data()[0].completed}.load() : 0;
        const bool stopping      = active && ::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>{runtime->session.control.data()[0].cancel}.load();
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
        ImGui::SetCursorPos({12 * scale, 4 * scale});
        if (page == Page::generation) {
            if (text_button("##Application", "GENESIA", scale)) ImGui::OpenPopup("Application");
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Right-click canvas: open Raw\n`: datasets\nTab: Prompt\nF11: fullscreen\nEsc: exit");
        } else {
            if (text_button("##Back", "\xE2\x80\xB9", scale)) back();
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Back one level\nRight-click the canvas to return");
            ImGui::SameLine(0, 4 * scale);
            const auto count    = collection ? collection->images.size() : 0;
            const auto position = positions.find(collection_key);
            const auto number   = count && position != positions.end() ? position->second.index + 1 : 0;
            const auto label    = std::format("{}  \xC2\xB7  {} / {}", collection_key, number, count);
            if (text_button("##Location", label.c_str(), scale, std::min(ImGui::CalcTextSize(label.c_str()).x + 24 * scale, size.x * 0.42F))) dataset_sidebar.open = !dataset_sidebar.open;
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s\n`: datasets\nTab: Prompt\nLeft-click image: center / inspect\nRight-click canvas: back\nEsc: exit", collection_key.c_str());
        }
        const float left_end = ImGui::GetItemRectMax().x + 4 * scale;
        if (ImGui::BeginPopup("Application")) {
            ImGui::MenuItem("Live preview", nullptr, &preview_enabled);
            ImGui::Separator();
            if (ImGui::MenuItem("Save prompt", "Ctrl+S", false, page == Page::generation)) save_prompt();
            if (ImGui::MenuItem("Open output folder")) ShellExecuteW(window.native_window, L"open", dataset::raw.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
            ImGui::EndPopup();
        }
        const bool queue_visible      = queued != 0 || paused || ImGui::IsPopupOpen("Queue");
        const std::string queue_label = queue_visible ? std::format("Queue  {}", queued) : "";
        const char* stop_label        = active ? "Stop" : paused && !failed ? "Resume" : "";
        const bool submission         = page == Page::generation || repaint.has_value();
        const float primary_width     = submission ? 150 * scale : 0;
        float controls_width          = primary_width;
        for (const char* label : {queue_label.c_str(), stop_label})
            if (*label) controls_width += ImGui::CalcTextSize(label).x + 28 * scale;
        const bool queue_paused  = paused && !active && !failed;
        const char* label        = queue_paused ? "Queue paused" : progress_alpha > 0 ? progress_label.c_str() : "";
        const float alpha        = queue_paused ? 1 : progress_alpha;
        const float right_start  = maximum.x - controls_width;
        const float label_width  = *label ? std::max(ImGui::CalcTextSize("Finishing image").x, ImGui::CalcTextSize(label).x) : 0;
        const auto time_text     = ImGui::CalcTextSize(progress_time.c_str());
        const float time_width   = std::max(ImGui::CalcTextSize("0000.0s").x, time_text.x);
        const bool show_time     = *label && !queue_paused && !progress_time.empty() && label_width + 16 * scale + time_width <= right_start - left_end - 32 * scale;
        const float status_width = label_width + (show_time ? 16 * scale + time_width : 0);
        const float group_left   = right_start - status_width - (*label ? 12 * scale : 0) - 8 * scale;
        control_shade({group_left, minimum.y + 4 * scale}, maximum, scale);
        if (*label) {
            ImGui::SetCursorScreenPos({group_left + 8 * scale, minimum.y + 4 * scale});
            ImGui::Dummy({status_width, control_height * scale});
            const auto origin = ImGui::GetItemRectMin();
            const auto text   = ImGui::CalcTextSize(label);
            const float y     = origin.y + (control_height * scale - text.y) / 2;
            draw->AddText({origin.x, y}, ImGui::GetColorU32(ImVec4{0.89F, 0.89F, 0.94F, alpha}), label);
            if (show_time) {
                draw->AddCircleFilled({origin.x + label_width + 8 * scale, origin.y + control_height * scale / 2}, 1.25F * scale, ImGui::GetColorU32(ImVec4{0.57F, 0.58F, 0.64F, alpha}));
                draw->AddText({origin.x + status_width - time_text.x, y}, ImGui::GetColorU32(ImVec4{0.67F, 0.68F, 0.74F, alpha}), progress_time.c_str());
            }
            if (ImGui::IsItemHovered() && !progress_time.empty()) ImGui::SetTooltip("Elapsed: %s", progress_time.c_str());
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
                auto& session = runtime->session;
                const std::lock_guard lock{session.mutex};
                if (session.queue.empty()) ImGui::TextUnformatted("No pending images");
                for (std::size_t i = 0; i < session.queue.size();) {
                    const auto& request = session.queue[i];
                    ImGui::PushID(static_cast<int>(request.id));
                    ImGui::Text("#%llu   %s   %d x %d", static_cast<unsigned long long>(request.id + 1), request.source ? "Repaint" : "Generate", request.parameters.width, request.parameters.height);
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
                if (active) runtime->session.stop();
                else runtime->session.resume();
            }
            x = ImGui::GetItemRectMax().x + 4 * scale;
        }
        if (submission) {
            const bool valid   = repaint ? root && root->ready && repaints.at(repaint->source.sha)->editor.valid : prompt_editor.valid;
            const bool enabled = !unavailable && !ImGui::GetTopMostPopupModal() && valid;
            if (ImGui::IsPopupOpen("Generation settings") && ImGui::IsWindowHovered(ImGuiHoveredFlags_AllowWhenBlockedByPopup | ImGuiHoveredFlags_AllowWhenBlockedByActiveItem) && ImGui::IsMouseHoveringRect({x, minimum.y}, maximum) && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                commit_parameters();
                ImGui::ClosePopupsOverWindow(ImGui::GetCurrentWindow(), false);
            }
            ImGui::SetCursorScreenPos({x, minimum.y});
            ImGui::PushFont(nullptr, 18);
            ImGui::BeginDisabled(!enabled);
            const bool clicked = text_button("##Submit", repaint ? "Repaint" : "Generate", scale, primary_width, false, repaint ? ImVec4{0.53F, 0.80F, 0.72F, 1} : ImVec4{0.70F, 0.65F, 0.97F, 1});
            ImGui::EndDisabled();
            const bool hovered = ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled | ImGuiHoveredFlags_AllowWhenBlockedByActiveItem);
            if (hovered && ImGui::IsMouseReleased(ImGuiMouseButton_Right)) {
                commit_parameters();
                ImGui::OpenPopup("Generation settings");
            }
            ImGui::PopFont();
            if (hovered && !ImGui::IsPopupOpen("Generation settings")) ImGui::SetTooltip("%s\nCtrl+Shift+`\nRight-click: settings", repaint ? "Repaint the green source into Raw" : "Generate a new image into Raw");
            generation_settings(scale, size);
            if (enabled && (clicked || ImGui::Shortcut(ImGuiMod_Ctrl | ImGuiMod_Shift | ImGuiKey_GraveAccent, ImGuiInputFlags_RouteGlobal | ImGuiInputFlags_RouteOverActive))) submit();
        }
        window.drag_region = {};
        if (!ImGui::IsPopupOpen(nullptr, ImGuiPopupFlags_AnyPopupId | ImGuiPopupFlags_AnyPopupLevel)) window.drag_region = {left_end, 0, group_left - 4 * scale, maximum.y};
        ImGui::End();
    }

    void UserInterface::sidebar(const bool left, const float scale, const ImVec2 size, const Picture& image) {
        const auto& panel = left ? dataset_sidebar : prompt_sidebar;
        if (!left) {
            if (!panel.open || page != Page::generation) prompt_editor.suspend();
            if (repaint && !panel.open) repaints.at(repaint->source.sha)->editor.suspend();
        }
        if (panel.amount == 0) return;
        const float visible = panel.width * panel.amount;
        const float top     = (top_strip_height + 24) * scale;
        ImGui::SetNextWindowPos({left ? visible - panel.width : size.x - visible, top});
        ImGui::SetNextWindowSize({panel.width, std::max(1.0F, size.y - top - (bottom_margin + control_height + 16) * scale)});
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {16 * scale, 0});
        ImGui::PushStyleVar(ImGuiStyleVar_Alpha, panel.amount);
        std::optional<std::string> selected;
        if (ImGui::Begin(left ? "##DatasetSidebar" : "##PromptSidebar", nullptr, overlay | ImGuiWindowFlags_NoBackground | (panel.open ? ImGuiWindowFlags_None : ImGuiWindowFlags_NoInputs))) {
            ImGui::PushFont(nullptr, 12);
            const float content_y = ImGui::GetCursorPosY() + ImGui::GetFontSize() + 12 * scale;
            if (left) ImGui::TextDisabled("DATASETS");
            else if (page == Page::generation) {
                ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, {0, 0});
                ImGui::PushStyleVar(ImGuiStyleVar_ButtonTextAlign, {0, 0.5F});
                ImGui::PushStyleColor(ImGuiCol_Button, {0, 0, 0, 0});
                ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
                if (ImGui::Button(preset.name.c_str(), {ImGui::CalcTextSize(preset.name.c_str()).x + 16 * scale, 0})) {
                    try {
                        preset_names = prompt::list_presets();
                        ImGui::OpenPopup("Prompt presets");
                    } catch (const std::exception& failure) {
                        preset_error = failure.what();
                    }
                }
                const auto minimum = ImGui::GetItemRectMin();
                const auto maximum = ImGui::GetItemRectMax();
                const ImVec2 arrow{maximum.x - 5 * scale, (minimum.y + maximum.y) / 2};
                ImGui::GetWindowDrawList()->AddTriangleFilled({arrow.x - 3 * scale, arrow.y - 1.5F * scale}, {arrow.x + 3 * scale, arrow.y - 1.5F * scale}, {arrow.x, arrow.y + 1.5F * scale}, ImGui::GetColorU32(ImGuiCol_Text));
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("Choose a prompt preset\nCtrl+S: save prompt");
                ImGui::PopStyleColor(2);
                ImGui::PopStyleVar(2);
                if (prompt != preset.prompt) {
                    ImGui::SameLine(0, 8 * scale);
                    const auto point = ImGui::GetCursorScreenPos();
                    ImGui::Dummy({8 * scale, ImGui::GetTextLineHeight()});
                    ImGui::GetWindowDrawList()->AddCircleFilled({point.x + 3 * scale, point.y + ImGui::GetTextLineHeight() / 2}, 2 * scale, ImGui::GetColorU32(ImGuiCol_CheckMark));
                    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Unsaved prompt changes");
                }
                if (ImGui::BeginPopup("Prompt presets")) {
                    for (const auto& name : preset_names) {
                        if (ImGui::Selectable(name.c_str(), name == preset.name) && name != preset.name) {
                            pending_preset = name;
                            preset_error.clear();
                        }
                    }
                    ImGui::Separator();
                    if (ImGui::MenuItem("Save as...")) save_as_requested = true;
                    ImGui::EndPopup();
                }
            } else if (repaint) {
                ImGui::TextDisabled("Repaint edits");
                if (ImGui::BeginPopupContextItem("Repaint edits")) {
                    if (ImGui::MenuItem("Reset changes")) {
                        ImGui::ClearActiveID();
                        repaints[repaint->source.sha] = std::make_unique<RepaintDraft>(library.textures.at(repaint->source.sha).record);
                    }
                    ImGui::EndPopup();
                }
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("Temporary draft for the green source\nRight-click: Reset changes");
            } else ImGui::TextDisabled("Image prompt");
            ImGui::PopFont();
            ImGui::SetCursorPosY(content_y);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {0, 0});
            ImGui::BeginChild("##SidebarContent", {0, 0}, ImGuiChildFlags_None, ImGuiWindowFlags_NoBackground);
            if (left) selected = dataset_contents();
            else if (page == Page::generation) prompt_editor.draw(prompt, tag_search, *catalog, scale);
            else if (repaint) {
                auto& edits = *repaints.at(repaint->source.sha);
                ImGui::PushID(repaint->source.sha.c_str());
                edits.editor.draw(edits.prompt, tag_search, *edits.catalog, scale);
                ImGui::PopID();
            } else if (image.record) show_prompt(image.record->prompt, *image.record->catalog, scale);
            else ImGui::TextDisabled(image.file ? "Loading prompt..." : "No image selected.");
            ImGui::EndChild();
            ImGui::PopStyleVar();
        }
        ImGui::End();
        ImGui::PopStyleVar(2);
        if (selected) select_collection(std::move(*selected));
    }

    std::optional<std::string> UserInterface::dataset_contents() {
        if (!library.error.empty()) ImGui::TextWrapped("%s", library.error.c_str());
        else if (page == Page::dataset && collection) {
            ImGui::TextWrapped("%s", collection->key.c_str());
            ImGui::TextDisabled("%zu images", collection->images.size());
            ImGui::TextDisabled("%zu concepts in %s", root->concepts.size(), root->all.name.c_str());
            ImGui::TextColored(root->ready ? ImVec4{0.4F, 0.76F, 0.58F, 1} : ImVec4{0.95F, 0.49F, 0.42F, 1}, "%s", root->ready ? "Ready" : "Index error");
        } else ImGui::TextDisabled(library.ready ? "Choose a dataset" : "Indexing...");
        ImGui::Spacing();
        ImGui::Separator();
        std::optional<std::string> selected;
        for (const auto& entry : library.roots) {
            ImGui::PushID(entry.all.key.c_str());
            const auto label = std::format("{}  \xC2\xB7  {}{}", entry.all.name == "raw" ? "Raw" : entry.all.name, entry.all.images.size(), entry.ready ? "" : "  !");
            if (page == Page::dataset && collection_key.starts_with(entry.all.key + "/")) ImGui::SetNextItemOpen(true, ImGuiCond_Once);
            const auto flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_SpanAvailWidth | (page == Page::dataset && collection_key == entry.all.key ? ImGuiTreeNodeFlags_Selected : 0) | (entry.concepts.empty() && entry.ready ? ImGuiTreeNodeFlags_Leaf : 0);
            const bool open  = ImGui::TreeNodeEx("##Root", flags, "%s", label.c_str());
            if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen()) selected = entry.all.key;
            if (open) {
                for (const auto& item : entry.concepts) {
                    const auto text = std::format("{}  \xC2\xB7  {}", item.name, item.images.size());
                    if (ImGui::Selectable(text.c_str(), page == Page::dataset && collection_key == item.key)) selected = item.key;
                }
                if (!entry.ready && ImGui::TreeNodeEx("##Issues", ImGuiTreeNodeFlags_SpanAvailWidth, "Index error")) {
                    if (!entry.error.empty()) ImGui::TextWrapped("%s", entry.error.c_str());
                    for (const auto& conflict : entry.conflicts) {
                        ImGui::TextColored({0.95F, 0.49F, 0.42F, 1}, "Duplicate image copies");
                        for (const auto& path : conflict) ImGui::TextWrapped("%s", path.string().c_str());
                        ImGui::Spacing();
                    }
                    ImGui::TreePop();
                }
                ImGui::TreePop();
            }
            ImGui::PopID();
        }
        return selected;
    }

    void UserInterface::bottom_controls(const float scale, const ImVec2 size, const ControlLayout& layout, const Picture& image) {
        if (page != Page::generation && !image.texture) return;
        const float y     = size.y - (bottom_margin + control_height) * scale;
        const float right = size.x - bottom_margin * scale;
        const float x     = right - layout.right_width;
        ImGui::SetNextWindowPos({x, y});
        ImGui::SetNextWindowSize({layout.right_width, control_height * scale});
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {0, 0});
        ImGui::Begin("##ImageControls", nullptr, overlay | ImGuiWindowFlags_NoBackground);
        ImGui::PopStyleVar();
        if (image.texture) control_shade({x, y}, {right, y + control_height * scale}, scale);
        float dimensions_x = x;
        auto* draw         = ImGui::GetWindowDrawList();
        if (layout.different) {
            const float info_x = layout.image_above ? right - layout.image_label_width : dimensions_x;
            const float info_y = layout.image_above ? y - (control_height + 8) * scale : y;
            if (layout.image_above) control_shade({info_x, info_y}, {right, info_y + control_height * scale}, scale);
            draw->PushClipRect({0, 0}, size, false);
            draw->AddText({info_x, info_y + (control_height * scale - ImGui::GetFontSize()) / 2}, ImGui::GetColorU32(ImVec4{0.54F, 0.55F, 0.61F, 1}), std::format("{} {} \xC3\x97 {} \xE2\x86\x92", image.preview ? "Preview" : "Image", image.width, image.height).c_str());
            draw->PopClipRect();
            if (!layout.image_above) dimensions_x += layout.image_label_width;
        }
        if (page == Page::generation) {
            const ImVec2 dimensions_origin{dimensions_x, y};
            if (layout.different) {
                draw->AddText({dimensions_x + 4 * scale, y + (control_height * scale - ImGui::GetFontSize()) / 2}, ImGui::GetColorU32(ImGuiCol_TextDisabled), "Next");
                dimensions_x += ImGui::CalcTextSize("Next").x + 12 * scale;
            }
            ImGui::SetCursorScreenPos({dimensions_x, y});
            constexpr int dimension_step = 64;
            number_field("##Width", "", ImGuiDataType_S32, &draft.width, {48 * scale, control_height * scale}, &dimension_step, "%d", scale, parameter_edit);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Next image width\nUp / Down: 64 px\nMiddle-click: swap dimensions\nRight-click: presets");
            draw->AddText({dimensions_x + 51 * scale, y + (control_height * scale - ImGui::GetFontSize()) / 2}, ImGui::GetColorU32(ImGuiCol_TextDisabled), "\xC3\x97");
            ImGui::SetCursorScreenPos({dimensions_x + 64 * scale, y});
            number_field("##Height", "", ImGuiDataType_S32, &draft.height, {48 * scale, control_height * scale}, &dimension_step, "%d", scale, parameter_edit);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Next image height\nUp / Down: 64 px\nMiddle-click: swap dimensions\nRight-click: presets");
            const bool dimensions_hovered = ImGui::IsWindowHovered(ImGuiHoveredFlags_AllowWhenBlockedByActiveItem) && ImGui::IsMouseHoveringRect(dimensions_origin, ImGui::GetItemRectMax());
            if (dimensions_hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Middle)) {
                commit_parameters();
                std::swap(draft.width, draft.height);
            }
            if (dimensions_hovered && ImGui::IsMouseReleased(ImGuiMouseButton_Right)) {
                commit_parameters();
                ImGui::OpenPopup("Resolution presets");
            }
            if (ImGui::BeginPopup("Resolution presets")) {
                for (const auto& [label, width, height] : std::array{std::tuple{"Portrait  1024 x 1536", 1024, 1536}, std::tuple{"Square  1024 x 1024", 1024, 1024}, std::tuple{"Landscape  1536 x 1024", 1536, 1024}}) {
                    if (ImGui::MenuItem(label, nullptr, draft.width == width && draft.height == height)) {
                        draft.width  = width;
                        draft.height = height;
                    }
                }
                if (image.texture) {
                    ImGui::Separator();
                    if (ImGui::MenuItem("Use image size")) {
                        draft.width  = image.width;
                        draft.height = image.height;
                    }
                }
                ImGui::EndPopup();
            }
        } else if (image.texture) {
            ImGui::SetCursorScreenPos({dimensions_x, y});
            const auto label = std::format("{} \xC3\x97 {}", image.width, image.height);
            ImGui::Dummy({ImGui::CalcTextSize(label.c_str()).x, control_height * scale});
            draw->AddText({dimensions_x, y + (control_height * scale - ImGui::GetFontSize()) / 2}, ImGui::GetColorU32(ImGuiCol_TextDisabled), label.c_str());
        }
        if (image.texture) {
            ImGui::SetCursorScreenPos({right - 104 * scale, y});
            const ImVec2 dimensions{float(image.width), float(image.height)};
            const auto& io         = ImGui::GetIO();
            const ImVec2 available = page == Page::dataset && viewing == View::browse ? ImVec2{canvas_size.x * io.DisplayFramebufferScale.x, canvas_size.y * io.DisplayFramebufferScale.y} : view_available;
            const float fitted     = std::min(available.x / dimensions.x, available.y / dimensions.y);
            if (page == Page::dataset && viewing == View::browse) view.update(available, dimensions, frame_time);
            if (text_button("##View", std::format("{}{:.0f}%", view.fit ? "Fit \xC2\xB7 " : "", view.zoom * 100).c_str(), scale, 104 * scale)) {
                if (page == Page::dataset && !repaint) viewing = View::inspect;
                const double now = glfwGetTime();
                view.scale_to(std::max(1.0F, fitted), {}, dimensions, fitted >= 1, now);
                animate_until = std::max(animate_until, now + 0.12);
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Left-click: view at 100%%\nMiddle-click: fit image\nFit is the minimum zoom\nImage: scroll to zoom, drag to pan, double-click to toggle fit / 100%%");
                if (ImGui::IsMouseClicked(ImGuiMouseButton_Middle)) {
                    const double now = glfwGetTime();
                    view.scale_to(fitted, {}, dimensions, true, now);
                    animate_until = std::max(animate_until, now + 0.12);
                }
            }
        }
        ImGui::End();
    }

    void UserInterface::classifier_controls(const float scale, const ImVec2 size, const ControlLayout& layout, const Picture& image) {
        if (layout.classifier_width <= 0 || layout.classifier_height <= 0) return;
        const ImVec2 origin{dataset_sidebar.width * dataset_sidebar.amount + bottom_margin * scale, size.y - layout.classifier_bottom - layout.classifier_height};
        const float row_height = control_height * scale;
        ImGui::SetNextWindowPos(origin);
        ImGui::SetNextWindowSize({layout.classifier_width, layout.classifier_height});
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {0, 0});
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0);
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {0, 0});
        ImGui::PushStyleVar(ImGuiStyleVar_ScrollbarSize, 3 * scale);
        ImGui::PushStyleColor(ImGuiCol_ScrollbarBg, {0, 0, 0, 0});
        ImGui::Begin("##ClassifierControls", nullptr, (overlay & ~ImGuiWindowFlags_NoScrollbar) | ImGuiWindowFlags_NoBackground);
        if (image.texture) control_shade(origin, {origin.x + layout.classifier_width, origin.y + layout.classifier_height}, scale);
        for (const auto& row : layout.classifiers) {
            ImGui::PushID(row.id.data(), row.id.data() + row.id.size());
            ImGui::BeginGroup();
            const auto position      = ImGui::GetCursorScreenPos();
            const float width        = ImGui::GetContentRegionAvail().x;
            const auto detail        = std::format("\xC2\xB7 {}", row.verdict);
            const float detail_width = ImGui::CalcTextSize(detail.c_str()).x;
            const float name_width   = std::min(ImGui::CalcTextSize(row.id.data(), row.id.data() + row.id.size()).x, std::max(0.0F, width - 36 * scale - std::min(detail_width, width * .6F)));
            const bool hovered       = ImGui::IsWindowHovered() && ImGui::IsMouseHoveringRect(position, {position.x + width, position.y + row_height});
            text_button("##Toggle", row.id.data(), scale, name_width + 24 * scale, row.enabled || hovered);
            ImGui::SameLine(0, 0);
            const auto verdict_position = ImGui::GetCursorScreenPos();
            ImGui::Dummy({std::max(1.0F, width - name_width - 24 * scale), row_height});
            control_text(detail, {verdict_position.x, position.y + (row_height - ImGui::GetFontSize()) / 2}, position.x + width - 12 * scale, row.ink, scale);
            ImGui::EndGroup();
            if (ImGui::IsItemHovered()) {
                if (row.available && ImGui::IsMouseClicked(ImGuiMouseButton_Middle)) {
                    std::unique_lock<std::mutex> lock;
                    if (runtime) lock = std::unique_lock{runtime->session.mutex};
                    if (row.enabled) std::erase(classification.enabled, row.id);
                    else classification.enabled.emplace_back(row.id);
                    animate_until = std::max(animate_until, glfwGetTime() + 0.16);
                }
                ImGui::BeginTooltip();
                ImGui::TextUnformatted(row.id.data(), row.id.data() + row.id.size());
                ImGui::TextDisabled(row.available ? (row.enabled ? "Next image: ON" : "Next image: OFF") : "Historical result only; model unavailable at startup");
                if (row.available) ImGui::TextDisabled("Middle-click: enable / disable for subsequent images");
                ImGui::TextDisabled("Result belongs to the displayed image");
                ImGui::TextColored(row.ink, "%s", row.result ? row.verdict.c_str() : row.verdict == "\xE2\x80\xA6" ? "Pending" : "Not checked");
                if (row.result) {
                    ImGui::Text("YES threshold: %.3f", row.result->threshold);
                    if (!row.result->error.empty()) ImGui::TextWrapped("%s", row.result->error.c_str());
                    for (std::size_t i = 0; i < row.result->scores.size(); ++i) ImGui::Text("%s: %.5f", row.result->classes[i].c_str(), row.result->scores[i]);
                }
                ImGui::EndTooltip();
            }
            ImGui::PopID();
        }
        ImGui::End();
        ImGui::PopStyleColor();
        ImGui::PopStyleVar(4);
    }

    void UserInterface::draw() {
        if (parameter_edit.id && ImGui::GetActiveID() != parameter_edit.id) commit_parameters();
        refresh_at                = std::numeric_limits<double>::infinity();
        frame_time                = glfwGetTime();
        const float scale         = renderer.dpi;
        const auto size           = ImGui::GetIO().DisplaySize;
        const float interpolation = std::min(1.0F, ImGui::GetIO().DeltaTime / 0.045F);
        for (auto [panel, maximum, fraction] : {std::tuple{&dataset_sidebar, 300.0F, 0.24F}, std::tuple{&prompt_sidebar, 800.0F, 0.40F}}) {
            panel->amount = std::lerp(panel->amount, float(panel->open), interpolation);
            if (std::abs(panel->amount - float(panel->open)) < 0.01F) panel->amount = float(panel->open);
            panel->width = std::min(maximum * scale, size.x * fraction);
        }
        canvas_origin     = {dataset_sidebar.width * dataset_sidebar.amount, 0};
        canvas_size       = {std::max(1.0F, size.x - dataset_sidebar.width * dataset_sidebar.amount - prompt_sidebar.width * prompt_sidebar.amount), size.y};
        std::string error = library.error.empty() ? action_error : library.error;
        if (runtime) {
            auto& session = runtime->session;
            const std::lock_guard lock{session.mutex};
            session.preview_enabled = preview_enabled;
            session.preview_visible = renderer.visible && session.active && ((!session.active->source && page == Page::generation) || (repaint && repaint->result.task == session.active->id && (viewing == View::comparison || viewing == View::result)));
            if (!session.error.empty()) error = session.error;
        }
        const auto editor_state = [&] {
            const PromptEditor* editor = page == Page::generation ? &prompt_editor : repaint ? &repaints.at(repaint->source.sha)->editor : nullptr;
            return std::pair{editor && editor->escape_owned, editor && editor->focus_input};
        };
        const bool dismissing = escape_owned || parameter_edit.id || ImGui::IsAnyItemActive() || ImGui::GetDragDropPayload() || ImGui::GetIO().WantTextInput || (prompt_sidebar.open && editor_state().first) || ImGui::IsPopupOpen(nullptr, ImGuiPopupFlags_AnyPopupId | ImGuiPopupFlags_AnyPopupLevel);
        if (!renderer.visible) return;
        canvas(scale, size);
        sidebar(true, scale, size, {});
        Picture image;
        if (page == Page::generation) image = resolve_output(generation);
        else if (repaint) image = viewing == View::result || (viewing == View::comparison && (repaint->result.texture || repaint->result.saved)) ? resolve_output(repaint->result, Role::result) : resolve_image(repaint->source, Role::source);
        else if (collection && root->ready && !collection->images.empty() && positions.contains(collection_key)) image = resolve_image(collection->images[positions.at(collection_key).index]);
        const auto controls = control_layout(scale, size, image);
        sidebar(false, scale, size, image);
        bottom_controls(scale, size, controls, image);
        classifier_controls(scale, size, controls, image);
        top_strip(scale, size);
        preset_dialogs(scale);
        if (!error.empty() && error != shown_error) {
            shown_error = error;
            ImGui::OpenPopup("Genesia error");
        }
        ImGui::SetNextWindowSize({540 * scale, 0});
        if (ImGui::BeginPopupModal("Genesia error", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::TextWrapped("%s", shown_error.c_str());
            ImGui::Spacing();
            if (ImGui::Button("Close") || ImGui::IsKeyPressed(ImGuiKey_Escape, false)) ImGui::CloseCurrentPopup();
            ImGui::EndPopup();
        }
        if (page == Page::generation && !ImGui::GetTopMostPopupModal() && ImGui::Shortcut(ImGuiMod_Ctrl | ImGuiKey_S, ImGuiInputFlags_RouteGlobal)) save_prompt();
        if (ImGui::Shortcut(ImGuiKey_F11, ImGuiInputFlags_RouteGlobal | ImGuiInputFlags_RouteOverActive)) window.toggle_fullscreen();
        if (!dismissing && !ImGui::IsPopupOpen(nullptr, ImGuiPopupFlags_AnyPopupId | ImGuiPopupFlags_AnyPopupLevel) && ImGui::Shortcut(ImGuiKey_Escape, ImGuiInputFlags_RouteGlobal)) window.request_close();
        escape_owned = parameter_edit.id || ImGui::IsAnyItemActive() || ImGui::GetDragDropPayload() || (prompt_sidebar.open && editor_state().first) || ImGui::IsPopupOpen(nullptr, ImGuiPopupFlags_AnyPopupId | ImGuiPopupFlags_AnyPopupLevel);
        if (!ImGui::GetIO().WantTextInput && !(prompt_sidebar.open && editor_state().second) && !ImGui::IsPopupOpen(nullptr, ImGuiPopupFlags_AnyPopupId | ImGuiPopupFlags_AnyPopupLevel)) {
            if (ImGui::Shortcut(ImGuiKey_Tab, ImGuiInputFlags_RouteGlobal)) prompt_sidebar.open = !prompt_sidebar.open;
            if (ImGui::Shortcut(ImGuiKey_GraveAccent, ImGuiInputFlags_RouteGlobal)) dataset_sidebar.open = !dataset_sidebar.open;
        }
        const auto& io = ImGui::GetIO();
        if (io.MouseDelta.x || io.MouseDelta.y || io.MouseWheel || io.InputQueueCharacters.Size || ImGui::IsAnyItemActive()) animate_until = glfwGetTime() + 0.16;
    }
} // namespace genesia::editor
