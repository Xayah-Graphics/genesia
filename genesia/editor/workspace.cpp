module;
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_internal.h>
module genesia.editor.workspace;
import genesia.project;
import genesia.generation.settings;
import genesia.prompt.library;
import genesia.generation.output;
import genesia.editor.platform.window;
import genesia.editor.graphics.renderer;
import genesia.editor.graphics.interop;
import genesia.editor.graphics.bridge;
import genesia.editor.viewing.canvas;
import genesia.editor.panels.application;
import genesia.editor.panels.sidebars;
import genesia.editor.panels.loras;
import std;
import genesia.io.files;

namespace genesia::editor {

    Workspace::Workspace(prompts::Preset preset, std::shared_ptr<const prompt::Catalog> catalog, std::shared_ptr<const prompts::Library> prompt_library, WindowPlatform& platform, Renderer& display) : catalog{std::move(catalog)}, prompt_library{std::move(prompt_library)}, preset_name{std::move(preset.name)}, window{platform}, renderer{display}, prompt_panel{*this->prompt_library, display, platform}, runtime{display.device}, textures{display}, prompt{std::move(preset.recipe)}, tag_search{*this->catalog} {
        prompt_editor.reset(prompt.free);
        try {
            const auto directory = std::filesystem::path{project::assets} / "loras";
            std::filesystem::create_directories(directory);
            for (const auto& entry : std::filesystem::directory_iterator{directory})
                if (entry.is_regular_file() && entry.path().extension() == ".safetensors") loras.push_back({files::utf8(entry.path().filename())});
            std::ranges::sort(loras, {}, &LoraSettings::file);
        } catch (const std::exception& error) {
            action_error = error.what();
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

    Workspace::~Workspace() {
        runtime.session.shutdown();
        textures.shutdown();
        if (generation.texture) renderer.retire(generation.texture);
    }

    void Workspace::receive() {
        auto delivered = runtime.session.drain();
        for (const auto& event : delivered.events) runtime.web.receive(event);
        {
            auto& session  = runtime.session;
            session_state  = session.snapshot();
            auto& events   = delivered.events;
            auto& previews = delivered.previews;
            for (const auto& frame : previews) {
                const auto presentation = std::static_pointer_cast<const PresentedFrame>(frame.frame);
                auto& source            = presentation->bridge->slots[presentation->slot];
                const bool final_ready  = std::ranges::any_of(events, [&](const runtime::Event& event) { return event.kind == runtime::EventKind::generated && event.id == frame.id; });
                if ((generation.task && frame.id < *generation.task) || !preview_enabled || !renderer.visible || final_ready || (generation.task == frame.id && (generation.record || frame.step <= generation.step))) {
                    renderer.discard(*source.timeline, presentation->ready);
                    continue;
                }
                if (generation.task != frame.id) begin_generation(frame.id, frame.width, frame.height);
                if (generation.texture) renderer.retire(generation.texture);
                generation.texture = renderer.texture({static_cast<std::uint32_t>(frame.width), static_cast<std::uint32_t>(frame.height)});
                renderer.copy(generation.texture, source.buffer, *source.timeline, presentation->ready);
                generation.width   = frame.width;
                generation.height  = frame.height;
                generation.preview = true;
                generation.step    = frame.step;
            }
            for (auto& event : events) {
                if (event.kind == runtime::EventKind::task) {
                    task = event.task;
                    if (task->state == runtime::State::failed) {
                        action_error = task->error;
                        shown_error.clear();
                    }
                    continue;
                }
                const bool current = !generation.task || event.id >= *generation.task;
                if (current && generation.task != event.id) begin_generation(event.id, event.record.parameters.width, event.record.parameters.height);
                if (event.kind == runtime::EventKind::generated) {
                    const auto presentation = std::static_pointer_cast<const PresentedFrame>(event.frame);
                    auto& source            = presentation->bridge->slots[presentation->slot];
                    if (current) {
                        if (generation.texture) renderer.retire(generation.texture);
                        generation.width   = event.record.parameters.width;
                        generation.height  = event.record.parameters.height;
                        generation.texture = renderer.texture({static_cast<std::uint32_t>(generation.width), static_cast<std::uint32_t>(generation.height)});
                        renderer.copy(generation.texture, source.buffer, *source.timeline, presentation->ready);
                        generation.record  = std::move(event.record);
                        generation.preview = false;
                    } else renderer.discard(*source.timeline, presentation->ready);
                } else if (current) {
                    if (generation.texture && generation.preview) renderer.retire(std::exchange(generation.texture, 0));
                    textures.adopt(event.record, generation.texture);
                    generation.saved   = event.record.path;
                    generation.record  = std::move(event.record);
                    generation.preview = false;
                }
            }
        }
        const bool refreshed = textures.receive();
        if (!textures.history_error.empty()) action_error = textures.history_error;
        if (refreshed && textures.history_error.empty() && generation.saved && !std::ranges::contains(textures.history, *generation.saved)) {
            generation.saved.reset();
            generation.record.reset();
        }
        if (!textures.history.empty()) {
            const auto selected = std::ranges::find(textures.history, position.selected);
            if (selected != textures.history.end()) {
                position.index = std::size_t(selected - textures.history.begin());
                if (refreshed) position.scroll = float(position.index);
            } else {
                position.index    = position.selected.empty() ? textures.history.size() - 1 : std::min(position.index, textures.history.size() - 1);
                position.selected = textures.history[position.index];
                position.scroll   = float(position.index);
            }
        } else position = {};
    }

    void Workspace::update_generation() {
        if (!continuous_generation || !task) return;
        const auto& current = *task;
        if (page != Page::generation || session_state.finished || !session_state.error.empty() || current.stopping || current.state == runtime::State::stopped || current.state == runtime::State::failed) {
            continuous_generation = false;
            return;
        }
        if (session_state.active || current.state != runtime::State::complete || ImGui::GetTopMostPopupModal()) return;
        try {
            continuous_generation = submit();
        } catch (const std::exception& failure) {
            continuous_generation = false;
            action_error          = failure.what();
            shown_error.clear();
        }
    }

    void Workspace::update_web() {
        auto& web = runtime.web;
        if (!web.enabled) return;
        auto state = runtime.session.snapshot();
        std::string unavailable, failure;
        if (!state.error.empty()) unavailable = state.error;
        else if (state.finished) unavailable = "主机正在关闭";
        else if (page != Page::generation) unavailable = "暂不可用 · 请将主机切换到文生图界面";
        else if (ImGui::GetTopMostPopupModal()) unavailable = "暂不可用 · 请先处理主机对话框";
        if (web.requested) {
            if (!unavailable.empty()) failure = unavailable;
            else if (state.active) failure = "主机忙碌";
            else {
                try {
                    if (!submit()) failure = preset_error;
                } catch (const std::exception& error) {
                    failure = error.what();
                }
            }
            web.requested = false;
            state         = runtime.session.snapshot();
        }
        web.update(state, std::move(unavailable), std::move(failure));
    }

    void Workspace::open_history() {
        commit_parameters();
        if (!prompt_editor.commit(prompt.free, *catalog)) {
            prompt_sidebar.open = true;
            return;
        }
        textures.refresh();
        prompt_editor.suspend();
        generation_view = view;
        page            = Page::history;
        viewing         = View::browse;
        view            = {};
        if (generation.saved) {
            const auto found = std::ranges::find(textures.history, *generation.saved);
            if (found != textures.history.end()) {
                center_image(std::size_t(found - textures.history.begin()));
                position.scroll = float(position.index);
            }
        }
        window.redraw = true;
    }
    void Workspace::center_image(const std::size_t index) {
        position.index    = index;
        position.selected = textures.history[index];
        animate_until     = glfwGetTime() + 0.3;
    }
    void Workspace::back() {
        commit_parameters();
        if (viewing == View::inspect) {
            viewing = View::browse;
            view    = {};
        } else {
            page = Page::generation;
            view = generation_view;
        }
    }
    Workspace::Picture Workspace::resolve_image(const std::filesystem::path& file) const {
        Picture picture{.width = 1, .height = 1, .file = file};
        const auto cached = textures.entries.find(file);
        if (cached != textures.entries.end()) {
            picture.width        = cached->second.width;
            picture.height       = cached->second.height;
            picture.texture      = cached->second.texture;
            picture.record       = cached->second.record ? &*cached->second.record : nullptr;
            picture.record_error = cached->second.record_error;
        }
        return picture;
    }

    Workspace::Picture Workspace::resolve_generation() const {
        if (generation.saved) {
            auto picture = resolve_image(*generation.saved);
            if (picture.texture) return picture;
        }
        return {generation.texture, generation.width, generation.height, generation.record ? &*generation.record : nullptr, generation.saved, generation.preview};
    }
    void Workspace::commit_parameters() {
        if (!parameter_edit.id) return;
        auto* input = ImGui::GetInputTextState(parameter_edit.id);
        ImGui::DataTypeApplyFromText(input->TextA.Data, parameter_edit.type, parameter_edit.value, ImGui::DataTypeGetInfo(parameter_edit.type)->ScanFmt);
        // The value is committed; do not replay the text after a parameter action.
        input->ID = 0;
        if (ImGui::GetCurrentContext()->InputTextDeactivatedState.ID == parameter_edit.id) ImGui::GetCurrentContext()->InputTextDeactivatedState.ID = 0;
        ImGui::ClearActiveID();
        parameter_edit = {};
    }

    bool Workspace::prepare_prompt() {
        if (!prompt_editor.commit(prompt.free, *catalog)) {
            prompt_sidebar.open = true;
            preset_error        = "Invalid free prompt";
            for (const auto& side : prompt_editor.groups)
                for (const auto& group : side)
                    if (group.error) preset_error = group.error->message;
            if (prompt_editor.addition.error) preset_error = prompt_editor.addition.error->message;
            return false;
        }
        prompt_panel.update(prompt, *catalog);
        if (!prompt_panel.ready) {
            preset_error = prompt_panel.error.empty() ? "Wait for the prompt previews to finish loading." : prompt_panel.error;
            return false;
        }
        return true;
    }

    bool Workspace::save_prompt() {
        if (!prompt_editor.commit(prompt.free, *catalog)) return false;
        try {
            prompts::write_preset(*prompt_library, {preset_name, prompt}, *catalog);
            preset_error.clear();
            return true;
        } catch (const std::exception& failure) {
            preset_error = failure.what();
            return false;
        }
    }

    void Workspace::switch_preset(std::string name) {
        try {
            auto next = prompts::read_preset(*prompt_library, std::move(name), *catalog);
            ImGui::ClearActiveID();
            preset_name = std::move(next.name);
            prompt      = std::move(next.recipe);
            prompt_editor.reset(prompt.free);
            prompt_panel.update(prompt, *catalog);
            preset_error.clear();
        } catch (const std::exception& failure) {
            preset_error = failure.what();
        }
    }

    void Workspace::begin_generation(const std::uint64_t task, const int width, const int height) {
        if (generation.texture) renderer.retire(generation.texture);
        generation = {.task = task, .width = width, .height = height};
        if (page == Page::generation) view = {};
        else generation_view = {};
    }

    generation::Settings Workspace::parameters() const {
        auto result     = draft;
        result.positive = prompt_panel.composition.text[0];
        result.negative = prompt_panel.composition.text[1];
        for (const auto& lora : loras)
            if (lora.active) result.loras.push_back({lora.file, {}, lora.weight, lora.start / 100});
        return result;
    }

    bool Workspace::submit() {
        commit_parameters();
        if (page != Page::generation || !prepare_prompt()) return false;
        if (random_seed) {
            std::random_device random;
            seed = std::uniform_int_distribution<std::uint64_t>{}(random);
        }
        task          = runtime.session.submit({parameters(), seed});
        session_state = runtime.session.snapshot();
        animate_until = glfwGetTime() + 0.2;
        return true;
    }

    void Workspace::draw() {
        prompt_panel.update(prompt, *catalog);
        if (parameter_edit.id && ImGui::GetActiveID() != parameter_edit.id) commit_parameters();
        refresh_at                = std::numeric_limits<double>::infinity();
        frame_time                = glfwGetTime();
        const float scale         = renderer.dpi;
        const auto size           = ImGui::GetIO().DisplaySize;
        const float interpolation = std::min(1.0F, ImGui::GetIO().DeltaTime / 0.045F);
        const float target        = float(prompt_sidebar.open || prompt_panel.incoming.has_value());
        prompt_sidebar.amount     = std::lerp(prompt_sidebar.amount, target, interpolation);
        if (std::abs(prompt_sidebar.amount - target) < 0.01F) prompt_sidebar.amount = target;
        prompt_sidebar.width = std::min(800.0F * scale, size.x * 0.40F);
        canvas_size          = {std::max(1.0F, size.x - prompt_sidebar.width * prompt_sidebar.amount), size.y};
        {
            auto& session          = runtime.session;
            const bool generating  = session_state.active.has_value();
            const bool visible     = renderer.visible && generating && page == Page::generation;
            const bool web_preview = runtime.web.watching() && generating;
            session.configure_preview(preview_enabled || web_preview, visible || web_preview);
        }
        if (!renderer.visible) return;
        canvas(*this, scale, size);
        Picture image;
        if (page == Page::generation) image = resolve_generation();
        else if (textures.history_ready && !textures.history.empty()) image = resolve_image(textures.history[position.index]);
        const auto controls = control_layout(*this, scale, size, image);
        sidebar(*this, scale, size, image);
        if (!window.drop_error.empty()) {
            action_error = std::exchange(window.drop_error, {});
            shown_error.clear();
        }
        // PNG drops outside a preview rectangle cancel the preview operation.
        window.dropped.clear();
        bottom_controls(*this, scale, size, controls, image);
        lora_controls(*this, scale, size, controls);
        top_strip(*this, scale, size);
        preset_dialogs(*this, scale);
        const bool inspecting = page != Page::generation && viewing == View::inspect;
        if (inspecting && image.texture && image.file && !image.preview && !pending_delete && !session_state.active && !parameter_edit.id && !ImGui::IsAnyItemActive() && !ImGui::GetDragDropPayload() && !ImGui::GetIO().WantTextInput && !ImGui::IsPopupOpen(nullptr, ImGuiPopupFlags_AnyPopupId | ImGuiPopupFlags_AnyPopupLevel) && (ImGui::Shortcut(ImGuiKey_Delete, ImGuiInputFlags_RouteGlobal) || ImGui::Shortcut(ImGuiKey_KeypadDecimal, ImGuiInputFlags_RouteGlobal))) {
            pending_delete = *image.file;
            ImGui::OpenPopup("Delete image?");
        }
        ImGui::SetNextWindowPos({size.x / 2, size.y / 2}, ImGuiCond_Appearing, {0.5F, 0.5F});
        ImGui::SetNextWindowSize({480 * scale, 0});
        if (ImGui::BeginPopupModal("Delete image?", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::TextWrapped("%s", files::utf8(*pending_delete).c_str());
            ImGui::Spacing();
            ImGui::TextWrapped("Permanently delete this file? This cannot be undone.");
            ImGui::Spacing();
            const bool cancel = ImGui::Button("Cancel") || ImGui::IsKeyPressed(ImGuiKey_Escape, false);
            ImGui::SetItemDefaultFocus();
            ImGui::SameLine();
            const bool confirm = ImGui::Button("Delete permanently");
            if (cancel) {
                pending_delete.reset();
                ImGui::CloseCurrentPopup();
            } else if (confirm) {
                try {
                    std::filesystem::remove(*pending_delete);
                    textures.discard(*pending_delete);
                    if (generation.saved == pending_delete) {
                        generation.saved.reset();
                        generation.record.reset();
                    }
                    if (textures.history.empty()) {
                        position = {};
                        viewing  = View::browse;
                    } else {
                        center_image(std::min(position.index, textures.history.size() - 1));
                        position.scroll = float(position.index);
                    }
                    view = {};
                    pending_delete.reset();
                } catch (const std::exception& failure) {
                    action_error = failure.what();
                    pending_delete.reset();
                }
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
        }
        const auto& error = session_state.error.empty() ? action_error : session_state.error;
        if (!error.empty() && error != shown_error && !ImGui::IsPopupOpen("Delete image?")) {
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
        if (ImGui::Shortcut(ImGuiKey_F11, ImGuiInputFlags_RouteGlobal | ImGuiInputFlags_RouteOverActive) || (!ImGui::GetIO().WantTextInput && !(page == Page::generation && prompt_sidebar.open && prompt_editor.focus_input) && !ImGui::IsPopupOpen(nullptr, ImGuiPopupFlags_AnyPopupId | ImGuiPopupFlags_AnyPopupLevel) && ImGui::Shortcut(ImGuiKey_F, ImGuiInputFlags_RouteGlobal))) window.toggle_fullscreen();
        if (!ImGui::GetIO().WantTextInput && !(page == Page::generation && prompt_sidebar.open && prompt_editor.focus_input) && !ImGui::IsPopupOpen(nullptr, ImGuiPopupFlags_AnyPopupId | ImGuiPopupFlags_AnyPopupLevel)) {
            if (ImGui::Shortcut(ImGuiKey_Tab, ImGuiInputFlags_RouteGlobal)) prompt_sidebar.open = !prompt_sidebar.open;
        }
        if (page == Page::history || generation.saved) refresh_at = std::min(refresh_at, frame_time + 2);
        const auto& io = ImGui::GetIO();
        if (io.MouseDelta.x || io.MouseDelta.y || io.MouseWheel || io.InputQueueCharacters.Size || ImGui::IsAnyItemActive()) animate_until = glfwGetTime() + 0.16;
    }
} // namespace genesia::editor
