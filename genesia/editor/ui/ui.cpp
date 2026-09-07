module;
#include <Windows.h>

#include <GLFW/glfw3.h>

#include "../../core/sdxl/control.h"
#include <genesia/cuda.h>
#include <imgui.h>
#include <misc/cpp/imgui_stdlib.h>
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
        constexpr float top_strip_height   = 36;
        enum class Icon { history, collapse, edit, settings, fit };

        bool icon_button(const char* id, const Icon icon, const char* tooltip, const float scale, const bool selected = false) {
            const ImVec2 p       = ImGui::GetCursorScreenPos();
            const float size     = 36 * scale;
            const bool clicked   = ImGui::InvisibleButton(id, {size, size});
            const bool hovered   = ImGui::IsItemHovered();
            const auto key       = ImGui::GetID(id);
            auto* storage        = ImGui::GetStateStorage();
            const float previous = storage->GetFloat(key);
            const float alpha    = std::lerp(previous, (hovered || selected) ? 1.0F : 0.0F, std::min(1.0F, ImGui::GetIO().DeltaTime / 0.12F));
            storage->SetFloat(key, alpha);
            auto* draw = ImGui::GetWindowDrawList();
            draw->AddRectFilled(p, {p.x + size, p.y + size}, ImGui::GetColorU32(ImVec4{0.6F, 0.6F, 0.8F, alpha * 0.12F}), 8 * scale);
            const ImVec2 c{p.x + size / 2, p.y + size / 2};
            const float r    = 7 * scale;
            const float line = 1.4F * scale;
            const ImU32 ink  = ImGui::GetColorU32(selected ? ImVec4{0.68F, 0.68F, 1, 1} : ImVec4{0.78F + alpha * 0.15F, 0.79F + alpha * 0.14F, 0.84F + alpha * 0.10F, 1});
            switch (icon) {
            case Icon::history:
                draw->AddRect({c.x - r, c.y - r}, {c.x + r, c.y + r}, ink, 2 * scale, 0, line);
                draw->AddLine({c.x - r + 4 * scale, c.y - r}, {c.x - r + 4 * scale, c.y + r}, ink, line);
                break;
            case Icon::collapse: draw->AddLine({c.x - r, c.y}, {c.x + r, c.y}, ink, line); break;
            case Icon::edit:
                draw->AddLine({c.x - r, c.y + r}, {c.x + r, c.y - r}, ink, 3 * scale);
                draw->AddLine({c.x - r, c.y + r + 2 * scale}, {c.x + r, c.y + r + 2 * scale}, ink, line);
                break;
            case Icon::settings:
                for (int i = -1; i <= 1; ++i) {
                    draw->AddLine({c.x - r, c.y + i * 5 * scale}, {c.x + r, c.y + i * 5 * scale}, ink, line);
                    draw->AddCircleFilled({c.x + (i == 0 ? 3 : -3) * scale, c.y + i * 5 * scale}, 2.4F * scale, ink);
                }
                break;
            case Icon::fit:
                draw->AddRect({c.x - r, c.y - r}, {c.x + r, c.y + r}, ink, 2 * scale, 0, line);
                draw->AddRect({c.x - 3 * scale, c.y - 3 * scale}, {c.x + 3 * scale, c.y + 3 * scale}, ink, 0, 0, line);
                break;
            }
            if (hovered) ImGui::SetTooltip("%s", tooltip);
            return clicked;
        }

        bool strip_button(const char* id, const char* label, const float scale) {
            const auto text    = ImGui::CalcTextSize(label);
            const auto origin  = ImGui::GetCursorScreenPos();
            const bool clicked = ImGui::InvisibleButton(id, {text.x + 24 * scale, 27 * scale});
            const float alpha  = ImGui::IsItemActive() ? 0.85F : ImGui::IsItemHovered() ? 1.0F : 0.75F;
            const ImVec2 position{origin.x + 12 * scale, origin.y + (27 * scale - text.y) / 2};
            auto* draw = ImGui::GetWindowDrawList();
            draw->AddText({position.x + scale, position.y + scale}, ImGui::GetColorU32(ImVec4{0, 0, 0, alpha * 0.8F}), label);
            draw->AddText(position, ImGui::GetColorU32(ImVec4{0.93F, 0.93F, 0.96F, alpha}), label);
            return clicked;
        }

    } // namespace

    UserInterface::UserInterface(Configuration settings, std::filesystem::path path, WindowPlatform& platform, Renderer& display, Interop& bridge, Session& generation) : configuration{std::move(settings)}, configuration_path{std::move(path)}, window{platform}, renderer{display}, interop{bridge}, session{generation}, draft{configuration.parameters}, seed{configuration.seeds.front()} {
        ImGui::StyleColorsDark();
        auto& style          = ImGui::GetStyle();
        style.WindowRounding = 16;
        style.ChildRounding = style.FrameRounding = style.GrabRounding = 8;
        style.PopupRounding                                            = 12;
        style.WindowBorderSize                                         = 1;
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

    void UserInterface::submit() {
        if (random_seed) {
            std::random_device source;
            seed = (std::uint64_t(source()) << 32) | source();
        }
        session.enqueue(draft, seed);
        edited_since_submit = false;
        animate_until       = glfwGetTime() + 0.2;
    }

    void UserInterface::canvas(const float scale, const ImVec2 size, const float composer_height) {
        ImGui::SetNextWindowPos({0, 0});
        ImGui::SetNextWindowSize(size);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {0, 0});
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0);
        ImGui::Begin("##Canvas", nullptr, overlay | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNav);
        ImGui::InvisibleButton("##ImageArea", size);
        const bool hovered = ImGui::IsItemHovered();
        const float left   = history_amount * 300 * scale;
        const float bottom = size.y - composer_amount * (composer_height + 24 * scale);
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
            draw->AddRectFilled({minimum.x - 1, minimum.y - 1}, {maximum.x + 1, maximum.y + 1}, IM_COL32(45, 46, 55, 255));
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
        if (image_texture) {
            ImGui::SetNextWindowPos({size.x - 178 * scale, size.y - 60 * scale});
            ImGui::SetNextWindowSize({154 * scale, 38 * scale});
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {0, 0});
            ImGui::Begin("##ImageTools", nullptr, overlay | ImGuiWindowFlags_NoBackground);
            if (icon_button("##Fit", Icon::fit, "Fit image", scale)) {
                fit_image = true;
                pan       = {};
            }
            ImGui::SameLine();
            if (ImGui::Button(std::format("{:.0f}%", zoom * 100).c_str(), {72 * scale, 36 * scale})) {
                fit_image = false;
                zoom      = 1;
                pan       = {};
            }
            ImGui::End();
            ImGui::PopStyleVar();
        }
    }

    void UserInterface::top_strip(const float scale, const ImVec2 size) {
        bool active, paused, loaded, failed;
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
            else if (stage == sdxl::Stage::sampling) progress_label = std::format("Generating  {} / {}", completed, steps);
            else if (stage == sdxl::Stage::decoding || stage == sdxl::Stage::transferring || stage == sdxl::Stage::complete) progress_label = "Finishing image";
            else progress_label = "Preparing";
            if (active) progress_label += std::format("   {:.1f}s", elapsed);
        } else progress_alpha = paused || failed ? 0 : std::max(0.0F, progress_alpha - ImGui::GetIO().DeltaTime / 0.15F);
        if (working) refresh_at = now + (indeterminate ? 1.0 / 30 : 0.1);
        else if (progress_alpha > 0) refresh_at = now + 1.0 / 60;

        ImGui::SetNextWindowPos({6 * scale, 5 * scale});
        ImGui::SetNextWindowSize({size.x - 12 * scale, top_strip_height * scale});
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {6 * scale, 4 * scale});
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
                draw->AddRectFilled({x, minimum.y}, {x + segment_width, maximum.y}, ImGui::GetColorU32(ImVec4{0.59F, 0.57F, 0.95F, 0.10F}));
            } else {
                const float fraction = working ? std::min(1.0F, float(completed) / steps) : 1.0F;
                const float x        = std::lerp(minimum.x, maximum.x, fraction);
                draw->AddRectFilled(minimum, {x, maximum.y}, ImGui::GetColorU32(ImVec4{0.59F, 0.57F, 0.95F, 0.07F * progress_alpha}));
                if (x > minimum.x) draw->AddLine({x, minimum.y + 2 * scale}, {x, maximum.y - 2 * scale}, ImGui::GetColorU32(ImVec4{0.59F, 0.57F, 0.95F, 0.72F * progress_alpha}), scale);
            }
            draw->PopClipRect();
        }
        if (icon_button("##History", Icon::history, "Session history", scale * 0.75F, history_open)) history_open = !history_open;
        ImGui::SameLine(0, 4 * scale);
        if (strip_button("##Application", "GENESIA", scale)) ImGui::OpenPopup("Application");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Application menu\nEsc to exit");
        const float left_end = ImGui::GetItemRectMax().x + 4 * scale;
        if (ImGui::BeginPopup("Application")) {
            ImGui::MenuItem("Live preview", nullptr, &configuration.preview.enabled);
            ImGui::Separator();
            if (ImGui::MenuItem("Save configuration", "Ctrl+S")) {
                configuration.parameters = draft;
                configuration.seeds      = {seed};
                write_configuration(configuration, configuration_path);
            }
            if (ImGui::MenuItem("Open output folder")) ShellExecuteW(window.native_window, L"open", std::filesystem::absolute(configuration.output).c_str(), nullptr, nullptr, SW_SHOWNORMAL);
            ImGui::EndPopup();
        }
        const bool queue_visible      = queued != 0 || paused || ImGui::IsPopupOpen("Queue");
        const std::string queue_label = queue_visible ? std::format("Queue  {}", queued) : "";
        const char* stop_label        = active ? "Stop" : paused && !failed ? "Resume" : "";
        const char* latest_label      = following_latest ? "" : active ? "View live" : "Latest";
        float controls_width{};
        for (const char* label : {queue_label.c_str(), stop_label, latest_label})
            if (*label) controls_width += ImGui::CalcTextSize(label).x + 28 * scale;
        const float right_start = maximum.x - 6 * scale - controls_width;
        float x                 = right_start;
        if (queue_visible) {
            ImGui::SetCursorScreenPos({x, minimum.y + 4 * scale});
            if (strip_button("##Queue", queue_label.c_str(), scale)) ImGui::OpenPopup("Queue");
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
            if (strip_button("##Playback", stop_label, scale)) {
                if (active) session.stop();
                else session.resume();
            }
            x = ImGui::GetItemRectMax().x + 4 * scale;
        }
        if (*latest_label) {
            ImGui::SetCursorScreenPos({x, minimum.y + 4 * scale});
            if (strip_button("##Latest", latest_label, scale)) {
                following_latest = true;
                preview_step     = 0;
                if (!active && !history.empty()) {
                    selected   = history.back().id;
                    image_live = false;
                    session.load(selected, history.back().record.directory / "image.png");
                }
            }
        }
        const char* label      = paused && !active && !failed ? "Queue paused" : progress_alpha > 0 ? progress_label.c_str() : "";
        const auto text        = ImGui::CalcTextSize(label);
        const float text_left  = std::clamp((size.x - text.x) / 2, left_end + 8 * scale, std::max(left_end + 8 * scale, right_start - 8 * scale - text.x));
        const float text_right = std::min(text_left + text.x, right_start - 8 * scale);
        if (*label) {
            const float alpha = paused && !active ? 1 : progress_alpha;
            const ImVec2 position{text_left, minimum.y + (top_strip_height * scale - text.y) / 2};
            draw->PushClipRect({left_end, minimum.y}, {right_start, maximum.y}, true);
            draw->AddText({position.x + scale, position.y + scale}, ImGui::GetColorU32(ImVec4{0, 0, 0, 0.75F * alpha}), label);
            draw->AddText(position, ImGui::GetColorU32(ImVec4{0.89F, 0.89F, 0.94F, alpha}), label);
            draw->PopClipRect();
            if (image_live && selected == active_id && ImGui::IsWindowHovered() && ImGui::IsMouseHoveringRect({text_left, minimum.y}, {text_right, maximum.y})) ImGui::SetTooltip("Displayed preview: step %u", preview_step);
        }
        window.drag_regions = {};
        if (!ImGui::IsPopupOpen(nullptr, ImGuiPopupFlags_AnyPopup)) {
            if (*label) window.drag_regions = {{{left_end, 0, text_left - 4 * scale, maximum.y}, {text_right + 4 * scale, 0, right_start - 4 * scale, maximum.y}}};
            else window.drag_regions[0] = {left_end, 0, right_start - 4 * scale, maximum.y};
        }
        ImGui::End();
    }

    void UserInterface::composer(const float scale, const ImVec2 size, const float height) {
        const float expanded_width = std::min(840 * scale, size.x - (48 + 300 * history_amount) * scale);
        const float compact_width  = 192 * scale;
        const float width          = std::lerp(compact_width, expanded_width, composer_amount);
        const float center         = (size.x + history_amount * 300 * scale) / 2;
        if (composer_amount < 0.03F) {
            ImGui::SetNextWindowPos({center - compact_width / 2, size.y - 68 * scale});
            ImGui::SetNextWindowSize({compact_width, 44 * scale});
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {4 * scale, 4 * scale});
            ImGui::Begin("##Composer", nullptr, overlay);
            ImGui::SetCursorPosX((compact_width - 172 * scale) / 2);
            if (icon_button("##Edit", Icon::edit, "Edit prompt", scale)) composer_open = true;
            ImGui::SameLine();
            if (ImGui::Button("Edit prompt", {128 * scale, 36 * scale})) composer_open = true;
            ImGui::End();
            ImGui::PopStyleVar();
            return;
        }
        const float collapsed    = 44 * scale;
        const float panel_height = std::lerp(collapsed, height, composer_amount);
        const float y            = size.y - 24 * scale - panel_height;
        ImGui::SetNextWindowPos({center - width / 2, y});
        ImGui::SetNextWindowSize({width, panel_height});
        ImGui::Begin("##Composer", nullptr, overlay);
        ImGui::PushStyleVar(ImGuiStyleVar_Alpha, composer_amount);
        ImGui::TextDisabled("PROMPT");
        ImGui::SameLine(width - 60 * scale);
        ImGui::SetCursorPosY(6 * scale);
        if (icon_button("##Collapse", Icon::collapse, "Hide prompt", scale)) composer_open = false;
        ImGui::SetCursorPosY(44 * scale);
        ImGui::PushStyleColor(ImGuiCol_FrameBg, {0, 0, 0, 0});
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, {0, 4 * scale});
        if (ImGui::InputTextMultiline("##Positive", &draft.positive, {width - 40 * scale, 112 * scale}, ImGuiInputTextFlags_WordWrap)) edited_since_submit = true;
        ImGui::PopStyleVar();
        ImGui::PopStyleColor();
        if (advanced) {
            ImGui::Separator();
            ImGui::TextDisabled("NEGATIVE PROMPT");
            if (ImGui::InputTextMultiline("##Negative", &draft.negative, {width - 40 * scale, 82 * scale}, ImGuiInputTextFlags_WordWrap)) edited_since_submit = true;
            ImGui::SetNextItemWidth(110 * scale);
            if (ImGui::InputInt("Steps", &draft.steps)) edited_since_submit = true;
            ImGui::SameLine(0, 24 * scale);
            ImGui::SetNextItemWidth(110 * scale);
            if (ImGui::InputFloat("CFG", &draft.cfg, 0, 0, "%.1f")) edited_since_submit = true;
            if (width >= 740 * scale) ImGui::SameLine(0, 24 * scale);
            ImGui::SetNextItemWidth(190 * scale);
            if (ImGui::InputScalar("Seed", ImGuiDataType_U64, &seed)) {
                random_seed         = false;
                edited_since_submit = true;
            }
        }
        ImGui::SetCursorPosY(height - 58 * scale);
        if (ImGui::Button(std::format("{} x {}", draft.width, draft.height).c_str(), {136 * scale, 38 * scale})) ImGui::OpenPopup("Image size");
        if (ImGui::BeginPopup("Image size")) {
            ImGui::TextDisabled("ASPECT RATIO");
            if (ImGui::Button("Portrait  2:3")) {
                draft.width         = 1024;
                draft.height        = 1536;
                edited_since_submit = true;
                ImGui::CloseCurrentPopup();
            }
            if (ImGui::Button("Square  1:1")) {
                draft.width = draft.height = 1024;
                edited_since_submit        = true;
                ImGui::CloseCurrentPopup();
            }
            if (ImGui::Button("Landscape  3:2")) {
                draft.width         = 1536;
                draft.height        = 1024;
                edited_since_submit = true;
                ImGui::CloseCurrentPopup();
            }
            ImGui::Separator();
            ImGui::SetNextItemWidth(160 * scale);
            if (ImGui::InputInt("Width", &draft.width, 64, 256)) edited_since_submit = true;
            ImGui::SetNextItemWidth(160 * scale);
            if (ImGui::InputInt("Height", &draft.height, 64, 256)) edited_since_submit = true;
            ImGui::EndPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button(random_seed ? "Random seed" : "Seed locked", {128 * scale, 38 * scale})) random_seed = !random_seed;
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Seed: %llu\nClick to %s", static_cast<unsigned long long>(seed), random_seed ? "lock" : "randomize each generation");
        ImGui::SameLine();
        if (icon_button("##Advanced", Icon::settings, "Generation parameters", scale, advanced)) {
            advanced      = !advanced;
            animate_until = glfwGetTime() + 0.2;
        }
        ImGui::SameLine(width - 182 * scale);
        ImGui::PushStyleColor(ImGuiCol_Button, {0.48F, 0.46F, 0.83F, 1});
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, {0.57F, 0.55F, 0.94F, 1});
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, {0.42F, 0.40F, 0.76F, 1});
        bool unavailable;
        {
            const std::lock_guard lock{session.mutex};
            unavailable = session.worker_done;
        }
        ImGui::BeginDisabled(unavailable);
        if (ImGui::Button("Generate", {162 * scale, 38 * scale})) submit();
        ImGui::EndDisabled();
        ImGui::PopStyleColor(3);
        ImGui::PopStyleVar();
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
            draw->AddRectFilled(p, {p.x + card.x, p.y + card.y}, IM_COL32(19, 20, 27, 255), 8 * scale);
            const float fit = std::min((card_width - 12 * scale) / item.record.parameters.width, 134 * scale / item.record.parameters.height);
            const float w   = fit * item.record.parameters.width;
            const float h   = fit * item.record.parameters.height;
            const ImVec2 minimum{p.x + (card_width - w) / 2, p.y + (142 * scale - h) / 2};
            draw->AddImage(item.texture, minimum, {minimum.x + w, minimum.y + h});
            draw->AddRect(p, {p.x + card.x, p.y + card.y}, chosen ? IM_COL32(151, 145, 241, 210) : hovered ? IM_COL32(110, 109, 137, 180) : IM_COL32(73, 74, 89, 80), 8 * scale, 0, chosen ? 1.5F * scale : scale);
            ImGui::PushFont(nullptr, 12);
            draw->AddText(ImGui::GetFont(), ImGui::GetFontSize(), {p.x + 8 * scale, p.y + 144 * scale}, IM_COL32(173, 175, 190, 255), std::format("#{:02}   {:.1f}s", item.id + 1, item.record.sample_seconds + item.record.decode_seconds).c_str());
            ImGui::PopFont();
            if (hovered) ImGui::SetTooltip("%d x %d\nSeed: %llu%s", item.record.parameters.width, item.record.parameters.height, static_cast<unsigned long long>(item.record.seed), item.saved ? "" : "\nSaving...");
            if (ImGui::BeginPopupContextItem()) {
                if (ImGui::MenuItem("Reuse settings")) {
                    draft               = item.record.parameters;
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
        refresh_at                = std::numeric_limits<double>::infinity();
        const float scale         = renderer.dpi;
        const auto size           = ImGui::GetIO().DisplaySize;
        const float interpolation = std::min(1.0F, ImGui::GetIO().DeltaTime / 0.045F);
        composer_amount           = std::lerp(composer_amount, composer_open ? 1.0F : 0.0F, interpolation);
        history_amount            = std::lerp(history_amount, history_open ? 1.0F : 0.0F, interpolation);
        if (std::abs(composer_amount - float(composer_open)) < 0.01F) composer_amount = float(composer_open);
        if (std::abs(history_amount - float(history_open)) < 0.01F) history_amount = float(history_open);
        const bool narrow         = size.x - (48 + 300 * history_amount) * scale < 740 * scale;
        std::string error;
        {
            const std::lock_guard lock{session.mutex};
            session.preview_enabled = configuration.preview.enabled;
            session.preview_visible = renderer.visible && following_latest;
            error                   = session.error;
        }
        const float composer_height = (advanced ? narrow ? 464 : 416 : 224) * scale;
        if (renderer.visible) {
            canvas(scale, size, composer_height);
            composer(scale, size, composer_height);
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
            if (ImGui::GetIO().KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_Enter)) submit();
            if (ImGui::GetIO().KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_S)) {
                configuration.parameters = draft;
                configuration.seeds      = {seed};
                write_configuration(configuration, configuration_path);
            }
            if (!ImGui::GetIO().WantTextInput && ImGui::IsKeyPressed(ImGuiKey_Tab)) composer_open = !composer_open;
            const auto& io = ImGui::GetIO();
            if (io.MouseDelta.x || io.MouseDelta.y || io.MouseWheel || io.InputQueueCharacters.Size || ImGui::IsAnyItemActive()) animate_until = glfwGetTime() + 0.16;
        }
    }
} // namespace genesia::editor
