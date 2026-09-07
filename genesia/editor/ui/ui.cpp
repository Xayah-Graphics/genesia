module;
#include <Windows.h>
#include <shellapi.h>
#include <imgui.h>
#include <misc/cpp/imgui_stdlib.h>
#include <GLFW/glfw3.h>
#include <genesia/cuda.h>
#include "../../core/sdxl/control.h"
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
        enum class Icon { history, menu, minimize, maximize, close, edit, settings, fit };

        bool icon_button(const char* id, const Icon icon, const char* tooltip, const float scale, const bool selected = false) {
            const ImVec2 p = ImGui::GetCursorScreenPos();
            const float size = 36 * scale;
            const bool clicked = ImGui::InvisibleButton(id, {size, size});
            const bool hovered = ImGui::IsItemHovered();
            const auto key = ImGui::GetID(id);
            auto* storage = ImGui::GetStateStorage();
            const float previous = storage->GetFloat(key);
            const float alpha = std::lerp(previous, (hovered || selected) ? 1.0F : 0.0F, std::min(1.0F, ImGui::GetIO().DeltaTime / 0.12F));
            storage->SetFloat(key, alpha);
            auto* draw = ImGui::GetWindowDrawList();
            draw->AddRectFilled(p, {p.x + size, p.y + size}, ImGui::GetColorU32(icon == Icon::close && hovered ? ImVec4{0.8F, 0.2F, 0.3F, alpha * 0.8F} : ImVec4{0.6F, 0.6F, 0.8F, alpha * 0.12F}), 8 * scale);
            const ImVec2 c{p.x + size / 2, p.y + size / 2};
            const float r = 7 * scale;
            const float line = 1.4F * scale;
            const ImU32 ink = ImGui::GetColorU32(selected ? ImVec4{0.68F, 0.68F, 1, 1} : ImVec4{0.78F + alpha * 0.15F, 0.79F + alpha * 0.14F, 0.84F + alpha * 0.10F, 1});
            switch (icon) {
            case Icon::history:
                draw->AddRect({c.x - r, c.y - r}, {c.x + r, c.y + r}, ink, 2 * scale, 0, line);
                draw->AddLine({c.x - r + 4 * scale, c.y - r}, {c.x - r + 4 * scale, c.y + r}, ink, line);
                break;
            case Icon::menu:
                for (int i = -1; i <= 1; ++i) draw->AddCircleFilled({c.x + i * 5 * scale, c.y}, 1.3F * scale, ink);
                break;
            case Icon::minimize: draw->AddLine({c.x - r, c.y}, {c.x + r, c.y}, ink, line); break;
            case Icon::maximize: draw->AddRect({c.x - r + scale, c.y - r + scale}, {c.x + r - scale, c.y + r - scale}, ink, scale, 0, line); break;
            case Icon::close:
                draw->AddLine({c.x - r + scale, c.y - r + scale}, {c.x + r - scale, c.y + r - scale}, ink, line);
                draw->AddLine({c.x - r + scale, c.y + r - scale}, {c.x + r - scale, c.y - r + scale}, ink, line);
                break;
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

    }

    UserInterface::UserInterface(Configuration settings, std::filesystem::path path, WindowPlatform& platform, Renderer& display, Interop& bridge, Session& generation)
        : configuration{std::move(settings)}, configuration_path{std::move(path)}, window{platform}, renderer{display}, interop{bridge}, session{generation}, draft{configuration.parameters}, seed{configuration.seeds.front()} {
        ImGui::StyleColorsDark();
        auto& style = ImGui::GetStyle();
        style.WindowRounding = 16;
        style.ChildRounding = style.FrameRounding = style.GrabRounding = 8;
        style.PopupRounding = 12;
        style.WindowBorderSize = 1;
        style.FrameBorderSize = 0;
        style.WindowPadding = {20, 16};
        style.FramePadding = {12, 9};
        style.ItemSpacing = {8, 10};
        style.ScrollbarSize = 8;
        style.ScrollbarRounding = 8;
        style.Colors[ImGuiCol_Text] = {0.93F, 0.93F, 0.96F, 1};
        style.Colors[ImGuiCol_TextDisabled] = {0.57F, 0.58F, 0.64F, 1};
        style.Colors[ImGuiCol_WindowBg] = {0.095F, 0.10F, 0.125F, 0.985F};
        style.Colors[ImGuiCol_PopupBg] = {0.12F, 0.125F, 0.15F, 1};
        style.Colors[ImGuiCol_Border] = {0.70F, 0.72F, 0.85F, 0.10F};
        style.Colors[ImGuiCol_FrameBg] = {0.07F, 0.075F, 0.095F, 1};
        style.Colors[ImGuiCol_FrameBgHovered] = {0.14F, 0.145F, 0.18F, 1};
        style.Colors[ImGuiCol_FrameBgActive] = {0.16F, 0.16F, 0.21F, 1};
        style.Colors[ImGuiCol_Button] = {0.17F, 0.175F, 0.215F, 1};
        style.Colors[ImGuiCol_ButtonHovered] = {0.23F, 0.23F, 0.29F, 1};
        style.Colors[ImGuiCol_ButtonActive] = {0.30F, 0.29F, 0.38F, 1};
        style.Colors[ImGuiCol_Header] = {0.35F, 0.34F, 0.55F, 0.35F};
        style.Colors[ImGuiCol_HeaderHovered] = {0.45F, 0.44F, 0.67F, 0.35F};
        style.Colors[ImGuiCol_CheckMark] = style.Colors[ImGuiCol_SliderGrab] = {0.63F, 0.62F, 1, 1};
        style.Colors[ImGuiCol_NavCursor] = {0.63F, 0.62F, 1, 0.8F};
    }

    void UserInterface::receive() {
        std::deque<Event> events;
        {
            const std::lock_guard lock{session.mutex};
            events.swap(session.events);
        }
        for (auto& event : events) {
            if (event.kind == EventKind::generated) {
                const auto texture = renderer.texture({std::uint32_t(event.record.parameters.width), std::uint32_t(event.record.parameters.height)});
                renderer.copy(texture, interop.slots[event.slot].buffer, *interop.slots[event.slot].timeline, event.ready);
                if (latest_texture && (latest_texture != image_texture || following_latest)) renderer.retire(latest_texture);
                latest_texture = texture;
                history.push_back({event.id, event.record, renderer.upload(event.image)});
                if (following_latest) {
                    image_texture = latest_texture;
                    image_width = event.record.parameters.width;
                    image_height = event.record.parameters.height;
                    selected = event.id;
                    fit_image = true;
                    pan = {};
                }
                if (!edited_since_submit) composer_open = false;
                animate_until = glfwGetTime() + 0.2;
                display_times.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - event.generated_at).count());
            } else if (event.kind == EventKind::saved) {
                for (auto& item : history) if (item.id == event.id) item.saved = true;
            } else if (event.id == selected && !following_latest) {
                if (image_texture && image_texture != latest_texture) renderer.retire(image_texture);
                image_texture = renderer.upload(event.image);
                image_width = event.image.width;
                image_height = event.image.height;
                fit_image = true;
                pan = {};
            }
        }
    }

    void UserInterface::submit() {
        if (random_seed) {
            std::random_device source;
            seed = (std::uint64_t(source()) << 32) | source();
        }
        session.enqueue(draft, seed);
        edited_since_submit = false;
        animate_until = glfwGetTime() + 0.2;
    }

    void UserInterface::canvas(const float scale, const ImVec2 size, const float composer_height) {
        ImGui::SetNextWindowPos({0, 0});
        ImGui::SetNextWindowSize(size);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {0, 0});
        ImGui::Begin("##Canvas", nullptr, overlay | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNav);
        ImGui::InvisibleButton("##ImageArea", size);
        const bool hovered = ImGui::IsItemHovered();
        const float left = 24 * scale + history_amount * 300 * scale;
        const float top = 68 * scale;
        const float bottom = size.y - 24 * scale - composer_amount * (composer_height + 24 * scale);
        const ImVec2 center{(left + size.x - 24 * scale) / 2, (top + bottom) / 2};
        auto* draw = ImGui::GetWindowDrawList();
        if (image_texture) {
            const float fitted = std::min((size.x - left - 24 * scale) / image_width, std::max(64 * scale, bottom - top) / image_height);
            if (fit_image) zoom = fitted;
            if (hovered && ImGui::GetIO().MouseWheel != 0) {
                const float next = std::clamp(zoom * std::pow(1.15F, ImGui::GetIO().MouseWheel), 0.05F, 16.0F);
                const auto mouse = ImGui::GetIO().MousePos;
                pan = {mouse.x - center.x - (mouse.x - center.x - pan.x) * next / zoom,
                    mouse.y - center.y - (mouse.y - center.y - pan.y) * next / zoom};
                zoom = next;
                fit_image = false;
            }
            if (hovered && ImGui::IsKeyDown(ImGuiKey_Space) && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
                pan.x += ImGui::GetIO().MouseDelta.x;
                pan.y += ImGui::GetIO().MouseDelta.y;
                fit_image = false;
                ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeAll);
            }
            if (hovered && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) { fit_image = true; pan = {}; }
            const ImVec2 minimum{center.x + pan.x - image_width * zoom / 2, center.y + pan.y - image_height * zoom / 2};
            const ImVec2 maximum{minimum.x + image_width * zoom, minimum.y + image_height * zoom};
            draw->PushClipRect({left, top}, {size.x - 24 * scale, std::max(top, bottom)}, true);
            draw->AddRectFilled({minimum.x - 1, minimum.y - 1}, {maximum.x + 1, maximum.y + 1}, IM_COL32(45, 46, 55, 255));
            draw->AddImage(image_texture, minimum, maximum);
            draw->PopClipRect();
        } else {
            const char* title = "Imagine something new.";
            ImGui::PushFont(nullptr, 30);
            const auto text = ImGui::CalcTextSize(title);
            draw->AddText(ImGui::GetFont(), ImGui::GetFontSize(), {center.x - text.x / 2, center.y - 30 * scale}, IM_COL32(218, 219, 230, 255), title);
            ImGui::PopFont();
            const char* subtitle = "A few words. A world of possibilities.";
            const auto hint = ImGui::CalcTextSize(subtitle);
            draw->AddText({center.x - hint.x / 2, center.y + 20 * scale}, IM_COL32(119, 121, 137, 255), subtitle);
            draw->AddCircle({center.x, center.y - 88 * scale}, 14 * scale, IM_COL32(145, 142, 225, 180), 32, 1.5F * scale);
            draw->AddCircleFilled({center.x + 14 * scale, center.y - 100 * scale}, 3 * scale, IM_COL32(184, 182, 250, 255));
        }
        ImGui::End();
        ImGui::PopStyleVar();
    }

    void UserInterface::chrome(const float scale, const ImVec2 size) {
        ImGui::SetNextWindowPos({20 * scale, 14 * scale});
        ImGui::SetNextWindowSize({230 * scale, 40 * scale});
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {0, 0});
        ImGui::Begin("##Navigation", nullptr, overlay | ImGuiWindowFlags_NoBackground);
        if (icon_button("##History", Icon::history, "Session history", scale, history_open)) { history_open = !history_open; animate_until = glfwGetTime() + 0.2; }
        ImGui::SameLine(0, 14 * scale);
        ImGui::SetCursorPosY(9 * scale);
        ImGui::TextUnformatted("GENESIA");
        ImGui::SameLine(0, 12 * scale);
        ImGui::SetCursorPosY(0);
        if (icon_button("##Menu", Icon::menu, "Menu", scale)) ImGui::OpenPopup("Application");
        if (ImGui::BeginPopup("Application")) {
            if (ImGui::MenuItem("Save configuration", "Ctrl+S")) {
                configuration.parameters = draft;
                configuration.seeds = {seed};
                write_configuration(configuration, configuration_path);
            }
            if (ImGui::MenuItem("Open output folder")) ShellExecuteW(window.native_window, L"open", std::filesystem::absolute(configuration.output).c_str(), nullptr, nullptr, SW_SHOWNORMAL);
            ImGui::Separator();
            if (ImGui::MenuItem("Quit", "Alt+F4")) window.request_close();
            ImGui::EndPopup();
        }
        ImGui::End();
        ImGui::SetNextWindowPos({size.x - 132 * scale, 14 * scale});
        ImGui::SetNextWindowSize({120 * scale, 40 * scale});
        ImGui::Begin("##Window", nullptr, overlay | ImGuiWindowFlags_NoBackground);
        if (icon_button("##Minimize", Icon::minimize, "Minimize", scale)) glfwIconifyWindow(window.window);
        ImGui::SameLine(0, 4 * scale);
        if (icon_button("##Maximize", Icon::maximize, "Maximize / restore", scale)) {
            if (glfwGetWindowAttrib(window.window, GLFW_MAXIMIZED)) glfwRestoreWindow(window.window);
            else glfwMaximizeWindow(window.window);
        }
        ImGui::SameLine(0, 4 * scale);
        if (icon_button("##Close", Icon::close, "Close", scale)) window.request_close();
        ImGui::End();
        ImGui::PopStyleVar();
        window.drag_region = {240 * scale, 0, size.x - 140 * scale, 54 * scale};
        if (image_texture) {
            ImGui::SetNextWindowPos({size.x - 178 * scale, size.y - 60 * scale});
            ImGui::SetNextWindowSize({154 * scale, 38 * scale});
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {0, 0});
            ImGui::Begin("##ImageTools", nullptr, overlay | ImGuiWindowFlags_NoBackground);
            if (icon_button("##Fit", Icon::fit, "Fit image", scale)) { fit_image = true; pan = {}; }
            ImGui::SameLine();
            if (ImGui::Button(std::format("{:.0f}%", zoom * 100).c_str(), {72 * scale, 36 * scale})) { fit_image = false; zoom = 1; pan = {}; }
            ImGui::End();
            ImGui::PopStyleVar();
        }
    }

    void UserInterface::composer(const float scale, const ImVec2 size, const float height) {
        const float width = std::min(840 * scale, size.x - (48 + 300 * history_amount) * scale);
        const float center = (size.x + history_amount * 300 * scale) / 2;
        if (composer_amount < 0.03F) {
            ImGui::SetNextWindowPos({center - 96 * scale, size.y - 68 * scale});
            ImGui::SetNextWindowSize({192 * scale, 44 * scale});
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {4 * scale, 4 * scale});
            ImGui::Begin("##EditPrompt", nullptr, overlay);
            if (icon_button("##Edit", Icon::edit, "Edit prompt", scale)) composer_open = true;
            ImGui::SameLine();
            if (ImGui::Button("Edit prompt", {136 * scale, 36 * scale})) composer_open = true;
            ImGui::End();
            ImGui::PopStyleVar();
            return;
        }
        const float y = size.y - 24 * scale - height + (1 - composer_amount) * (height + 30 * scale);
        ImGui::SetNextWindowPos({center - width / 2, y});
        ImGui::SetNextWindowSize({width, height});
        ImGui::PushStyleVar(ImGuiStyleVar_Alpha, composer_amount);
        ImGui::Begin("##Composer", nullptr, overlay);
        ImGui::TextDisabled("PROMPT");
        ImGui::SameLine(width - 60 * scale);
        ImGui::SetCursorPosY(6 * scale);
        if (icon_button("##Collapse", Icon::minimize, "Hide prompt", scale)) composer_open = false;
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
            if (ImGui::InputScalar("Seed", ImGuiDataType_U64, &seed)) { random_seed = false; edited_since_submit = true; }
        }
        ImGui::SetCursorPosY(height - 58 * scale);
        if (ImGui::Button(std::format("{} x {}", draft.width, draft.height).c_str(), {136 * scale, 38 * scale})) ImGui::OpenPopup("Image size");
        if (ImGui::BeginPopup("Image size")) {
            ImGui::TextDisabled("ASPECT RATIO");
            if (ImGui::Button("Portrait  2:3")) { draft.width = 1024; draft.height = 1536; edited_since_submit = true; ImGui::CloseCurrentPopup(); }
            if (ImGui::Button("Square  1:1")) { draft.width = draft.height = 1024; edited_since_submit = true; ImGui::CloseCurrentPopup(); }
            if (ImGui::Button("Landscape  3:2")) { draft.width = 1536; draft.height = 1024; edited_since_submit = true; ImGui::CloseCurrentPopup(); }
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
        if (icon_button("##Advanced", Icon::settings, "Generation parameters", scale, advanced)) { advanced = !advanced; animate_until = glfwGetTime() + 0.2; }
        ImGui::SameLine(width - 182 * scale);
        ImGui::PushStyleColor(ImGuiCol_Button, {0.48F, 0.46F, 0.83F, 1});
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, {0.57F, 0.55F, 0.94F, 1});
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, {0.42F, 0.40F, 0.76F, 1});
        bool unavailable;
        { const std::lock_guard lock{session.mutex}; unavailable = session.worker_done; }
        ImGui::BeginDisabled(unavailable);
        if (ImGui::Button("Generate", {162 * scale, 38 * scale})) submit();
        ImGui::EndDisabled();
        ImGui::PopStyleColor(3);
        ImGui::End();
        ImGui::PopStyleVar();
    }

    void UserInterface::history_drawer(const float scale, const ImVec2 size) {
        if (history_amount < 0.03F) return;
        ImGui::SetNextWindowPos({16 * scale - (1 - history_amount) * 300 * scale, 68 * scale});
        ImGui::SetNextWindowSize({280 * scale, size.y - 92 * scale});
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
            const ImVec2 p = ImGui::GetCursorScreenPos();
            const ImVec2 card{card_width, 164 * scale};
            if (ImGui::InvisibleButton("Image", card) && item.saved) {
                following_latest = item.id == history.back().id;
                selected = item.id;
                if (following_latest) {
                    if (image_texture && image_texture != latest_texture) renderer.retire(image_texture);
                    image_texture = latest_texture;
                    image_width = item.record.parameters.width;
                    image_height = item.record.parameters.height;
                    fit_image = true;
                    pan = {};
                } else session.load(item.id, item.record.directory / "image.png");
            }
            const bool hovered = ImGui::IsItemHovered();
            const bool chosen = item.id == selected;
            auto* draw = ImGui::GetWindowDrawList();
            draw->AddRectFilled(p, {p.x + card.x, p.y + card.y}, IM_COL32(19, 20, 27, 255), 8 * scale);
            const float fit = std::min((card_width - 12 * scale) / item.record.parameters.width, 134 * scale / item.record.parameters.height);
            const float w = fit * item.record.parameters.width;
            const float h = fit * item.record.parameters.height;
            const ImVec2 minimum{p.x + (card_width - w) / 2, p.y + (142 * scale - h) / 2};
            draw->AddImage(item.texture, minimum, {minimum.x + w, minimum.y + h});
            draw->AddRect(p, {p.x + card.x, p.y + card.y}, chosen ? IM_COL32(151, 145, 241, 210) : hovered ? IM_COL32(110, 109, 137, 180) : IM_COL32(73, 74, 89, 80), 8 * scale, 0, chosen ? 1.5F * scale : scale);
            ImGui::PushFont(nullptr, 12);
            draw->AddText(ImGui::GetFont(), ImGui::GetFontSize(), {p.x + 8 * scale, p.y + 144 * scale}, IM_COL32(173, 175, 190, 255),
                std::format("#{:02}   {:.1f}s", item.id + 1, item.record.sample_seconds + item.record.decode_seconds).c_str());
            ImGui::PopFont();
            if (hovered) ImGui::SetTooltip("%d x %d\nSeed: %llu%s", item.record.parameters.width, item.record.parameters.height, static_cast<unsigned long long>(item.record.seed), item.saved ? "" : "\nSaving...");
            if (ImGui::BeginPopupContextItem()) {
                if (ImGui::MenuItem("Reuse settings")) {
                    draft = item.record.parameters;
                    seed = item.record.seed;
                    random_seed = false;
                    composer_open = true;
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

    void UserInterface::status(const float scale, const ImVec2 size, const float composer_height) {
        std::optional<Request> active;
        std::size_t queued;
        bool paused, loaded;
        double elapsed{};
        std::string error;
        {
            const std::lock_guard lock{session.mutex};
            active = session.active;
            queued = session.queue.size();
            paused = session.paused;
            loaded = session.model_ready;
            error = session.error;
            if (active) elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - session.started).count();
        }
        const auto stage = static_cast<sdxl::Stage>(::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>{session.control.data()[0].stage}.load());
        const auto completed = ::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>{session.control.data()[0].completed}.load();
        if (active || queued || paused || !loaded || !following_latest) {
            const float width = (following_latest ? 476 : 568) * scale;
            const float bottom = size.y - 126 * scale - composer_amount * (composer_height - 46 * scale);
            ImGui::SetNextWindowPos({(size.x + history_amount * 300 * scale - width) / 2, bottom});
            ImGui::SetNextWindowSize({width, 46 * scale});
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {14 * scale, 6 * scale});
            ImGui::Begin("##GenerationStatus", nullptr, overlay);
            std::string label;
            if (!loaded) label = "Loading model";
            else if (active) {
                if (stage == sdxl::Stage::preparing) label = "Preparing";
                else if (stage == sdxl::Stage::decoding || stage == sdxl::Stage::transferring) label = "Finishing image";
                else label = std::format("Generating  {} / {}", completed, active->parameters.steps);
                label += std::format("   {:.1f}s", elapsed);
            } else label = paused ? "Queue paused" : "Ready";
            ImGui::AlignTextToFramePadding();
            ImGui::TextUnformatted(label.c_str());
            ImGui::SameLine(width - (following_latest ? 202 : 294) * scale);
            if (ImGui::Button(std::format("Queue  {}", queued).c_str(), {92 * scale, 32 * scale})) ImGui::OpenPopup("Queue");
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
            ImGui::SameLine();
            if (active) {
                if (ImGui::Button("Stop", {82 * scale, 32 * scale})) session.stop();
            } else if (paused) {
                if (ImGui::Button("Resume", {82 * scale, 32 * scale})) session.resume();
            }
            if (!following_latest && !history.empty()) {
                if (active || paused) ImGui::SameLine();
                if (ImGui::Button("Latest", {82 * scale, 32 * scale})) {
                    following_latest = true;
                    selected = history.back().id;
                    if (image_texture && image_texture != latest_texture) renderer.retire(image_texture);
                    image_texture = latest_texture;
                    image_width = history.back().record.parameters.width;
                    image_height = history.back().record.parameters.height;
                    fit_image = true;
                    pan = {};
                }
            }
            if (active && stage == sdxl::Stage::sampling) {
                const auto p = ImGui::GetWindowPos();
                ImGui::GetWindowDrawList()->AddLine({p.x + 14 * scale, p.y + 45 * scale}, {p.x + 14 * scale + (width - 28 * scale) * std::min(1.0F, float(completed) / active->parameters.steps), p.y + 45 * scale}, IM_COL32(151, 145, 241, 240), 1.5F * scale);
            }
            ImGui::End();
            ImGui::PopStyleVar();
        }
        if (!error.empty() && error != shown_error) { shown_error = error; ImGui::OpenPopup("Generation failed"); }
        ImGui::SetNextWindowSize({540 * scale, 0});
        if (ImGui::BeginPopupModal("Generation failed", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::TextWrapped("%s", shown_error.c_str());
            ImGui::Spacing();
            if (ImGui::Button("Close")) ImGui::CloseCurrentPopup();
            ImGui::EndPopup();
        }
    }

    void UserInterface::draw() {
        const float scale = renderer.dpi;
        const auto size = ImGui::GetIO().DisplaySize;
        const float interpolation = std::min(1.0F, ImGui::GetIO().DeltaTime / 0.045F);
        composer_amount = std::lerp(composer_amount, composer_open ? 1.0F : 0.0F, interpolation);
        history_amount = std::lerp(history_amount, history_open ? 1.0F : 0.0F, interpolation);
        const bool narrow = size.x - (48 + 300 * history_amount) * scale < 740 * scale;
        const float composer_height = (advanced ? narrow ? 464 : 416 : 224) * scale;
        if (renderer.visible) {
            canvas(scale, size, composer_height);
            chrome(scale, size);
            composer(scale, size, composer_height);
            history_drawer(scale, size);
            status(scale, size, composer_height);
            if (ImGui::GetIO().KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_Enter)) submit();
            if (ImGui::GetIO().KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_S)) {
                configuration.parameters = draft;
                configuration.seeds = {seed};
                write_configuration(configuration, configuration_path);
            }
            if (!ImGui::GetIO().WantTextInput && ImGui::IsKeyPressed(ImGuiKey_Tab)) composer_open = !composer_open;
            const auto& io = ImGui::GetIO();
            if (io.MouseDelta.x || io.MouseDelta.y || io.MouseWheel || io.InputQueueCharacters.Size || ImGui::IsAnyItemActive()) animate_until = glfwGetTime() + 0.16;
        }
    }
}
