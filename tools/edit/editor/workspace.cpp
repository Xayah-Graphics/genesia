module;
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_internal.h>
#include <misc/cpp/imgui_stdlib.h>
module edit.editor.workspace;
import tools.files;
import std;
namespace edit::editor {
    Workspace::Workspace(tools::editor::WindowPlatform& platform, tools::editor::Renderer& display) : window{platform}, renderer{display}, session{[] { glfwPostEmptyEvent(); }} {
        load_preset(read_preset("default"));
    }
    void Workspace::receive() {
        const auto now = std::chrono::steady_clock::now();
        auto delivery       = session.drain();
        const auto previous = state.revision;
        state               = std::move(delivery.state);
        if (state.revision != previous && !state.result.error.empty()) error = state.result.error;
        if (save_at && now >= *save_at) {
            try {
                save_preset(preset_name);
            } catch (const std::exception& failure) {
                error         = failure.what();
                window.redraw = true;
            }
        }
    }
    void Workspace::drop() {
        if (!window.drop_error.empty()) error = std::exchange(window.drop_error, {});
        auto paths = std::exchange(window.dropped, {});
        if (paths.empty()) return;
        try {
            if (state.busy) throw std::runtime_error{"Stop the current batch before dropping another input"};
            if (paths.size() != 1) throw std::runtime_error{"Drop one PNG image or one image folder"};
            const auto path   = std::filesystem::canonical(paths.front());
            const auto target = std::filesystem::is_directory(path) ? 0 : drop_target(float(window.drop_position.x), float(window.drop_position.y));
            if (!target) throw std::runtime_error{"Drop the PNG onto a fixed image slot"};
            if (*target == 0) {
                Request request{path, prompt};
                for (const auto& image : references) {
                    if (!image.error.empty()) throw std::runtime_error{image.error};
                    request.references.push_back(image.path);
                }
                save_preset(preset_name);
                session.submit(std::move(request));
                state = session.drain().state;
                error = state.result.error;
            } else {
                ImageSlot image{.path = path};
                load_image(image);
                error = image.error;
                if (*target == references.size() + 1) {
                    references.push_back(std::move(image));
                    reveal_reference = true;
                } else {
                    auto& destination = references[*target - 1];
                    if (destination.texture) retired.push_back(destination.texture);
                    destination = std::move(image);
                }
                save_preset(preset_name);
            }
        } catch (const std::exception& failure) {
            error = failure.what();
        }
        window.redraw = true;
    }
    void Workspace::draw() {
        for (const auto texture : retired) renderer.retire(texture);
        retired.clear();
        drop_bounds.clear();
        const float dpi   = renderer.dpi;
        const float width = std::min(420 * dpi, float(window.extent_limit.width));
        const auto& style = ImGui::GetStyle();
        ImGui::SetNextWindowPos({0, 0});
        ImGui::SetNextWindowSize({width, float(window.extent_limit.height)});
        ImGui::Begin("Edit", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoBringToFrontOnFocus);
        ImGui::TextUnformatted("Edit Images with Qwen");
        ImGui::BeginDisabled(state.busy);
        ImGui::SetNextItemWidth(-90 * dpi);
        if (ImGui::BeginCombo("##Preset", preset_name.c_str())) {
            try {
                if (ImGui::IsWindowAppearing()) preset_names = list_presets();
                for (const auto& name : preset_names) {
                    if (!ImGui::Selectable(name.c_str(), name == preset_name) || name == preset_name) continue;
                    save_preset(preset_name);
                    load_preset(read_preset(name));
                }
            } catch (const std::exception& failure) {
                error = failure.what();
            }
            ImGui::EndCombo();
        }
        ImGui::SameLine();
        if (ImGui::Button("Save as...")) {
            new_preset_name.fill(0);
            preset_error.clear();
            ImGui::OpenPopup("Save edit preset as");
        }
        ImGui::EndDisabled();
        ImGui::SetNextWindowSize({360 * dpi, 0});
        if (ImGui::BeginPopupModal("Save edit preset as", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::TextDisabled("Preset name");
            if (ImGui::IsWindowAppearing()) ImGui::SetKeyboardFocusHere();
            ImGui::SetNextItemWidth(-1);
            const bool enter = ImGui::InputText("##PresetName", new_preset_name.data(), new_preset_name.size(), ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_CallbackCharFilter, [](ImGuiInputTextCallbackData* data) {
                const auto c = data->EventChar;
                return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-' || c == '_' ? 0 : 1;
            });
            ImGui::TextDisabled("Letters, numbers, hyphens and underscores");
            if (!preset_error.empty()) ImGui::TextWrapped("%s", preset_error.c_str());
            ImGui::BeginDisabled(state.busy || new_preset_name[0] == 0);
            if ((ImGui::Button("Save") || enter) && !state.busy && new_preset_name[0]) {
                try {
                    save_preset(preset_name);
                    save_preset(new_preset_name.data(), false);
                    ImGui::CloseCurrentPopup();
                } catch (const std::exception& failure) {
                    preset_error = failure.what();
                }
            }
            ImGui::EndDisabled();
            ImGui::SameLine();
            if (ImGui::Button("Cancel") || ImGui::IsKeyPressed(ImGuiKey_Escape, false)) ImGui::CloseCurrentPopup();
            ImGui::EndPopup();
        }
        ImGui::TextDisabled("Positive prompt");
        const float text_width    = ImGui::GetContentRegionAvail().x - 2 * style.FramePadding.x - style.ScrollbarSize;
        const float text_height   = ImGui::CalcTextSize(prompt.c_str(), nullptr, false, text_width).y;
        const float prompt_height = std::clamp(text_height, 3 * ImGui::GetTextLineHeight(), 8 * ImGui::GetTextLineHeight()) + 2 * style.FramePadding.y;
        ImGui::BeginDisabled(state.busy);
        if (ImGui::InputTextMultiline("##prompt", &prompt, {-1, prompt_height}, ImGuiInputTextFlags_WordWrap)) {
            save_at       = std::chrono::steady_clock::now() + std::chrono::milliseconds{300};
            window.redraw = true;
        }
        ImGui::EndDisabled();
        const auto reference_origin = ImGui::GetCursorScreenPos();
        ImGui::TextDisabled("Fixed images  /  %zu of 9", references.size());
        const auto count              = std::min(references.size() + 1, 9uz);
        const float row_height        = 100 * dpi + style.ItemSpacing.y;
        const float references_height = std::min(float((count + 1) / 2) * row_height - style.ItemSpacing.y, std::max(100 * dpi, std::min(320 * dpi, float(window.extent_limit.height) - ImGui::GetCursorPosY() - 150 * dpi)));
        ImGui::BeginChild("##References", {0, references_height});
        const float card_width = (ImGui::GetContentRegionAvail().x - style.ItemSpacing.x) / 2;
        std::optional<std::size_t> removed;
        for (std::size_t i = 0; i < count; ++i) {
            if (i % 2) ImGui::SameLine();
            if (draw_image(i < references.size() ? &references[i] : nullptr, i + 2, {card_width, 100 * dpi})) removed = i;
        }
        if (std::exchange(reveal_reference, false)) ImGui::SetScrollHereY(1);
        ImGui::EndChild();
        const auto reference_end = ImGui::GetItemRectMax();
        reference_bounds         = {reference_origin.x, reference_origin.y, reference_end.x, reference_end.y};
        if (removed) {
            if (references[*removed].texture) renderer.retire(references[*removed].texture);
            references.erase(references.begin() + *removed);
            window.redraw = true;
            try {
                save_preset(preset_name);
            } catch (const std::exception& failure) {
                error = failure.what();
            }
        }
        if (state.busy) {
            ImGui::TextUnformatted(state.stopping || state.progress.stage == Stage::stopping ? "Stopping..." : state.progress.stage == Stage::preparing ? "Preparing..." : "Editing");
            ImGui::SameLine();
            ImGui::TextDisabled("%zu / %zu", state.progress.completed, state.progress.total);
            const float ratio = state.progress.total ? float(state.progress.completed) / float(state.progress.total) : 0;
            ImGui::ProgressBar(ratio, {-1, 4 * dpi}, "");
            if (!state.progress.file.empty()) ImGui::TextWrapped("%s", tools::files::utf8(state.progress.file.filename()).c_str());
            ImGui::BeginDisabled(state.stopping || state.progress.stage == Stage::stopping);
            if (ImGui::Button("Stop")) session.cancel();
            ImGui::EndDisabled();
        } else ImGui::TextWrapped("Drop a PNG outside Fixed images to start.\nDrop a folder anywhere to start a batch.");
        if (!state.busy && state.state != runtime::State::idle) {
            ImGui::Separator();
            const auto label = state.state == runtime::State::complete ? "Complete" : state.state == runtime::State::stopped ? "Stopped" : "Failed";
            ImGui::Text("Last batch: %s  /  %zu of %zu saved", label, state.result.completed, state.result.total);
            if (state.result.completed) {
                ImGui::PushTextWrapPos();
                ImGui::TextDisabled("%s", tools::files::utf8(state.result.output).c_str());
                ImGui::PopTextWrapPos();
            }
        }
        if (!error.empty()) {
            ImGui::PushStyleColor(ImGuiCol_Text, {0.95F, 0.59F, 0.55F, 1});
            ImGui::TextWrapped("%s", error.c_str());
            ImGui::PopStyleColor();
            if (ImGui::SmallButton("Dismiss")) error.clear();
        }
        const auto* layout           = ImGui::GetCurrentWindow();
        const float height           = layout->DC.CursorMaxPos.y - layout->DC.CursorStartPos.y + 2 * style.WindowPadding.y;
        window.content_extent.width  = static_cast<std::uint32_t>(std::ceil(width));
        window.content_extent.height = static_cast<std::uint32_t>(std::ceil(std::min(height, float(window.extent_limit.height))));
        if (!state.busy && !window.dragged.empty()) {
            const auto target  = std::filesystem::is_directory(window.dragged.front()) ? 0 : drop_target(float(window.drag_position.x), float(window.drag_position.y));
            const auto display = ImGui::GetIO().DisplaySize;
            const auto bounds  = !target ? reference_bounds : *target ? drop_bounds[*target - 1] : std::array{5 * dpi, 5 * dpi, display.x - 5 * dpi, display.y - 5 * dpi};
            const ImVec2 minimum{bounds[0], bounds[1]}, maximum{bounds[2], bounds[3]};
            auto* draw = ImGui::GetForegroundDrawList();
            draw->PushClipRect(minimum, maximum, true);
            draw->AddRectFilled(minimum, {maximum.x, minimum.y + 28 * dpi}, ImGui::GetColorU32(ImGuiCol_WindowBg), 7 * dpi);
            draw->AddRect(minimum, maximum, ImGui::GetColorU32(ImGuiCol_PlotHistogram), 7 * dpi, 0, 2 * dpi);
            const auto label = !target ? std::string{"Drop onto a fixed image slot"} : *target ? std::format("{} <image{}>", *target <= references.size() ? "Replace" : "Add", *target + 1) : std::string{"Drop to start"};
            draw->AddText({minimum.x + 10 * dpi, minimum.y + 7 * dpi}, ImGui::GetColorU32(ImGuiCol_PlotHistogram), label.c_str());
            draw->PopClipRect();
        }
        window.drag_requested = !ImGui::IsAnyItemHovered() && !ImGui::IsAnyItemActive() && ImGui::IsMouseClicked(ImGuiMouseButton_Left);
        ImGui::End();
    }
    void Workspace::save_preset(const std::string& name, const bool replace) {
        save_at.reset();
        Preset preset{name, prompt};
        for (const auto& image : references) preset.references.push_back(image.path);
        write_preset(preset, replace);
        preset_name = name;
    }
    void Workspace::load_preset(Preset preset) {
        std::vector<ImageSlot> images;
        std::string image_errors;
        for (auto& path : preset.references) {
            auto& image = images.emplace_back();
            image.path  = std::move(path);
            load_image(image);
            if (!image.error.empty()) {
                if (!image_errors.empty()) image_errors += '\n';
                image_errors += image.error;
            }
        }
        for (const auto& image : references)
            if (image.texture) retired.push_back(image.texture);
        references       = std::move(images);
        preset_name      = std::move(preset.name);
        prompt           = std::move(preset.prompt);
        error            = std::move(image_errors);
        reveal_reference = false;
        window.redraw    = true;
    }
    void Workspace::load_image(ImageSlot& image) {
        try {
            image.thumbnail = tools::read_thumbnail(image.path, 256);
        } catch (const std::exception& failure) {
            image.error = std::format("{}\n{}", tools::files::utf8(image.path), failure.what());
        }
    }
    std::optional<std::size_t> Workspace::drop_target(const float x, const float y) const {
        for (std::size_t i = 0; i < drop_bounds.size(); ++i) {
            const auto& bounds = drop_bounds[i];
            if (x >= bounds[0] && x < bounds[2] && y >= bounds[1] && y < bounds[3]) return i + 1;
        }
        if (x >= reference_bounds[0] && x < reference_bounds[2] && y >= reference_bounds[1] && y < reference_bounds[3]) return {};
        return 0;
    }
    bool Workspace::draw_image(ImageSlot* image, const std::size_t number, const ImVec2 size) {
        const float dpi = renderer.dpi;
        if (image && !image->thumbnail.pixels.empty()) {
            image->texture          = renderer.upload(image->thumbnail);
            image->thumbnail.pixels = {};
        }
        ImGui::PushID(static_cast<int>(number));
        ImGui::BeginGroup();
        const auto origin = ImGui::GetCursorScreenPos();
        const ImVec2 end{origin.x + size.x, origin.y + size.y};
        const auto& clip = ImGui::GetCurrentWindow()->ClipRect;
        drop_bounds.push_back({std::max(origin.x, clip.Min.x), std::max(origin.y, clip.Min.y), std::min(end.x, clip.Max.x), std::min(end.y, clip.Max.y)});
        ImGui::SetNextItemAllowOverlap();
        ImGui::InvisibleButton("##Image", size);
        const bool hovered = ImGui::IsItemHovered();
        auto* draw         = ImGui::GetWindowDrawList();
        draw->AddRectFilled(origin, end, ImGui::GetColorU32(hovered ? ImGuiCol_FrameBgHovered : ImGuiCol_FrameBg), 7 * dpi);
        const auto label = std::format("<image{}>", number);
        draw->AddText({origin.x + 10 * dpi, origin.y + 9 * dpi}, ImGui::GetColorU32(ImGuiCol_Text), label.c_str());
        if (image) {
            float text_x = origin.x + 10 * dpi;
            if (image->texture) {
                const float scale = 54 * dpi / std::max(image->thumbnail.width, image->thumbnail.height);
                const ImVec2 minimum{origin.x + 10 * dpi + (54 * dpi - image->thumbnail.width * scale) / 2, origin.y + 36 * dpi + (54 * dpi - image->thumbnail.height * scale) / 2};
                draw->AddImage(image->texture, minimum, {minimum.x + image->thumbnail.width * scale, minimum.y + image->thumbnail.height * scale});
                text_x = origin.x + 74 * dpi;
            }
            const auto filename = tools::files::utf8(image->path.filename());
            ImGui::RenderTextEllipsis(draw, {text_x, origin.y + 40 * dpi}, {end.x - 10 * dpi, origin.y + 62 * dpi}, end.x - 10 * dpi, filename.c_str(), nullptr, nullptr);
            draw->AddText({text_x, origin.y + 67 * dpi}, ImGui::GetColorU32(image->error.empty() ? ImGuiCol_TextDisabled : ImGuiCol_PlotHistogram), image->error.empty() ? "PNG image" : "Unavailable");
            if (hovered) ImGui::SetTooltip("%s", image->error.empty() ? tools::files::utf8(image->path).c_str() : image->error.c_str());
        } else {
            draw->AddText({origin.x + 10 * dpi, origin.y + 40 * dpi}, ImGui::GetColorU32(ImGuiCol_TextDisabled), "Drop a PNG here");
            draw->AddText({origin.x + 10 * dpi, origin.y + 67 * dpi}, ImGui::GetColorU32(ImGuiCol_TextDisabled), "Optional");
        }
        bool removed{};
        if (image) {
            ImGui::SetCursorScreenPos({end.x - 30 * dpi, origin.y + 5 * dpi});
            ImGui::BeginDisabled(state.busy);
            removed = ImGui::Button("x", {24 * dpi, 24 * dpi});
            ImGui::EndDisabled();
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Remove <image%zu>; later images move forward. Update the prompt as needed.", number);
        }
        ImGui::EndGroup();
        ImGui::PopID();
        return removed;
    }
} // namespace edit::editor
