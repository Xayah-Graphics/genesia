module;
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_internal.h>
module genesia.editor.viewing.canvas;
import genesia.editor.widgets.controls;
import genesia.editor.widgets.tags;
import genesia.io.files;
import genesia.runtime.session;
import genesia.runtime.catalog;
import genesia.project;
import genesia.prompt.preset;
import genesia.editor.graphics.bridge;
import genesia.editor.viewing.camera;
import std;
namespace genesia::editor {
    std::vector<Workspace::ImageResult> image_results(const Workspace& workspace, const Workspace::Picture& image) {
        std::vector<Workspace::ImageResult> results;
        if (!image.file || image.preview) return results;
        auto models = workspace.activated;
        std::string annotation;
        if (workspace.page == Workspace::Page::audit && !workspace.audit_report.concept_key.empty()) {
            const auto& rows = workspace.audit_report.rows;
            const auto row   = std::ranges::find_if(rows, [&](const auto& value) { return value.sample.file.sha == image.file->sha; });
            if (row != rows.end()) {
                std::erase(models, workspace.audit_key);
                models.insert(models.begin(), workspace.audit_key);
                annotation = std::format("Label: {} · {:.1f}%", row->label, row->confidence * 100);
            }
        }
        for (const auto& key : models) {
            const auto source = workspace.library.classifiers.find(key);
            if (source == workspace.library.classifiers.end() || !source->second.model) continue;
            const auto identity = key + "|" + source->second.model->sha + "|" + image.file->sha;
            Workspace::ImageResult entry;
            if (key == workspace.audit_key) entry.annotation = annotation;
            const auto result = workspace.prediction_cache.entries.find({source->second.model->sha, image.file->sha});
            if (result != workspace.prediction_cache.entries.end()) {
                const auto& prediction = result->second;
                const auto index       = std::ranges::find(prediction.classes, prediction.label) - prediction.classes.begin();
                entry.summary          = std::format("{} · {} · {:.1f}%", key, prediction.label, prediction.scores[index] * 100);
                for (std::size_t i = 0; i < prediction.classes.size(); ++i) entry.distribution.push_back(std::format("{}  {:.2f}%", prediction.classes[i], prediction.scores[i] * 100));
            } else if (const auto error = workspace.prediction_errors.find(identity); error != workspace.prediction_errors.end()) {
                entry.summary    = key + " · Inference failed";
                entry.annotation = error->second;
                entry.failed     = true;
            } else entry.summary = key + " · Pending...";
            results.push_back(std::move(entry));
        }
        return results;
    }

    Workspace::ImageAction image_panel(Workspace& workspace, const char* id, const Workspace::Picture& image, const ImVec2 origin, const ImVec2 size, const float scale, const bool interactive, const float brightness) {
        if (!image.width || !image.height) return Workspace::ImageAction::none;
        auto* draw = ImGui::GetWindowDrawList();
        ImGui::SetCursorScreenPos(origin);
        ImGui::InvisibleButton(id, size, ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonMiddle);
        const bool hovered = ImGui::IsItemHovered();
        const auto& io     = ImGui::GetIO();
        const float inset  = image.role == Workspace::Role::image ? 0 : 8 * scale;
        const ImVec2 available{(size.x - 2 * inset) * io.DisplayFramebufferScale.x, (size.y - 2 * inset) * io.DisplayFramebufferScale.y};
        const ImVec2 dimensions{float(workspace.repaint && interactive ? workspace.repaint->source.width : image.width), float(workspace.repaint && interactive ? workspace.repaint->source.height : image.height)};
        const float fitted = std::min(available.x / dimensions.x, available.y / dimensions.y);
        const double now   = workspace.frame_time;
        if (interactive || (image.file && workspace.collection && image.file->sha == workspace.current_position().selected)) {
            workspace.view_available = available;
            workspace.view.update(available, dimensions, now);
        }
        const float zoom    = interactive ? workspace.view.zoom : fitted;
        const ImVec2 center = interactive ? workspace.view.center : ImVec2{0.5F, 0.5F};
        const ImVec2 extent{dimensions.x * zoom / io.DisplayFramebufferScale.x, dimensions.y * zoom / io.DisplayFramebufferScale.y};
        ImVec2 minimum{origin.x + size.x / 2 - center.x * extent.x, origin.y + size.y / 2 - center.y * extent.y};
        const bool over_image         = hovered && ImGui::IsMouseHoveringRect(minimum, {minimum.x + extent.x, minimum.y + extent.y});
        Workspace::ImageAction action = Workspace::ImageAction::none;
        if (image.texture && over_image && ImGui::IsMouseClicked(ImGuiMouseButton_Middle) && image.file) action = Workspace::ImageAction::repaint;
        if (interactive && image.texture) {
            const bool movable = extent.x > size.x - 2 * inset + 1 || extent.y > size.y - 2 * inset + 1;
            if (over_image && ImGui::IsMouseClicked(ImGuiMouseButton_Left) && movable && !ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
                workspace.view.dragging = true;
                workspace.view.started  = -1;
            }
            if (workspace.view.dragging && ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
                workspace.view.center.x -= io.MouseDelta.x * io.DisplayFramebufferScale.x / (dimensions.x * workspace.view.zoom);
                workspace.view.center.y -= io.MouseDelta.y * io.DisplayFramebufferScale.y / (dimensions.y * workspace.view.zoom);
                workspace.view.constrain(available, dimensions);
            }
            if (!ImGui::IsMouseDown(ImGuiMouseButton_Left)) workspace.view.dragging = false;
            if (hovered && io.MouseWheel && !workspace.view.dragging) {
                ImGui::SetKeyOwner(ImGuiKey_MouseWheelY, ImGui::GetItemID());
                const float next = std::clamp(workspace.view.target_zoom * std::pow(1.15F, io.MouseWheel), fitted, std::max(16.0F, fitted));
                const ImVec2 pivot{(io.MousePos.x - origin.x - size.x / 2) * io.DisplayFramebufferScale.x, (io.MousePos.y - origin.y - size.y / 2) * io.DisplayFramebufferScale.y};
                workspace.view.scale_to(next, pivot, dimensions, next <= fitted, now);
            }
            if (over_image && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) workspace.view.scale_to(workspace.view.fit ? std::max(1.0F, fitted) : fitted, {}, dimensions, !workspace.view.fit || fitted >= 1, now);
            if (over_image && movable) workspace.renderer.hand_cursor = workspace.view.dragging ? 1 : 0;
            minimum = {origin.x + size.x / 2 - workspace.view.center.x * extent.x, origin.y + size.y / 2 - workspace.view.center.y * extent.y};
        }
        if (image.texture && over_image && ImGui::IsMouseReleased(ImGuiMouseButton_Left) && !ImGui::IsMouseDragPastThreshold(ImGuiMouseButton_Left) && io.MouseClickedLastCount[ImGuiMouseButton_Left] == 1) action = Workspace::ImageAction::click;
        if (zoom == 1) {
            minimum.x = std::round(minimum.x * io.DisplayFramebufferScale.x) / io.DisplayFramebufferScale.x;
            minimum.y = std::round(minimum.y * io.DisplayFramebufferScale.y) / io.DisplayFramebufferScale.y;
        }
        const ImVec2 maximum{minimum.x + extent.x, minimum.y + extent.y};
        draw->PushClipRect(origin, {origin.x + size.x, origin.y + size.y}, true);
        if (image.texture) draw->AddImage(image.texture, minimum, maximum, {0, 0}, {1, 1}, ImGui::GetColorU32(ImVec4{brightness, brightness, brightness, 1}));
        else {
            const char* label = image.role == Workspace::Role::result ? "Waiting for image" : "Loading image";
            if (image.file) {
                const auto cached = workspace.textures.entries.find(image.file->sha);
                if (cached != workspace.textures.entries.end() && !cached->second.error.empty()) label = "Image read error";
            }
            const auto text = ImGui::CalcTextSize(label);
            draw->AddText({origin.x + (size.x - text.x) / 2, origin.y + (size.y - text.y) / 2}, ImGui::GetColorU32(ImGuiCol_TextDisabled), label);
        }
        if (image.confidence && image.role == Workspace::Role::image) {
            const float probability = *image.confidence;
            const ImVec4 color{std::lerp(0.94F, 0.25F, probability), std::lerp(0.25F, 0.82F, probability), std::lerp(0.28F, 0.52F, probability), 0.95F};
            draw->AddRect(minimum, maximum, ImGui::GetColorU32(color), 0, ImDrawFlags_None, 2 * scale);
        }
        if (image.role != Workspace::Role::image) {
            const ImVec4 first  = image.role == Workspace::Role::source ? ImVec4{0.25F, 0.86F, 0.57F, 1} : ImVec4{0.65F, 0.43F, 0.97F, 1};
            const ImVec4 second = image.role == Workspace::Role::source ? ImVec4{0.36F, 0.72F, 0.70F, 1} : ImVec4{0.32F, 0.58F, 0.97F, 1};
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
        const auto results = image.texture ? image_results(workspace, image) : std::vector<Workspace::ImageResult>{};
        if (!results.empty()) {
            const float left    = std::max(origin.x, minimum.x);
            const float right   = std::min(origin.x + size.x, maximum.x);
            const float top     = std::max(origin.y, minimum.y);
            const float bottom  = std::min(origin.y + size.y, maximum.y);
            const float padding = 12 * scale;
            float height{}, width{};
            for (const auto& entry : results) {
                height += ImGui::GetFontSize() + 8 * scale;
                width = std::max(width, ImGui::CalcTextSize(entry.summary.c_str()).x);
                if (!entry.annotation.empty()) {
                    height += ImGui::GetFontSize() + 4 * scale;
                    width = std::max(width, ImGui::CalcTextSize(entry.annotation.c_str()).x);
                }
            }
            width = std::min(width + 8 * scale, right - left - 2 * padding);
            ImVec2 position{left + padding, bottom - padding - height};
            const float text_right = position.x + width;
            draw->PushClipRect({left, top}, {right, bottom}, true);
            ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * brightness);
            const auto shade = ImGui::GetColorU32(ImVec4{0.025F, 0.03F, 0.04F, 0.62F});
            const auto clear = ImGui::GetColorU32(ImVec4{0.025F, 0.03F, 0.04F, 0});
            draw->AddRectFilledMultiColor({position.x - 6 * scale, position.y - 4 * scale}, {text_right + 6 * scale, bottom - padding + 4 * scale}, shade, clear, clear, shade);
            for (const auto& entry : results) {
                const float row_height = ImGui::GetFontSize() + 8 * scale + (entry.annotation.empty() ? 0 : ImGui::GetFontSize() + 4 * scale);
                const auto ink         = entry.failed ? ImVec4{0.95F, 0.49F, 0.42F, 1} : ImGui::GetStyleColorVec4(ImGuiCol_Text);
                control_text(entry.summary, {position.x + 4 * scale, position.y + 4 * scale}, text_right, ink, scale);
                if (!entry.annotation.empty()) control_text(entry.annotation, {position.x + 4 * scale, position.y + ImGui::GetFontSize() + 8 * scale}, text_right, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled), scale);
                if (hovered && !workspace.view.dragging && ImGui::IsMouseHoveringRect(position, {text_right, position.y + row_height})) {
                    ImGui::BeginTooltip();
                    ImGui::TextUnformatted(entry.summary.c_str());
                    if (!entry.annotation.empty()) ImGui::TextUnformatted(entry.annotation.c_str());
                    if (!entry.distribution.empty()) ImGui::Separator();
                    for (const auto& probability : entry.distribution) ImGui::TextUnformatted(probability.c_str());
                    ImGui::EndTooltip();
                }
                position.y += row_height;
            }
            ImGui::PopStyleVar();
            draw->PopClipRect();
        }
        return action;
    }

    void canvas(Workspace& workspace, const float scale, const ImVec2 size) {
        ImGui::SetNextWindowPos({0, 0});
        ImGui::SetNextWindowSize(size);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {0, 0});
        ImGui::Begin("##Canvas", nullptr, overlay | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoBringToFrontOnFocus);
        ImGui::PopStyleVar();
        const auto origin    = workspace.canvas_origin;
        const auto available = workspace.canvas_size;
        // Navigate on press so releasing a popup-dismissal click cannot also change the view.
        const bool navigating = ImGui::IsWindowHovered() && ImGui::IsMouseHoveringRect(origin, {origin.x + available.x, origin.y + available.y}) && !ImGui::IsPopupOpen(nullptr, ImGuiPopupFlags_AnyPopupId | ImGuiPopupFlags_AnyPopupLevel) && ImGui::IsMouseClicked(ImGuiMouseButton_Right);
        ImGui::PushClipRect(origin, {origin.x + available.x, origin.y + available.y}, true);
        std::vector<dataset::File> wanted;
        std::optional<dataset::File> repaint_source;
        if (workspace.page == Workspace::Page::generation) {
            if (workspace.generation.saved) wanted.push_back(*workspace.generation.saved);
            const auto image = workspace.resolve_output(workspace.generation);
            if (image.texture) {
                if (image_panel(workspace, "##GeneratedImage", image, origin, available, scale, true) == Workspace::ImageAction::repaint) repaint_source = image.file;
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
        } else if (!workspace.collection || !workspace.root || !workspace.root->ready) {
            ImGui::SetCursorScreenPos({origin.x + 16 * scale, (top_strip_height + 24) * scale});
            if (!workspace.library.ready) ImGui::TextDisabled("Indexing datasets...");
            else if (!workspace.collection || !workspace.root) ImGui::TextWrapped("Dataset does not exist: %s", workspace.collection_key.c_str());
            else {
                ImGui::TextColored({0.95F, 0.49F, 0.42F, 1}, "Dataset needs attention");
                if (!workspace.root->error.empty()) ImGui::TextWrapped("%s", workspace.root->error.c_str());
                if (!workspace.root->conflicts.empty()) ImGui::TextWrapped("Identical images are stored as independent files. Open the dataset sidebar to see all conflict paths.");
            }
        } else if (workspace.repaint) {
            wanted.push_back(workspace.repaint->source);
            if (workspace.repaint->result.saved) wanted.push_back(*workspace.repaint->result.saved);
            const auto source = workspace.resolve_image(workspace.repaint->source, Workspace::Role::source);
            const auto result = workspace.resolve_output(workspace.repaint->result, Workspace::Role::result);
            if (workspace.viewing == Workspace::View::comparison) {
                const bool horizontal = workspace.repaint->source.width <= workspace.repaint->source.height;
                const float gap       = 20 * scale;
                const ImVec2 pane{horizontal ? (available.x - gap) / 2 : available.x, horizontal ? available.y : (available.y - gap) / 2};
                const ImVec2 next{origin.x + (horizontal ? pane.x + gap : 0), origin.y + (horizontal ? 0 : pane.y + gap)};
                Workspace::ImageAction first, second;
                // Process the hovered pane first so both draws use the same camera this frame.
                if (ImGui::GetActiveID() == ImGui::GetID("##Result") || (ImGui::GetActiveID() != ImGui::GetID("##Source") && ImGui::IsMouseHoveringRect(next, {next.x + pane.x, next.y + pane.y}))) {
                    second = image_panel(workspace, "##Result", result, next, pane, scale, true);
                    first  = image_panel(workspace, "##Source", source, origin, pane, scale, true);
                } else {
                    first  = image_panel(workspace, "##Source", source, origin, pane, scale, true);
                    second = image_panel(workspace, "##Result", result, next, pane, scale, true);
                }
                if (first == Workspace::ImageAction::repaint) repaint_source = source.file;
                else if (second == Workspace::ImageAction::repaint) repaint_source = result.file;
                else if (first == Workspace::ImageAction::click) workspace.viewing = Workspace::View::source;
                else if (second == Workspace::ImageAction::click) workspace.viewing = Workspace::View::result;
            } else {
                const auto image = workspace.viewing == Workspace::View::result ? result : source;
                if (image_panel(workspace, "##RepaintImage", image, origin, available, scale, true) == Workspace::ImageAction::repaint) repaint_source = image.file;
            }
        } else if (workspace.collection->images.empty()) {
            ImGui::SetCursorScreenPos({origin.x + 16 * scale, (top_strip_height + 24) * scale});
            ImGui::TextDisabled("This dataset is empty.");
        } else {
            auto& position = workspace.current_position();
            if (workspace.viewing == Workspace::View::inspect) {
                const auto& file = workspace.collection->images[position.index];
                wanted.push_back(file);
                const auto image = workspace.resolve_image(file);
                if (image_panel(workspace, "##InspectedImage", image, origin, available, scale, true) == Workspace::ImageAction::repaint) repaint_source = file;
            } else {
                wanted.push_back(workspace.collection->images[position.index]);
                const auto& io = ImGui::GetIO();
                if (ImGui::IsWindowHovered() && ImGui::IsMouseHoveringRect(origin, {origin.x + available.x, origin.y + available.y}) && io.MouseWheel) {
                    const auto index = std::clamp<std::ptrdiff_t>(static_cast<std::ptrdiff_t>(position.index) - (io.MouseWheel > 0 ? 1 : -1), 0, workspace.collection->images.size() - 1);
                    workspace.center_image(static_cast<std::size_t>(index));
                    ImGui::SetKeyOwner(ImGuiKey_MouseWheelY, ImGui::GetID("##DatasetScroll"));
                }
                position.scroll = std::lerp(position.scroll, static_cast<float>(position.index), std::min(1.0F, io.DeltaTime / 0.055F));
                if (std::abs(position.scroll - position.index) < 0.001F) position.scroll = static_cast<float>(position.index);
                else workspace.animate_until = glfwGetTime() + 0.1;
                const auto anchor = static_cast<std::size_t>(std::floor(position.scroll));
                const auto first  = anchor > 3 ? anchor - 3 : 0;
                const auto last   = std::min(workspace.collection->images.size(), anchor + 5);
                const float gap   = 24 * scale;
                const auto width  = [&](const std::size_t index) {
                    const auto& file = workspace.collection->images[index];
                    return std::min(available.x * 0.68F, available.y * file.width / file.height);
                };
                const float advance = anchor + 1 < workspace.collection->images.size() ? (width(anchor) + width(anchor + 1)) / 2 + gap : 0;
                float x             = origin.x + available.x / 2 - (position.scroll - anchor) * advance;
                for (auto i = anchor; i > first; --i) x -= (width(i) + width(i - 1)) / 2 + gap;
                for (auto i = first; i < last; ++i) {
                    const auto& file = workspace.collection->images[i];
                    wanted.push_back(file);
                    const float image_width = width(i);
                    const float brightness  = std::lerp(1.0F, 0.60F, std::min(1.0F, std::abs(float(i) - position.scroll)));
                    ImGui::PushID(static_cast<int>(i));
                    const auto action = image_panel(workspace, "##DatasetImage", workspace.resolve_image(file), {x - image_width / 2, origin.y}, {image_width, available.y}, scale, false, brightness);
                    ImGui::PopID();
                    if (action == Workspace::ImageAction::repaint) repaint_source = file;
                    else if (action == Workspace::ImageAction::click) {
                        if (i == position.index && position.scroll == float(position.index)) {
                            workspace.viewing = Workspace::View::inspect;
                            workspace.view    = {};
                        } else workspace.center_image(i);
                    }
                    if (i + 1 < last) x += (image_width + width(i + 1)) / 2 + gap;
                }
            }
        }
        for (const auto& file : wanted) {
            const auto cached = workspace.textures.entries.find(file.sha);
            if (cached != workspace.textures.entries.end() && !cached->second.error.empty()) workspace.action_error = cached->second.error;
        }
        if ((workspace.page == Workspace::Page::dataset || workspace.page == Workspace::Page::audit) && !workspace.repaint && workspace.collection && !workspace.collection->images.empty()) {
            const auto selected = std::ranges::find(wanted, workspace.collection->images[workspace.current_position().index].sha, &dataset::File::sha);
            if (selected != wanted.end()) std::rotate(wanted.begin(), selected, selected + 1);
        }
        workspace.visible_images.clear();
        for (const auto& file : wanted)
            if (!std::ranges::contains(workspace.visible_images, file.sha, &dataset::File::sha)) workspace.visible_images.push_back(file);
        if (workspace.pending_delete) {
            const auto root = *workspace.pending_delete->path.lexically_relative(project::directory).begin();
            std::erase_if(wanted, [&](const dataset::File& file) { return file.sha == workspace.pending_delete->sha && *file.path.lexically_relative(project::directory).begin() == root; });
        }
        workspace.textures.request(std::move(wanted));
        ImGui::PopClipRect();
        ImGui::End();
        if (navigating) {
            if (workspace.page == Workspace::Page::generation) workspace.select_collection("raw", workspace.generation.saved ? workspace.generation.saved->sha : std::string{});
            else workspace.back();
        } else if (repaint_source) workspace.start_repaint(*repaint_source);
    }
} // namespace genesia::editor
