module;
#include <Windows.h>
#include <GLFW/glfw3.h>
#include "../../core/sdxl/control.h"
#include <genesia/cuda.h>
#include <imgui.h>
#include <imgui_internal.h>
#include <shellapi.h>
#include <nlohmann/json.hpp>
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
import genesia.files;

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

    UserInterface::TrainingDraft::TrainingDraft(const classifier::TrainingData& source) : config{classifier::configuration()} {
        if (!source.training.is_null()) {
            config = source.training.at("config");
            steps  = source.training.at("target").get<int>();
            if (source.training.at("state") == "complete") steps = source.training.at("step").get<int>() + 400;
        }
        const auto directory = source.root / ".genesia" / "classifier" / "metrics";
        if (std::filesystem::exists(directory / "history.json")) {
            const auto history = files::read_json(directory / "history.json");
            if (!history.empty()) metrics = history.back();
        }
        if (std::filesystem::exists(directory / "steps.csv")) {
            std::ifstream log{directory / "steps.csv"};
            std::string line;
            std::getline(log, line);
            while (std::getline(log, line)) {
                const auto first  = line.find(',');
                const auto second = line.find(',', first + 1);
                losses.push_back(std::stof(line.substr(first + 1, second - first - 1)));
            }
        }
    }

    UserInterface::UserInterface(prompt::Preset preset, std::shared_ptr<const prompt::Catalog> catalog, WindowPlatform& platform, Renderer& display, std::string dataset) : catalog{std::move(catalog)}, preset{std::move(preset)}, window{platform}, renderer{display}, library{this->catalog, display}, prompt{this->preset.prompt}, tag_search{*this->catalog} {
        prompt_editor.reset(prompt);
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
        if (revision != library.revision) {
            std::erase_if(activated, [&](const auto& key) {
                const auto found = library.classifiers.find(key);
                return found == library.classifiers.end() || found->second.training.is_null() || found->second.training.at("published").is_null();
            });
            if (!audit_key.empty()) rebuild_audit();
        }
        synchronize_collection();
        if (runtime) {
            auto& session = runtime->session;
            std::deque<work::Event> events;
            std::deque<work::PreviewFrame> previews;
            {
                const std::lock_guard lock{session.mutex};
                events.swap(session.events);
                previews.swap(session.previews);
            }
            for (const auto& frame : previews) {
                auto& source           = runtime->preview->slots[frame.slot];
                Output* output         = !frame.from_image && (!generation.task || frame.id >= *generation.task) ? &generation : repaint && repaint->result.task == frame.id ? &repaint->result : nullptr;
                const bool final_ready = std::ranges::any_of(events, [&](const work::Event& event) { return event.kind == work::EventKind::generated && event.id == frame.id; });
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
                if (event.kind == work::EventKind::task) {
                    task_event(event.data);
                    continue;
                }
                Output* output = event.record.source.empty() && (!generation.task || event.id >= *generation.task) ? &generation : repaint && repaint->result.task == event.id ? &repaint->result : nullptr;
                if (output && output->task != event.id) begin_output(*output, event.id, event.record.parameters.width, event.record.parameters.height);
                if (event.kind == work::EventKind::generated) {
                    auto& source = runtime->interop->slots[event.slot];
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
                    if (output) {
                        output->record  = std::move(event.record);
                        output->preview = false;
                    }
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
        if (page == Page::audit) collection = &audit_collection;
        if (library.ready && page == Page::dataset && !collection) action_error = "Dataset does not exist: " + collection_key;
        if (collection && root && root->ready && !collection->images.empty()) {
            if (current_position().selected.empty()) {
                const auto index   = collection->images.size() - 1;
                current_position() = {collection->images[index].sha, index, static_cast<float>(index)};
            }
            if (locate) {
                const auto member = std::ranges::find(root->files, *locate, &dataset::File::path);
                if (member != root->files.end()) {
                    const auto found = std::ranges::find(collection->images, member->sha, &dataset::File::sha);
                    if (found != collection->images.end()) {
                        const auto index   = static_cast<std::size_t>(found - collection->images.begin());
                        current_position() = {found->sha, index, static_cast<float>(index)};
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
        page    = Page::dataset;
        viewing = View::browse;
        if (concept_tool == ConceptTool::audit) concept_tool = ConceptTool::none;
        collection_key = std::move(key);
        choosing_type  = false;
        type_error.clear();
        locate = std::move(target);
        synchronize_collection();
        view          = {};
        window.redraw = true;
    }

    UserInterface::Position& UserInterface::current_position() {
        return page == Page::audit ? audit_position : positions[collection_key];
    }

    void UserInterface::center_image(const std::size_t index) {
        locate.reset();
        auto& position    = current_position();
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
        } else if (page == Page::audit) {
            if (concept_tool == ConceptTool::audit) concept_tool = ConceptTool::none;
            page           = audit_return;
            viewing        = audit_return_view;
            collection_key = audit_return_collection;
            view           = audit_return_camera;
            synchronize_collection();
        } else {
            page = Page::generation;
            view = generation_view;
        }
    }

    UserInterface::Picture UserInterface::resolve_image(const dataset::File& file, const Role role) const {
        Picture picture{0, file.width, file.height, nullptr, file, false, role};
        if (page == Page::audit && role == Role::image && !audit_report.is_null()) {
            const auto& rows = audit_report.at("rows");
            const auto row   = std::ranges::find_if(rows, [&](const auto& item) { return item.at("sha") == file.sha; });
            if (row != rows.end()) picture.confidence = row->at("confidence").template get<float>();
        }
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
        std::optional<work::RepaintSource> source;
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
        work::Request request;
        request.parameters = parameters;
        request.seed       = seed;
        request.prompt     = std::move(submitted);
        request.catalog    = std::move(prompt_catalog);
        request.source     = std::move(source);
        const auto task    = submit_task(std::move(request));
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
        return layout;
    }

    std::vector<UserInterface::ImageResult> UserInterface::image_results(const Picture& image) const {
        std::vector<ImageResult> results;
        if (!image.file || image.preview) return results;
        auto models = activated;
        std::string annotation;
        if (page == Page::audit && !audit_report.is_null()) {
            const auto& rows = audit_report.at("rows");
            const auto row   = std::ranges::find_if(rows, [&](const auto& value) { return value.at("sha") == image.file->sha; });
            if (row != rows.end()) {
                std::erase(models, audit_key);
                models.insert(models.begin(), audit_key);
                annotation = std::format("Label: {} · {:.1f}%", row->at("label").get_ref<const std::string&>(), row->at("confidence").get<float>() * 100);
            }
        }
        for (const auto& key : models) {
            const auto source = library.classifiers.find(key);
            if (source == library.classifiers.end() || source->second.training.is_null() || source->second.training.at("published").is_null()) continue;
            const auto identity = key + "|" + source->second.training.at("published").at("sha").get<std::string>() + "|" + image.file->sha;
            ImageResult entry;
            if (key == audit_key) entry.annotation = annotation;
            const auto result = predictions.find(identity);
            if (result != predictions.end()) {
                const auto& prediction = result->second;
                const auto index       = std::ranges::find(prediction.classes, prediction.label) - prediction.classes.begin();
                entry.summary          = std::format("{} · {} · {:.1f}%", key, prediction.label, prediction.scores[index] * 100);
                for (std::size_t i = 0; i < prediction.classes.size(); ++i) entry.distribution.push_back(std::format("{}  {:.2f}%", prediction.classes[i], prediction.scores[i] * 100));
            } else if (const auto error = prediction_errors.find(identity); error != prediction_errors.end()) {
                entry.summary    = key + " · Inference failed";
                entry.annotation = error->second;
                entry.failed     = true;
            } else entry.summary = key + " · Pending...";
            results.push_back(std::move(entry));
        }
        return results;
    }

    UserInterface::ImageAction UserInterface::image_panel(const char* id, const Picture& image, const ImVec2 origin, const ImVec2 size, const float scale, const bool interactive, const float brightness) {
        if (!image.width || !image.height) return ImageAction::none;
        auto* draw = ImGui::GetWindowDrawList();
        ImGui::SetCursorScreenPos(origin);
        ImGui::InvisibleButton(id, size, ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonMiddle);
        const bool hovered = ImGui::IsItemHovered();
        const auto& io     = ImGui::GetIO();
        const float inset  = image.role == Role::image ? 0 : 8 * scale;
        const ImVec2 available{(size.x - 2 * inset) * io.DisplayFramebufferScale.x, (size.y - 2 * inset) * io.DisplayFramebufferScale.y};
        const ImVec2 dimensions{float(repaint && interactive ? repaint->source.width : image.width), float(repaint && interactive ? repaint->source.height : image.height)};
        const float fitted = std::min(available.x / dimensions.x, available.y / dimensions.y);
        const double now   = frame_time;
        if (interactive || (image.file && collection && image.file->sha == current_position().selected)) {
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
        if (image.confidence && image.role == Role::image) {
            const float probability = *image.confidence;
            const ImVec4 color{std::lerp(0.94F, 0.25F, probability), std::lerp(0.25F, 0.82F, probability), std::lerp(0.28F, 0.52F, probability), 0.95F};
            draw->AddRect(minimum, maximum, ImGui::GetColorU32(color), 0, ImDrawFlags_None, 2 * scale);
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
        const auto results = image.texture ? image_results(image) : std::vector<ImageResult>{};
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
                if (hovered && !view.dragging && ImGui::IsMouseHoveringRect(position, {text_right, position.y + row_height})) {
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
        } else if (!library.error.empty() || !collection || !root || !root->ready) {
            ImGui::SetCursorScreenPos({origin.x + 16 * scale, (top_strip_height + 24) * scale});
            if (!library.error.empty()) ImGui::TextWrapped("%s", library.error.c_str());
            else if (!library.ready) ImGui::TextDisabled("Indexing datasets...");
            else if (!collection || !root) ImGui::TextWrapped("Dataset does not exist: %s", collection_key.c_str());
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
            auto& position = current_position();
            if (viewing == View::inspect) {
                const auto& file = collection->images[position.index];
                wanted.push_back(file);
                const auto image = resolve_image(file);
                if (image_panel("##InspectedImage", image, origin, available, scale, true) == ImageAction::repaint) repaint_source = file;
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
        if ((page == Page::dataset || page == Page::audit) && !repaint && collection && !collection->images.empty()) {
            const auto selected = std::ranges::find(wanted, collection->images[current_position().index].sha, &dataset::File::sha);
            if (selected != wanted.end()) std::rotate(wanted.begin(), selected, selected + 1);
        }
        visible_images.clear();
        for (const auto& file : wanted)
            if (!std::ranges::contains(visible_images, file.sha, &dataset::File::sha)) visible_images.push_back(file);
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
        if (std::ranges::any_of(task_status, [](const auto& entry) { return entry.second.at("kind") == static_cast<int>(work::Kind::generate); })) {
            ImGui::Separator();
            ImGui::TextDisabled("IMAGE GENERATION");
            operation_activity({work::Kind::generate}, "");
        }
        if (ImGui::IsKeyPressed(ImGuiKey_Escape, false)) ImGui::CloseCurrentPopup();
        ImGui::PopStyleVar();
        ImGui::EndPopup();
    }

    void UserInterface::top_strip(const float scale, const ImVec2 size) {
        bool active{}, loaded{true}, failed{}, unavailable{};
        std::optional<std::uint64_t> stopping_image;
        int steps{};
        work::Kind active_kind{work::Kind::generate};
        double elapsed{};
        if (runtime) {
            const auto& session = runtime->session;
            const std::lock_guard lock{runtime->session.mutex};
            active      = session.active.has_value();
            loaded      = session.model_ready;
            failed      = !session.error.empty();
            unavailable = session.worker_done;
            if (active) {
                active_kind = session.active->kind;
                steps       = session.active->parameters.steps;
                elapsed     = std::chrono::duration<double>(std::chrono::steady_clock::now() - session.started).count();
                if (active_kind == work::Kind::generate && session.jobs.at(session.active->id).at("state") == "running") stopping_image = session.active->id;
            }
        }
        const auto stage         = runtime ? static_cast<sdxl::Stage>(::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>{runtime->session.control.data()[0].stage}.load()) : sdxl::Stage::idle;
        const auto completed     = runtime ? ::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>{runtime->session.control.data()[0].completed}.load() : 0;
        const bool stopping      = active && ::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>{runtime->session.control.data()[0].cancel}.load();
        const bool working       = active && !failed;
        const bool indeterminate = working && (active_kind != work::Kind::generate || stage != sdxl::Stage::sampling || stopping);
        const double now         = glfwGetTime();
        if (working) {
            progress_alpha = 1;
            if (stopping) progress_label = "Stopping";
            else if (!loaded) progress_label = "Loading model";
            else if (stage == sdxl::Stage::sampling) progress_label = std::format("{} / {}", completed, steps);
            else if (stage == sdxl::Stage::decoding || stage == sdxl::Stage::transferring || stage == sdxl::Stage::complete) progress_label = "Finishing image";
            else progress_label = "Preparing";
            if (!stopping && active_kind != work::Kind::generate) progress_label = std::array{"Generate", "Training", "Inference", "Audit", "Moving image", "Undo", "Classifying folder"}[static_cast<int>(active_kind)];
            progress_time = active ? std::format("{:.1f}s", elapsed) : "";
        } else {
            progress_alpha = failed ? 0 : std::max(0.0F, progress_alpha - ImGui::GetIO().DeltaTime / 0.15F);
            if (!failed && (page == Page::generation || repaint)) {
                bool latest = true;
                for (const auto& [id, task] : std::views::reverse(task_status)) {
                    if (task.at("kind") != static_cast<int>(work::Kind::generate)) continue;
                    const auto& state = task.at("state");
                    if (state == "saving" || state == "queued" || (latest && state == "failed")) {
                        progress_label = state == "saving" ? "Saving image" : state == "queued" ? "Image queued" : "Generation failed";
                        progress_time.clear();
                        progress_alpha = 1;
                        active_kind    = work::Kind::generate;
                        break;
                    }
                    latest = false;
                }
            }
        }
        if (working) refresh_at = now + (indeterminate ? 1.0 / 30 : 0.1);
        else if (progress_alpha > 0 && progress_alpha < 1) refresh_at = now + 1.0 / 60;

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
            const auto count  = collection ? collection->images.size() : 0;
            const auto number = count ? current_position().index + 1 : 0;
            const auto label  = std::format("{}  \xC2\xB7  {} / {}", collection_key, number, count);
            if (text_button("##Location", label.c_str(), scale, std::min(ImGui::CalcTextSize(label.c_str()).x + 24 * scale, size.x * 0.42F))) {
                dataset_sidebar.open = !dataset_sidebar.open;
                if (dataset_sidebar.open) expand_dataset_roots = true;
            }
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
        const bool submission     = page == Page::generation || repaint.has_value();
        const float primary_width = submission ? 150 * scale : 0;
        const float stop_width    = stopping_image ? ImGui::CalcTextSize("Stop image").x + 28 * scale : 0;
        const char* label         = progress_alpha > 0 ? progress_label.c_str() : "";
        const float alpha         = progress_alpha;
        const float right_start   = maximum.x - primary_width - stop_width;
        const float label_width   = *label ? std::max(ImGui::CalcTextSize("Finishing image").x, ImGui::CalcTextSize(label).x) : 0;
        const auto time_text      = ImGui::CalcTextSize(progress_time.c_str());
        const float time_width    = std::max(ImGui::CalcTextSize("0000.0s").x, time_text.x);
        const bool show_time      = *label && !progress_time.empty() && label_width + 16 * scale + time_width <= right_start - left_end - 32 * scale;
        const float status_width  = label_width + (show_time ? 16 * scale + time_width : 0);
        const float group_left    = right_start - status_width - (*label ? 12 * scale : 0) - 8 * scale;
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
            if (ImGui::IsItemHovered()) {
                if (submission && active_kind == work::Kind::generate) {
                    ImGui::SetTooltip("Click: image generation progress and cancellation");
                    if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) ImGui::OpenPopup("Generation settings");
                } else if (!progress_time.empty()) ImGui::SetTooltip("Elapsed: %s", progress_time.c_str());
            }
        }
        if (stopping_image) {
            ImGui::SetCursorScreenPos({right_start, minimum.y + 4 * scale});
            ImGui::BeginDisabled(stopping);
            if (text_button("##StopImage", "Stop image", scale)) runtime->session.cancel(*stopping_image);
            ImGui::EndDisabled();
        }
        if (submission) {
            const bool valid   = repaint ? root && root->ready && repaints.at(repaint->source.sha)->editor.valid : prompt_editor.valid;
            const bool enabled = !unavailable && !ImGui::GetTopMostPopupModal() && valid;
            if (ImGui::IsPopupOpen("Generation settings") && ImGui::IsWindowHovered(ImGuiHoveredFlags_AllowWhenBlockedByPopup | ImGuiHoveredFlags_AllowWhenBlockedByActiveItem) && ImGui::IsMouseHoveringRect({maximum.x - primary_width, minimum.y}, maximum) && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                commit_parameters();
                ImGui::ClosePopupsOverWindow(ImGui::GetCurrentWindow(), false);
            }
            ImGui::SetCursorScreenPos({maximum.x - primary_width, minimum.y});
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
            if (hovered && !ImGui::IsPopupOpen("Generation settings")) ImGui::SetTooltip("%s\nCtrl+Shift+`\nRight-click: settings, image progress and cancellation", repaint ? "Repaint the green source into Raw" : "Generate a new image into Raw");
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
        if (left) {
            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, {8 * scale, 5 * scale});
            ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {8 * scale, 6 * scale});
            ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 4 * scale);
            ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 0);
            ImGui::PushStyleVar(ImGuiStyleVar_ScrollbarSize, 5 * scale);
            ImGui::PushStyleColor(ImGuiCol_Header, {0.32F, 0.60F, 0.65F, 0.18F});
            ImGui::PushStyleColor(ImGuiCol_HeaderHovered, {0.55F, 0.65F, 0.69F, 0.12F});
            ImGui::PushStyleColor(ImGuiCol_HeaderActive, {0.32F, 0.60F, 0.65F, 0.25F});
            ImGui::PushStyleColor(ImGuiCol_Button, {0.18F, 0.21F, 0.24F, 0.65F});
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, {0.24F, 0.30F, 0.33F, 0.75F});
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, {0.28F, 0.37F, 0.40F, 0.85F});
            ImGui::PushStyleColor(ImGuiCol_PlotLines, {0.44F, 0.80F, 0.87F, 0.85F});
            ImGui::PushStyleColor(ImGuiCol_PlotHistogram, {0.44F, 0.80F, 0.87F, 0.75F});
            ImGui::PushStyleColor(ImGuiCol_Separator, {0.70F, 0.72F, 0.85F, 0.12F});
        }
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
            if (left) dataset_controls(scale);
            if (ImGui::BeginChild("##SidebarContent", {0, 0}, ImGuiChildFlags_None, ImGuiWindowFlags_NoBackground)) {
                if (left) selected = dataset_contents();
                else if (page == Page::generation) prompt_editor.draw(prompt, tag_search, *catalog, scale);
                else if (repaint) {
                    auto& edits = *repaints.at(repaint->source.sha);
                    ImGui::PushID(repaint->source.sha.c_str());
                    edits.editor.draw(edits.prompt, tag_search, *edits.catalog, scale);
                    ImGui::PopID();
                } else if (image.record) show_prompt(image.record->prompt, *image.record->catalog, scale);
                else ImGui::TextDisabled(image.file ? "Loading prompt..." : "No image selected.");
            }
            ImGui::EndChild();
            ImGui::PopStyleVar();
        }
        ImGui::End();
        if (left) {
            ImGui::PopStyleColor(9);
            ImGui::PopStyleVar(5);
        }
        ImGui::PopStyleVar(2);
        if (selected) select_collection(std::move(*selected));
    }

    void UserInterface::dataset_controls(const float scale) {
        if (!library.error.empty()) ImGui::TextWrapped("%s", library.error.c_str());
        else if (collection && root) {
            const auto title = ImGui::GetCursorScreenPos();
            const ImVec2 title_size{ImGui::GetContentRegionAvail().x, ImGui::GetTextLineHeight()};
            ImGui::Dummy(title_size);
            ImGui::RenderTextEllipsis(ImGui::GetWindowDrawList(), title, {title.x + title_size.x, title.y + title_size.y}, title.x + title_size.x, collection->name.c_str(), nullptr, nullptr);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", collection_key.c_str());
            if (collection_key == root->all.key) ImGui::TextDisabled("%zu images", collection->images.size());
            else {
                ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
                ImGui::TextWrapped("%s / %zu images", root->all.name.c_str(), collection->images.size());
                ImGui::PopStyleColor();
            }
            const auto metadata = library.concepts.find(collection_key);
            const auto failure  = library.concept_errors.find(collection_key);
            if (metadata != library.concepts.end()) {
                const auto& assigned = metadata->second;
                const bool busy      = std::ranges::any_of(task_status, [&](const auto& entry) { return entry.second.at("concept") == collection_key && (entry.second.at("state") == "queued" || entry.second.at("state") == "running"); });
                static constexpr std::array names{"Unassigned", "Classifier", "LoRA"};
                const auto color = assigned.type == dataset::ConceptType::classifier ? ImVec4{0.44F, 0.80F, 0.87F, 1} : assigned.type == dataset::ConceptType::lora ? ImVec4{0.92F, 0.69F, 0.36F, 1} : ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled);
                ImGui::BeginDisabled(assigned.locked || busy);
                ImGui::PushStyleColor(ImGuiCol_Text, color);
                ImGui::PushStyleColor(ImGuiCol_Button, {color.x, color.y, color.z, choosing_type ? 0.22F : 0.10F});
                ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 5 * scale);
                if (ImGui::SmallButton(std::format("{}{}###ConceptType", names[static_cast<std::size_t>(assigned.type)], assigned.locked ? "" : "  v").c_str())) choosing_type = !choosing_type;
                ImGui::PopStyleVar();
                ImGui::PopStyleColor(2);
                ImGui::EndDisabled();
                if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) ImGui::SetTooltip("%s", assigned.locked ? "Type is permanently locked by training history. Restart keeps this type." : busy ? "Finish or cancel this concept's queued operations before changing its type." : "Choose this concept's purpose. Assigning a type does not start training.");
                if (assigned.locked || busy) choosing_type = false;
                if (choosing_type) {
                    for (std::size_t i = 0; i < names.size(); ++i)
                        if (ImGui::Selectable(names[i], static_cast<std::size_t>(assigned.type) == i)) {
                            if (assigned.type == static_cast<dataset::ConceptType>(i)) {
                                choosing_type = false;
                                type_error.clear();
                                break;
                            }
                            try {
                                metadata->second = dataset::assign_type(collection_key, static_cast<dataset::ConceptType>(i));
                                library.classifiers.erase(collection_key);
                                training_drafts.erase(collection_key);
                                concept_tool  = ConceptTool::none;
                                choosing_type = false;
                                type_error.clear();
                                library.refresh();
                            } catch (const std::exception& error) {
                                type_error = error.what();
                            }
                            break;
                        }
                }
                if (!type_error.empty()) ImGui::TextWrapped("%s", type_error.c_str());
                if (failure != library.concept_errors.end()) ImGui::TextWrapped("%s", failure->second.c_str());
                else if (assigned.type == dataset::ConceptType::lora) {
                    ImGui::TextWrapped("LoRA training is not available yet.");
                    ImGui::BeginDisabled();
                    ImGui::Button("Train LoRA", {-FLT_MIN, 0});
                    ImGui::EndDisabled();
                } else if (assigned.type == dataset::ConceptType::classifier) {
                    const auto info = library.classifiers.find(collection_key);
                    if (info == library.classifiers.end()) ImGui::TextDisabled("Reading classifier data...");
                    else {
                        const auto& source   = info->second;
                        const bool published = !source.training.is_null() && !source.training.at("published").is_null();
                        const bool active    = std::ranges::contains(activated, collection_key);
                        if (!choosing_type && type_error.empty()) ImGui::SameLine(0, 8 * scale);
                        if (published) ImGui::TextColored(color, "%s", active ? "Active" : "Model available");
                        else ImGui::TextDisabled("No published model");
                        if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", published ? "Middle-click this concept in the dataset list to activate or deactivate its model." : "A model becomes available after training reaches its target and finishes saving.");
                        const bool audit_available    = published || std::ranges::any_of(task_status, [&](const auto& entry) {
                            const auto kind = static_cast<work::Kind>(entry.second.at("kind").template get<int>());
                            return entry.second.at("concept") == collection_key && (kind == work::Kind::audit || kind == work::Kind::fix || kind == work::Kind::undo);
                        });
                        const bool classify_available = published || std::ranges::any_of(task_status, [&](const auto& entry) { return entry.second.at("concept") == collection_key && entry.second.at("kind") == static_cast<int>(work::Kind::classify); });
                        const int buttons             = 1 + int(audit_available) + int(classify_available);
                        const float gap               = 4 * scale;
                        const float width             = (ImGui::GetContentRegionAvail().x - (buttons - 1) * gap) / buttons;
                        ImGui::Spacing();
                        for (const auto [tool, label] : {std::pair{ConceptTool::train, "Train"}, std::pair{ConceptTool::audit, "Audit"}, std::pair{ConceptTool::classify, "Classify"}}) {
                            if ((tool == ConceptTool::audit && !audit_available) || (tool == ConceptTool::classify && !classify_available)) continue;
                            if (tool != ConceptTool::train) ImGui::SameLine(0, gap);
                            const bool current = concept_tool == tool;
                            ImGui::PushStyleColor(ImGuiCol_Button, {color.x, color.y, color.z, current ? 0.08F : 0});
                            ImGui::PushStyleColor(ImGuiCol_Text, current ? ImGui::GetStyleColorVec4(ImGuiCol_Text) : ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
                            if (ImGui::Button(label, {width, 30 * scale})) {
                                if (tool == ConceptTool::audit && (page != Page::audit || repaint)) {
                                    open_audit(collection_key);
                                    if (page == Page::audit && !repaint) concept_tool = ConceptTool::audit;
                                } else concept_tool = current ? ConceptTool::none : tool;
                            }
                            ImGui::PopStyleColor(2);
                            if (concept_tool == tool) {
                                const auto minimum = ImGui::GetItemRectMin();
                                const auto maximum = ImGui::GetItemRectMax();
                                ImGui::GetWindowDrawList()->AddLine({minimum.x + 8 * scale, maximum.y - scale}, {maximum.x - 8 * scale, maximum.y - scale}, ImGui::GetColorU32(color), 2 * scale);
                            }
                        }
                        if (concept_tool != ConceptTool::none) {
                            ImGui::PushID(collection_key.c_str());
                            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {4 * scale, 6 * scale});
                            ImGui::SetNextWindowSizeConstraints({0, 0}, {FLT_MAX, std::max(1.0F, ImGui::GetContentRegionAvail().y * 0.45F)});
                            if (ImGui::BeginChild("##ConceptTool", {0, 0}, ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_AlwaysUseWindowPadding, ImGuiWindowFlags_NoBackground)) {
                                if (concept_tool == ConceptTool::train) training_controls(source, scale);
                                else if (concept_tool == ConceptTool::audit) audit_controls(source);
                                else classify_controls(source);
                            }
                            ImGui::EndChild();
                            ImGui::PopStyleVar();
                            ImGui::PopID();
                        }
                    }
                }
            } else if (failure != library.concept_errors.end()) ImGui::TextWrapped("%s", failure->second.c_str());
            else if (collection_key == root->all.key) ImGui::TextDisabled("%zu concepts", root->concepts.size());
            if (!root->ready) ImGui::TextColored({0.95F, 0.49F, 0.42F, 1}, "Index error / details below");
        } else ImGui::TextDisabled(library.ready ? "Choose a dataset" : "Indexing...");
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
    }

    std::string UserInterface::concept_activity(const std::string_view key) const {
        static constexpr std::array names{"Generate", "Train", "Infer", "Audit", "Fix", "Undo", "Classify"};
        std::array<bool, names.size()> latest{};
        std::array<std::size_t, names.size()> queued{}, running{};
        std::string result;
        for (const auto& [id, task] : std::views::reverse(task_status)) {
            if (task.at("concept").get_ref<const std::string&>() != key || task.at("kind") == static_cast<int>(work::Kind::infer)) continue;
            const auto kind   = task.at("kind").get<std::size_t>();
            const auto& state = task.at("state");
            if (state == "queued") ++queued[kind];
            else if (state == "running") ++running[kind];
            else if (!latest[kind] && state == "failed") result += std::format("{} failed / ", names[kind]);
            latest[kind] = true;
        }
        for (std::size_t kind = 0; kind < names.size(); ++kind) {
            if (running[kind]) result += std::format("{} running / ", names[kind]);
            if (queued[kind]) result += std::format("{} queued {} / ", names[kind], queued[kind]);
        }
        if (!result.empty()) result.resize(result.size() - 3);
        return result;
    }

    std::optional<std::string> UserInterface::dataset_contents() {
        std::optional<std::string> selected;
        const float scale = renderer.dpi;
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {0, 4 * scale});
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, {6 * scale, 4 * scale});
        ImGui::PushStyleVar(ImGuiStyleVar_IndentSpacing, ImGui::GetTreeNodeToLabelSpacing() + 6 * scale);
        for (const auto& entry : library.roots) {
            ImGui::PushID(entry.all.key.c_str());
            const auto origin = ImGui::GetCursorScreenPos();
            const float width = ImGui::GetContentRegionAvail().x;
            ImGui::SetNextItemOpen(true, expand_dataset_roots ? ImGuiCond_Always : ImGuiCond_Once);
            const auto flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_FramePadding | (collection_key == entry.all.key ? ImGuiTreeNodeFlags_Selected : 0) | (entry.concepts.empty() && entry.ready ? ImGuiTreeNodeFlags_Leaf : 0);
            const bool open  = ImGui::TreeNodeEx("##Root", flags, "");
            if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen()) selected = entry.all.key;
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s\n%zu images / %zu concepts", entry.all.key.c_str(), entry.all.images.size(), entry.concepts.size());
            auto* draw       = ImGui::GetWindowDrawList();
            const float y    = origin.y + ImGui::GetStyle().FramePadding.y;
            const auto count = std::to_string(entry.all.images.size());
            float right      = std::min(origin.x + width, draw->GetClipRectMax().x) - 6 * scale - ImGui::CalcTextSize(count.c_str()).x;
            draw->AddText({right, y}, ImGui::GetColorU32(ImGuiCol_TextDisabled), count.c_str());
            right -= 8 * scale;
            if (!entry.ready) {
                right -= ImGui::CalcTextSize("!").x;
                draw->AddText({right, y}, ImGui::GetColorU32(ImVec4{0.95F, 0.49F, 0.42F, 1}), "!");
                right -= 8 * scale;
            }
            const float left = origin.x + ImGui::GetTreeNodeToLabelSpacing();
            ImGui::RenderTextEllipsis(draw, {left, y}, {std::max(left, right), y + ImGui::GetFontSize()}, std::max(left, right), entry.all.name == "raw" ? "Raw" : entry.all.name.c_str(), nullptr, nullptr);
            if (open) {
                for (const auto& item : entry.concepts) {
                    ImGui::PushID(item.key.c_str());
                    const auto metadata  = library.concepts.find(item.key);
                    const auto info      = library.classifiers.find(item.key);
                    const auto failure   = library.concept_errors.find(item.key);
                    const bool published = info != library.classifiers.end() && !info->second.training.is_null() && !info->second.training.at("published").is_null();
                    const bool enabled   = std::ranges::contains(activated, item.key);
                    const auto origin    = ImGui::GetCursorScreenPos();
                    const float width    = ImGui::GetContentRegionAvail().x;
                    const float height   = ImGui::GetTextLineHeight() + 8 * scale;
                    if (ImGui::Selectable("##Concept", collection_key == item.key, ImGuiSelectableFlags_None, {width, height})) selected = item.key;
                    const bool hovered = ImGui::IsItemHovered();
                    if (published && ImGui::IsItemClicked(ImGuiMouseButton_Middle)) {
                        if (enabled) std::erase(activated, item.key);
                        else activated.push_back(item.key);
                    }
                    auto* draw = ImGui::GetWindowDrawList();
                    const ImVec2 minimum{origin.x, origin.y};
                    const ImVec2 maximum{std::min(origin.x + width, draw->GetClipRectMax().x), origin.y + height};
                    const float y    = origin.y + 4 * scale;
                    const auto count = std::to_string(item.images.size());
                    float right      = maximum.x - 6 * scale - ImGui::CalcTextSize(count.c_str()).x;
                    draw->AddText({right, y}, ImGui::GetColorU32(ImGuiCol_TextDisabled), count.c_str());
                    right -= 8 * scale;
                    if (failure != library.concept_errors.end()) {
                        right -= ImGui::CalcTextSize("!").x;
                        draw->AddText({right, y}, ImGui::GetColorU32(ImVec4{0.95F, 0.49F, 0.42F, 1}), "!");
                        right -= 8 * scale;
                    } else if (metadata != library.concepts.end() && metadata->second.type != dataset::ConceptType::none) {
                        const bool classifier_type = metadata->second.type == dataset::ConceptType::classifier;
                        const char* tag            = classifier_type ? "C" : "LoRA";
                        const auto color           = classifier_type ? ImVec4{0.44F, 0.80F, 0.87F, 1} : ImVec4{0.92F, 0.69F, 0.36F, 1};
                        const float width          = ImGui::CalcTextSize(tag).x + 10 * scale;
                        right -= width;
                        draw->AddRectFilled({right, y - scale}, {right + width, y + ImGui::GetFontSize() + scale}, ImGui::GetColorU32(ImVec4{color.x, color.y, color.z, 0.12F}), 4 * scale);
                        draw->AddText({right + 5 * scale, y}, ImGui::GetColorU32(color), tag);
                        right -= 8 * scale;
                        if (published) {
                            right -= 12 * scale;
                            ImGui::RenderCheckMark(draw, {right, y + 2 * scale}, ImGui::GetColorU32(color), 10 * scale);
                            right -= 6 * scale;
                        }
                    }
                    const float left = minimum.x + 6 * scale;
                    if (enabled) draw->AddCircleFilled({left - 4 * scale, (minimum.y + maximum.y) / 2}, 2 * scale, ImGui::GetColorU32(ImGuiCol_Text));
                    ImGui::RenderTextEllipsis(draw, {left, y}, {std::max(left, right), y + ImGui::GetFontSize()}, std::max(left, right), item.name.c_str(), nullptr, nullptr);
                    if (hovered) {
                        ImGui::BeginTooltip();
                        ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + 320 * scale);
                        ImGui::TextUnformatted(item.key.c_str());
                        ImGui::TextDisabled("%zu images", item.images.size());
                        if (metadata != library.concepts.end()) {
                            ImGui::Text("Type: %s%s", metadata->second.type == dataset::ConceptType::none ? "Unassigned" : metadata->second.type == dataset::ConceptType::classifier ? "Classifier" : "LoRA", metadata->second.locked ? " / Locked" : "");
                            if (published) {
                                ImGui::TextUnformatted(enabled ? "Model available / Active" : "Model available");
                                ImGui::TextDisabled("Middle-click to activate or deactivate.");
                            }
                            if (metadata->second.type == dataset::ConceptType::classifier) {
                                const auto activity = concept_activity(item.key);
                                if (!activity.empty()) ImGui::TextUnformatted(activity.c_str());
                                if (info != library.classifiers.end() && !info->second.training.is_null()) {
                                    ImGui::Text("Training: %s", info->second.training.at("state").get_ref<const std::string&>().c_str());
                                    if (info->second.training.at("fingerprint") != info->second.fingerprint) ImGui::TextUnformatted("Training data changed. Restart is required to train again.");
                                }
                            }
                        }
                        if (failure != library.concept_errors.end()) ImGui::TextColored({0.95F, 0.49F, 0.42F, 1}, "%s", failure->second.c_str());
                        ImGui::PopTextWrapPos();
                        ImGui::EndTooltip();
                    }
                    ImGui::PopID();
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
        if (!library.roots.empty()) expand_dataset_roots = false;
        ImGui::PopStyleVar(3);
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
            const float fitted = std::min(view_available.x / dimensions.x, view_available.y / dimensions.y);
            if (text_button("##View", std::format("{}{:.0f}%", view.fit ? "Fit \xC2\xB7 " : "", view.zoom * 100).c_str(), scale, 104 * scale)) {
                if ((page == Page::dataset || page == Page::audit) && !repaint) viewing = View::inspect;
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

    void UserInterface::open_audit(std::string key, const bool refresh) {
        if (!leave_repaint()) return;
        if (page == Page::generation || page == Page::dataset) {
            audit_return            = page;
            audit_return_collection = collection_key;
            audit_return_view       = viewing;
            audit_return_camera     = view;
        }
        if (audit_key != key) {
            audit_report   = nullptr;
            audit_position = {};
            audit_category.clear();
        }
        audit_key = std::move(key);
        audit_task.reset();
        for (const auto& [id, task] : std::views::reverse(task_status))
            if (task.at("concept") == audit_key && task.at("kind") == static_cast<int>(work::Kind::audit) && (task.at("state") == "queued" || task.at("state") == "running")) {
                audit_task = id;
                break;
            }
        collection_key    = audit_key;
        page              = Page::audit;
        viewing           = View::browse;
        const auto stored = dataset::directory / files::path(audit_key) / ".genesia" / "classifier" / "audit.json";
        if (!refresh && audit_report.is_null() && std::filesystem::exists(stored)) audit_report = files::read_json(stored);
        rebuild_audit();
        synchronize_collection();
        if (audit_task) return;
        const auto source = library.classifiers.find(audit_key);
        if (source == library.classifiers.end() || source->second.training.is_null() || source->second.training.at("published").is_null()) return;
        if (!refresh && !audit_report.is_null() && !audit_dirty) return;
        work::Request request;
        request.kind        = work::Kind::audit;
        request.concept_key = audit_key;
        request.refresh     = refresh;
        audit_task          = submit_task(std::move(request));
        audit_dirty         = false;
        rebuild_audit();
        synchronize_collection();
    }
    std::uint64_t UserInterface::submit_task(work::Request request) {
        if (!runtime) runtime = std::make_unique<WorkspaceRuntime>(renderer.device);
        const auto id = runtime->session.enqueue(std::move(request));
        const std::lock_guard lock{runtime->session.mutex};
        task_status[id] = runtime->session.jobs.at(id);
        return id;
    }
    void UserInterface::task_event(const nlohmann::json& event) {
        const auto kind  = static_cast<work::Kind>(event.at("kind").get<int>());
        const auto id    = event.at("id").get<std::uint64_t>();
        const auto state = event.at("state").get<std::string>();
        const auto key   = event.at("concept").get<std::string>();
        if (kind != work::Kind::infer) task_status[id] = event;
        if (kind == work::Kind::train && state == "running" && event.contains("progress") && event.at("progress").is_object()) {
            const auto found = training_drafts.find(key);
            if (found != training_drafts.end()) {
                const auto& progress = event.at("progress");
                if (progress.contains("loss")) found->second.losses.push_back(progress.at("loss").get<float>());
                if (progress.contains("metrics")) found->second.metrics = progress.at("metrics");
            }
        }
        if (state == "complete") {
            if (kind == work::Kind::infer) {
                const auto result                                                        = event.at("result").get<classifier::Result>();
                predictions[result.id + "|" + result.model_sha + "|" + result.image_sha] = result;
            } else if (kind == work::Kind::audit && key == audit_key && audit_task == id) {
                audit_report = event.at("result");
                audit_task.reset();
                rebuild_audit();
            } else if (kind == work::Kind::fix || kind == work::Kind::undo) {
                library.refresh();
                audit_dirty = key == audit_key;
            } else if (kind == work::Kind::train || kind == work::Kind::classify) library.refresh();
        } else if (state == "failed" || state == "stopped") {
            if (audit_task == id) {
                audit_task.reset();
                audit_dirty = false;
            }
            if (state == "failed") {
                if (kind == work::Kind::infer) prediction_errors[key + "|" + event.value("model_sha", "") + "|" + event.value("image_sha", "")] = event.at("error").get<std::string>();
            }
            if (kind == work::Kind::train) library.refresh();
        }
    }
    void UserInterface::rebuild_audit() {
        audit_collection = {.key = audit_key, .name = audit_key};
        if (audit_report.is_null()) return;
        const auto source = library.classifiers.find(audit_key);
        if (source == library.classifiers.end()) return;
        if (source->second.training.is_null() || source->second.training.at("published").is_null() || source->second.training.at("published").at("sha") != audit_report.at("model_sha")) {
            audit_report = nullptr;
            audit_dirty  = true;
            return;
        }
        std::vector<nlohmann::json> rows;
        const auto model_sha = audit_report.at("model_sha").get<std::string>();
        bool missing{};
        for (const auto& sample : source->second.samples) {
            const auto& previous = audit_report.at("rows");
            const auto found     = std::ranges::find_if(previous, [&](const auto& row) { return row.at("sha") == sample.file.sha; });
            if (found == previous.end()) {
                missing = true;
                continue;
            }
            auto row            = *found;
            const auto& label   = source->second.classes[sample.label];
            const auto classes  = row.at("classes").template get<std::vector<std::string>>();
            const auto category = std::ranges::find(classes, label);
            if (category == classes.end()) {
                missing = true;
                continue;
            }
            row["paths"] = nlohmann::json::array();
            for (const auto& path : sample.paths) row["paths"].push_back(files::utf8(path));
            row["label"]        = label;
            row["path"]         = files::utf8(sample.file.path);
            row["confidence"]   = row.at("scores").at(category - classes.begin());
            const auto identity = audit_key + "|" + model_sha + "|" + sample.file.sha;
            predictions.insert_or_assign(identity, classifier::Result{audit_key, model_sha, sample.file.sha, row.at("predicted").get<std::string>(), classes, row.at("scores").get<std::vector<float>>()});
            prediction_errors.erase(identity);
            rows.push_back(std::move(row));
        }
        std::ranges::sort(rows, [](const auto& a, const auto& b) { return a.at("confidence") == b.at("confidence") ? a.at("path") < b.at("path") : a.at("confidence") < b.at("confidence"); });
        audit_dirty          = missing || audit_report.at("fingerprint") != source->second.fingerprint;
        audit_report["rows"] = rows;
        for (const auto& row : rows) {
            if (!audit_category.empty() && row.at("label") != audit_category) continue;
            const auto file = std::ranges::find_if(source->second.samples, [&](const auto& sample) { return row.at("sha") == sample.file.sha; });
            audit_collection.images.push_back(file->file);
        }
        if (!audit_collection.images.empty()) {
            const auto selected     = std::ranges::find(audit_collection.images, audit_position.selected, &dataset::File::sha);
            audit_position.index    = selected == audit_collection.images.end() ? std::min(audit_position.index, audit_collection.images.size() - 1) : static_cast<std::size_t>(selected - audit_collection.images.begin());
            audit_position.selected = audit_collection.images[audit_position.index].sha;
            audit_position.scroll   = static_cast<float>(audit_position.index);
        }
    }
    void UserInterface::operation_activity(const std::initializer_list<work::Kind> kinds, const std::string_view key, const std::optional<std::uint64_t> exclude) {
        const auto matches = [&](const auto& entry) { return entry.first != exclude && entry.second.at("concept").template get_ref<const std::string&>() == key && std::ranges::contains(kinds, static_cast<work::Kind>(entry.second.at("kind").template get<int>())); };
        if (!std::ranges::any_of(task_status, matches)) return;
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {8 * renderer.dpi, 6 * renderer.dpi});
        if (std::ranges::contains(kinds, work::Kind::generate)) ImGui::SetNextWindowSizeConstraints({0, 0}, {FLT_MAX, ImGui::GetIO().DisplaySize.y * 0.3F});
        if (ImGui::BeginChild("##OperationActivity", {0, 0}, ImGuiChildFlags_AutoResizeY, ImGuiWindowFlags_NoBackground)) {
            const auto draw_job = [&](const nlohmann::json& task) {
                const auto kind    = static_cast<work::Kind>(task.at("kind").get<int>());
                const auto& state  = task.at("state").get_ref<const std::string&>();
                const bool pending = state == "queued" || state == "running" || state == "saving";
                std::string title;
                switch (kind) {
                case work::Kind::generate: title = task.at("source") == "" ? "Generate" : "Repaint"; break;
                case work::Kind::train: title = std::format("Train / {} steps", task.at("target").get<int>()); break;
                case work::Kind::classify: title = files::utf8(files::path(task.at("input").get_ref<const std::string&>()).filename()); break;
                case work::Kind::audit: title = "Audit"; break;
                case work::Kind::fix: title = "Move to " + task.at("category").get<std::string>(); break;
                case work::Kind::undo: title = "Undo move"; break;
                case work::Kind::infer: break;
                }
                ImGui::PushID(task.at("id").dump().c_str());
                if (ImGui::TreeNodeEx("##Operation", pending || state == "failed" ? ImGuiTreeNodeFlags_DefaultOpen : ImGuiTreeNodeFlags_None, "%s / %s", title.c_str(), state.c_str())) {
                    if (kind == work::Kind::generate) {
                        ImGui::Text("%d x %d / seed %llu", task.at("width").get<int>(), task.at("height").get<int>(), task.at("seed").get<std::uint64_t>());
                        if (task.at("source") != "") ImGui::TextWrapped("Source: %s", task.at("source").get_ref<const std::string&>().c_str());
                        if (state == "running") {
                            auto& session = runtime->session;
                            const std::lock_guard lock{session.mutex};
                            if (session.active && session.active->id == task.at("id").get<std::uint64_t>()) {
                                const auto stage = static_cast<sdxl::Stage>(::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>{session.control.data()[0].stage}.load());
                                if (stage == sdxl::Stage::sampling) {
                                    const auto step = ::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>{session.control.data()[0].completed}.load();
                                    ImGui::ProgressBar(float(step) / session.active->parameters.steps, {-1, 0});
                                    ImGui::Text("Step %u / %d", step, session.active->parameters.steps);
                                }
                            }
                        }
                    } else if (kind == work::Kind::classify) ImGui::TextWrapped("%s", task.at("input").get_ref<const std::string&>().c_str());
                    const auto progress = task.find("progress");
                    const bool moving   = kind == work::Kind::fix || kind == work::Kind::undo || (progress != task.end() && progress->is_object() && progress->value("stage", "") == "moving");
                    if (state == "queued" || (state == "running" && !moving)) {
                        if (ImGui::Button(state == "queued" ? "Cancel queued operation" : "Stop this operation", {-FLT_MIN, 0})) runtime->session.cancel(task.at("id").get<std::uint64_t>());
                    } else if (state == "saving") ImGui::TextWrapped("Saving the completed image.");
                    else if (state == "running" && moving) ImGui::TextWrapped("Finishing the file move.");
                    if (progress != task.end() && progress->is_object()) {
                        if (progress->contains("completed")) {
                            ImGui::ProgressBar(progress->at("completed").get<float>() / progress->at("total").get<float>(), {-1, 0});
                            ImGui::Text("%zu / %zu images", progress->at("completed").get<std::size_t>(), progress->at("total").get<std::size_t>());
                        }
                        if (progress->contains("step")) ImGui::Text("Step %d / %d", progress->at("step").get<int>(), progress->at("target").get<int>());
                    }
                    if (state == "failed") ImGui::TextWrapped("%s", task.at("error").get_ref<const std::string&>().c_str());
                    if (state == "complete" && kind == work::Kind::classify)
                        for (const auto& [label, count] : task.at("result").at("classes").items()) ImGui::TextWrapped("%s / %zu images", label.c_str(), count.get<std::size_t>());
                    if (state == "complete" && kind == work::Kind::generate) ImGui::TextWrapped("%s", task.at("result").at("path").get_ref<const std::string&>().c_str());
                    ImGui::TreePop();
                }
                ImGui::PopID();
            };
            bool history{};
            for (const auto& entry : std::views::reverse(task_status)) {
                if (!matches(entry)) continue;
                const auto& state = entry.second.at("state");
                if (state == "complete" || state == "stopped") history = true;
                else draw_job(entry.second);
            }
            if (history && ImGui::CollapsingHeader(exclude ? "Earlier training" : "Completed and stopped"))
                for (const auto& entry : std::views::reverse(task_status))
                    if (matches(entry) && (entry.second.at("state") == "complete" || entry.second.at("state") == "stopped")) draw_job(entry.second);
        }
        ImGui::EndChild();
        ImGui::PopStyleVar();
    }

    void UserInterface::training_controls(const classifier::TrainingData& source, const float scale) {
        auto& edits = training_drafts.try_emplace(source.key, source).first->second;
        const nlohmann::json* task{};
        for (auto entry = task_status.rbegin(); entry != task_status.rend(); ++entry)
            if (entry->second.at("concept") == source.key && entry->second.at("kind") == static_cast<int>(work::Kind::train)) {
                task = &entry->second;
                break;
            }
        const bool busy       = task && (task->at("state") == "queued" || task->at("state") == "running");
        const bool recorded   = !source.training.is_null();
        const bool changed    = recorded && source.training.at("fingerprint") != source.fingerprint;
        const bool configured = recorded && source.training.at("config") != edits.config;
        const bool checkpoint = recorded && std::filesystem::exists(source.root / ".genesia" / "classifier" / "latest.safetensors");
        const int completed   = recorded ? source.training.at("step").get<int>() : 0;
        std::string issue     = source.issue;
        std::vector<std::size_t> counts(source.classes.size());
        for (const auto& sample : source.samples) ++counts[sample.label];
        if (source.classes.size() < 2) issue = "A classifier needs at least two nonempty categories.";
        for (std::size_t i = 0; i < counts.size(); ++i)
            if (!counts[i]) issue = "Empty category: " + source.classes[i];
        if (!issue.empty()) ImGui::TextWrapped("%s", issue.c_str());
        if (changed) ImGui::TextWrapped("Training data changed. Restart training is required.");
        else if (configured) ImGui::TextWrapped("Training parameters changed. Restart training is required.");
        else if (recorded && !checkpoint && !busy) ImGui::TextWrapped("Checkpoint missing. Restart training is required.");
        ImGui::Spacing();
        if (task) {
            const auto& state = task->at("state");
            if (!busy && recorded) ImGui::TextDisabled("%s / step %d", state.get_ref<const std::string&>().c_str(), task->contains("result") ? task->at("result").at("step").get<int>() : completed);
            else ImGui::TextDisabled("%s", state.get_ref<const std::string&>().c_str());
            if (state == "failed") ImGui::TextWrapped("%s", task->at("error").get_ref<const std::string&>().c_str());
            if (busy && task->contains("progress")) {
                const auto& progress = task->at("progress");
                const auto& stage    = progress.at("stage");
                if (stage == "preparing") {
                    ImGui::TextDisabled("Preparing %zu / %zu images", progress.at("completed").get<std::size_t>(), progress.at("total").get<std::size_t>());
                    ImGui::ProgressBar(progress.at("completed").get<float>() / progress.at("total").get<float>(), {-1, 3 * scale}, "");
                } else if (stage == "training") {
                    ImGui::Text("Step %d / %d", progress.at("step").get<int>(), progress.at("target").get<int>());
                    ImGui::ProgressBar(progress.at("step").get<float>() / progress.at("target").get<float>(), {-1, 3 * scale}, "");
                    ImGui::TextDisabled("Loss %.4f / %.1fs", progress.at("loss").get<float>(), progress.at("seconds").get<double>());
                } else if (stage == "evaluation") ImGui::TextDisabled("Evaluating / step %d", progress.at("metrics").at("step").get<int>());
            }
        } else if (recorded) ImGui::TextDisabled("%s / step %d", source.training.at("state").get_ref<const std::string&>().c_str(), completed);
        bool submit{}, restart{};
        if (busy) {
            if (ImGui::Button(task->at("state") == "queued" ? "Cancel queued training" : "Stop training", {-FLT_MIN, 0})) runtime->session.cancel(task->at("id").get<std::uint64_t>());
        } else {
            ImGui::AlignTextToFramePadding();
            ImGui::TextDisabled("Target steps");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(-FLT_MIN);
            ImGui::InputInt("##TrainingSteps", &edits.steps, 0, 0);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Cumulative training target, including completed steps.");
            const bool target = edits.steps > completed && (!recorded || edits.steps >= source.training.at("target").get<int>());
            if (!target) ImGui::TextWrapped("Increase the cumulative target to continue training.");
            ImGui::BeginDisabled(!issue.empty() || changed || configured || (recorded && !checkpoint) || !target);
            submit = ImGui::Button(recorded ? "Continue training" : "Start training", {-FLT_MIN, 0});
            ImGui::EndDisabled();
            if (recorded) {
                ImGui::BeginDisabled(!issue.empty());
                if (ImGui::Selectable("Restart training...", edits.restart_confirm)) edits.restart_confirm = !edits.restart_confirm;
                if (edits.restart_confirm) {
                    ImGui::TextWrapped("Delete this concept's weights, checkpoints, snapshots, metrics and audit history. Train from the initial pretrained model. Images are retained.");
                    if (ImGui::Button("Delete state and restart", {-FLT_MIN, 0})) submit = restart = true;
                    if (ImGui::Button("Cancel", {-FLT_MIN, 0})) edits.restart_confirm = false;
                }
                ImGui::EndDisabled();
            }
        }
        if (submit) {
            work::Request request;
            request.kind        = work::Kind::train;
            request.concept_key = source.key;
            request.training    = {.concept_key = source.key, .steps = edits.steps, .restart = restart, .overrides = edits.config};
            submit_task(std::move(request));
            edits.restart_confirm = false;
            if (restart) {
                edits.losses.clear();
                edits.metrics = nullptr;
            }
        }
        if (!edits.losses.empty()) {
            ImGui::Spacing();
            ImGui::TextDisabled("Training loss / %.4f", edits.losses.back());
            ImGui::PushStyleColor(ImGuiCol_FrameBg, {0, 0, 0, 0});
            ImGui::PlotLines("##TrainingLoss", edits.losses.data(), static_cast<int>(edits.losses.size()), 0, nullptr, FLT_MAX, FLT_MAX, {-FLT_MIN, 60 * scale});
            ImGui::PopStyleColor();
        }
        ImGui::PushStyleColor(ImGuiCol_Header, {0, 0, 0, 0});
        if (ImGui::CollapsingHeader("Categories & checks")) {
            for (std::size_t i = 0; i < counts.size(); ++i) ImGui::TextWrapped("%s / %zu images", source.classes[i].c_str(), counts[i]);
            ImGui::TextWrapped("Training checks metadata, original sizes, labels and independent train / validation groups before loading the model.");
        }
        if (ImGui::CollapsingHeader("Parameters")) {
            ImGui::BeginDisabled(busy);
            for (auto entry = edits.config.begin(); entry != edits.config.end(); ++entry) {
                std::string label = entry.key();
                std::ranges::replace(label, '_', ' ');
                ImGui::TextWrapped("%s", label.c_str());
                ImGui::PushID(entry.key().c_str());
                ImGui::SetNextItemWidth(-FLT_MIN);
                if (entry.key() == "seed") {
                    auto value = entry.value().get<std::uint64_t>();
                    if (ImGui::InputScalar("##Value", ImGuiDataType_U64, &value)) entry.value() = value;
                } else if (entry.value().is_number_integer()) {
                    int value = entry.value().get<int>();
                    if (ImGui::InputInt("##Value", &value, 0, 0)) entry.value() = value;
                } else {
                    double value = entry.value().get<double>();
                    if (ImGui::InputDouble("##Value", &value, 0, 0, "%.8g")) entry.value() = value;
                }
                ImGui::PopID();
            }
            ImGui::EndDisabled();
        }
        if (ImGui::CollapsingHeader("Evaluation")) {
            const auto& metrics = edits.metrics;
            if (metrics.is_null()) ImGui::TextWrapped("Evaluation appears after a training checkpoint.");
            else {
                ImGui::Text("Step %d / %zu samples", metrics.at("step").get<int>(), metrics.at("samples").get<std::size_t>());
                ImGui::Text("Cross-entropy %.4f", metrics.at("loss").get<double>());
                ImGui::Text("Accuracy %.2f%%", metrics.at("accuracy").get<double>() * 100);
                ImGui::Text("Macro precision %.3f", metrics.at("precision").get<double>());
                ImGui::Text("Macro recall %.3f", metrics.at("recall").get<double>());
                ImGui::Text("Macro F1 %.3f", metrics.at("f1").get<double>());
                if (ImGui::TreeNode("Per-category metrics")) {
                    for (const auto& row : metrics.at("classes")) {
                        ImGui::TextWrapped("%s / %zu samples", row.at("name").get_ref<const std::string&>().c_str(), row.at("samples").get<std::size_t>());
                        ImGui::TextWrapped("P %.3f / R %.3f / F1 %.3f", row.at("precision").get<double>(), row.at("recall").get<double>(), row.at("f1").get<double>());
                    }
                    ImGui::TreePop();
                }
                if (ImGui::TreeNode("Confusion matrix")) {
                    const auto& classes = metrics.at("classes");
                    ImGui::TextWrapped("Rows: human category. Columns: predicted category.");
                    if (ImGui::BeginTable("##Confusion", static_cast<int>(classes.size()) + 1, ImGuiTableFlags_Borders | ImGuiTableFlags_ScrollX | ImGuiTableFlags_SizingFixedFit, {0, 0}, (120 + 72 * classes.size()) * scale)) {
                        ImGui::TableSetupColumn("Category", ImGuiTableColumnFlags_WidthFixed, 120 * scale);
                        for (const auto& label : classes) ImGui::TableSetupColumn(label.at("name").get_ref<const std::string&>().c_str(), ImGuiTableColumnFlags_WidthFixed, 72 * scale);
                        ImGui::TableHeadersRow();
                        for (std::size_t i = 0; i < classes.size(); ++i) {
                            ImGui::TableNextRow();
                            ImGui::TableNextColumn();
                            ImGui::TextWrapped("%s", classes[i].at("name").get_ref<const std::string&>().c_str());
                            for (const auto& count : metrics.at("confusion")[i]) {
                                ImGui::TableNextColumn();
                                ImGui::Text("%d", count.get<int>());
                            }
                        }
                        ImGui::EndTable();
                    }
                    ImGui::TreePop();
                }
            }
        }
        if (task) operation_activity({work::Kind::train}, source.key, task->at("id").get<std::uint64_t>());
        ImGui::PopStyleColor();
    }
    void UserInterface::classify_controls(const classifier::TrainingData& source) {
        const bool trained = !source.training.is_null() && !source.training.at("published").is_null();
        auto& path         = classify_paths[source.key];
        if (!trained) ImGui::TextWrapped("A published classifier is required to submit another folder.");
        ImGui::BeginDisabled(!trained);
        ImGui::TextWrapped("Drop one folder into the window, or paste its path below.");
        ImGui::TextWrapped("Direct PNG images move into predicted category folders. Original names are retained.");
        ImGui::TextUnformatted("Directory");
        ImGui::SetNextItemWidth(-FLT_MIN);
        ImGui::InputText("##ClassifyDirectory", path.data(), path.size());
        ImGui::BeginDisabled(!path.front());
        if (ImGui::Button("Classify folder", {-FLT_MIN, 0})) {
            work::Request request;
            request.kind        = work::Kind::classify;
            request.concept_key = source.key;
            request.input       = files::path(path.data());
            submit_task(std::move(request));
        }
        ImGui::EndDisabled();
        ImGui::EndDisabled();
        operation_activity({work::Kind::classify}, source.key);
    }
    void UserInterface::audit_controls(const classifier::TrainingData& source) {
        if (repaint) {
            ImGui::TextWrapped("Return from Repaint to edit audit labels.");
            return;
        }
        ImGui::TextWrapped("%s", audit_task ? "Updating audit..." : audit_dirty || audit_report.is_null() ? "Results incomplete" : "Current labels, lowest confidence first");
        ImGui::TextUnformatted("Category");
        ImGui::SetNextItemWidth(-FLT_MIN);
        if (ImGui::BeginCombo("##AuditCategory", audit_category.empty() ? "All categories" : audit_category.c_str())) {
            if (ImGui::Selectable("All categories", audit_category.empty())) {
                audit_category.clear();
                audit_position = {};
                rebuild_audit();
            }
            for (const auto& name : source.classes)
                if (ImGui::Selectable(name.c_str(), name == audit_category)) {
                    audit_category = name;
                    audit_position = {};
                    rebuild_audit();
                }
            ImGui::EndCombo();
        }
        const bool busy = audit_task.has_value() || std::ranges::any_of(task_status, [&](const auto& entry) { return entry.second.at("concept") == audit_key && (entry.second.at("state") == "running" || entry.second.at("state") == "queued"); });
        ImGui::BeginDisabled(busy);
        ImGui::BeginDisabled(source.training.is_null() || source.training.at("published").is_null());
        if (ImGui::Button("Refresh audit", {-FLT_MIN, 0})) open_audit(audit_key, true);
        ImGui::EndDisabled();
        if (ImGui::Button("Undo move", {-FLT_MIN, 0})) {
            work::Request request;
            request.kind        = work::Kind::undo;
            request.concept_key = audit_key;
            submit_task(std::move(request));
        }
        ImGui::EndDisabled();
        if (!audit_report.is_null() && !audit_collection.images.empty()) {
            const auto& file = audit_collection.images[audit_position.index];
            const auto& rows = audit_report.at("rows");
            const auto row   = std::ranges::find_if(rows, [&](const auto& value) { return value.at("sha") == file.sha; });
            if (row != rows.end()) {
                const auto label     = row->at("label").get<std::string>();
                const auto predicted = row->at("predicted").get<std::string>();
                const auto fix_to    = [&](const std::string& category) {
                    work::Request request;
                    request.kind        = work::Kind::fix;
                    request.concept_key = audit_key;
                    request.sha         = file.sha;
                    request.category    = category;
                    if (audit_position.index + 1 < audit_collection.images.size()) audit_position.selected = audit_collection.images[audit_position.index + 1].sha;
                    submit_task(std::move(request));
                };
                ImGui::BeginDisabled(busy);
                if (label != predicted && ImGui::Button(("Move to " + predicted).c_str(), {-FLT_MIN, 0})) fix_to(predicted);
                ImGui::TextUnformatted("Move to category");
                ImGui::SetNextItemWidth(-FLT_MIN);
                if (ImGui::BeginCombo("##AuditMoveCategory", "Choose...")) {
                    for (const auto& name : source.classes)
                        if (name != label && ImGui::Selectable(name.c_str())) fix_to(name);
                    ImGui::EndCombo();
                }
                ImGui::EndDisabled();
            }
        }
        operation_activity({work::Kind::audit, work::Kind::fix, work::Kind::undo}, audit_key);
    }
    void UserInterface::observe_inference() {
        std::vector<work::Request> wanted;
        for (const auto& file : visible_images)
            for (const auto& key : activated) {
                const auto found = library.classifiers.find(key);
                if (found == library.classifiers.end() || found->second.training.is_null() || found->second.training.at("published").is_null()) continue;
                const auto sha      = found->second.training.at("published").at("sha").get<std::string>();
                const auto identity = key + "|" + sha + "|" + file.sha;
                if (predictions.contains(identity) || prediction_errors.contains(identity)) continue;
                work::Request request;
                request.kind        = work::Kind::infer;
                request.concept_key = key;
                request.file        = file;
                request.descriptor  = {key, sha, found->second.root / ".genesia" / "classifier" / "models" / (sha + ".safetensors")};
                wanted.push_back(std::move(request));
            }
        if (!wanted.empty() && !runtime) runtime = std::make_unique<WorkspaceRuntime>(renderer.device);
        if (runtime) runtime->session.observe(std::move(wanted));
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
        canvas_origin = {dataset_sidebar.width * dataset_sidebar.amount, 0};
        canvas_size   = {std::max(1.0F, size.x - dataset_sidebar.width * dataset_sidebar.amount - prompt_sidebar.width * prompt_sidebar.amount), size.y};
        if (!window.dropped.empty()) {
            const auto info = library.classifiers.find(collection_key);
            if (!dataset_sidebar.open || concept_tool != ConceptTool::classify || info == library.classifiers.end() || info->second.training.is_null() || info->second.training.at("published").is_null()) action_error = "Open Classify at the top of the Dataset panel before dropping a folder.";
            else if (window.dropped.size() != 1) action_error = "Drop one directory at a time.";
            else {
                work::Request request;
                request.kind        = work::Kind::classify;
                request.concept_key = collection_key;
                request.input       = window.dropped.front();
                submit_task(std::move(request));
            }
            window.dropped.clear();
        }
        std::string error = library.error.empty() ? action_error : library.error;
        if (runtime) {
            auto& session = runtime->session;
            const std::lock_guard lock{session.mutex};
            session.preview_enabled = preview_enabled;
            session.activated       = activated;
            session.preview_visible = renderer.visible && session.active && ((!session.active->source && page == Page::generation) || (repaint && repaint->result.task == session.active->id && (viewing == View::comparison || viewing == View::result)));
            if (!session.error.empty()) error = session.error;
        }
        if (page == Page::audit && !repaint && audit_dirty && !audit_task) {
            bool stopped{};
            for (const auto& [id, task] : std::views::reverse(task_status))
                if (task.at("concept") == audit_key && task.at("kind") == static_cast<int>(work::Kind::audit)) {
                    stopped = task.at("state") == "stopped" || task.at("state") == "failed";
                    break;
                }
            if (!stopped) open_audit(audit_key);
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
        else if (collection && root && root->ready && !collection->images.empty()) image = resolve_image(collection->images[current_position().index]);
        const auto controls = control_layout(scale, size, image);
        sidebar(false, scale, size, image);
        bottom_controls(scale, size, controls, image);
        observe_inference();
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
            if (ImGui::Shortcut(ImGuiKey_GraveAccent, ImGuiInputFlags_RouteGlobal)) {
                dataset_sidebar.open = !dataset_sidebar.open;
                if (dataset_sidebar.open) expand_dataset_roots = true;
            }
        }
        const auto& io = ImGui::GetIO();
        if (io.MouseDelta.x || io.MouseDelta.y || io.MouseWheel || io.InputQueueCharacters.Size || ImGui::IsAnyItemActive()) animate_until = glfwGetTime() + 0.16;
    }
} // namespace genesia::editor
