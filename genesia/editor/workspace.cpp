module;
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_internal.h>
module genesia.editor.workspace;
import genesia.project;
import genesia.generation.defaults;
import genesia.prompt.preset;
import genesia.generation.output;
import genesia.editor.platform.window;
import genesia.editor.graphics.renderer;
import genesia.editor.graphics.interop;
import genesia.editor.graphics.bridge;
import genesia.editor.viewing.canvas;
import genesia.editor.panels.application;
import genesia.editor.panels.sidebars;
import genesia.editor.panels.datasets;
import std;
import genesia.io.files;

namespace genesia::editor {

    Workspace::RepaintDraft::RepaintDraft(const Record& source, std::shared_ptr<const prompt::Catalog> catalog) : document{prompt::resolve(source.prompt, std::move(catalog))} {
        editor.tracking = true;
        editor.reset(document.prompt);
    }

    Workspace::TrainingDraft::TrainingDraft(const training::TrainingData& source) {
        if (source.training) {
            config = source.training->config;
            steps  = source.training->phase == training::Phase::complete ? source.training->step + 400 : source.training->target;
        }
        const auto history = training::read_history(source.root);
        for (const auto& step : history.steps) losses.push_back(step.loss);
        if (!history.evaluations.empty()) metrics = history.evaluations.back();
    }

    Workspace::Workspace(prompt::Preset preset, std::shared_ptr<const prompt::Catalog> catalog, WindowPlatform& platform, Renderer& display, std::string dataset) : catalog{std::move(catalog)}, preset{std::move(preset)}, window{platform}, renderer{display}, dataset_index{[] { glfwPostEmptyEvent(); }}, textures{display}, prompt{this->preset.prompt}, tag_search{*this->catalog} {
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

    Workspace::~Workspace() {
        if (generation.texture) renderer.retire(generation.texture);
        if (repaint && repaint->result.texture) renderer.retire(repaint->result.texture);
    }

    void Workspace::receive() {
        const auto revision = library->revision;
        if (auto next = dataset_index.poll()) library = std::move(next);
        textures.receive();
        if (revision != library->revision) {
            for (auto& entry : library->roots) {
                std::vector<const dataset::Collection*> collections{&entry.all};
                for (auto& candidate : entry.concepts) collections.push_back(&candidate);
                for (const auto* candidate : collections) {
                    const auto remembered = positions.find(candidate->key);
                    if (remembered == positions.end() || candidate->images.empty()) continue;
                    auto& position      = remembered->second;
                    const auto selected = std::ranges::find(candidate->images, position.selected, &dataset::File::sha);
                    const auto index    = selected == candidate->images.end() ? std::min(position.index, candidate->images.size() - 1) : static_cast<std::size_t>(selected - candidate->images.begin());
                    position.scroll     = std::clamp(position.scroll + static_cast<float>(index) - position.index, 0.0F, static_cast<float>(candidate->images.size() - 1));
                    position.index      = index;
                    position.selected   = candidate->images[index].sha;
                }
            }
        }
        if (revision != library->revision) {
            std::erase_if(activated, [&](const auto& key) {
                const auto found = library->classifiers.find(key);
                return found == library->classifiers.end() || !found->second.model;
            });
            if (!audit_key.empty()) rebuild_audit();
        }
        synchronize_collection();
        if (runtime) {
            auto& session           = runtime->session;
            session_state           = session.snapshot();
            auto [events, previews] = session.drain();
            for (const auto& frame : previews) {
                const auto presentation = std::static_pointer_cast<const PresentedFrame>(frame.frame);
                auto& source            = presentation->bridge->slots[presentation->slot];
                Output* output          = !frame.from_image && (!generation.task || frame.id >= *generation.task) ? &generation : repaint && repaint->result.task == frame.id ? &repaint->result : nullptr;
                const bool final_ready  = std::ranges::any_of(events, [&](const runtime::Event& event) { return event.kind == runtime::EventKind::generated && event.id == frame.id; });
                if (!output || !preview_enabled || !renderer.visible || final_ready || (output->task == frame.id && (output->record || frame.step <= output->step))) {
                    renderer.discard(*source.timeline, presentation->ready);
                    continue;
                }
                if (output->task != frame.id) begin_output(*output, frame.id, frame.width, frame.height);
                if (output->texture) renderer.retire(output->texture);
                output->texture = renderer.texture({static_cast<std::uint32_t>(frame.width), static_cast<std::uint32_t>(frame.height)});
                renderer.copy(output->texture, source.buffer, *source.timeline, presentation->ready);
                output->width   = frame.width;
                output->height  = frame.height;
                output->preview = true;
                output->step    = frame.step;
            }
            for (auto& event : events) {
                if (event.kind == runtime::EventKind::task) {
                    task_event(event.task);
                    continue;
                }
                Output* output = event.record.source.empty() && (!generation.task || event.id >= *generation.task) ? &generation : repaint && repaint->result.task == event.id ? &repaint->result : nullptr;
                if (output && output->task != event.id) begin_output(*output, event.id, event.record.parameters.width, event.record.parameters.height);
                if (event.kind == runtime::EventKind::generated) {
                    const auto presentation = std::static_pointer_cast<const PresentedFrame>(event.frame);
                    auto& source            = presentation->bridge->slots[presentation->slot];
                    if (output) {
                        if (output->texture) renderer.retire(output->texture);
                        output->width   = event.record.parameters.width;
                        output->height  = event.record.parameters.height;
                        output->texture = renderer.texture({static_cast<std::uint32_t>(output->width), static_cast<std::uint32_t>(output->height)});
                        renderer.copy(output->texture, source.buffer, *source.timeline, presentation->ready);
                        output->record  = std::move(event.record);
                        output->preview = false;
                    } else renderer.discard(*source.timeline, presentation->ready);
                } else {
                    if (output) {
                        output->record  = std::move(event.record);
                        output->preview = false;
                    }
                    dataset_index.refresh();
                }
            }
        }
        for (auto* output : {&generation, repaint ? &repaint->result : nullptr}) {
            if (!output || !output->record || output->record->path.empty()) continue;
            if (!output->saved) {
                const auto raw = std::ranges::find_if(library->roots, [](const auto& value) { return value.all.key == "raw"; });
                if (raw != library->roots.end() && raw->ready) {
                    const auto file = std::ranges::find(raw->files, output->record->path, &dataset::File::path);
                    if (file != raw->files.end()) output->saved = *file;
                }
            }
            if (output->saved && output->texture) {
                const auto cached = textures.entries.find(output->saved->sha);
                if (cached != textures.entries.end() && cached->second.texture) renderer.retire(std::exchange(output->texture, 0));
            }
        }
    }

    void Workspace::synchronize_collection() {
        root       = nullptr;
        collection = nullptr;
        for (auto& entry : library->roots) {
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
        if (library->ready && page == Page::dataset && !collection) action_error = "Dataset does not exist: " + collection_key;
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

    void Workspace::select_collection(std::string key, std::optional<std::filesystem::path> target) {
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

    Workspace::Position& Workspace::current_position() {
        return page == Page::audit ? audit_position : positions[collection_key];
    }

    void Workspace::center_image(const std::size_t index) {
        locate.reset();
        auto& position    = current_position();
        position.index    = index;
        position.selected = collection->images[index].sha;
        animate_until     = glfwGetTime() + 0.3;
    }

    void Workspace::start_repaint(const dataset::File& file) {
        const auto source = file;
        if (repaint && repaint->source.sha == source.sha) {
            leave_repaint();
            return;
        }
        const auto cached = textures.entries.find(source.sha);
        if (cached == textures.entries.end() || !cached->second.texture) return;
        const Page return_page  = repaint ? repaint->return_page : page;
        View return_view        = repaint ? repaint->return_view : viewing;
        ImageView return_camera = repaint ? repaint->return_camera : view;
        if (!leave_repaint()) return;
        if (page == Page::generation) {
            select_collection("raw", source.path);
            if (page != Page::dataset) return;
        }
        auto& edits = repaints[source.sha];
        if (!edits) edits = std::make_unique<RepaintDraft>(cached->second.record, catalog);
        repaint.emplace(source, return_page, return_view, return_camera);
        viewing             = View::repaint;
        view                = {};
        prompt_sidebar.open = true;
    }

    bool Workspace::leave_repaint() {
        if (!repaint) return true;
        auto& edits = *repaints.at(repaint->source.sha);
        if (!edits.editor.commit(edits.document.prompt, *edits.document.catalog)) {
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

    void Workspace::back() {
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

    Workspace::Picture Workspace::resolve_image(const dataset::File& file, const Role role) const {
        Picture picture{0, file.width, file.height, nullptr, file, false, role};
        if (page == Page::audit && role == Role::image && !audit_report.concept_key.empty()) {
            const auto& rows = audit_report.rows;
            const auto row   = std::ranges::find_if(rows, [&](const auto& item) { return item.sample.file.sha == file.sha; });
            if (row != rows.end()) picture.confidence = row->confidence;
        }
        const auto cached = textures.entries.find(file.sha);
        if (cached != textures.entries.end() && cached->second.error.empty()) {
            picture.texture = cached->second.texture;
            picture.record  = &cached->second.record;
        }
        return picture;
    }

    Workspace::Picture Workspace::resolve_output(const Output& output, const Role role) const {
        if (output.saved) {
            auto picture = resolve_image(*output.saved, role);
            if (picture.texture) return picture;
        }
        return {output.texture, output.width, output.height, output.record ? &*output.record : nullptr, output.saved, output.preview, role};
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

    bool Workspace::save_prompt() {
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

    void Workspace::switch_preset() {
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

    void Workspace::begin_output(Output& output, const std::uint64_t task, const int width, const int height) {
        if (output.texture) renderer.retire(output.texture);
        output = {.task = task, .width = width, .height = height};
        if (&output == &generation) {
            if (page == Page::generation) view = {};
            else generation_view = {};
        }
    }

    void Workspace::submit() {
        commit_parameters();
        auto parameters     = draft;
        auto prompt_catalog = catalog;
        prompt::Pair submitted;
        std::optional<runtime::RepaintSource> source;
        if (repaint) {
            auto& edits = *repaints.at(repaint->source.sha);
            if (!edits.editor.commit(edits.document.prompt, *edits.document.catalog)) {
                prompt_sidebar.open = true;
                return;
            }
            parameters.width   = repaint->source.width;
            parameters.height  = repaint->source.height;
            parameters.denoise = denoise;
            submitted          = edits.editor.materialize(edits.document.prompt);
            prompt_catalog     = edits.document.catalog;
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
        runtime::Generate request;
        request.parameters = parameters;
        request.seed       = seed;
        request.prompt     = std::move(submitted);
        request.catalog    = std::move(prompt_catalog);
        request.source     = std::move(source);
        const auto task    = submit_task({std::move(request)});
        if (repaint) {
            begin_output(repaint->result, task, parameters.width, parameters.height);
            viewing = View::comparison;
            view    = {};
        }
        animate_until = glfwGetTime() + 0.2;
    }

    void Workspace::open_audit(std::string key, const bool refresh) {
        if (!leave_repaint()) return;
        if (page == Page::generation || page == Page::dataset) {
            audit_return            = page;
            audit_return_collection = collection_key;
            audit_return_view       = viewing;
            audit_return_camera     = view;
        }
        if (audit_key != key) {
            audit_report   = {};
            audit_position = {};
            audit_category.clear();
        }
        audit_key = std::move(key);
        audit_task.reset();
        for (const auto& [id, task] : std::views::reverse(task_status))
            if (task.concept_key == audit_key && task.kind == runtime::Kind::audit && (task.state == runtime::State::queued || task.state == runtime::State::running)) {
                audit_task = id;
                break;
            }
        collection_key = audit_key;
        page           = Page::audit;
        viewing        = View::browse;
        rebuild_audit();
        synchronize_collection();
        if (audit_task) return;
        const auto source = library->classifiers.find(audit_key);
        if (source == library->classifiers.end() || !source->second.model) return;
        if (!refresh && audit_report.complete) return;
        audit_task  = submit_task({runtime::Audit{audit_key, refresh}});
        audit_dirty = false;
    }
    std::uint64_t Workspace::submit_task(runtime::Request request) {
        if (!runtime) runtime = std::make_unique<WorkspaceRuntime>(renderer.device);
        const auto id   = runtime->session.enqueue(std::move(request));
        session_state   = runtime->session.snapshot();
        task_status[id] = session_state.jobs.at(id);
        return id;
    }
    void Workspace::task_event(const runtime::TaskStatus& event) {
        const auto kind  = event.kind;
        const auto id    = event.id;
        const auto state = event.state;
        const auto& key  = event.concept_key;
        if (kind != runtime::Kind::infer) task_status[id] = event;
        if (kind == runtime::Kind::train && state == runtime::State::running) {
            const auto found = training_drafts.find(key);
            if (found != training_drafts.end()) {
                if (const auto* step = std::get_if<training::Step>(&event.progress.value)) found->second.losses.push_back(step->loss);
                if (const auto* metrics = std::get_if<training::Metrics>(&event.progress.value)) found->second.metrics = *metrics;
            }
        }
        if (state == runtime::State::complete) {
            if (kind == runtime::Kind::infer) {
                const auto& result                                                       = std::get<classification::Result>(event.result.value);
                predictions[result.id + "|" + result.model_sha + "|" + result.image_sha] = result;
            } else if (kind == runtime::Kind::audit && key == audit_key && audit_task == id) {
                audit_task.reset();
                prediction_cache.entries.clear();
                rebuild_audit();
            } else if (kind == runtime::Kind::fix || kind == runtime::Kind::undo) {
                dataset_index.refresh();
            } else if (kind == runtime::Kind::train || kind == runtime::Kind::classify || kind == runtime::Kind::assign) dataset_index.refresh();
        } else if (state == runtime::State::failed || state == runtime::State::stopped) {
            if (audit_task == id) {
                audit_task.reset();
                audit_dirty = false;
            }
            if (state == runtime::State::failed && kind == runtime::Kind::infer) prediction_errors[key + "|" + event.model_sha + "|" + event.image_sha] = event.error;
            if (kind == runtime::Kind::train) dataset_index.refresh();
            if (kind == runtime::Kind::assign && key == collection_key) type_error = event.error;
        }
    }
    void Workspace::rebuild_audit() {
        audit_collection  = {.key = audit_key, .name = audit_key};
        const auto source = library->classifiers.find(audit_key);
        if (source == library->classifiers.end()) {
            audit_report = {};
            return;
        }
        audit_report = classification::view(source->second, prediction_cache, audit_category);
        audit_dirty  = !audit_report.complete;
        for (const auto& row : audit_report.rows) {
            audit_collection.images.push_back(row.sample.file);
            const auto identity   = audit_key + "|" + audit_report.model_sha + "|" + row.sample.file.sha;
            predictions[identity] = row.prediction;
            prediction_errors.erase(identity);
        }
        if (audit_collection.images.empty()) audit_position = {};
        else {
            const auto selected = std::ranges::find(audit_collection.images, audit_position.selected, &dataset::File::sha);
            const auto index    = audit_position.selected.empty() ? 0 : selected == audit_collection.images.end() ? std::min(audit_position.index, audit_collection.images.size() - 1) : std::size_t(selected - audit_collection.images.begin());
            audit_position      = {audit_collection.images[index].sha, index, static_cast<float>(index)};
        }
    }

    void Workspace::observe_inference() {
        std::vector<runtime::Infer> wanted;
        for (const auto& file : visible_images)
            for (const auto& key : activated) {
                const auto found = library->classifiers.find(key);
                if (found == library->classifiers.end() || !found->second.model) continue;
                const auto& model   = *found->second.model;
                const auto identity = key + "|" + model.sha + "|" + file.sha;
                if (predictions.contains(identity) || prediction_errors.contains(identity)) continue;
                wanted.push_back({.concept_key = key, .descriptor = model, .file = file});
            }
        if (!wanted.empty() && !runtime) runtime = std::make_unique<WorkspaceRuntime>(renderer.device);
        if (runtime) runtime->session.observe(std::move(wanted));
    }
    void Workspace::draw() {
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
            const auto info = library->classifiers.find(collection_key);
            if (!dataset_sidebar.open || concept_tool != ConceptTool::classify || info == library->classifiers.end() || !info->second.model) action_error = "Open Classify at the top of the Dataset panel before dropping a folder.";
            else if (window.dropped.size() != 1) action_error = "Drop one directory at a time.";
            else {
                submit_task({runtime::Classify{collection_key, window.dropped.front()}});
            }
            window.dropped.clear();
        }
        std::string error = library->error.empty() ? action_error : library->error;
        if (runtime) {
            auto& session = runtime->session;
            session.activate(activated);
            const auto* generate = session_state.active ? std::get_if<runtime::Generate>(&session_state.active->request.operation) : nullptr;
            const bool visible   = renderer.visible && generate && ((!generate->source && page == Page::generation) || (repaint && repaint->result.task == session_state.active->id && (viewing == View::comparison || viewing == View::result)));
            session.configure_preview(preview_enabled, visible);
            if (!session_state.error.empty()) error = session_state.error;
        }
        if (page == Page::audit && !repaint && audit_dirty && !audit_task) {
            bool stopped{};
            for (const auto& [id, task] : std::views::reverse(task_status))
                if (task.concept_key == audit_key && task.kind == runtime::Kind::audit) {
                    stopped = task.state == runtime::State::stopped || task.state == runtime::State::failed;
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
        canvas(*this, scale, size);
        sidebar(*this, true, scale, size, {});
        Picture image;
        if (page == Page::generation) image = resolve_output(generation);
        else if (repaint) image = viewing == View::result || (viewing == View::comparison && (repaint->result.texture || repaint->result.saved)) ? resolve_output(repaint->result, Role::result) : resolve_image(repaint->source, Role::source);
        else if (collection && root && root->ready && !collection->images.empty()) image = resolve_image(collection->images[current_position().index]);
        const auto controls = control_layout(*this, scale, size, image);
        sidebar(*this, false, scale, size, image);
        bottom_controls(*this, scale, size, controls, image);
        observe_inference();
        top_strip(*this, scale, size);
        preset_dialogs(*this, scale);
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
