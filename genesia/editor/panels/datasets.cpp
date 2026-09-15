module;
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_internal.h>
#include <misc/cpp/imgui_stdlib.h>
module genesia.editor.panels.datasets;
import genesia.editor.widgets.controls;
import genesia.editor.widgets.tags;
import genesia.io.files;
import genesia.runtime.session;
import genesia.runtime.catalog;
import genesia.project;
import genesia.editor.graphics.bridge;
import std;
namespace genesia::editor {
    void dataset_controls(Workspace& workspace, const float scale) {
        if (workspace.collection && workspace.root) {
            const auto concept_item = std::ranges::find(workspace.root->concepts, workspace.collection_key, &dataset::Collection::key);
            const auto& selected    = concept_item == workspace.root->concepts.end() ? workspace.root->all : *concept_item;
            const auto title        = ImGui::GetCursorScreenPos();
            const ImVec2 title_size{ImGui::GetContentRegionAvail().x, ImGui::GetTextLineHeight()};
            ImGui::Dummy(title_size);
            ImGui::RenderTextEllipsis(ImGui::GetWindowDrawList(), title, {title.x + title_size.x, title.y + title_size.y}, title.x + title_size.x, selected.name.c_str(), nullptr, nullptr);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", workspace.collection_key.c_str());
            if (workspace.collection_key == workspace.root->all.key) ImGui::TextDisabled("%zu images", selected.images.size());
            else {
                ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
                ImGui::TextWrapped("%s / %zu images", workspace.root->all.name.c_str(), selected.images.size());
                ImGui::PopStyleColor();
            }
            const auto metadata = workspace.library.concepts.find(workspace.collection_key);
            const auto failure  = workspace.library.concept_errors.find(workspace.collection_key);
            if (metadata != workspace.library.concepts.end()) {
                const auto& assigned = metadata->second;
                const bool busy      = workspace.session_state.active.has_value();
                static constexpr std::array names{"Unassigned", "Classifier", "LoRA"};
                const auto color = assigned.type == dataset::ConceptType::classifier ? ImVec4{0.44F, 0.80F, 0.87F, 1} : assigned.type == dataset::ConceptType::lora ? ImVec4{0.92F, 0.69F, 0.36F, 1} : ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled);
                ImGui::BeginDisabled(assigned.locked || busy);
                ImGui::PushStyleColor(ImGuiCol_Text, color);
                ImGui::PushStyleColor(ImGuiCol_Button, {color.x, color.y, color.z, workspace.choosing_type ? 0.22F : 0.10F});
                ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 5 * scale);
                if (ImGui::SmallButton(std::format("{}###ConceptType", names[static_cast<std::size_t>(assigned.type)]).c_str())) workspace.choosing_type = !workspace.choosing_type;
                ImGui::PopStyleVar();
                ImGui::PopStyleColor(2);
                ImGui::EndDisabled();
                if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) ImGui::SetTooltip("%s", assigned.locked ? "Type is permanently locked by training or an imported model." : busy ? "Finish the current operation before changing its type." : "Choose this concept's purpose. Assigning a type does not start training.");
                if (assigned.locked || busy) workspace.choosing_type = false;
                if (workspace.choosing_type) {
                    for (std::size_t i = 0; i < names.size(); ++i)
                        if (ImGui::Selectable(names[i], static_cast<std::size_t>(assigned.type) == i)) {
                            if (assigned.type == static_cast<dataset::ConceptType>(i)) {
                                workspace.choosing_type = false;
                                workspace.type_error.clear();
                                break;
                            }
                            const auto assign = [&workspace, type = static_cast<dataset::ConceptType>(i)] {
                                try {
                                    workspace.submit_task({runtime::Assign{workspace.collection_key, type}});
                                    workspace.training_drafts.erase(workspace.collection_key);
                                    workspace.concept_tool  = Workspace::ConceptTool::none;
                                    workspace.choosing_type = false;
                                    workspace.type_error.clear();
                                } catch (const std::exception& error) {
                                    workspace.type_error = error.what();
                                }
                            };
                            if (workspace.save_caption(assign)) assign();
                            break;
                        }
                }
                if (!workspace.type_error.empty()) ImGui::TextWrapped("%s", workspace.type_error.c_str());
                if (failure != workspace.library.concept_errors.end()) ImGui::TextWrapped("%s", failure->second.c_str());
                else if (assigned.type != dataset::ConceptType::none) {
                    const bool lora       = assigned.type == dataset::ConceptType::lora;
                    const auto captions   = workspace.library.captions.find(workspace.collection_key);
                    const auto classifier = workspace.library.classifiers.find(workspace.collection_key);
                    const bool available  = lora ? captions != workspace.library.captions.end() : classifier != workspace.library.classifiers.end();
                    if (!available) ImGui::TextDisabled("Reading concept data...");
                    else {
                        const auto registered = workspace.library.loras.find(workspace.collection_key);
                        const bool published  = lora ? registered != workspace.library.loras.end() && registered->second.has_value() : classifier->second.model.has_value();
                        const bool active     = workspace.model_settings.at(workspace.collection_key).active;
                        if (!workspace.choosing_type && workspace.type_error.empty()) ImGui::SameLine(0, 8 * scale);
                        if (published) ImGui::TextColored(color, "%s", active ? "Active" : "Model available");
                        else ImGui::TextDisabled("%s", lora ? "No imported model" : "No published model");
                        if (published && ImGui::IsItemHovered()) ImGui::SetTooltip("Middle-click this concept in the dataset list to activate or deactivate its model.");
                        const int buttons = lora || published ? 3 : 1;
                        const float gap   = 4 * scale;
                        const float width = (ImGui::GetContentRegionAvail().x - (buttons - 1) * gap) / buttons;
                        ImGui::Spacing();
                        int column{};
                        for (const auto [tool, label] : {std::pair{Workspace::ConceptTool::train, "Train"}, std::pair{Workspace::ConceptTool::audit, "Audit"}, std::pair{Workspace::ConceptTool::classify, "Classify"}, std::pair{Workspace::ConceptTool::tags, "Tags"}, std::pair{Workspace::ConceptTool::export_dataset, "Export"}, std::pair{Workspace::ConceptTool::model, "Model"}}) {
                            if (lora != (tool == Workspace::ConceptTool::tags || tool == Workspace::ConceptTool::export_dataset || tool == Workspace::ConceptTool::model)) continue;
                            if (!lora && !published && tool != Workspace::ConceptTool::train) continue;
                            if (column++) ImGui::SameLine(0, gap);
                            const bool current = workspace.concept_tool == tool;
                            ImGui::PushStyleColor(ImGuiCol_Button, {color.x, color.y, color.z, current ? 0.08F : 0});
                            ImGui::PushStyleColor(ImGuiCol_Text, current ? ImGui::GetStyleColorVec4(ImGuiCol_Text) : ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
                            if (ImGui::Button(label, {width, 30 * scale})) workspace.select_tool(tool);
                            ImGui::PopStyleColor(2);
                            if (workspace.concept_tool == tool) {
                                const auto minimum = ImGui::GetItemRectMin();
                                const auto maximum = ImGui::GetItemRectMax();
                                ImGui::GetWindowDrawList()->AddLine({minimum.x + 8 * scale, maximum.y - scale}, {maximum.x - 8 * scale, maximum.y - scale}, ImGui::GetColorU32(color), 2 * scale);
                            }
                        }
                        if (workspace.concept_tool != Workspace::ConceptTool::none) {
                            ImGui::PushID(workspace.collection_key.c_str());
                            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {4 * scale, 6 * scale});
                            const bool tags    = lora && workspace.concept_tool == Workspace::ConceptTool::tags;
                            const float height = std::max(1.0F, ImGui::GetContentRegionAvail().y * 0.45F);
                            ImGui::SetNextWindowSizeConstraints({0, 0}, {FLT_MAX, height});
                            if (ImGui::BeginChild("##ConceptTool", {0, tags ? height : 0}, ImGuiChildFlags_AlwaysUseWindowPadding | (tags ? ImGuiChildFlags_None : ImGuiChildFlags_AutoResizeY), ImGuiWindowFlags_NoBackground | (tags ? ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse : 0))) {
                                if (lora) {
                                    if (workspace.concept_tool == Workspace::ConceptTool::model) model_controls(workspace);
                                    else if (workspace.concept_tool == Workspace::ConceptTool::tags) caption_controls(workspace, captions->second, scale);
                                    else export_controls(workspace, captions->second);
                                } else if (!classifier->second.inspected && workspace.concept_tool != Workspace::ConceptTool::classify) ImGui::TextDisabled("Reading concept samples...");
                                else if (workspace.concept_tool == Workspace::ConceptTool::train) training_controls(workspace, classifier->second, scale);
                                else if (workspace.concept_tool == Workspace::ConceptTool::audit) audit_controls(workspace, classifier->second);
                                else classify_controls(workspace, classifier->second);
                            }
                            ImGui::EndChild();
                            ImGui::PopStyleVar();
                            ImGui::PopID();
                        }
                    }
                }
            } else if (failure != workspace.library.concept_errors.end()) ImGui::TextWrapped("%s", failure->second.c_str());
            else if (workspace.collection_key == workspace.root->all.key) {
                ImGui::TextDisabled("%zu concepts", workspace.root->concepts.size());
                ImGui::Spacing();
                ImGui::BeginDisabled(workspace.session_state.active.has_value());
                if (ImGui::Button("Normalize Dataset", {-FLT_MIN, 0})) workspace.submit_task({runtime::Normalize{workspace.collection_key}});
                ImGui::EndDisabled();
                if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) ImGui::SetTooltip("Merge identical PNG copies into hard links, then number each folder's images\n00001.png, 00002.png, ... by modification time. Dot directories are excluded.\nImage bytes stay unchanged.\nRenaming changes training fingerprints; existing models remain available.");
                operation_activity(workspace, {runtime::Kind::normalize}, workspace.collection_key);
            }
            if (!workspace.root->ready) ImGui::TextColored({0.95F, 0.49F, 0.42F, 1}, "Index error / details below");
        } else if (workspace.library.ready && workspace.page != Workspace::Page::generation) ImGui::TextWrapped("Dataset does not exist: %s", workspace.collection_key.c_str());
        else ImGui::TextDisabled(workspace.library.ready ? "Choose a dataset" : "Indexing...");
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
    }

    std::string concept_activity(const Workspace& workspace, const std::string_view key) {
        std::string result;
        for (const auto& [identity, task] : workspace.activity) {
            if (task.concept_key != key || task.kind == runtime::Kind::infer || task.state == runtime::State::complete || task.state == runtime::State::stopped) continue;
            if (!result.empty()) result += " / ";
            result += std::format("{} {}", runtime::kinds[std::size_t(task.kind)], runtime::states[std::size_t(task.state)]);
        }
        return result;
    }

    std::optional<std::string> dataset_contents(Workspace& workspace) {
        std::optional<std::string> selected;
        const float scale = workspace.renderer.dpi;
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {0, 4 * scale});
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, {6 * scale, 4 * scale});
        ImGui::PushStyleVar(ImGuiStyleVar_IndentSpacing, ImGui::GetTreeNodeToLabelSpacing() + 6 * scale);
        for (const auto& entry : workspace.library.roots) {
            ImGui::PushID(entry.all.key.c_str());
            const auto origin = ImGui::GetCursorScreenPos();
            const float width = ImGui::GetContentRegionAvail().x;
            ImGui::SetNextItemOpen(true, workspace.expand_dataset_roots ? ImGuiCond_Always : ImGuiCond_Once);
            const auto flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_FramePadding | (workspace.collection_key == entry.all.key ? ImGuiTreeNodeFlags_Selected : 0) | (entry.concepts.empty() && entry.ready ? ImGuiTreeNodeFlags_Leaf : 0);
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
                    const auto metadata     = workspace.library.concepts.find(item.key);
                    const auto info         = workspace.library.classifiers.find(item.key);
                    const auto failure      = workspace.library.concept_errors.find(item.key);
                    const auto caption_data = workspace.library.captions.find(item.key);
                    const bool lora_invalid = caption_data != workspace.library.captions.end() && !caption_data->second.issue.empty();
                    const bool lora_type    = metadata != workspace.library.concepts.end() && metadata->second.type == dataset::ConceptType::lora;
                    const auto lora_model   = workspace.library.loras.find(item.key);
                    const bool published    = lora_type ? lora_model != workspace.library.loras.end() && lora_model->second.has_value() : info != workspace.library.classifiers.end() && info->second.model.has_value();
                    const auto controls     = workspace.model_settings.find(item.key);
                    const bool enabled      = published && controls != workspace.model_settings.end() && controls->second.active;
                    const bool replacing    = workspace.session_state.active && workspace.session_state.active->concept_key == item.key && (workspace.session_state.active->kind == runtime::Kind::lora || workspace.session_state.active->kind == runtime::Kind::assign);
                    const auto origin       = ImGui::GetCursorScreenPos();
                    const float width       = ImGui::GetContentRegionAvail().x;
                    const float height      = ImGui::GetTextLineHeight() + 8 * scale;
                    auto* draw              = ImGui::GetWindowDrawList();
                    const ImVec2 minimum{origin.x, origin.y};
                    const ImVec2 maximum{std::min(origin.x + width, draw->GetClipRectMax().x), origin.y + height};
                    if (ImGui::Selectable("##Concept", workspace.collection_key == item.key, ImGuiSelectableFlags_None, {width, height})) selected = item.key;
                    const bool hovered = ImGui::IsItemHovered();
                    if (published && !replacing && failure == workspace.library.concept_errors.end() && controls != workspace.model_settings.end() && ImGui::IsItemClicked(ImGuiMouseButton_Middle)) {
                        controls->second.active = !enabled;
                        controls->second.dirty  = true;
                        workspace.save_model_settings(item.key);
                    }
                    const float y    = origin.y + 4 * scale;
                    const auto count = std::to_string(item.images.size());
                    float right      = maximum.x - 6 * scale - ImGui::CalcTextSize(count.c_str()).x;
                    draw->AddText({right, y}, ImGui::GetColorU32(ImGuiCol_TextDisabled), count.c_str());
                    right -= 8 * scale;
                    float left = minimum.x + 6 * scale;
                    if (enabled) draw->AddCircleFilled({left - 4 * scale, (minimum.y + maximum.y) / 2}, 2 * scale, ImGui::GetColorU32(ImGuiCol_Text));
                    if (failure != workspace.library.concept_errors.end() || lora_invalid) {
                        right -= ImGui::CalcTextSize("!").x;
                        draw->AddText({right, y}, ImGui::GetColorU32(ImVec4{0.95F, 0.49F, 0.42F, 1}), "!");
                        right -= 8 * scale;
                    } else if (metadata != workspace.library.concepts.end() && metadata->second.type != dataset::ConceptType::none) {
                        const bool classifier_type = metadata->second.type == dataset::ConceptType::classifier;
                        const char* tag            = classifier_type ? "C" : "L";
                        const auto color           = classifier_type ? ImVec4{0.44F, 0.80F, 0.87F, 1} : ImVec4{0.92F, 0.69F, 0.36F, 1};
                        const float width          = ImGui::CalcTextSize(tag).x + 10 * scale;
                        draw->AddRectFilled({left, y - scale}, {left + width, y + ImGui::GetFontSize() + scale}, ImGui::GetColorU32(ImVec4{color.x, color.y, color.z, 0.12F}), 4 * scale);
                        draw->AddText({left + 5 * scale, y}, ImGui::GetColorU32(color), tag);
                        left += width + 8 * scale;
                        if (published) {
                            ImGui::RenderCheckMark(draw, {left, y + 2 * scale}, ImGui::GetColorU32(color), 10 * scale);
                            left += 18 * scale;
                        }
                    }
                    ImGui::RenderTextEllipsis(draw, {left, y}, {std::max(left, right), y + ImGui::GetFontSize()}, std::max(left, right), item.name.c_str(), nullptr, nullptr);
                    if (hovered) {
                        ImGui::BeginTooltip();
                        ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + 320 * scale);
                        ImGui::TextUnformatted(item.key.c_str());
                        ImGui::TextDisabled("%zu images", item.images.size());
                        if (metadata != workspace.library.concepts.end()) {
                            ImGui::Text("Type: %s%s", metadata->second.type == dataset::ConceptType::none ? "Unassigned" : metadata->second.type == dataset::ConceptType::classifier ? "Classifier" : "LoRA", metadata->second.locked ? " / Locked" : "");
                            if (published) {
                                ImGui::TextUnformatted(enabled ? "Model available / Active" : "Model available");
                                ImGui::TextDisabled("Middle-click to activate or deactivate.");
                            }
                            if (metadata->second.type == dataset::ConceptType::classifier) {
                                const auto activity = concept_activity(workspace, item.key);
                                if (!activity.empty()) ImGui::TextUnformatted(activity.c_str());
                                if (info != workspace.library.classifiers.end() && info->second.training.has_value()) {
                                    ImGui::Text("Training: %s", training::phases[std::size_t(info->second.training->phase)].data());
                                    if (info->second.training->fingerprint != info->second.fingerprint) ImGui::TextUnformatted("Training data changed. Restart is required to train again.");
                                }
                            }
                        }
                        if (failure != workspace.library.concept_errors.end()) ImGui::TextColored({0.95F, 0.49F, 0.42F, 1}, "%s", failure->second.c_str());
                        if (lora_invalid) ImGui::TextColored({0.95F, 0.49F, 0.42F, 1}, "Duplicate images in LoRA concept");
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
        if (!workspace.library.roots.empty()) workspace.expand_dataset_roots = false;
        ImGui::PopStyleVar(3);
        return selected;
    }

    void operation_activity(Workspace& workspace, const std::initializer_list<runtime::Kind> kinds, const std::string_view key) {
        const runtime::TaskStatus* task{};
        for (const auto kind : kinds) {
            const auto found = workspace.activity.find({std::string{key}, kind});
            if (found != workspace.activity.end() && (!task || found->second.id > task->id)) task = &found->second;
        }
        if (!task) return;
        const auto* batch = std::get_if<runtime::BatchProgress>(&task->progress.value);
        ImGui::TextDisabled("%s", task->state < runtime::State::complete && task->stopping ? "Stopping..." : task->state < runtime::State::complete && batch ? runtime::stages[std::size_t(batch->stage)].data() : runtime::states[std::size_t(task->state)].data());
        if (task->state < runtime::State::complete) {
            if (batch && batch->total) {
                ImGui::ProgressBar(float(batch->completed) / batch->total, {-1, 3 * workspace.renderer.dpi}, "");
                ImGui::Text("%zu / %zu", batch->completed, batch->total);
            }
            const bool committing = task->kind == runtime::Kind::fix || task->kind == runtime::Kind::assign || task->kind == runtime::Kind::lora || (batch && batch->stage == runtime::Stage::moving);
            if (!committing && ImGui::Button("Stop", {-FLT_MIN, 0})) workspace.runtime.session.cancel(task->id);
        }
        if (!task->error.empty()) ImGui::TextWrapped("%s", task->error.c_str());
        if (const auto* normalized = std::get_if<dataset::NormalizeResult>(&task->result.value)) {
            ImGui::TextWrapped("%zu PNG entries checked", normalized->files);
            ImGui::TextWrapped("%zu copies replaced with hard links", normalized->linked);
            ImGui::TextWrapped("%zu images renamed", normalized->renamed.size());
        }
        if (const auto* classified = std::get_if<classification::Classification>(&task->result.value))
            for (const auto& [label, count] : classified->classes) ImGui::TextWrapped("%s / %zu images", label.c_str(), count);
    }

    void training_controls(Workspace& workspace, const training::TrainingData& source, const float scale) {
        auto& edits                     = workspace.training_drafts.try_emplace(source.key, source).first->second;
        const auto found                = workspace.activity.find({source.key, runtime::Kind::train});
        const runtime::TaskStatus* task = found == workspace.activity.end() ? nullptr : &found->second;
        const bool busy                 = task && task->state < runtime::State::complete;
        const bool recorded             = source.training.has_value();
        const bool changed              = recorded && source.training->fingerprint != source.fingerprint;
        const bool configured           = recorded && source.training->config != edits.config;
        const bool checkpoint           = recorded && source.training->checkpoint;
        const int completed             = recorded ? source.training->step : 0;
        const auto& issue               = source.issue.empty() ? source.training_issue : source.issue;
        const auto& counts              = source.counts;
        if (!issue.empty()) ImGui::TextWrapped("%s", issue.c_str());
        if (changed) ImGui::TextWrapped("Training data changed. Restart training is required.");
        else if (configured) ImGui::TextWrapped("Training parameters changed. Restart training is required.");
        else if (recorded && !checkpoint && !busy) ImGui::TextWrapped("Checkpoint missing. Restart training is required.");
        ImGui::Spacing();
        if (task) {
            if (!busy && recorded) {
                const auto* result = std::get_if<training::State>(&task->result.value);
                ImGui::TextDisabled("%s / step %d", runtime::states[std::size_t(task->state)].data(), result ? result->step : completed);
            } else ImGui::TextDisabled("%s", runtime::states[std::size_t(task->state)].data());
            if (task->state == runtime::State::failed) ImGui::TextWrapped("%s", task->error.c_str());
            if (busy) {
                if (const auto* batch = std::get_if<runtime::BatchProgress>(&task->progress.value)) {
                    ImGui::TextDisabled("Preparing %zu / %zu images", batch->completed, batch->total);
                    ImGui::ProgressBar(float(batch->completed) / batch->total, {-1, 3 * scale}, "");
                } else if (const auto* step = std::get_if<training::Step>(&task->progress.value)) {
                    ImGui::Text("Step %d / %d", step->step, step->target);
                    ImGui::ProgressBar(float(step->step) / step->target, {-1, 3 * scale}, "");
                    ImGui::TextDisabled("Loss %.4f / %.1fs", step->loss, step->seconds);
                } else if (const auto* metrics = std::get_if<training::Metrics>(&task->progress.value)) ImGui::TextDisabled("Evaluating / step %d", metrics->step);
            }
        } else if (recorded) ImGui::TextDisabled("%s / step %d", training::phases[std::size_t(source.training->phase)].data(), completed);
        bool submit{}, restart{};
        if (busy) {
            if (ImGui::Button("Stop training", {-FLT_MIN, 0})) workspace.runtime.session.cancel(task->id);
        } else {
            ImGui::AlignTextToFramePadding();
            ImGui::TextDisabled("Target steps");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(-FLT_MIN);
            ImGui::InputInt("##TrainingSteps", &edits.steps, 0, 0);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Cumulative training target, including completed steps.");
            const bool target = edits.steps > completed && (!recorded || edits.steps >= source.training->target);
            if (!target) ImGui::TextWrapped("Increase the cumulative target to continue training.");
            ImGui::BeginDisabled(workspace.session_state.active.has_value() || !issue.empty() || changed || configured || (recorded && !checkpoint) || !target);
            submit = ImGui::Button(recorded ? "Continue training" : "Start training", {-FLT_MIN, 0});
            ImGui::EndDisabled();
            if (recorded) {
                ImGui::BeginDisabled(workspace.session_state.active.has_value() || !issue.empty());
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
            workspace.submit_task({runtime::Train{{.concept_key = source.key, .steps = edits.steps, .restart = restart, .config = edits.config}}});
            edits.restart_confirm = false;
            if (restart) {
                edits.losses.clear();
                edits.metrics.reset();
            }
        }
        if (!edits.losses.empty()) {
            ImGui::Spacing();
            ImGui::TextDisabled("Training loss / %.4f", edits.losses.back());
            ImGui::PushStyleColor(ImGuiCol_FrameBg, {0, 0, 0, 0});
            ImGui::PlotLines("##TrainingLoss", edits.losses.data(), int(edits.losses.size()), 0, nullptr, FLT_MAX, FLT_MAX, {-FLT_MIN, 60 * scale});
            ImGui::PopStyleColor();
        }
        ImGui::PushStyleColor(ImGuiCol_Header, {0, 0, 0, 0});
        if (ImGui::CollapsingHeader("Categories & checks")) {
            for (std::size_t i = 0; i < counts.size(); ++i) ImGui::TextWrapped("%s / %zu images", source.classes[i].c_str(), counts[i]);
            ImGui::TextWrapped("Training checks original sizes, labels and independent train / validation groups before loading the model. Generation records are optional.");
        }
        if (ImGui::CollapsingHeader("Parameters")) {
            ImGui::BeginDisabled(busy);
            const auto parameter = [&]<typename T>(const char* name, T& value) {
                ImGui::TextWrapped("%s", name);
                ImGui::PushID(name);
                ImGui::SetNextItemWidth(-FLT_MIN);
                if constexpr (std::same_as<T, int>) ImGui::InputInt("##Value", &value, 0, 0);
                else if constexpr (std::same_as<T, std::uint64_t>) ImGui::InputScalar("##Value", ImGuiDataType_U64, &value);
                else ImGui::InputFloat("##Value", &value, 0, 0, "%.8g");
                ImGui::PopID();
            };
            parameter("Physical batch", edits.config.physical_batch);
            parameter("Effective batch", edits.config.effective_batch);
            parameter("Head only steps", edits.config.head_only_steps);
            parameter("Warmup steps", edits.config.warmup_steps);
            parameter("Head only learning rate", edits.config.head_only_lr);
            parameter("Backbone learning rate", edits.config.backbone_lr);
            parameter("Head learning rate", edits.config.head_lr);
            parameter("Weight decay", edits.config.weight_decay);
            parameter("Clip norm", edits.config.clip_norm);
            parameter("Seed", edits.config.seed);
            parameter("Evaluation interval", edits.config.eval_interval);
            parameter("Save interval", edits.config.save_interval);
            parameter("Log interval", edits.config.log_interval);
            ImGui::EndDisabled();
        }
        if (ImGui::CollapsingHeader("Evaluation")) {
            if (!edits.metrics) ImGui::TextWrapped("Evaluation appears after a training checkpoint.");
            else {
                const auto& metrics = *edits.metrics;
                ImGui::Text("Step %d / %zu samples", metrics.step, metrics.samples);
                ImGui::Text("Cross-entropy %.4f", metrics.loss);
                ImGui::Text("Accuracy %.2f%%", metrics.accuracy * 100);
                ImGui::Text("Macro precision %.3f", metrics.precision);
                ImGui::Text("Macro recall %.3f", metrics.recall);
                ImGui::Text("Macro F1 %.3f", metrics.f1);
                if (ImGui::TreeNode("Per-category metrics")) {
                    for (const auto& row : metrics.classes) {
                        ImGui::TextWrapped("%s / %zu samples", row.name.c_str(), row.samples);
                        ImGui::TextWrapped("P %.3f / R %.3f / F1 %.3f", row.precision, row.recall, row.f1);
                    }
                    ImGui::TreePop();
                }
                if (ImGui::TreeNode("Confusion matrix")) {
                    const auto& classes = metrics.classes;
                    ImGui::TextWrapped("Rows: human category. Columns: predicted category.");
                    if (ImGui::BeginTable("##Confusion", int(classes.size()) + 1, ImGuiTableFlags_Borders | ImGuiTableFlags_ScrollX | ImGuiTableFlags_SizingFixedFit, {0, 0}, (120 + 72 * classes.size()) * scale)) {
                        ImGui::TableSetupColumn("Category", ImGuiTableColumnFlags_WidthFixed, 120 * scale);
                        for (const auto& label : classes) ImGui::TableSetupColumn(label.name.c_str(), ImGuiTableColumnFlags_WidthFixed, 72 * scale);
                        ImGui::TableHeadersRow();
                        for (std::size_t i = 0; i < classes.size(); ++i) {
                            ImGui::TableNextRow();
                            ImGui::TableNextColumn();
                            ImGui::TextWrapped("%s", classes[i].name.c_str());
                            for (const auto count : metrics.confusion[i]) {
                                ImGui::TableNextColumn();
                                ImGui::Text("%d", count);
                            }
                        }
                        ImGui::EndTable();
                    }
                    ImGui::TreePop();
                }
            }
        }
        ImGui::PopStyleColor();
    }

    void classify_controls(Workspace& workspace, const training::TrainingData& source) {
        const bool trained = source.model.has_value();
        auto& path         = workspace.classify_paths[source.key];
        if (!trained) ImGui::TextWrapped("A published classifier is required to submit another folder.");
        ImGui::BeginDisabled(!trained || workspace.session_state.active.has_value());
        ImGui::TextWrapped("Drop one folder into the window, or paste its path below.");
        ImGui::TextWrapped("Direct PNG images move into predicted category folders. Original names are retained.");
        ImGui::TextUnformatted("Directory");
        ImGui::SetNextItemWidth(-FLT_MIN);
        ImGui::InputText("##ClassifyDirectory", path.data(), path.size());
        ImGui::BeginDisabled(!path.front());
        if (ImGui::Button("Classify folder", {-FLT_MIN, 0})) {
            workspace.submit_task({runtime::Classify{source.key, files::path(path.data())}});
        }
        ImGui::EndDisabled();
        ImGui::EndDisabled();
        operation_activity(workspace, {runtime::Kind::classify}, source.key);
    }

    void caption_controls(Workspace& workspace, const caption::Dataset& source, const float scale) {
        auto& editor        = workspace.caption_editor;
        const auto identity = source.key + "/" + workspace.caption_folder;
        const auto found    = source.document.folders.find(workspace.caption_folder);
        const auto text     = found == source.document.folders.end() ? std::string{} : caption::compose(found->second);
        const bool changed  = editor.key != identity;
        if (changed) editor = {.key = identity, .input = text, .saved = text};
        else if (!editor.task && editor.input == editor.saved) editor.input = editor.saved = text;
        const auto& folder  = workspace.caption_folder;
        const auto red      = ImVec4{0.98F, 0.34F, 0.32F, 1};
        const auto bypassed = std::ranges::count(source.folders, true, &caption::Folder::excluded);
        const auto included = source.folders.size() - bypassed;
        const auto status   = std::format("{} / {} directories complete / {} bypassed{}", included - source.missing_tags.size(), included, bypassed, source.missing_tags.empty() ? std::string{} : std::format(" / {} missing tags", source.missing_tags.size()));
        ImGui::PushStyleColor(ImGuiCol_Text, source.missing_tags.empty() ? ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled) : red);
        ImGui::TextWrapped("%s", status.c_str());
        ImGui::PopStyleColor();
        const auto available    = ImGui::GetContentRegionAvail();
        const bool horizontal   = available.x >= 560 * scale;
        const float gap         = 12 * scale;
        const float tree_width  = horizontal ? (available.x - gap) * 0.40F : available.x;
        const float tree_height = std::max(1.0F, horizontal ? available.y : (available.y - gap) * 0.45F);
        std::optional<std::string> selected, toggled;
        if (ImGui::BeginChild("##CaptionTree", {tree_width, tree_height}, ImGuiChildFlags_None, ImGuiWindowFlags_NoBackground)) {
            const bool reveal = ImGui::IsWindowAppearing();
            std::map<std::string, std::vector<const caption::Folder*>> children;
            std::map<std::string, std::size_t> missing_descendants;
            for (const auto& directory : source.folders) {
                if (directory.path == ".") continue;
                const auto path = files::path(directory.path);
                children[path.has_parent_path() ? files::utf8(path.parent_path()) : "."].push_back(&directory);
            }
            for (const auto& missing : source.missing_tags) {
                auto path = files::path(missing);
                while (path != ".") {
                    path = path.has_parent_path() ? path.parent_path() : std::filesystem::path{"."};
                    ++missing_descendants[files::utf8(path)];
                }
            }
            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, {4 * scale, 4 * scale});
            ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {0, 2 * scale});
            const auto draw_directory = [&](this auto&& draw_directory, const caption::Folder& directory) -> void {
                ImGui::PushID(directory.path.c_str());
                const auto descendants = children.find(directory.path);
                const bool branch      = descendants != children.end();
                const bool missing     = source.missing_tags.contains(directory.path);
                const bool current     = directory.path == folder;
                const auto position    = ImGui::GetCursorScreenPos();
                const float width      = ImGui::GetContentRegionAvail().x;
                const float height     = ImGui::GetTextLineHeight() + 8 * scale;
                auto* draw             = ImGui::GetWindowDrawList();
                if (missing) draw->AddRectFilled(position, {position.x + width, position.y + height}, ImGui::GetColorU32(ImVec4{red.x, red.y, red.z, 0.10F}), 3 * scale);
                const bool ancestor = directory.path == "." || folder.starts_with(directory.path + "/");
                ImGui::SetNextItemOpen(true, reveal && ancestor ? ImGuiCond_Always : ImGuiCond_Once);
                ImGui::PushStyleColor(ImGuiCol_Text, missing ? red : ImGui::GetStyleColorVec4(directory.excluded ? ImGuiCol_TextDisabled : ImGuiCol_Text));
                ImGui::PushStyleColor(ImGuiCol_HeaderHovered, missing ? ImVec4{red.x, red.y, red.z, 0.18F} : ImVec4{0.92F, 0.69F, 0.36F, 0.10F});
                const bool open = ImGui::TreeNodeEx("##Directory", ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_FramePadding | ImGuiTreeNodeFlags_NoTreePushOnOpen | (branch ? 0 : ImGuiTreeNodeFlags_Leaf), "");
                if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen()) selected = directory.path;
                if (ImGui::IsItemClicked(ImGuiMouseButton_Middle) && !workspace.session_state.active) toggled = directory.path;
                const bool hovered = ImGui::IsItemHovered();
                if (current && reveal) ImGui::SetScrollHereY(0.5F);
                const float y         = position.y + 4 * scale;
                float right           = std::min(position.x + width, draw->GetClipRectMax().x) - 4 * scale;
                const auto incomplete = missing_descendants.find(directory.path);
                if (incomplete != missing_descendants.end()) {
                    const auto count = std::format("{} below", incomplete->second);
                    right -= ImGui::CalcTextSize(count.c_str()).x;
                    draw->AddText({right, y}, ImGui::GetColorU32(red), count.c_str());
                    right -= 6 * scale;
                }
                if (missing) {
                    right -= ImGui::CalcTextSize("!").x;
                    draw->AddText({right, y}, ImGui::GetColorU32(red), "!");
                    right -= 6 * scale;
                }
                if (!directory.total) {
                    right -= ImGui::CalcTextSize("empty").x;
                    draw->AddText({right, y}, ImGui::GetColorU32(ImGuiCol_TextDisabled), "empty");
                    right -= 6 * scale;
                }
                const float left        = position.x + ImGui::GetTreeNodeToLabelSpacing();
                const auto label        = files::utf8(files::path(directory.path == "." ? source.key : directory.path).filename());
                const auto tags         = source.document.folders.find(directory.path);
                const auto own_tags     = tags == source.document.folders.end() ? std::string{} : caption::compose(tags->second);
                const float label_right = own_tags.empty() ? std::max(left, right) : left + std::min(ImGui::CalcTextSize(label.data(), label.data() + label.size()).x, std::max(0.0F, (right - left - 8 * scale) * 0.5F));
                ImGui::RenderTextEllipsis(draw, {left, y}, {label_right, y + ImGui::GetFontSize()}, label_right, label.data(), label.data() + label.size(), nullptr);
                if (directory.excluded) {
                    const float end = std::min(label_right, left + ImGui::CalcTextSize(label.c_str()).x);
                    if (end > left) draw->AddLine({left, y + ImGui::GetFontSize() * 0.5F}, {end, y + ImGui::GetFontSize() * 0.5F}, ImGui::GetColorU32(ImGuiCol_Text), scale);
                }
                const float tags_left = label_right + 8 * scale;
                if (!own_tags.empty() && tags_left < right) {
                    ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
                    ImGui::RenderTextEllipsis(draw, {tags_left, y}, {right, y + ImGui::GetFontSize()}, right, own_tags.data(), own_tags.data() + own_tags.size(), nullptr);
                    ImGui::PopStyleColor();
                }
                if (current) draw->AddRectFilled({position.x, position.y + 2 * scale}, {position.x + 2 * scale, position.y + height - 2 * scale}, ImGui::GetColorU32(ImVec4{0.96F, 0.76F, 0.42F, 1}), scale);
                ImGui::PopStyleColor(2);
                if (hovered) {
                    ImGui::BeginTooltip();
                    ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + 420 * scale);
                    ImGui::TextUnformatted((source.key + (directory.path == "." ? "" : "/" + directory.path)).c_str());
                    if (directory.excluded) ImGui::TextDisabled(source.document.bypass.contains(directory.path) ? "Bypassed: this folder and its descendants are excluded from export." : "Excluded from export by a parent folder's bypass.");
                    ImGui::TextDisabled(workspace.session_state.active ? "Finish the current operation before changing bypass." : "Middle-click toggles this folder's own bypass. A parent's bypass still applies.");
                    if (missing) ImGui::TextColored(red, "This directory needs its own tags, even when empty.");
                    else ImGui::TextWrapped("%s", own_tags.c_str());
                    if (incomplete != missing_descendants.end()) ImGui::TextColored(red, "%zu descendant directories still need tags.", incomplete->second);
                    ImGui::PopTextWrapPos();
                    ImGui::EndTooltip();
                }
                if (open && branch) {
                    ImGui::Indent(14 * scale);
                    for (const auto* child : descendants->second) draw_directory(*child);
                    ImGui::Unindent(14 * scale);
                    const float guide = position.x + 7 * scale;
                    draw->AddLine({guide, position.y + height}, {guide, ImGui::GetCursorScreenPos().y - 2 * scale}, ImGui::GetColorU32(ImVec4{0.70F, 0.72F, 0.75F, 0.16F}), scale);
                }
                ImGui::PopID();
            };
            draw_directory(*std::ranges::find(source.folders, std::string_view{"."}, &caption::Folder::path));
            ImGui::PopStyleVar(2);
        }
        ImGui::EndChild();
        if (toggled) {
            const auto toggle = [&workspace, key = source.key, path = *toggled, bypass = !source.document.bypass.contains(*toggled)] {
                try {
                    workspace.caption_editor.task = workspace.submit_task({runtime::Caption{key, path, std::nullopt, bypass}});
                    workspace.caption_editor.error.clear();
                } catch (const std::exception& failure) {
                    workspace.caption_editor.error = failure.what();
                }
            };
            if (workspace.save_caption(toggle)) toggle();
        }
        if (selected && *selected != folder) {
            workspace.select_collection(source.key + (*selected == "." ? "" : "/" + *selected));
            if (workspace.caption_editor.key != identity) return;
        }
        if (horizontal) ImGui::SameLine(0, gap);
        else ImGui::SetCursorPosY(ImGui::GetCursorPosY() + gap - ImGui::GetStyle().ItemSpacing.y);
        const float edit_height = std::max(1.0F, horizontal ? available.y : available.y - tree_height - gap);
        if (changed) ImGui::SetNextWindowScroll({0, 0});
        if (ImGui::BeginChild("##CaptionDetails", {0, edit_height}, ImGuiChildFlags_None, ImGuiWindowFlags_NoBackground)) {
            ImGui::TextWrapped("%s", folder == "." ? "Concept root" : folder.c_str());
            const auto directory = std::ranges::find(source.folders, folder, &caption::Folder::path);
            ImGui::TextDisabled("%zu direct / %zu including descendants", directory->direct, directory->total);
            if (directory->excluded) ImGui::TextDisabled("Bypassed / excluded from export");
            if (folder != ".") {
                const auto path      = files::path(folder);
                const auto inherited = caption::compose(caption::resolve(source.document, source.key, path.has_parent_path() ? files::utf8(path.parent_path()) : "."));
                ImGui::Spacing();
                ImGui::TextDisabled("Trigger + inherited tags");
                const auto position = ImGui::GetCursorScreenPos();
                const float width   = ImGui::GetContentRegionAvail().x;
                const auto* begin   = inherited.data();
                const auto* end     = begin + inherited.size();
                const auto* wrap    = ImGui::GetFont()->CalcWordWrapPosition(ImGui::GetFontSize(), begin, end, width);
                const float height  = ImGui::GetTextLineHeight();
                ImGui::Dummy({width, height * (wrap == end ? 1 : 2)});
                auto* draw = ImGui::GetWindowDrawList();
                ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
                ImGui::RenderTextEllipsis(draw, position, {position.x + width, position.y + height}, position.x + width, begin, wrap, nullptr);
                while (wrap != end && *wrap == ' ') ++wrap;
                if (wrap != end) ImGui::RenderTextEllipsis(draw, {position.x, position.y + height}, {position.x + width, position.y + height * 2}, position.x + width, wrap, end, nullptr);
                ImGui::PopStyleColor();
                if (ImGui::IsItemHovered()) {
                    ImGui::BeginTooltip();
                    ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + 480 * scale);
                    ImGui::TextUnformatted(inherited.c_str());
                    ImGui::PopTextWrapPos();
                    ImGui::EndTooltip();
                }
            }
            ImGui::Spacing();
            ImGui::TextDisabled("This directory's tags");
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Comma-separated tags. Enter or leave the editor to save.\nEsc discards unsaved changes. The concept folder name is the trigger word, followed by tags from this directory up to the concept root.");
            if (editor.task || editor.saved_at > 0 && workspace.frame_time < editor.saved_at + 2) {
                ImGui::SameLine();
                ImGui::TextDisabled(editor.task ? "Saving..." : "Saved");
                if (!editor.task) workspace.refresh_at = std::min(workspace.refresh_at, editor.saved_at + 2);
            }
            ImGui::BeginDisabled(workspace.session_state.active.has_value());
            if (editor.draw(workspace.tag_search, scale)) workspace.save_caption();
            ImGui::EndDisabled();
            if (!editor.error.empty()) {
                ImGui::PushStyleColor(ImGuiCol_Text, {0.95F, 0.49F, 0.42F, 1});
                ImGui::TextWrapped("%s", editor.error.c_str());
                ImGui::PopStyleColor();
            }
            if (!source.issue.empty()) {
                ImGui::TextColored({0.95F, 0.49F, 0.42F, 1}, "Duplicate images / export unavailable");
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", source.issue.c_str());
            }
        }
        ImGui::EndChild();
    }

    void export_controls(Workspace& workspace, const caption::Dataset& source) {
        auto& path = workspace.export_paths.try_emplace(source.key, files::utf8(project::directory.parent_path() / "exports" / "lora" / files::path(source.key))).first->second;
        ImGui::Spacing();
        ImGui::TextDisabled("Export concept");
        ImGui::TextWrapped("%s", source.key.c_str());
        ImGui::TextDisabled("New output directory");
        ImGui::SetNextItemWidth(-1);
        ImGui::InputText("##ExportPath", &path);
        ImGui::TextWrapped("Independent PNG copies, complete .txt captions and foreground -masklabel.png files for OneTrainer. Missing masks are computed automatically. Source images stay unchanged.");
        const auto total = std::ranges::find(source.folders, std::string_view{"."}, &caption::Folder::path)->total;
        ImGui::TextDisabled("%zu images to export / %zu bypassed", source.export_images, total - source.export_images);
        if (!source.issue.empty()) {
            ImGui::TextColored({0.95F, 0.49F, 0.42F, 1}, "Duplicate images / export unavailable");
            if (ImGui::TreeNodeEx("Conflict paths", ImGuiTreeNodeFlags_NoTreePushOnOpen)) ImGui::TextWrapped("%s", source.issue.c_str());
        }
        if (!source.missing_tags.empty()) {
            ImGui::TextColored({0.98F, 0.34F, 0.32F, 1}, "%zu directories need their own tags", source.missing_tags.size());
            ImGui::TextWrapped("Inherited tags do not count. Empty directories also need their own tags. Bypassed subtrees are excluded from this check.");
            for (const auto& folder : source.missing_tags) {
                const auto label = folder == "." ? std::string{"Concept root"} : folder;
                ImGui::PushID(folder.c_str());
                const auto position = ImGui::GetCursorScreenPos();
                const float width   = ImGui::GetContentRegionAvail().x;
                const bool clicked  = ImGui::Selectable("##MissingDirectory");
                ImGui::RenderTextEllipsis(ImGui::GetWindowDrawList(), position, {position.x + width, position.y + ImGui::GetTextLineHeight()}, position.x + width, label.data(), label.data() + label.size(), nullptr);
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", label.c_str());
                ImGui::PopID();
                if (clicked) {
                    const auto open = [&workspace, key = source.key, folder] {
                        workspace.select_collection(key + (folder == "." ? "" : "/" + folder));
                        if (workspace.collection_key == key && workspace.caption_folder == folder && !workspace.repaint) workspace.concept_tool = Workspace::ConceptTool::tags;
                    };
                    if (workspace.save_caption(open)) open();
                    break;
                }
            }
        }
        ImGui::BeginDisabled(workspace.session_state.active.has_value() || !source.issue.empty() || !source.missing_tags.empty() || !workspace.root->ready || source.export_images == 0 || path.empty());
        if (ImGui::Button("Export", {-FLT_MIN, 0})) {
            try {
                workspace.submit_task({runtime::Export{source.key, files::path(path)}});
            } catch (const std::exception& failure) {
                workspace.type_error = failure.what();
            }
        }
        ImGui::EndDisabled();
        operation_activity(workspace, {runtime::Kind::export_dataset}, source.key);
        const auto exported = workspace.activity.find({source.key, runtime::Kind::export_dataset});
        if (exported != workspace.activity.end())
            if (const auto* result = std::get_if<caption::Exported>(&exported->second.result.value)) {
                ImGui::TextWrapped("%zu images exported", result->images);
                ImGui::TextWrapped("%s", files::utf8(result->path).c_str());
                if (ImGui::SmallButton("Copy output path")) ImGui::SetClipboardText(files::utf8(result->path).c_str());
            }
    }

    void model_controls(Workspace& workspace) {
        const auto& key       = workspace.collection_key;
        auto& controls        = workspace.model_settings.at(key);
        auto& path            = workspace.lora_paths[key];
        const bool replacing  = workspace.session_state.active && workspace.session_state.active->concept_key == key && workspace.session_state.active->kind == runtime::Kind::lora;
        const auto registered = workspace.library.loras.find(key);
        const auto* model     = registered != workspace.library.loras.end() && registered->second ? &*registered->second : nullptr;
        if (model) {
            ImGui::TextWrapped("%s", model->name.c_str());
            ImGui::TextDisabled("SHA %.12s", model->sha.c_str());
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", model->sha.c_str());
            ImGui::BeginDisabled(replacing);
            // Keep the numeric value current so closing the window can save an active edit.
            ImGui::PushItemFlag(ImGuiItemFlags_LiveEditOnInputScalar, true);
            for (const bool start : {false, true}) {
                const char* id   = start ? "##LoraStart" : "##LoraWeight";
                auto& value      = start ? controls.lora->start : controls.lora->weight;
                const auto input = ImGui::GetID(id);
                ImGui::TextUnformatted(start ? "Start" : "Strength");
                ImGui::SetNextItemWidth(-FLT_MIN);
                const bool changed = ImGui::InputFloat(id, &value, start ? 5 : 0.05F, start ? 10 : 0.1F, start ? "%.1f%%" : "%.2f");
                if (changed && start) value = std::clamp(value, 0.0F, 100.0F);
                controls.dirty |= changed;
                const bool editing = ImGui::GetActiveID() == input;
                if (editing) workspace.editing_model = key;
                if ((changed && !editing) || ImGui::IsItemDeactivatedAfterEdit()) workspace.save_model_settings(key);
                if (ImGui::IsItemHovered()) {
                    if (start) ImGui::SetTooltip("Enable LoRA at this point in the full denoising schedule.\n0%%: entire image; 100%%: never.\nRepaint starts at its existing noise level. Changes apply to the next submission.");
                    else ImGui::SetTooltip("LoRA strength for the next submission. 0 disables it; 1 uses its original strength.");
                }
            }
            ImGui::PopItemFlag();
            ImGui::EndDisabled();
            ImGui::TextDisabled("%s", controls.active ? "Active / middle-click concept to deactivate" : "Middle-click concept to activate");
        }
        ImGui::Spacing();
        ImGui::TextDisabled("Drop a .safetensors file here or paste its path");
        ImGui::BeginDisabled(workspace.session_state.active.has_value());
        ImGui::SetNextItemWidth(-FLT_MIN);
        ImGui::InputTextWithHint("##LoraPath", "Model path", path.data(), path.size());
        ImGui::BeginDisabled(path[0] == '\0');
        const bool import_model = ImGui::Button(model ? "Replace model" : "Import model", {-FLT_MIN, 0});
        ImGui::EndDisabled();
        const bool remove = model && ImGui::Button("Remove model", {-FLT_MIN, 0});
        ImGui::EndDisabled();
        if (import_model || remove) {
            try {
                workspace.submit_task({runtime::LoraModel{key, remove ? std::filesystem::path{} : files::path(path.data()), remove}});
            } catch (const std::exception& failure) {
                workspace.action_error = failure.what();
                workspace.shown_error.clear();
            }
        }
        operation_activity(workspace, {runtime::Kind::lora}, key);
    }

    void audit_controls(Workspace& workspace, const training::TrainingData& source) {
        if (workspace.repaint) {
            ImGui::TextWrapped("Return from Repaint to edit audit labels.");
            return;
        }
        ImGui::TextWrapped("%s", workspace.audit_task ? "Updating audit..." : workspace.audit_dirty || workspace.audit_report.concept_key.empty() ? "Results incomplete" : "Current labels, lowest confidence first");
        ImGui::TextUnformatted("Category");
        ImGui::SetNextItemWidth(-FLT_MIN);
        if (ImGui::BeginCombo("##AuditCategory", workspace.audit_category.empty() ? "All categories" : workspace.audit_category.c_str())) {
            if (ImGui::Selectable("All categories", workspace.audit_category.empty())) {
                workspace.audit_category.clear();
                workspace.audit_position = {};
                workspace.rebuild_audit();
            }
            for (const auto& name : source.classes)
                if (ImGui::Selectable(name.c_str(), name == workspace.audit_category)) {
                    workspace.audit_category = name;
                    workspace.audit_position = {};
                    workspace.rebuild_audit();
                }
            ImGui::EndCombo();
        }
        const bool busy = workspace.session_state.active.has_value();
        ImGui::BeginDisabled(busy || !source.model);
        if (ImGui::Button("Refresh audit", {-FLT_MIN, 0})) workspace.open_audit(workspace.audit_key, true);
        ImGui::EndDisabled();
        if (!workspace.audit_report.concept_key.empty() && !workspace.audit_collection.images.empty()) {
            const auto& file = workspace.audit_collection.images[workspace.audit_position.index];
            const auto& rows = workspace.audit_report.rows;
            const auto row   = std::ranges::find_if(rows, [&](const auto& value) { return value.sample.file.sha == file.sha; });
            if (row != rows.end()) {
                const auto label     = row->label;
                const auto predicted = row->prediction.label;
                const auto fix_to    = [&](const std::string& category) {
                    if (workspace.audit_position.index + 1 < workspace.audit_collection.images.size()) workspace.audit_position.selected = workspace.audit_collection.images[workspace.audit_position.index + 1].sha;
                    workspace.submit_task({runtime::Fix{workspace.audit_key, file.sha, category}});
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
        operation_activity(workspace, {runtime::Kind::audit, runtime::Kind::fix}, workspace.audit_key);
    }
} // namespace genesia::editor
