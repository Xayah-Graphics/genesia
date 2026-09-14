module;
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_internal.h>
module genesia.editor.panels.datasets;
import genesia.editor.widgets.controls;
import genesia.editor.widgets.tags;
import genesia.io.files;
import genesia.runtime.session;
import genesia.runtime.catalog;
import genesia.project;
import genesia.prompt.preset;
import genesia.editor.graphics.bridge;
import std;
namespace genesia::editor {
    void dataset_controls(Workspace& workspace, const float scale) {
        if (workspace.collection && workspace.root) {
            const auto title = ImGui::GetCursorScreenPos();
            const ImVec2 title_size{ImGui::GetContentRegionAvail().x, ImGui::GetTextLineHeight()};
            ImGui::Dummy(title_size);
            ImGui::RenderTextEllipsis(ImGui::GetWindowDrawList(), title, {title.x + title_size.x, title.y + title_size.y}, title.x + title_size.x, workspace.collection->name.c_str(), nullptr, nullptr);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", workspace.collection_key.c_str());
            if (workspace.collection_key == workspace.root->all.key) ImGui::TextDisabled("%zu images", workspace.collection->images.size());
            else {
                ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
                ImGui::TextWrapped("%s / %zu images", workspace.root->all.name.c_str(), workspace.collection->images.size());
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
                if (ImGui::SmallButton(std::format("{}{}###ConceptType", names[static_cast<std::size_t>(assigned.type)], assigned.locked ? "" : "  v").c_str())) workspace.choosing_type = !workspace.choosing_type;
                ImGui::PopStyleVar();
                ImGui::PopStyleColor(2);
                ImGui::EndDisabled();
                if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) ImGui::SetTooltip("%s", assigned.locked ? "Type is permanently locked by training history. Restart keeps this type." : busy ? "Finish the current operation before changing its type." : "Choose this concept's purpose. Assigning a type does not start training.");
                if (assigned.locked || busy) workspace.choosing_type = false;
                if (workspace.choosing_type) {
                    for (std::size_t i = 0; i < names.size(); ++i)
                        if (ImGui::Selectable(names[i], static_cast<std::size_t>(assigned.type) == i)) {
                            if (assigned.type == static_cast<dataset::ConceptType>(i)) {
                                workspace.choosing_type = false;
                                workspace.type_error.clear();
                                break;
                            }
                            try {
                                workspace.submit_task({runtime::Assign{workspace.collection_key, static_cast<dataset::ConceptType>(i)}});
                                workspace.training_drafts.erase(workspace.collection_key);
                                workspace.concept_tool  = Workspace::ConceptTool::none;
                                workspace.choosing_type = false;
                                workspace.type_error.clear();
                            } catch (const std::exception& error) {
                                workspace.type_error = error.what();
                            }
                            break;
                        }
                }
                if (!workspace.type_error.empty()) ImGui::TextWrapped("%s", workspace.type_error.c_str());
                if (failure != workspace.library.concept_errors.end()) ImGui::TextWrapped("%s", failure->second.c_str());
                else if (assigned.type == dataset::ConceptType::lora) {
                    ImGui::TextWrapped("LoRA training is not available yet.");
                    ImGui::BeginDisabled();
                    ImGui::Button("Train LoRA", {-FLT_MIN, 0});
                    ImGui::EndDisabled();
                } else if (assigned.type == dataset::ConceptType::classifier) {
                    const auto info = workspace.library.classifiers.find(workspace.collection_key);
                    if (info == workspace.library.classifiers.end()) ImGui::TextDisabled("Reading classifier data...");
                    else {
                        const auto& source   = info->second;
                        const bool published = source.model.has_value();
                        const bool active    = std::ranges::contains(workspace.activated, workspace.collection_key);
                        if (!workspace.choosing_type && workspace.type_error.empty()) ImGui::SameLine(0, 8 * scale);
                        if (published) ImGui::TextColored(color, "%s", active ? "Active" : "Model available");
                        else ImGui::TextDisabled("No published model");
                        if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", published ? "Middle-click this concept in the dataset list to activate or deactivate its model." : "A model becomes available after training reaches its target and finishes saving.");
                        const bool audit_available    = published;
                        const bool classify_available = published;
                        const int buttons             = 1 + int(audit_available) + int(classify_available);
                        const float gap               = 4 * scale;
                        const float width             = (ImGui::GetContentRegionAvail().x - (buttons - 1) * gap) / buttons;
                        ImGui::Spacing();
                        for (const auto [tool, label] : {std::pair{Workspace::ConceptTool::train, "Train"}, std::pair{Workspace::ConceptTool::audit, "Audit"}, std::pair{Workspace::ConceptTool::classify, "Classify"}}) {
                            if ((tool == Workspace::ConceptTool::audit && !audit_available) || (tool == Workspace::ConceptTool::classify && !classify_available)) continue;
                            if (tool != Workspace::ConceptTool::train) ImGui::SameLine(0, gap);
                            const bool current = workspace.concept_tool == tool;
                            ImGui::PushStyleColor(ImGuiCol_Button, {color.x, color.y, color.z, current ? 0.08F : 0});
                            ImGui::PushStyleColor(ImGuiCol_Text, current ? ImGui::GetStyleColorVec4(ImGuiCol_Text) : ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
                            if (ImGui::Button(label, {width, 30 * scale})) {
                                if (tool == Workspace::ConceptTool::audit && (workspace.page != Workspace::Page::audit || workspace.repaint)) {
                                    workspace.open_audit(workspace.collection_key);
                                    if (workspace.page == Workspace::Page::audit && !workspace.repaint) workspace.concept_tool = Workspace::ConceptTool::audit;
                                } else workspace.concept_tool = current ? Workspace::ConceptTool::none : tool;
                            }
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
                            ImGui::SetNextWindowSizeConstraints({0, 0}, {FLT_MAX, std::max(1.0F, ImGui::GetContentRegionAvail().y * 0.45F)});
                            if (ImGui::BeginChild("##ConceptTool", {0, 0}, ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_AlwaysUseWindowPadding, ImGuiWindowFlags_NoBackground)) {
                                if (!source.inspected && workspace.concept_tool != Workspace::ConceptTool::classify) ImGui::TextDisabled("Reading concept samples...");
                                else if (workspace.concept_tool == Workspace::ConceptTool::train) training_controls(workspace, source, scale);
                                else if (workspace.concept_tool == Workspace::ConceptTool::audit) audit_controls(workspace, source);
                                else classify_controls(workspace, source);
                            }
                            ImGui::EndChild();
                            ImGui::PopStyleVar();
                            ImGui::PopID();
                        }
                    }
                }
            } else if (failure != workspace.library.concept_errors.end()) ImGui::TextWrapped("%s", failure->second.c_str());
            else if (workspace.collection_key == workspace.root->all.key) ImGui::TextDisabled("%zu concepts", workspace.root->concepts.size());
            if (!workspace.root->ready) ImGui::TextColored({0.95F, 0.49F, 0.42F, 1}, "Index error / details below");
        } else ImGui::TextDisabled(workspace.library.ready ? "Choose a dataset" : "Indexing...");
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
                    const auto metadata  = workspace.library.concepts.find(item.key);
                    const auto info      = workspace.library.classifiers.find(item.key);
                    const auto failure   = workspace.library.concept_errors.find(item.key);
                    const bool published = info != workspace.library.classifiers.end() && info->second.model.has_value();
                    const bool enabled   = std::ranges::contains(workspace.activated, item.key);
                    const auto origin    = ImGui::GetCursorScreenPos();
                    const float width    = ImGui::GetContentRegionAvail().x;
                    const float height   = ImGui::GetTextLineHeight() + 8 * scale;
                    if (ImGui::Selectable("##Concept", workspace.collection_key == item.key, ImGuiSelectableFlags_None, {width, height})) selected = item.key;
                    const bool hovered = ImGui::IsItemHovered();
                    if (published && ImGui::IsItemClicked(ImGuiMouseButton_Middle)) {
                        if (enabled) std::erase(workspace.activated, item.key);
                        else workspace.activated.push_back(item.key);
                    }
                    auto* draw = ImGui::GetWindowDrawList();
                    const ImVec2 minimum{origin.x, origin.y};
                    const ImVec2 maximum{std::min(origin.x + width, draw->GetClipRectMax().x), origin.y + height};
                    const float y    = origin.y + 4 * scale;
                    const auto count = std::to_string(item.images.size());
                    float right      = maximum.x - 6 * scale - ImGui::CalcTextSize(count.c_str()).x;
                    draw->AddText({right, y}, ImGui::GetColorU32(ImGuiCol_TextDisabled), count.c_str());
                    right -= 8 * scale;
                    if (failure != workspace.library.concept_errors.end()) {
                        right -= ImGui::CalcTextSize("!").x;
                        draw->AddText({right, y}, ImGui::GetColorU32(ImVec4{0.95F, 0.49F, 0.42F, 1}), "!");
                        right -= 8 * scale;
                    } else if (metadata != workspace.library.concepts.end() && metadata->second.type != dataset::ConceptType::none) {
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
        ImGui::TextDisabled("%s", runtime::states[std::size_t(task->state)].data());
        const auto* batch = std::get_if<runtime::BatchProgress>(&task->progress.value);
        if (task->state < runtime::State::complete) {
            if (batch) {
                ImGui::ProgressBar(float(batch->completed) / batch->total, {-1, 3 * workspace.renderer.dpi}, "");
                ImGui::Text("%zu / %zu", batch->completed, batch->total);
            }
            const bool moving = task->kind == runtime::Kind::fix || task->kind == runtime::Kind::undo || task->kind == runtime::Kind::assign || (batch && batch->stage == runtime::Stage::moving);
            if (!moving && ImGui::Button("Stop", {-FLT_MIN, 0})) workspace.runtime.session.cancel(task->id);
        }
        if (!task->error.empty()) ImGui::TextWrapped("%s", task->error.c_str());
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
            ImGui::TextWrapped("Training checks metadata, original sizes, labels and independent train / validation groups before loading the model.");
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
        ImGui::BeginDisabled(busy);
        ImGui::BeginDisabled(!source.model);
        if (ImGui::Button("Refresh audit", {-FLT_MIN, 0})) workspace.open_audit(workspace.audit_key, true);
        ImGui::EndDisabled();
        if (ImGui::Button("Undo move", {-FLT_MIN, 0})) {
            workspace.submit_task({runtime::Undo{workspace.audit_key}});
        }
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
        operation_activity(workspace, {runtime::Kind::audit, runtime::Kind::fix, runtime::Kind::undo}, workspace.audit_key);
    }
} // namespace genesia::editor
