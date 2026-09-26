module;
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_internal.h>
module classifier.editor.workspace;
import tools.files;
import tools.editor.platform.files;
import tools.editor.style;
import std;
namespace classifier::editor {
    Workspace::Workspace(tools::editor::WindowPlatform& platform, const float& scale) : window{platform}, dpi{scale}, session{[] { glfwPostEmptyEvent(); }} {}
    void Workspace::submit(runtime::Request request) {
        try {
            const auto kind = runtime::Kind(request.operation.index());
            session.submit(std::move(request));
            error.clear();
            state.busy     = true;
            state.stopping = false;
            state.state    = runtime::State::running;
            state.kind     = kind;
            state.progress = {};
            window.redraw  = true;
            if (kind == runtime::Kind::load) {
                mode               = Mode::inference;
                section            = Section::train;
                reset_popups       = true;
                training_requested = false;
                audit_requested    = false;
            } else if (kind == runtime::Kind::infer || kind == runtime::Kind::classify) {
                mode         = Mode::inference;
                reset_popups = true;
                inference.reset();
            } else if (kind == runtime::Kind::training_details) training_requested = true;
            else if (kind == runtime::Kind::audit) audit_requested = true;
            else if (kind == runtime::Kind::train) audit_requested = false;
        } catch (const std::exception& failure) {
            error = failure.what();
        }
    }
    void Workspace::receive() {
        auto delivery       = session.drain();
        const auto previous = state.revision;
        state               = std::move(delivery.state);
        if (state.revision != previous && !state.error.empty()) error = state.error;
        if (state.classifier != classifier) {
            const bool changed = !classifier || !state.classifier || classifier->root != state.classifier->root;
            classifier         = state.classifier;
            if (changed || state.kind == runtime::Kind::load) {
                reset_popups = true;
                report.reset();
                inference.reset();
                category.clear();
                audit_rows.clear();
            }
        }
        if (state.training != training) {
            training = state.training;
            losses.clear();
            metrics.reset();
            last_step = 0;
            if (training) {
                config              = training->state ? training->state->config : training::Config{};
                const int completed = training->state ? training->state->step : 0;
                const int target    = training->state ? training->state->target : 400;
                steps               = target > completed ? target : completed + 400;
                for (const auto& step : training->history.steps) {
                    losses.push_back(step.loss);
                    last_step = step.step;
                }
                if (!training->history.evaluations.empty()) metrics = training->history.evaluations.back();
            }
        }
        for (const auto& progress : delivery.progress) {
            if (const auto* step = std::get_if<training::Step>(&progress.value); step && step->step > last_step) {
                losses.push_back(step->loss);
                last_step = step->step;
            } else if (const auto* evaluation = std::get_if<training::Metrics>(&progress.value)) metrics = *evaluation;
        }
        if (state.audit != report) {
            report = state.audit;
            filter_audit();
        }
        if (state.result && (std::holds_alternative<runtime::Inferred>(state.result->value) || std::holds_alternative<classification::Classification>(state.result->value))) inference = state.result;
    }
    void Workspace::drop() {
        if (!window.drop_error.empty()) error = std::exchange(window.drop_error, {});
        auto paths = std::exchange(window.dropped, {});
        if (paths.empty()) return;
        try {
            if (state.busy) throw std::runtime_error{"Stop the current task before dropping another input"};
            if (paths.size() != 1) throw std::runtime_error{"Drop one classifier directory, image or image folder"};
            const auto path = std::filesystem::canonical(paths.front());
            if (std::filesystem::is_directory(path)) {
                if (std::filesystem::exists(path / ".genesia")) submit({runtime::Load{path}});
                else {
                    bool images{}, categories{};
                    for (const auto& entry : std::filesystem::directory_iterator(path)) {
                        if (tools::files::utf8(entry.path().filename()).starts_with('.')) continue;
                        categories |= entry.is_directory();
                        auto extension = tools::files::utf8(entry.path().extension());
                        std::ranges::transform(extension, extension.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
                        images |= entry.is_regular_file() && extension == ".png";
                    }
                    if (images) {
                        if (!classifier || !classifier->model) throw std::runtime_error{"Drag a trained classifier before classifying images"};
                        submit({runtime::Classify{classifier->root, path}});
                    } else if (categories) {
                        submit({runtime::Load{path}});
                        mode = Mode::training;
                    } else throw std::runtime_error{"The folder has no categories or PNG images"};
                }
            } else {
                if (!classifier || !classifier->model) throw std::runtime_error{"Drag a trained classifier before an image"};
                submit({runtime::Infer{classifier->root, path}});
            }
        } catch (const std::exception& failure) {
            error = failure.what();
        }
    }
    void Workspace::draw() {
        const auto size   = ImGui::GetIO().DisplaySize;
        const float scale = dpi;
        const float width = content_width();
        ImGui::SetNextWindowPos({0, 0});
        // Lay out against the screen limit, not the previous native height, to avoid resize feedback.
        ImGui::SetNextWindowSize({width, float(window.extent_limit.height)});
        ImGui::Begin("Classifier", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoBringToFrontOnFocus);
        ImGui::PushItemFlag(ImGuiItemFlags_LiveEditOnInputScalar, true);
        if (std::exchange(reset_popups, false)) {
            if (ImGui::IsPopupOpen(nullptr, ImGuiPopupFlags_AnyPopupId | ImGuiPopupFlags_AnyPopupLevel)) ImGui::ClosePopupToLevel(0, true);
            ImGui::ClearActiveID();
            ImGui::SetScrollY(0);
        }
        if (!classifier && !state.busy) tools::editor::draw_welcome("Train and Classify", "Drag in a classifier folder");
        else {
            if (classifier) {
                const auto label = mode == Mode::inference ? "/ Inference" : "/ Training";
                draw_path(classifier->root, ImGui::CalcTextSize(label).x + ImGui::GetStyle().ItemSpacing.x);
                ImGui::SameLine();
                ImGui::TextDisabled("%s", label);
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("Tab to switch to %s", mode == Mode::inference ? "Training" : "Inference");
            } else ImGui::TextUnformatted("Loading classifier");
            if (classifier && classifier->model) {
                ImGui::TextDisabled("%zu classes", classifier->classes.size());
                if (ImGui::IsItemHovered()) {
                    ImGui::BeginTooltip();
                    if (ImGui::BeginTable("Model classes", 1, ImGuiTableFlags_SizingFixedFit)) {
                        for (const auto& name : classifier->classes) {
                            ImGui::TableNextRow();
                            ImGui::TableNextColumn();
                            ImGui::TextUnformatted(name.c_str());
                        }
                        ImGui::EndTable();
                    }
                    ImGui::EndTooltip();
                }
                ImGui::SameLine();
                ImGui::TextDisabled("/ step %d", classifier->step);
            }
            ImGui::Spacing();
            if (!classifier || (state.kind == runtime::Kind::load && state.busy)) draw_progress();
            else if (mode == Mode::training) draw_training();
            else draw_inference();
        }
        if (!error.empty()) {
            ImGui::Spacing();
            ImGui::PushStyleColor(ImGuiCol_Text, {0.95F, 0.59F, 0.55F, 1});
            ImGui::TextWrapped("%s", error.c_str());
            ImGui::PopStyleColor();
            if (ImGui::SmallButton("Dismiss")) error.clear();
        }
        const auto* layout           = ImGui::GetCurrentWindow();
        const float height           = layout->DC.CursorMaxPos.y - layout->DC.CursorStartPos.y + 2 * ImGui::GetStyle().WindowPadding.y;
        window.content_extent.width  = static_cast<std::uint32_t>(std::ceil(width));
        window.content_extent.height = static_cast<std::uint32_t>(std::ceil(std::min(height, float(window.extent_limit.height))));
        // Draw inputs before switching so their latest edits remain in the settings draft.
        if (classifier && ImGui::IsKeyChordPressed(ImGuiKey_Tab)) {
            mode = mode == Mode::inference ? Mode::training : Mode::inference;
            if (ImGui::IsPopupOpen(nullptr, ImGuiPopupFlags_AnyPopupId | ImGuiPopupFlags_AnyPopupLevel)) ImGui::ClosePopupToLevel(0, true);
            ImGui::ClearActiveID();
            ImGui::SetScrollY(0);
            window.redraw = true;
        }
        if (!window.dragged.empty()) {
            auto* draw = ImGui::GetForegroundDrawList();
            draw->AddRect({5 * scale, 5 * scale}, {size.x - 5 * scale, size.y - 5 * scale}, ImGui::GetColorU32(ImGuiCol_PlotLines), 10 * scale, 0, 2 * scale);
        }
        const bool drag       = !ImGui::IsAnyItemHovered() && !ImGui::IsAnyItemActive() && !ImGui::IsPopupOpen(nullptr, ImGuiPopupFlags_AnyPopupId);
        window.drag_requested = drag && ImGui::IsMouseClicked(ImGuiMouseButton_Left);
        ImGui::PopItemFlag();
        ImGui::End();
    }
    void Workspace::filter_audit() {
        audit_rows.clear();
        if (!report) return;
        for (std::size_t i = 0; i < report->rows.size(); ++i)
            if (report->rows[i].confidence < threshold / 100 && (category.empty() || report->rows[i].label == category)) audit_rows.push_back(i);
    }
    float Workspace::content_width() const {
        const auto& style = ImGui::GetStyle();
        float width       = (!classifier && !state.busy ? 320 : 240) * dpi;
        if (classifier) {
            const auto heading = tools::files::utf8(classifier->root.filename()) + (mode == Mode::inference ? " / Inference" : " / Training");
            width              = std::max(width, ImGui::CalcTextSize(heading.c_str()).x + style.ItemSpacing.x);
            if (classifier->model) width = std::max(width, ImGui::CalcTextSize(std::format("{} classes / step {}", classifier->classes.size(), classifier->step).c_str()).x + style.ItemSpacing.x);
            if (mode == Mode::training) {
                width = std::max(width, ImGui::CalcTextSize("Images are scanned when training starts.").x);
                if (section == Section::audit) {
                    const float controls = 196 * dpi + ImGui::CalcTextSize("BelowRefresh").x + 2 * style.FramePadding.x + 3 * style.ItemSpacing.x;
                    width                = std::max(width, controls);
                    if (report && !audit_rows.empty()) {
                        float label{};
                        for (const auto& name : report->classes) label = std::max(label, ImGui::CalcTextSize(name.c_str()).x);
                        const float actions = ImGui::CalcTextSize("OpenApplyDelete").x + 6 * style.FramePadding.x + 3 * style.ItemSpacing.x;
                        width               = std::max(width, actions + std::max(180 * dpi, 2 * label + ImGui::CalcTextSize(" 100.0% ->  100.0%").x) + style.ScrollbarSize);
                    }
                } else {
                    width = std::max(width, 120 * dpi + style.ItemInnerSpacing.x + ImGui::CalcTextSize("Evaluation interval").x);
                    if (state.dataset) width = std::max(width, ImGui::CalcTextSize(std::format("Last scan: {} classes / {} images", state.dataset->classes.size(), state.dataset->samples.size()).c_str()).x + style.ItemSpacing.x);
                    if (metrics && metrics_expanded) {
                        width = std::max(width, ImGui::CalcTextSize("Precision 100.0%  /  Recall 100.0%  /  F1 100.0%").x);
                        for (std::size_t i = 0; i < metrics->classes.size(); ++i) {
                            const auto& metric = metrics->classes[i];
                            width              = std::max(width, ImGui::CalcTextSize(std::format("{}: {} 100.0 100.0 100.0 {}", i + 1, metric.name, metric.samples).c_str()).x + 10 * style.CellPadding.x + style.ScrollbarSize);
                        }
                    }
                }
            } else if (inference) {
                if (const auto* image = std::get_if<runtime::Inferred>(&inference->value)) {
                    for (const auto& name : image->prediction.classes) width = std::max(width, ImGui::CalcTextSize((name + "  /  100.0%").c_str()).x);
                } else if (const auto* classified = std::get_if<classification::Classification>(&inference->value)) {
                    for (const auto& [label, count] : classified->classes) width = std::max(width, ImGui::CalcTextSize(std::format("{}  /  {}", label, count).c_str()).x);
                }
            }
        }
        return std::ceil(std::min({width + 2 * style.WindowPadding.x, 720 * dpi, float(window.extent_limit.width)}));
    }
    void Workspace::draw_path(const std::filesystem::path& path, const float reserved) {
        auto name         = tools::files::utf8(path.filename());
        const float width = ImGui::GetContentRegionAvail().x - reserved;
        if (ImGui::CalcTextSize(name.c_str()).x > width) {
            const char* end{};
            ImGui::GetFont()->CalcTextSizeA(ImGui::GetFontSize(), std::max(0.0F, width - ImGui::CalcTextSize("...").x), 0, name.data(), name.data() + name.size(), &end);
            name.resize(end - name.data());
            name += "...";
        }
        ImGui::TextUnformatted(name.c_str());
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tools::files::utf8(path).c_str());
    }
    void Workspace::draw_progress() {
        const float scale = dpi;
        if (state.kind == runtime::Kind::train && !losses.empty()) {
            ImGui::TextDisabled("Training loss");
            ImGui::PushFont(nullptr, 26);
            ImGui::Text("%.4f", losses.back());
            ImGui::PopFont();
            ImGui::PushStyleColor(ImGuiCol_FrameBg, {0, 0, 0, 0});
            ImGui::PlotLines("##Loss", losses.data(), int(losses.size()), 0, nullptr, std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), {ImGui::GetContentRegionAvail().x, 105 * scale});
            ImGui::PopStyleColor();
        }
        std::visit(
            [&]<typename T>(const T& value) {
                if constexpr (std::same_as<T, training::Step>) {
                    ImGui::ProgressBar(float(value.step) / value.target, {-1, 3 * scale}, "");
                    ImGui::TextDisabled("%d / %d steps  /  %.1fs", value.step, value.target, value.seconds);
                } else if constexpr (std::same_as<T, training::Metrics>) ImGui::TextDisabled("Evaluating / step %d", value.step);
                else if constexpr (std::same_as<T, runtime::BatchProgress>) {
                    ImGui::TextDisabled("%s", runtime::stages[std::size_t(value.stage)].data());
                    if (value.total) ImGui::ProgressBar(float(value.completed) / value.total, {-1, 3 * scale}, "");
                    if (value.total) ImGui::TextDisabled("%zu / %zu", value.completed, value.total);
                    else if (value.stage == runtime::Stage::scanning) ImGui::TextDisabled("%zu images", value.completed);
                    if (!value.file.empty()) draw_path(tools::files::path(value.file));
                } else ImGui::TextDisabled("Preparing...");
            },
            state.progress.value);
        ImGui::Spacing();
        ImGui::BeginDisabled(state.stopping);
        if (ImGui::Button(state.stopping ? "Stopping..." : state.kind == runtime::Kind::train ? "Stop & save" : "Stop")) session.cancel();
        ImGui::EndDisabled();
    }
    void Workspace::draw_training() {
        const std::array sections{"Train##Section", "Audit##Section"};
        const float width = (ImGui::GetContentRegionAvail().x - 4 * dpi) / 2;
        for (std::size_t i = 0; i < sections.size(); ++i) {
            if (i) ImGui::SameLine(0, 4 * dpi);
            const bool selected = section == Section(i);
            ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyleColorVec4(selected ? ImGuiCol_ButtonActive : ImGuiCol_FrameBg));
            ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(selected ? ImGuiCol_Text : ImGuiCol_TextDisabled));
            if (ImGui::Button(sections[i], {width, 0})) {
                section = Section(i);
                ImGui::SetScrollY(0);
            }
            ImGui::PopStyleColor(2);
        }
        ImGui::Spacing();
        if (state.busy) {
            const bool inferring = state.kind == runtime::Kind::infer || state.kind == runtime::Kind::classify;
            const bool auditing  = state.kind == runtime::Kind::audit || state.kind == runtime::Kind::fix || state.kind == runtime::Kind::recycle;
            if (inferring || (section == Section::audit) != auditing) ImGui::TextDisabled("%s in progress", inferring ? "Inference" : auditing ? "Audit" : "Training");
        }
        if (section == Section::audit) {
            if (state.busy && state.kind == runtime::Kind::audit) draw_progress();
            else if (!classifier->model) ImGui::TextDisabled("Train a model to audit this dataset.");
            else if (report) draw_audit();
            else if (state.busy) ImGui::TextDisabled("Waiting for the current task before Audit.");
            else if (!audit_requested) submit({runtime::Audit{classifier->root}});
            else {
                ImGui::TextDisabled("No Audit report.");
                if (ImGui::Button("Refresh")) submit({runtime::Audit{classifier->root, true}});
            }
            return;
        }
        if (state.busy && (state.kind == runtime::Kind::train || state.kind == runtime::Kind::training_details)) {
            draw_progress();
            return;
        }
        if (!training) {
            if (state.busy) ImGui::TextDisabled("Waiting to load training details.");
            else if (!training_requested) submit({runtime::TrainingDetails{classifier->root}});
            else if (ImGui::Button("Load training details")) submit({runtime::TrainingDetails{classifier->root}});
            return;
        }
        if (!losses.empty()) {
            ImGui::TextDisabled("Training loss");
            ImGui::SameLine(ImGui::GetWindowContentRegionMax().x * .55F);
            ImGui::TextDisabled("Validation F1");
            ImGui::PushFont(nullptr, 25);
            ImGui::Text("%.4f", losses.back());
            ImGui::SameLine(ImGui::GetWindowContentRegionMax().x * .55F);
            if (metrics) ImGui::Text("%.1f%%", metrics->f1 * 100);
            else ImGui::TextDisabled("--");
            ImGui::PopFont();
            ImGui::Spacing();
        }
        if (training->state) ImGui::TextDisabled("%s / step %d", training::phases[std::size_t(training->state->phase)].data(), training->state->step);
        if (const auto& dataset = state.dataset) {
            ImGui::TextDisabled("Last scan:");
            ImGui::SameLine(0, ImGui::CalcTextSize(" ").x);
            ImGui::TextDisabled("%zu classes", dataset->classes.size());
            if (ImGui::IsItemHovered()) {
                ImGui::BeginTooltip();
                if (ImGui::BeginTable("Dataset classes", 2, ImGuiTableFlags_SizingFixedFit)) {
                    for (std::size_t i = 0; i < dataset->classes.size(); ++i) {
                        ImGui::TableNextRow();
                        ImGui::TableNextColumn();
                        ImGui::TextUnformatted(dataset->classes[i].c_str());
                        ImGui::TableNextColumn();
                        ImGui::Text("%zu", dataset->counts[i]);
                    }
                    ImGui::EndTable();
                }
                ImGui::EndTooltip();
            }
            ImGui::SameLine();
            ImGui::TextDisabled("/ %zu images", dataset->samples.size());
            if (!dataset->issue.empty()) ImGui::TextWrapped("%s", dataset->issue.c_str());
            if (!dataset->training_issue.empty()) ImGui::TextWrapped("%s", dataset->training_issue.c_str());
            if (training->state && training->state->fingerprint != dataset->fingerprint) ImGui::TextWrapped("Dataset changed since training. Restart is required to use these changes.");
        }
        ImGui::TextDisabled("Images are scanned when training starts.");
        ImGui::Spacing();
        draw_settings();
        if (metrics && (metrics_expanded = ImGui::CollapsingHeader("Validation metrics"))) draw_metrics();
    }
    void Workspace::draw_audit() {
        ImGui::BeginDisabled(state.busy);
        ImGui::SetNextItemWidth(126 * dpi);
        if (ImGui::BeginCombo("##Category", category.empty() ? "All classes" : category.c_str())) {
            if (ImGui::Selectable("All classes", category.empty())) {
                category.clear();
                filter_audit();
            }
            for (const auto& name : report->classes)
                if (ImGui::Selectable(name.c_str(), category == name)) {
                    category = name;
                    filter_audit();
                }
            ImGui::EndCombo();
        }
        ImGui::SameLine();
        ImGui::AlignTextToFramePadding();
        ImGui::TextDisabled("Below");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(70 * dpi);
        if (ImGui::DragFloat("##Confidence", &threshold, .5F, 0, 100, "%.0f%%", ImGuiSliderFlags_AlwaysClamp)) filter_audit();
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Score for the current category. Drag or Ctrl+click to edit.");
        ImGui::SameLine();
        if (ImGui::Button("Refresh")) submit({runtime::Audit{classifier->root, true}});
        ImGui::EndDisabled();
        if (audit_rows.empty()) {
            ImGui::Spacing();
            ImGui::TextDisabled("No samples below this threshold.");
            return;
        }
        const auto& style      = ImGui::GetStyle();
        const float actions    = ImGui::CalcTextSize("OpenApplyDelete").x + 6 * style.FramePadding.x + 2 * style.ItemSpacing.x;
        const float row_height = 2 * ImGui::GetTextLineHeight() + 16 * dpi;
        const float available  = float(window.extent_limit.height) - ImGui::GetCursorPosY() - style.WindowPadding.y - ImGui::GetTextLineHeightWithSpacing() - style.ItemSpacing.y;
        const float height     = std::min(std::min(float(audit_rows.size()), 8.0F) * row_height, std::max(row_height, available));
        ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, {0, 6 * dpi});
        if (ImGui::BeginTable("Audit", 2, ImGuiTableFlags_ScrollY | ImGuiTableFlags_SizingStretchProp, {0, height})) {
            ImGui::TableSetupColumn("Sample", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn("Actions", ImGuiTableColumnFlags_WidthFixed, actions);
            ImGuiListClipper clipper;
            clipper.Begin(int(audit_rows.size()), row_height);
            while (clipper.Step()) {
                for (int i = clipper.DisplayStart; i < clipper.DisplayEnd; ++i) {
                    const auto& row      = report->rows[audit_rows[i]];
                    const auto copies    = row.sample.paths.size();
                    const auto predicted = std::ranges::find(row.prediction.classes, row.prediction.label) - row.prediction.classes.begin();
                    ImGui::PushID(row.sample.file.sha.c_str());
                    ImGui::TableNextRow(ImGuiTableRowFlags_None, row_height);
                    ImGui::TableNextColumn();
                    auto name         = tools::files::utf8(row.sample.file.path.filename());
                    const auto suffix = copies > 1 ? std::format("  x{}", copies) : std::string{};
                    if (ImGui::CalcTextSize((name + suffix).c_str()).x > ImGui::GetContentRegionAvail().x) {
                        const char* end{};
                        const float width = std::max(0.0F, ImGui::GetContentRegionAvail().x - ImGui::CalcTextSize(("..." + suffix).c_str()).x);
                        ImGui::GetFont()->CalcTextSizeA(ImGui::GetFontSize(), width, 0, name.data(), name.data() + name.size(), &end);
                        name.resize(end - name.data());
                        name += "...";
                    }
                    name += suffix;
                    ImGui::TextUnformatted(name.c_str());
                    if (ImGui::IsItemHovered()) {
                        ImGui::BeginTooltip();
                        for (const auto& path : row.sample.paths) ImGui::TextUnformatted(tools::files::utf8(path).c_str());
                        ImGui::EndTooltip();
                    }
                    ImGui::SetCursorPosY(ImGui::GetCursorPosY() - style.ItemSpacing.y + 4 * dpi);
                    const auto scores = std::format("{} {:.1f}% -> {} {:.1f}%", row.label, row.confidence * 100, row.prediction.label, row.prediction.scores[predicted] * 100);
                    ImGui::TextDisabled("%s", scores.c_str());
                    if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", scores.c_str());
                    ImGui::TableNextColumn();
                    ImGui::BeginDisabled(state.busy);
                    if (ImGui::SmallButton("Open")) {
                        try {
                            tools::editor::shell::reveal(row.sample.file.path);
                        } catch (const std::exception& failure) {
                            error = failure.what();
                        }
                    }
                    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Show the representative image in File Explorer.");
                    ImGui::SameLine();
                    ImGui::BeginDisabled(row.label == row.prediction.label);
                    if (ImGui::SmallButton("Apply")) submit({runtime::Fix{classifier->root, row.sample.file.sha, row.prediction.label}});
                    ImGui::EndDisabled();
                    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) ImGui::SetTooltip("Move all %zu file(s) to %s.", copies, row.prediction.label.c_str());
                    ImGui::SameLine();
                    ImGui::PushStyleColor(ImGuiCol_Text, {0.88F, 0.57F, 0.55F, 1});
                    if (ImGui::SmallButton("Delete")) submit({runtime::Recycle{classifier->root, row.sample.file.sha, tools::editor::shell::recycle}});
                    ImGui::PopStyleColor();
                    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Move all %zu file(s) to the Recycle Bin.", copies);
                    ImGui::EndDisabled();
                    ImGui::PopID();
                }
            }
            ImGui::EndTable();
        }
        ImGui::PopStyleVar();
        if (state.busy && (state.kind == runtime::Kind::fix || state.kind == runtime::Kind::recycle)) ImGui::TextDisabled("%s", state.kind == runtime::Kind::recycle ? "Moving to Recycle Bin..." : "Applying prediction...");
        else ImGui::TextDisabled("%zu low-confidence samples", audit_rows.size());
    }
    void Workspace::draw_inference() {
        const float scale = dpi;
        if (classifier->model) {
            ImGui::BeginChild("Drop", {0, 64 * scale}, ImGuiChildFlags_None, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
            const auto text       = "Drop images or a folder";
            const auto dimensions = ImGui::CalcTextSize(text);
            ImGui::SetCursorPos({std::max(0.0F, (ImGui::GetWindowWidth() - dimensions.x) / 2), (64 * scale - dimensions.y) / 2});
            ImGui::TextDisabled("%s", text);
            ImGui::EndChild();
        } else ImGui::TextDisabled("No published model. Press Tab to train.");
        if (state.busy) {
            if (state.kind == runtime::Kind::infer || state.kind == runtime::Kind::classify) {
                draw_progress();
                return;
            }
            ImGui::TextDisabled("%s in progress", state.kind == runtime::Kind::train || state.kind == runtime::Kind::training_details ? "Training" : "Audit");
        }
        if (!inference) return;
        if (const auto* image = std::get_if<runtime::Inferred>(&inference->value)) {
            const auto& result = image->prediction;
            draw_path(image->path);
            ImGui::Spacing();
            std::vector<std::size_t> order(result.classes.size());
            std::iota(order.begin(), order.end(), 0);
            std::ranges::sort(order, [&](std::size_t a, std::size_t b) { return result.scores[a] > result.scores[b]; });
            const float row_height = ImGui::GetTextLineHeight() + 3 * scale + 2 * ImGui::GetStyle().ItemSpacing.y;
            ImGui::BeginChild("Prediction", {0, std::min(float(order.size()), 8.0F) * row_height}, ImGuiChildFlags_None, ImGuiWindowFlags_NoSavedSettings);
            for (const auto index : order) {
                ImGui::TextWrapped("%s  /  %.1f%%", result.classes[index].c_str(), result.scores[index] * 100);
                ImGui::ProgressBar(result.scores[index], {-1, 3 * scale}, "");
            }
            ImGui::EndChild();
        } else if (const auto* classified = std::get_if<classification::Classification>(&inference->value)) {
            ImGui::Text("%zu images classified", classified->movement.paths.size());
            draw_path(classified->input);
            ImGui::Spacing();
            if (!classified->classes.empty()) {
                ImGui::BeginChild("Classification", {0, std::min(float(classified->classes.size()), 8.0F) * ImGui::GetTextLineHeightWithSpacing()}, ImGuiChildFlags_None, ImGuiWindowFlags_NoSavedSettings);
                for (const auto& [label, count] : classified->classes) ImGui::TextWrapped("%s  /  %zu", label.c_str(), count);
                ImGui::EndChild();
            }
            ImGui::Spacing();
            ImGui::TextDisabled("Moved into category folders.");
        }
    }
    void Workspace::draw_settings() {
        const float scale = dpi;
        ImGui::SetNextItemWidth(120 * scale);
        ImGui::InputInt("Target steps", &steps, 0, 0);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Cumulative target, including completed steps.");
        if (ImGui::CollapsingHeader("Parameters")) {
            const auto parameter = [&]<typename T>(const char* label, T& value) {
                ImGui::SetNextItemWidth(120 * scale);
                if constexpr (std::same_as<T, int>) ImGui::InputInt(label, &value, 0, 0);
                else if constexpr (std::same_as<T, std::uint64_t>) ImGui::InputScalar(label, ImGuiDataType_U64, &value);
                else ImGui::InputFloat(label, &value, 0, 0, "%.8g");
            };
            parameter("Physical batch", config.physical_batch);
            parameter("Effective batch", config.effective_batch);
            parameter("Head-only steps", config.head_only_steps);
            parameter("Warmup steps", config.warmup_steps);
            parameter("Head-only LR", config.head_only_lr);
            parameter("Backbone LR", config.backbone_lr);
            parameter("Head LR", config.head_lr);
            parameter("Weight decay", config.weight_decay);
            parameter("Gradient clipping", config.clip_norm);
            parameter("Seed", config.seed);
            parameter("Evaluation interval", config.eval_interval);
            parameter("Save interval", config.save_interval);
            parameter("Log interval", config.log_interval);
        }
        const bool changed   = training->state && training->state->config != config;
        const bool resumable = !training->state || training->state->checkpoint;
        const bool target    = steps > (training->state ? training->state->step : 0) && (!training->state || steps >= training->state->target);
        if (changed) ImGui::TextWrapped("Parameters changed. Restart is required.");
        if (!resumable) ImGui::TextWrapped("Checkpoint missing. Restart is required.");
        if (!target) ImGui::TextWrapped("Increase the cumulative target steps to continue.");
        ImGui::BeginDisabled(state.busy || changed || !resumable || !target);
        if (ImGui::Button(training->state ? "Continue training" : "Train")) submit({runtime::Train{{classifier->root, steps, false, config}}});
        ImGui::EndDisabled();
        if (training->state && ImGui::CollapsingHeader("Restart training")) {
            ImGui::TextWrapped("Replace this classifier's weights, checkpoint and history. Keep all source images.");
            ImGui::BeginDisabled(state.busy || steps <= 0);
            if (ImGui::Button("Replace state & restart")) {
                losses.clear();
                last_step = 0;
                metrics.reset();
                submit({runtime::Train{{classifier->root, steps, true, config}}});
            }
            ImGui::EndDisabled();
        }
    }
    void Workspace::draw_metrics() {
        const float scale = dpi;
        ImGui::Text("Validation / step %d", metrics->step);
        ImGui::Text("Loss %.4f  /  Accuracy %.1f%%", metrics->loss, metrics->accuracy * 100);
        ImGui::Text("Precision %.1f%%  /  Recall %.1f%%  /  F1 %.1f%%", metrics->precision * 100, metrics->recall * 100, metrics->f1 * 100);
        ImGui::Spacing();
        const float row_height = ImGui::GetTextLineHeight() + 2 * ImGui::GetStyle().CellPadding.y;
        if (ImGui::BeginTable("Classes", 5, ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY | ImGuiTableFlags_SizingStretchProp, {0, (std::min(float(metrics->classes.size()), 8.0F) + 1) * row_height})) {
            ImGui::TableSetupScrollFreeze(0, 1);
            ImGui::TableSetupColumn("Class", ImGuiTableColumnFlags_WidthStretch);
            for (const auto label : {"P", "R", "F1", "N"}) ImGui::TableSetupColumn(label, ImGuiTableColumnFlags_WidthFixed);
            ImGui::TableHeadersRow();
            for (std::size_t i = 0; i < metrics->classes.size(); ++i) {
                const auto& metric = metrics->classes[i];
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("%zu: %s", i + 1, metric.name.c_str());
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", metric.name.c_str());
                ImGui::TableNextColumn();
                ImGui::Text("%.1f", metric.precision * 100);
                ImGui::TableNextColumn();
                ImGui::Text("%.1f", metric.recall * 100);
                ImGui::TableNextColumn();
                ImGui::Text("%.1f", metric.f1 * 100);
                ImGui::TableNextColumn();
                ImGui::Text("%zu", metric.samples);
            }
            ImGui::EndTable();
        }
        if (ImGui::CollapsingHeader("Confusion matrix")) {
            ImGui::TextDisabled("Rows: actual / columns: predicted");
            const float cell = 48 * scale, line = ImGui::GetTextLineHeightWithSpacing();
            ImGui::BeginChild("Matrix", {0, std::min(float(metrics->classes.size() + 1), 9.0F) * line + ImGui::GetStyle().ScrollbarSize}, ImGuiChildFlags_None, ImGuiWindowFlags_HorizontalScrollbar);
            const auto origin = ImGui::GetCursorScreenPos();
            auto* draw        = ImGui::GetWindowDrawList();
            const auto color  = ImGui::GetColorU32(ImGuiCol_Text);
            for (std::size_t column = 0; column < metrics->classes.size(); ++column) draw->AddText({origin.x + (column + 1) * cell, origin.y}, color, std::to_string(column + 1).c_str());
            for (std::size_t row = 0; row < metrics->confusion.size(); ++row) {
                draw->AddText({origin.x, origin.y + (row + 1) * line}, color, std::to_string(row + 1).c_str());
                for (std::size_t column = 0; column < metrics->confusion[row].size(); ++column) draw->AddText({origin.x + (column + 1) * cell, origin.y + (row + 1) * line}, color, std::to_string(metrics->confusion[row][column]).c_str());
            }
            ImGui::Dummy({(metrics->classes.size() + 1) * cell, (metrics->classes.size() + 1) * line});
            ImGui::EndChild();
        }
    }
} // namespace classifier::editor
