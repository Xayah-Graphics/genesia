module;
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_internal.h>
#include <misc/cpp/imgui_stdlib.h>
#include <nlohmann/json.hpp>
#include <ShlObj.h>
module qwen.editor.workspace;
import tools.files;
import tools.editor.style;
import std;
namespace qwen::editor {
    Workspace::Workspace(tools::editor::WindowPlatform& platform, const float& scale) : window{platform}, dpi{scale}, session{[] { glfwPostEmptyEvent(); }} {
        PWSTR directory{};
        const auto status = SHGetKnownFolderPath(FOLDERID_LocalAppData, 0, nullptr, &directory);
        if (FAILED(status)) throw std::system_error{static_cast<int>(status), std::system_category(), "Find Qwen settings directory"};
        settings = std::filesystem::path{directory} / "Genesia" / "Qwen" / "settings.json";
        CoTaskMemFree(directory);
        if (std::filesystem::exists(settings)) {
            const auto value = tools::files::read_json(settings);
            prompt          = value.at("prompt").get<std::string>();
            choices         = value.at("choices").get<std::vector<std::string>>();
            std::erase_if(choices, [](const std::string& value) { return value.find_first_not_of(" \t\r\n") == std::string::npos; });
        }
    }
    void Workspace::receive() {
        const auto now = std::chrono::steady_clock::now();
        if (connection_check.valid() && connection_check.wait_for(std::chrono::seconds{0}) == std::future_status::ready) {
            connected             = connection_check.get();
            next_connection_check = now + std::chrono::seconds{3};
            window.redraw         = true;
        }
        if (!connection_check.valid() && now >= next_connection_check) connection_check = std::async(std::launch::async, check_connection);
        auto delivery       = session.drain();
        const auto previous = state.revision;
        state               = std::move(delivery.state);
        for (const auto& progress : delivery.progress)
            if (!progress.result.empty()) results.push_back({progress.file, progress.result});
        if (state.revision != previous && !state.result.error.empty()) error = state.result.error;
        if (save_at && now >= *save_at) {
            try {
                save_settings();
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
            if (prompt.find_first_not_of(" \t\r\n") == std::string::npos) throw std::runtime_error{"Enter a prompt before dropping images"};
            if (choices.size() < 2 || std::ranges::any_of(choices, [](const std::string& value) { return value.find_first_not_of(" \t\r\n") == std::string::npos; })) throw std::runtime_error{"Enter at least two candidate results"};
            if (std::ranges::unique(choices).begin() != choices.end()) throw std::runtime_error{"Candidate results must be unique"};
            save_settings();
            results.clear();
            session.submit({paths.front(), prompt, choices});
            state = session.drain().state;
            error.clear();
        } catch (const std::exception& failure) {
            error = failure.what();
        }
        window.redraw = true;
    }
    void Workspace::draw() {
        const float width = std::min(520 * dpi, float(window.extent_limit.width));
        const auto& style = ImGui::GetStyle();
        ImGui::SetNextWindowPos({0, 0});
        ImGui::SetNextWindowSize({width, float(window.extent_limit.height)});
        ImGui::Begin("Qwen", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoBringToFrontOnFocus);
        if (state.state == runtime::State::idle && results.empty()) tools::editor::draw_welcome("Analyze Images with Qwen", "Drop an image or folder");
        ImGui::TextDisabled("Prompt");
        const auto connection_label = !connected ? "LM Studio: Checking..." : *connected ? "LM Studio: Connected" : "LM Studio: Offline";
        const auto connection_color = !connected ? style.Colors[ImGuiCol_TextDisabled] : *connected ? style.Colors[ImGuiCol_PlotHistogram] : ImVec4{0.95F, 0.59F, 0.55F, 1};
        ImGui::SameLine(width - style.WindowPadding.x - ImGui::CalcTextSize(connection_label).x);
        ImGui::TextColored(connection_color, "%s", connection_label);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", server_url.data());
        ImGui::BeginDisabled(state.busy);
        if (ImGui::InputTextMultiline("##prompt", &prompt, {-1, 3 * ImGui::GetTextLineHeightWithSpacing()}, ImGuiInputTextFlags_WordWrap)) save_at = std::chrono::steady_clock::now() + std::chrono::milliseconds{300};
        ImGui::TextDisabled("Candidates");
        const auto editing_index = editing_choice;
        const float group_width = ImGui::GetContentRegionAvail().x;
        constexpr float tag_height = 26;
        constexpr float tag_gap    = 4;
        constexpr float tag_padding = 8;
        const float chip_height     = tag_height * dpi;
        const float chip_gap        = tag_gap * dpi;
        const float chip_padding    = tag_padding * dpi;
        const float group_padding   = 12 * dpi;
        const float content_width   = group_width - 2 * group_padding;
        struct CandidateLayout final {
            ImVec2 position;
            float width;
        };
        std::vector<CandidateLayout> candidate_layout;
        candidate_layout.reserve(choices.size() + 1);
        float x{}, y{};
        for (std::size_t i = 0; i < choices.size(); ++i) {
            const float text_width = ImGui::CalcTextSize(choices[i].c_str()).x + 2 * chip_padding;
            const float chip_width = std::min(content_width, std::max(editing_choice && *editing_choice == i ? 220 * dpi : chip_height, text_width));
            if (x && x + chip_width > content_width) {
                x = 0;
                y += chip_height + chip_gap;
            }
            candidate_layout.push_back({{x, y}, chip_width});
            x += chip_width + chip_gap;
        }
        if (choices.size() < 8 && !editing_choice) {
            if (x && x + chip_height > content_width) {
                x = 0;
                y += chip_height + chip_gap;
            }
            candidate_layout.push_back({{x, y}, chip_height});
        }
        const auto origin = ImGui::GetCursorScreenPos();
        const ImVec2 maximum{origin.x + group_width, origin.y + y + chip_height + 2 * group_padding};
        auto* draw = ImGui::GetWindowDrawList();
        const bool group_hovered = ImGui::IsMouseHoveringRect(origin, maximum);
        draw->AddRectFilled(origin, maximum, ImGui::GetColorU32(ImVec4{0.071F, 0.075F, 0.086F, 1}), 12 * dpi);
        draw->AddRect(origin, maximum, ImGui::GetColorU32(ImVec4{0.55F, 0.55F, 0.62F, group_hovered ? 0.45F : 0.22F}), 12 * dpi, 0, dpi);
        std::optional<std::size_t> remove;
        bool editing_hovered{};
        ImGuiID editing_id{};
        for (std::size_t i = 0; i < choices.size(); ++i) {
            const auto& item = candidate_layout[i];
            ImGui::SetCursorScreenPos({origin.x + group_padding + item.position.x, origin.y + group_padding + item.position.y});
            ImGui::PushID(static_cast<int>(i));
            if (editing_choice && *editing_choice == i) {
                ImGui::PushStyleColor(ImGuiCol_FrameBg, {0.12F, 0.125F, 0.155F, 1});
                ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, {0.14F, 0.145F, 0.18F, 1});
                ImGui::PushStyleColor(ImGuiCol_FrameBgActive, {0.16F, 0.16F, 0.21F, 1});
                ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, {chip_padding, 3 * dpi});
                ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 5 * dpi);
                if (focus_choice) {
                    ImGui::SetKeyboardFocusHere();
                    focus_choice = false;
                }
                const bool committed = ImGui::InputText("##candidate", &choices[i], ImGuiInputTextFlags_EnterReturnsTrue);
                const bool canceled = ImGui::IsKeyPressed(ImGuiKey_Escape, false);
                editing_hovered = ImGui::IsItemHovered();
                editing_id = ImGui::GetItemID();
                if (ImGui::IsItemEdited()) save_at = std::chrono::steady_clock::now() + std::chrono::milliseconds{300};
                if (committed || canceled) {
                    const bool empty = choices[i].find_first_not_of(" \t\r\n") == std::string::npos;
                    if (canceled) {
                        if (choice_created || choice_original.empty()) remove = i;
                        else choices[i] = choice_original;
                    } else if (empty) remove = i;
                    editing_choice.reset();
                    choice_original.clear();
                    choice_created = false;
                    window.redraw = true;
                }
                ImGui::PopStyleVar(2);
                ImGui::PopStyleColor(3);
            } else {
                const auto minimum = ImGui::GetCursorScreenPos();
                ImGui::InvisibleButton("##candidate", {item.width, chip_height});
                const bool hovered = ImGui::IsItemHovered();
                draw->AddRectFilled(minimum, {minimum.x + item.width, minimum.y + chip_height}, ImGui::GetColorU32(hovered ? ImVec4{0.23F, 0.22F, 0.30F, 0.90F} : ImVec4{0.20F, 0.205F, 0.25F, 0.55F}), 5 * dpi);
                draw->AddText({minimum.x + chip_padding, minimum.y + (chip_height - ImGui::GetFontSize()) / 2}, ImGui::GetColorU32(ImVec4{0.76F, 0.76F, 0.81F, 1}), choices[i].c_str());
                if (hovered) {
                    ImGui::BeginTooltip();
                    ImGui::TextUnformatted(choices[i].c_str());
                    ImGui::TextUnformatted("Middle-click to remove. Double-click to edit.");
                    ImGui::EndTooltip();
                    if (choices.size() > 2 && ImGui::IsMouseClicked(ImGuiMouseButton_Middle)) remove = i;
                    if (!editing_index && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
                        editing_choice = i;
                        choice_original = choices[i];
                        focus_choice = true;
                        window.redraw = true;
                    }
                }
            }
            ImGui::PopID();
        }
        if (candidate_layout.size() > choices.size()) {
            const auto& item = candidate_layout.back();
            ImGui::SetCursorScreenPos({origin.x + group_padding + item.position.x, origin.y + group_padding + item.position.y});
            const bool clicked = ImGui::InvisibleButton("##add_candidate", {item.width, chip_height});
            const auto minimum = ImGui::GetItemRectMin();
            const bool hovered = ImGui::IsItemHovered();
            draw->AddRectFilled(minimum, {minimum.x + item.width, minimum.y + chip_height}, ImGui::GetColorU32(hovered ? ImVec4{0.23F, 0.22F, 0.30F, 0.90F} : ImVec4{0.20F, 0.205F, 0.25F, 0.55F}), 5 * dpi);
            const auto ink = ImGui::GetColorU32(hovered ? ImGuiCol_Text : ImGuiCol_TextDisabled);
            const auto center = ImVec2{minimum.x + item.width / 2, minimum.y + chip_height / 2};
            draw->AddLine({center.x - 5 * dpi, center.y}, {center.x + 5 * dpi, center.y}, ink, dpi);
            draw->AddLine({center.x, center.y - 5 * dpi}, {center.x, center.y + 5 * dpi}, ink, dpi);
            if (hovered) ImGui::SetTooltip("Add candidate");
            if (clicked && !editing_choice) {
                const auto empty = std::ranges::find_if(choices, [](const std::string& value) { return value.find_first_not_of(" \t\r\n") == std::string::npos; });
                if (empty == choices.end()) {
                    choices.emplace_back();
                    editing_choice = choices.size() - 1;
                    choice_created = true;
                } else {
                    editing_choice = static_cast<std::size_t>(empty - choices.begin());
                    choice_created = false;
                }
                choice_original = choices[*editing_choice];
                focus_choice = true;
                window.redraw = true;
            }
        }
        if (editing_index && ImGui::IsMouseClicked(ImGuiMouseButton_Left) && !editing_hovered) {
            const auto index = *editing_index;
            const bool empty = choices[index].find_first_not_of(" \t\r\n") == std::string::npos;
            if (empty) remove = index;
            if (ImGui::GetActiveID() == editing_id) ImGui::ClearActiveID();
            editing_choice.reset();
            choice_original.clear();
            choice_created = false;
            save_at = std::chrono::steady_clock::now() + std::chrono::milliseconds{300};
            window.redraw = true;
        }
        ImGui::SetCursorScreenPos(origin);
        ImGui::Dummy({group_width, maximum.y - origin.y});
        if (remove) {
            choices.erase(choices.begin() + *remove);
            if (editing_choice && *editing_choice > *remove) --*editing_choice;
            save_at      = std::chrono::steady_clock::now() + std::chrono::milliseconds{300};
            window.redraw = true;
        }
        ImGui::EndDisabled();
        if (!state.busy) {
            const auto origin     = ImGui::GetCursorPos();
            constexpr auto label  = "Drop a PNG or folder";
            const auto dimensions = ImGui::CalcTextSize(label);
            ImGui::SetCursorPos({(width - dimensions.x) / 2, origin.y + (56 * dpi - dimensions.y) / 2});
            ImGui::TextUnformatted(label);
            ImGui::SetCursorPos(origin);
            ImGui::Dummy({ImGui::GetContentRegionAvail().x, 56 * dpi});
        } else {
            ImGui::TextUnformatted(state.stopping ? "Stopping..." : "Analyzing");
            ImGui::SameLine();
            ImGui::TextDisabled("%zu / %zu", state.progress.completed, state.progress.total);
            ImGui::ProgressBar(state.progress.total ? float(state.progress.completed) / state.progress.total : 0, {-1, 4 * dpi}, "");
            ImGui::BeginDisabled(state.stopping);
            if (ImGui::Button("Stop")) session.cancel();
            ImGui::EndDisabled();
        }
        if (!results.empty()) {
            ImGui::Separator();
            ImGui::TextDisabled("Results");
            for (const auto& item : results) ImGui::TextWrapped("%s  /  %s", tools::files::utf8(item.file.filename()).c_str(), item.result.c_str());
            if (ImGui::Button("Copy Results")) {
                std::string text;
                for (const auto& item : results) text += nlohmann::json{{"image", tools::files::utf8(item.file)}, {"result", item.result}}.dump() + "\n";
                ImGui::SetClipboardText(text.c_str());
            }
        }
        if (!state.busy && state.state == runtime::State::complete && !results.empty()) ImGui::TextDisabled("%zu images analyzed", results.size());
        if (!error.empty()) {
            ImGui::PushStyleColor(ImGuiCol_Text, {0.95F, 0.59F, 0.55F, 1});
            ImGui::TextWrapped("%s", error.c_str());
            ImGui::PopStyleColor();
            if (ImGui::SmallButton("Dismiss")) error.clear();
        }
        const auto* layout           = ImGui::GetCurrentWindow();
        window.content_extent.width  = static_cast<std::uint32_t>(std::ceil(width));
        window.content_extent.height = static_cast<std::uint32_t>(std::ceil(std::min(layout->DC.CursorMaxPos.y - layout->DC.CursorStartPos.y + 2 * style.WindowPadding.y, float(window.extent_limit.height))));
        window.drag_requested = !ImGui::IsAnyItemHovered() && !ImGui::IsAnyItemActive() && ImGui::IsMouseClicked(ImGuiMouseButton_Left);
        ImGui::End();
    }
    void Workspace::save_settings() {
        save_at.reset();
        std::vector<std::string> saved_choices;
        for (const auto& choice : choices)
            if (choice.find_first_not_of(" \t\r\n") != std::string::npos) saved_choices.push_back(choice);
        tools::files::write_json(settings, {{"prompt", prompt}, {"choices", saved_choices}});
    }
} // namespace qwen::editor
