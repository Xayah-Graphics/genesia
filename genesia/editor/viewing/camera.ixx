module;
#include <imgui.h>
export module genesia.editor.viewing.camera;
import std;
export namespace genesia::editor {
    struct ImageView final {
        float zoom{1};
        bool fit{true}, dragging{};
        ImVec2 center{0.5F, 0.5F};
        float initial_zoom{1}, target_zoom{1};
        ImVec2 anchor{}, pivot{};
        double started{-1};
        void scale_to(float ratio, ImVec2 position, ImVec2 image, bool fitting, double now);
        void update(ImVec2 available, ImVec2 image, double now);
        void constrain(ImVec2 available, ImVec2 image);
    };
} // namespace genesia::editor
