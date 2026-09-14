module;
#include <imgui.h>
module genesia.editor.viewing.camera;
import std;
namespace genesia::editor {
    void ImageView::scale_to(const float ratio, const ImVec2 position, const ImVec2 image, const bool fitting, const double now) {
        fit          = fitting;
        dragging     = false;
        initial_zoom = zoom;
        target_zoom  = ratio;
        anchor       = {center.x + position.x / (image.x * zoom), center.y + position.y / (image.y * zoom)};
        pivot        = position;
        started      = now;
    }

    void ImageView::update(const ImVec2 available, const ImVec2 image, const double now) {
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

    void ImageView::constrain(const ImVec2 available, const ImVec2 image) {
        const float horizontal = std::min(0.5F, available.x / (2 * image.x * zoom));
        const float vertical   = std::min(0.5F, available.y / (2 * image.y * zoom));
        center.x               = std::clamp(center.x, horizontal, 1 - horizontal);
        center.y               = std::clamp(center.y, vertical, 1 - vertical);
    }

} // namespace genesia::editor
