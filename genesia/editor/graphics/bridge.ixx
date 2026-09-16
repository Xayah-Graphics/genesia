module;
#include <genesia/cuda.h>
export module genesia.editor.graphics.bridge;
export import genesia.runtime.session;
export import genesia.editor.web;
import genesia.editor.graphics.interop;
import genesia.editor.graphics.device;
import std;
export namespace genesia::editor {
    struct PresentedFrame final {
        std::shared_ptr<Interop> bridge;
        std::size_t slot;
        std::uint64_t ready;
    };
    struct WorkspaceRuntime final {
        graphics::Device& device;
        std::shared_ptr<Interop> interop, preview;
        Web web;
        runtime::Session session;
        explicit WorkspaceRuntime(graphics::Device& device);
    };
} // namespace genesia::editor
