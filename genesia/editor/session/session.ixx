module;
#include "../../core/sdxl/control.h"
#include <genesia/cuda.h>
#include <nlohmann/json.hpp>
export module genesia.editor.session;
export import genesia.work;
import genesia.editor.platform.interop;
import genesia.editor.runtime.device;
import std;
export namespace genesia::editor {
    struct WorkspaceRuntime final {
        runtime::Device& device;
        std::unique_ptr<Interop> interop, preview;
        work::Session session;
        explicit WorkspaceRuntime(runtime::Device& device);
    };
} // namespace genesia::editor
