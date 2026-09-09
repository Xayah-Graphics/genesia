export module genesia.editor.runtime.images;
export import genesia.generation.images;
import std;

export namespace genesia::editor {
    Image thumbnail(std::span<const std::uint8_t> pixels, int width, int height);
} // namespace genesia::editor
