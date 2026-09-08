export module genesia.headless;

export import genesia.generation.defaults;
export import genesia.prompt.preset;
import std;

export namespace genesia::headless {
    void run(const prompt::Preset& preset, const std::shared_ptr<const prompt::Catalog>& catalog);
}
