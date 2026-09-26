module genesia.generation.settings;
import genesia.io.files;
import std;
namespace genesia::generation {
    std::string positive_prompt(const Settings& parameters) {
        std::string result;
        for (const auto& lora : parameters.loras) {
            if (!result.empty()) result += ", ";
            result += files::utf8(files::path(lora.file).stem());
        }
        if (!result.empty() && !parameters.positive.empty()) result += ", ";
        result += parameters.positive;
        return result;
    }
} // namespace genesia::generation
