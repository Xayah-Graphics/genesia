export module tools.application;
import std;

export namespace tools::application {
    int run(const char* name, int argc, char** argv, bool editor, int (*execute)(bool, std::span<const std::string_view>));
} // namespace tools::application
