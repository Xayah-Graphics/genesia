export module foreground.headless;
import std;
export namespace foreground::headless {
    int run(std::span<const std::string_view> arguments);
}
