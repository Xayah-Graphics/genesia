module;
#include <genesia/cuda.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
module foreground.processing;
import tools.images;
import tools.files;
import std;
namespace foreground {
    namespace {
        // Pillow's separable, antialiased bicubic filter: byte rounding after each axis.
        std::vector<std::uint8_t> resize(const std::span<const std::uint8_t> input, int width, int height, const int channels, const int target_width, const int target_height) {
            std::vector<std::uint8_t> pixels{input.begin(), input.end()};
            for (int axis = 0; axis < 2; ++axis) {
                const int source = axis ? height : width, target = axis ? target_height : target_width;
                if (source == target) continue;
                const double scale = double(source) / target, filter = std::max(1.0, scale), support = 2 * filter;
                struct Filter final {
                    int first;
                    std::vector<int> weights;
                };
                std::vector<Filter> filters;
                for (int i = 0; i < target; ++i) {
                    const double center = (i + .5) * scale;
                    const int first = std::max(0, int(center - support + .5)), last = std::min(source, int(center + support + .5));
                    std::vector<double> weights;
                    double sum{};
                    for (int j = first; j < last; ++j) {
                        const double x     = std::abs((j + .5 - center) / filter);
                        const double value = x < 1 ? ((1.5 * x - 2.5) * x) * x + 1 : x < 2 ? ((-.5 * x + 2.5) * x - 4) * x + 2 : 0;
                        weights.push_back(value);
                        sum += value;
                    }
                    Filter coefficients{first};
                    for (const auto weight : weights) coefficients.weights.push_back(int(std::round(weight / sum * (1 << 22))));
                    filters.push_back(std::move(coefficients));
                }
                const int out_width = axis ? width : target, out_height = axis ? target : height;
                std::vector<std::uint8_t> output(std::size_t(out_width) * out_height * channels);
                for (int y = 0; y < out_height; ++y)
                    for (int x = 0; x < out_width; ++x) {
                        const auto& coefficients = filters[axis ? y : x];
                        for (int c = 0; c < channels; ++c) {
                            int value = 1 << 21;
                            for (int j = 0; j < int(coefficients.weights.size()); ++j) {
                                const int sx = axis ? x : coefficients.first + j, sy = axis ? coefficients.first + j : y;
                                value += pixels[(std::size_t(sy) * width + sx) * channels + c] * coefficients.weights[j];
                            }
                            output[(std::size_t(y) * out_width + x) * channels + c] = static_cast<std::uint8_t>(std::clamp(value >> 22, 0, 255));
                        }
                    }
                width  = out_width;
                height = out_height;
                pixels = std::move(output);
            }
            return pixels;
        }
    } // namespace

    Pipeline::Pipeline()  = default;
    Pipeline::~Pipeline() = default;

    Result Pipeline::process(const Request& request, const std::atomic_bool& interrupted, const std::function<void(const Progress&)>& progress) {
        Result result{.output = std::filesystem::absolute(request.input)};
        auto current = result.output;
        try {
            if (!std::filesystem::is_directory(result.output)) throw std::runtime_error{"Foreground requires a folder containing PNG images. Individual files are not supported."};
            progress({.stage = Stage::scanning});
            std::vector<std::filesystem::path> inputs;
            for (const auto& entry : std::filesystem::recursive_directory_iterator{result.output}) {
                if (interrupted) break;
                if (!entry.is_regular_file()) continue;
                auto extension = entry.path().extension().string();
                auto stem      = tools::files::utf8(entry.path().stem());
                std::ranges::transform(extension, extension.begin(), [](unsigned char value) { return char(std::tolower(value)); });
                std::ranges::transform(stem, stem.begin(), [](unsigned char value) { return char(std::tolower(value)); });
                if (extension == ".png" && !stem.ends_with("-masklabel")) inputs.push_back(entry.path());
            }
            std::ranges::sort(inputs);
            result.total = inputs.size();
            if (interrupted) {
                result.stopped = true;
                return result;
            }
            if (inputs.empty()) throw std::runtime_error{"No PNG images in this folder"};
            if (!network) {
                progress({.stage = Stage::loading_model, .total = result.total});
                network = std::make_unique<birefnet::Network>(std::filesystem::path{FOREGROUND_ASSET_DIRECTORY} / "weights" / "BiRefNet-general.safetensors");
            }
            for (const auto& source : inputs) {
                if (interrupted) break;
                current = source;
                progress({Stage::masking, result.completed, result.total, source});
                const auto image = tools::read_image(source);
                const auto input = resize(image.pixels, image.width, image.height, 3, 1024, 1024);
                const auto mask  = resize(network->infer(input), 1024, 1024, 1, image.width, image.height);
                std::vector<std::uint8_t> encoded;
                const auto write = [](void* context, void* data, int size) {
                    auto& bytes       = *static_cast<std::vector<std::uint8_t>*>(context);
                    const auto* start = static_cast<std::uint8_t*>(data);
                    bytes.insert(bytes.end(), start, start + size);
                };
                if (!stbi_write_png_to_func(write, &encoded, image.width, image.height, 1, mask.data(), image.width)) throw std::runtime_error{"Cannot encode foreground mask"};
                const auto target = source.parent_path() / (source.stem().native() + std::filesystem::path{"-masklabel.png"}.native());
                tools::files::write_bytes(target, encoded);
                ++result.completed;
                progress({Stage::masking, result.completed, result.total, source});
            }
            result.stopped = interrupted && result.completed != result.total;
        } catch (const std::exception& failure) {
            result.error = tools::files::utf8(current) + ": " + failure.what();
        }
        return result;
    }
} // namespace foreground
