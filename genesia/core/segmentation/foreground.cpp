module;
#include <genesia/cuda.h>
#include <stb_image_write.h>
module genesia.segmentation.foreground;
import std;

namespace genesia::foreground {
    namespace {
        const std::filesystem::path model = std::filesystem::path{project::assets} / "birefnet" / "BiRefNet-general.safetensors";
        std::mutex cache_files;

        const std::string& identity() {
            static const auto sha = files::digest(model);
            return sha;
        }

        std::filesystem::path mask_path(const dataset::File& image) {
            return std::filesystem::path{project::cache} / "foreground" / ("birefnet-v1-" + identity()) / (image.sha + ".png");
        }

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

    std::optional<Image> read_cached(const dataset::File& image) {
        const auto path = mask_path(image);
        // A Windows rename exposes the final name before its write handle closes.
        const std::lock_guard lock{cache_files};
        if (std::filesystem::exists(path)) return read_image(path, 1);
        return {};
    }

    Result Pipeline::infer(const dataset::File& image, std::span<const std::uint8_t> rgb) {
        const auto path = mask_path(image);
        if (std::filesystem::exists(path)) return {image, identity(), path, true};
        Image decoded;
        if (rgb.empty()) {
            decoded = read_image(image.path);
            rgb     = decoded.pixels;
        }
        const auto input = resize(rgb, image.width, image.height, 3, 1024, 1024);
        if (!network) network = std::make_unique<birefnet::Network>(model);
        const auto mask = resize(network->infer(input), 1024, 1024, 1, image.width, image.height);
        std::vector<std::uint8_t> encoded;
        const auto write = [](void* context, void* data, int size) {
            auto& bytes       = *static_cast<std::vector<std::uint8_t>*>(context);
            const auto* start = static_cast<std::uint8_t*>(data);
            bytes.insert(bytes.end(), start, start + size);
        };
        if (!stbi_write_png_to_func(write, &encoded, image.width, image.height, 1, mask.data(), image.width)) throw std::runtime_error{"Cannot encode foreground mask"};
        const std::lock_guard lock{cache_files};
        files::write_bytes(path, encoded);
        return {image, identity(), path, false};
    }
} // namespace genesia::foreground
