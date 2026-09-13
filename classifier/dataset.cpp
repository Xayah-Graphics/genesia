module;
#include <nlohmann/json.hpp>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
module classifier.dataset;
import std;
import genesia.hash;
namespace classifier {
    Dataset::Dataset(const std::filesystem::path& path) : root(std::filesystem::absolute(path)) {
        std::ifstream f(root / "dataset.json");
        f >> metadata;
        classes = metadata["classes"].get<std::vector<std::string>>();
        for (const auto& r : metadata["records"]) {
            int index = int(records.size());
            records.push_back({r["id"], r["path"], r["split"], r["group"].is_string() ? r["group"].get<std::string>() : r["group"].dump(), r["cache"], r["label"], r["width"], r["height"], r["offset"]});
            if (records.back().split == "train") train.push_back(index);
            else if (records.back().split == "val") val.push_back(index);
            const auto& cache = records.back().cache;
            if (!caches.contains(cache)) caches[cache] = std::make_unique<Mapping>(root / cache);
        }
    }
    const unsigned char* Dataset::rgb(int index) const {
        const auto& r = records[index];
        return static_cast<unsigned char*>(caches.at(r.cache)->data) + r.offset;
    }
    Image load_image(const std::filesystem::path& path) {
        Mapping file(path);
        Image image;
        int channels;
        unsigned char* data = stbi_load_from_memory(static_cast<unsigned char*>(file.data), int(file.size), &image.width, &image.height, &channels, 3);
        if (!data) throw std::runtime_error(stbi_failure_reason());
        image.rgb.assign(data, data + std::size_t(image.width) * image.height * 3);
        stbi_image_free(data);
        return image;
    }
    std::vector<std::filesystem::path> image_paths(const std::filesystem::path& folder) {
        std::vector<std::filesystem::path> paths;
        for (const auto& entry : std::filesystem::recursive_directory_iterator(folder)) {
            auto extension = path_utf8(entry.path().extension());
            std::ranges::transform(extension, extension.begin(), [](unsigned char c) { return char(std::tolower(c)); });
            if (entry.is_regular_file() && (extension == ".png" || extension == ".jpg" || extension == ".jpeg" || extension == ".bmp" || extension == ".tga")) paths.push_back(entry.path());
        }
        std::ranges::sort(paths);
        return paths;
    }
    void prepare(const std::vector<std::filesystem::path>& folders, const std::filesystem::path& output) {
        if (!std::filesystem::create_directories(output)) throw std::runtime_error("Prepare requires a new output directory for an immutable data snapshot");
        std::vector<std::filesystem::path> sorted = folders;
        std::ranges::sort(sorted, [](const auto& a, const auto& b) { return a.filename() == "YES" ? b.filename() != "YES" : b.filename() == "YES" ? false : a.filename() < b.filename(); });
        if (sorted.front().filename() != "YES") throw std::runtime_error("A YES category is required");
        nlohmann::json records = nlohmann::json::array();
        std::vector<std::string> classes, hashes;
        std::vector<std::uint64_t> phashes;
        std::map<std::string, std::ofstream> caches;
        for (const auto& folder : sorted) {
            int label = int(classes.size());
            classes.push_back(path_utf8(folder.filename()));
            for (const auto& path : image_paths(folder)) {
                Image image = load_image(path);
                if (!((image.width == 1024 && image.height == 1536) || (image.width == 1536 && image.height == 1024) || (image.width == 1024 && image.height == 1024))) throw std::runtime_error("Unsupported original size: " + path_utf8(path));
                std::string hex = genesia::sha256(image.rgb);
                hashes.push_back(hex);
                std::array<double, 1024> small{};
                for (int y = 0; y < 32; ++y)
                    for (int x = 0; x < 32; ++x) {
                        double sum = 0;
                        int count  = 0;
                        for (int iy = y * image.height / 32; iy < (y + 1) * image.height / 32; ++iy)
                            for (int ix = x * image.width / 32; ix < (x + 1) * image.width / 32; ++ix) {
                                auto p = image.rgb.data() + (std::size_t(iy) * image.width + ix) * 3;
                                sum += .299 * p[0] + .587 * p[1] + .114 * p[2];
                                ++count;
                            }
                        small[y * 32 + x] = sum / count;
                    }
                std::array<double, 64> dct{};
                for (int v = 0; v < 8; ++v)
                    for (int u = 0; u < 8; ++u)
                        for (int y = 0; y < 32; ++y)
                            for (int x = 0; x < 32; ++x) dct[v * 8 + u] += small[y * 32 + x] * std::cos(std::numbers::pi * (2 * x + 1) * u / 64) * std::cos(std::numbers::pi * (2 * y + 1) * v / 64);
                auto sorted_dct = dct;
                std::ranges::sort(sorted_dct);
                double median       = (sorted_dct[31] + sorted_dct[32]) * .5;
                std::uint64_t phash = 0;
                for (int i = 0; i < 64; ++i) phash |= std::uint64_t(dct[i] > median) << i;
                phashes.push_back(phash);
                std::string cache = std::format("{}x{}.rgb", image.width, image.height);
                if (!caches.contains(cache)) caches[cache] = std::ofstream(output / cache, std::ios::binary);
                auto& file           = caches.at(cache);
                std::uint64_t offset = file.tellp();
                file.write(reinterpret_cast<const char*>(image.rgb.data()), image.rgb.size());
                records.push_back({{"id", classes.back() + "/" + path_utf8(std::filesystem::relative(path, folder))}, {"path", path_utf8(std::filesystem::absolute(path))}, {"label", label}, {"width", image.width}, {"height", image.height}, {"cache", cache}, {"offset", offset}, {"sha256", hex}, {"phash", std::format("{:016x}", phash)}});
            }
        }
        caches.clear();
        std::vector<int> group(records.size());
        std::iota(group.begin(), group.end(), 0);
        auto root = [&](int i) {
            while (group[i] != i) {
                group[i] = group[group[i]];
                i        = group[i];
            }
            return i;
        };
        for (int i = 0; i < int(records.size()); ++i)
            for (int j = 0; j < i; ++j)
                if (hashes[i] == hashes[j] || std::popcount(phashes[i] ^ phashes[j]) <= 5) group[root(i)] = root(j);
        std::map<int, std::vector<int>> groups;
        for (int i = 0; i < int(records.size()); ++i) groups[root(i)].push_back(i);
        std::vector<int> order;
        for (const auto& [key, members] : groups) order.push_back(key);
        std::mt19937 random(42);
        std::shuffle(order.begin(), order.end(), random);
        std::vector<int> total(classes.size(), 0), val(classes.size(), 0);
        for (const auto& r : records) ++total[r["label"].get<int>()];
        for (int key : order) {
            std::vector<int> count(classes.size(), 0);
            for (int index : groups[key]) ++count[records[index]["label"].get<int>()];
            double before = 0, after = 0;
            for (int c = 0; c < int(classes.size()); ++c) {
                before += std::abs(val[c] - .2 * total[c]);
                after += std::abs(val[c] + count[c] - .2 * total[c]);
            }
            int selected = std::accumulate(val.begin(), val.end(), 0);
            before += std::abs(selected - .2 * records.size());
            after += std::abs(selected + groups[key].size() - .2 * records.size());
            bool validation = after < before;
            for (int index : groups[key]) {
                records[index]["split"] = validation ? "val" : "train";
                records[index]["group"] = std::to_string(key);
            }
            if (validation)
                for (int c = 0; c < int(classes.size()); ++c) val[c] += count[c];
        }
        write_json(output / "dataset.json", {{"version", 1}, {"seed", 42}, {"classes", classes}, {"records", records}, {"threshold", .9}, {"grouping", "SHA256 RGB and 64-bit DCT pHash <= 5, transitive closure"}});
        std::println("Prepared {} images in {} categories", records.size(), classes.size());
    }
    std::vector<std::filesystem::path> categories(const std::filesystem::path& root) {
        std::vector<std::filesystem::path> folders;
        for (const auto& entry : std::filesystem::directory_iterator(root))
            if (entry.is_directory() && entry.path().filename() != ".classifier") folders.push_back(entry.path());
        std::ranges::sort(folders, [](const auto& a, const auto& b) { return a.filename() == "YES" ? b.filename() != "YES" : b.filename() == "YES" ? false : a.filename() < b.filename(); });
        if (folders.size() < 2 || folders.front().filename() != "YES") throw std::runtime_error("Dataset requires YES and at least one rejection category");
        return folders;
    }
    std::string fingerprint(const std::filesystem::path& root) {
        genesia::Sha256 hash;
        for (const auto& folder : categories(root)) {
            auto label = folder.filename().generic_u8string();
            hash.update({reinterpret_cast<const unsigned char*>(label.data()), label.size() + 1});
            for (const auto& path : image_paths(folder)) {
                auto relative = std::filesystem::relative(path, root).generic_u8string();
                hash.update({reinterpret_cast<const unsigned char*>(relative.data()), relative.size() + 1});
                Mapping file(path);
                genesia::Sha256 content;
                content.update({static_cast<const unsigned char*>(file.data), file.size});
                hash.update(content.finish());
            }
        }
        std::string result;
        for (auto byte : hash.finish()) result += std::format("{:02x}", byte);
        return result;
    }
} // namespace classifier
