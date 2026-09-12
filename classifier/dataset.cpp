module;
#include <nlohmann/json.hpp>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
module classifier.dataset;
import std;
namespace classifier {
    struct Sha256 {
        std::array<std::uint32_t, 8> state{0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
        std::array<unsigned char, 64> buffer{};
        std::uint64_t bytes = 0;
        void update(std::span<const unsigned char> input) {
            const std::size_t used = bytes % 64;
            bytes += input.size();
            if (used && !input.empty()) {
                const auto count = std::min(input.size(), 64 - used);
                std::memcpy(buffer.data() + used, input.data(), count);
                input = input.subspan(count);
                if (used + count < 64) return;
                compress(buffer.data());
            }
            while (input.size() >= 64) {
                compress(input.data());
                input = input.subspan(64);
            }
            std::ranges::copy(input, buffer.begin());
        }
        std::array<unsigned char, 32> finish() {
            std::size_t used = bytes % 64;
            buffer[used++]   = 0x80;
            if (used > 56) {
                std::fill(buffer.begin() + used, buffer.end(), 0);
                compress(buffer.data());
                used = 0;
            }
            std::fill(buffer.begin() + used, buffer.begin() + 56, 0);
            const std::uint64_t bits = bytes * 8;
            for (int i = 0; i < 8; ++i) buffer[63 - i] = static_cast<unsigned char>(bits >> (8 * i));
            compress(buffer.data());
            std::array<unsigned char, 32> digest{};
            for (int i = 0; i < 32; ++i) digest[i] = static_cast<unsigned char>(state[i / 4] >> (24 - 8 * (i % 4)));
            return digest;
        }

    private:
        void compress(const unsigned char* input) {
            static constexpr std::array<std::uint32_t, 64> constants{0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
            std::array<std::uint32_t, 64> words{};
            for (int i = 0; i < 16; ++i) words[i] = (std::uint32_t(input[4 * i]) << 24) | (std::uint32_t(input[4 * i + 1]) << 16) | (std::uint32_t(input[4 * i + 2]) << 8) | input[4 * i + 3];
            for (int i = 16; i < 64; ++i) {
                const auto a = words[i - 15], b = words[i - 2];
                words[i] = words[i - 16] + (std::rotr(a, 7) ^ std::rotr(a, 18) ^ (a >> 3)) + words[i - 7] + (std::rotr(b, 17) ^ std::rotr(b, 19) ^ (b >> 10));
            }
            auto [a, b, c, d, e, f, g, h] = state;
            for (int i = 0; i < 64; ++i) {
                const auto first  = h + (std::rotr(e, 6) ^ std::rotr(e, 11) ^ std::rotr(e, 25)) + ((e & f) ^ (~e & g)) + constants[i] + words[i];
                const auto second = (std::rotr(a, 2) ^ std::rotr(a, 13) ^ std::rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
                h                 = g;
                g                 = f;
                f                 = e;
                e                 = d + first;
                d                 = c;
                c                 = b;
                b                 = a;
                a                 = first + second;
            }
            state[0] += a;
            state[1] += b;
            state[2] += c;
            state[3] += d;
            state[4] += e;
            state[5] += f;
            state[6] += g;
            state[7] += h;
        }
    };
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
            std::vector<std::filesystem::path> paths;
            for (const auto& entry : std::filesystem::recursive_directory_iterator(folder)) {
                auto ext = path_utf8(entry.path().extension());
                std::ranges::transform(ext, ext.begin(), [](unsigned char c) { return char(std::tolower(c)); });
                if (entry.is_regular_file() && (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp" || ext == ".tga")) paths.push_back(entry.path());
            }
            std::ranges::sort(paths);
            for (const auto& path : paths) {
                Image image = load_image(path);
                if (!((image.width == 1024 && image.height == 1536) || (image.width == 1536 && image.height == 1024) || (image.width == 1024 && image.height == 1024))) throw std::runtime_error("Unsupported original size: " + path_utf8(path));
                Sha256 hash;
                hash.update(image.rgb);
                std::string hex;
                for (auto byte : hash.finish()) hex += std::format("{:02x}", byte);
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
        Sha256 hash;
        for (const auto& folder : categories(root)) {
            auto label = folder.filename().generic_u8string();
            hash.update({reinterpret_cast<const unsigned char*>(label.data()), label.size() + 1});
            std::vector<std::filesystem::path> paths;
            for (const auto& entry : std::filesystem::recursive_directory_iterator(folder)) {
                auto ext = path_utf8(entry.path().extension());
                std::ranges::transform(ext, ext.begin(), [](unsigned char c) { return char(std::tolower(c)); });
                if (entry.is_regular_file() && (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp" || ext == ".tga")) paths.push_back(entry.path());
            }
            std::ranges::sort(paths);
            for (const auto& path : paths) {
                auto relative = std::filesystem::relative(path, root).generic_u8string();
                hash.update({reinterpret_cast<const unsigned char*>(relative.data()), relative.size() + 1});
                Mapping file(path);
                Sha256 content;
                content.update({static_cast<const unsigned char*>(file.data), file.size});
                hash.update(content.finish());
            }
        }
        std::string result;
        for (auto byte : hash.finish()) result += std::format("{:02x}", byte);
        return result;
    }
} // namespace classifier
