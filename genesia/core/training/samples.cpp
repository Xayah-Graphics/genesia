module;
#include <nlohmann/json.hpp>
module genesia.training.samples;
import genesia.io.hash;
import genesia.io.files;
import std;
namespace genesia::training {
    void to_json(nlohmann::json& json, const Record& value) {
        json = {{"id", value.id}, {"path", value.path}, {"split", value.split}, {"group", value.group}, {"label", value.label}, {"width", value.width}, {"height", value.height}, {"rgb_sha", value.rgb_sha}, {"phash", value.phash}, {"members", value.members}};
    }
    void from_json(const nlohmann::json& json, Record& value) {
        json.at("id").get_to(value.id);
        json.at("path").get_to(value.path);
        json.at("split").get_to(value.split);
        json.at("group").get_to(value.group);
        json.at("label").get_to(value.label);
        json.at("width").get_to(value.width);
        json.at("height").get_to(value.height);
        json.at("rgb_sha").get_to(value.rgb_sha);
        json.at("phash").get_to(value.phash);
        json.at("members").get_to(value.members);
    }
    void to_json(nlohmann::json& json, const Snapshot& value) {
        json = {{"version", 1}, {"fingerprint", value.fingerprint}, {"classes", value.classes}, {"records", value.records}};
    }
    void from_json(const nlohmann::json& json, Snapshot& value) {
        if (json.at("version") != 1) throw std::runtime_error{"Unsupported training snapshot format"};
        json.at("fingerprint").get_to(value.fingerprint);
        json.at("classes").get_to(value.classes);
        json.at("records").get_to(value.records);
    }

    TrainingData inspect(const dataset::Concept& source, const dataset::Root& root) {
        TrainingData result;
        result.inspected = true;
        result.training  = read_state(source.path);
        result.model     = models::find(source.key);
        result.key       = source.key;
        result.root      = source.path;
        if (!root.ready) result.issue = root.error.empty() ? "Root has independent image copies" : root.error;
        for (const auto& entry : std::filesystem::directory_iterator(result.root))
            if (entry.is_directory() && !files::utf8(entry.path().filename()).starts_with('.')) result.classes.push_back(files::utf8(entry.path().filename()));
        std::ranges::sort(result.classes);
        std::vector<std::pair<std::string, std::string>> members;
        std::map<std::string, std::size_t> resources;
        for (const auto& file : root.files) {
            const auto relative = file.path.lexically_relative(result.root);
            if (relative.empty() || *relative.begin() == "..") continue;
            members.push_back({files::utf8(relative), file.sha});
            if (std::distance(relative.begin(), relative.end()) < 2) {
                result.issue = "Unclassified image: " + files::utf8(file.path);
                continue;
            }
            const auto name  = files::utf8(*relative.begin());
            const auto label = static_cast<int>(std::ranges::find(result.classes, name) - result.classes.begin());
            if (!((file.width == 1024 && file.height == 1536) || (file.width == 1536 && file.height == 1024) || (file.width == 1024 && file.height == 1024))) result.issue = "Unsupported classifier size: " + files::utf8(file.path);
            const auto [found, added] = resources.emplace(file.sha, result.samples.size());
            if (added) result.samples.push_back({file, {file.path}, label});
            else {
                auto& sample = result.samples[found->second];
                if (sample.label != label) result.issue = "Conflicting class labels for " + file.sha + ": " + files::utf8(sample.file.path) + " / " + files::utf8(file.path);
                sample.paths.push_back(file.path);
            }
        }
        result.counts.resize(result.classes.size());
        for (const auto& sample : result.samples) ++result.counts[sample.label];
        if (result.classes.size() < 2) result.training_issue = "A classifier needs at least two nonempty categories";
        for (std::size_t i = 0; i < result.classes.size(); ++i)
            if (!result.counts[i]) result.training_issue = "Empty category: " + result.classes[i];
        std::ranges::sort(result.samples, {}, [](const Sample& sample) { return sample.file.path; });
        std::ranges::sort(members);
        const auto text    = nlohmann::json{{"classes", result.classes}, {"members", members}}.dump();
        result.fingerprint = sha256({reinterpret_cast<const unsigned char*>(text.data()), text.size()});
        return result;
    }
    Dataset::Dataset(const std::filesystem::path& path) : root{path}, snapshot{files::read_json(root / "snapshot.json").get<Snapshot>()} {
        const auto layout = files::read_json(root / "pixels.json");
        if (layout.at("version") != 1) throw std::runtime_error{"Unsupported RGB cache layout"};
        for (std::size_t i = 0; i < snapshot.records.size(); ++i) {
            const auto& record = snapshot.records[i];
            if (record.split == "train") train.push_back(int(i));
            else if (record.split == "val") val.push_back(int(i));
            else throw std::runtime_error{"Unknown training partition: " + record.split};
            const auto& entry = layout.at("images").at(record.id);
            auto& pixel       = pixels.emplace_back(entry.at("cache").get<std::string>(), entry.at("offset").get<std::uint64_t>());
            if (!caches.contains(pixel.cache)) caches[pixel.cache] = std::make_unique<files::Mapping>(root / pixel.cache);
        }
    }
    const unsigned char* Dataset::rgb(const int index) const {
        const auto& pixel = pixels[index];
        return static_cast<unsigned char*>(caches.at(pixel.cache)->data) + pixel.offset;
    }

    void prepare(const TrainingData& source, const std::filesystem::path& output, const std::atomic_bool& interrupted, const std::function<void(std::size_t, std::size_t)>& progress) {
        if (!source.issue.empty()) throw std::runtime_error{source.issue};
        if (!source.training_issue.empty()) throw std::runtime_error{source.training_issue};
        if (!std::filesystem::create_directories(output)) throw std::runtime_error{"Snapshot output already exists"};
        std::vector<Record> records;
        nlohmann::json pixels = nlohmann::json::object();
        const auto& classes   = source.classes;
        std::vector<std::string> hashes;
        std::vector<std::uint64_t> phashes;
        std::map<std::string, std::ofstream> caches;
        for (const auto& sample : source.samples) {
            if (interrupted.load()) throw runtime::Stopped{};
            const auto& path = sample.file.path;
            const auto image = read_image(path);
            const auto hex   = sha256(image.pixels);
            hashes.push_back(hex);
            std::array<double, 1024> small{};
            for (int y = 0; y < 32; ++y)
                for (int x = 0; x < 32; ++x) {
                    double sum = 0;
                    int count  = 0;
                    for (int iy = y * image.height / 32; iy < (y + 1) * image.height / 32; ++iy)
                        for (int ix = x * image.width / 32; ix < (x + 1) * image.width / 32; ++ix) {
                            auto p = image.pixels.data() + (std::size_t(iy) * image.width + ix) * 3;
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

            const auto cache = std::format("{}x{}.rgb", image.width, image.height);
            if (!caches.contains(cache)) caches[cache] = std::ofstream(output / cache, std::ios::binary);
            auto& file = caches.at(cache);
            file.exceptions(std::ios::badbit | std::ios::failbit);
            const std::uint64_t offset = file.tellp();
            file.write(reinterpret_cast<const char*>(image.pixels.data()), image.pixels.size());
            std::vector<std::string> members;
            for (const auto& member : sample.paths) members.push_back(files::utf8(member.lexically_relative(source.root)));
            records.push_back({sample.file.sha, files::utf8(path.lexically_relative(source.root)), {}, {}, sample.label, image.width, image.height, hex, phash, std::move(members)});
            pixels[sample.file.sha] = {{"cache", cache}, {"offset", offset}};
            progress(records.size(), source.samples.size());
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
        for (const auto& r : records) ++total[r.label];
        for (int key : order) {
            std::vector<int> count(classes.size(), 0);
            for (int index : groups[key]) ++count[records[index].label];
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
                records[index].split = validation ? "val" : "train";
                records[index].group = std::to_string(key);
            }
            if (validation)
                for (int c = 0; c < int(classes.size()); ++c) val[c] += count[c];
        }

        for (int label = 0; label < static_cast<int>(classes.size()); ++label) {
            for (const std::string_view partition : {"train", "val"})
                if (!std::ranges::any_of(records, [&](const auto& record) { return record.label == label && record.split == partition; })) throw std::runtime_error{"Category needs independent train and validation groups: " + classes[label]};
        }
        files::write_json(output / "snapshot.json", Snapshot{source.fingerprint, classes, std::move(records)});
        files::write_json(output / "pixels.json", {{"version", 1}, {"images", std::move(pixels)}});
    }
} // namespace genesia::training
