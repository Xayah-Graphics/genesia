module;
#include <nlohmann/json.hpp>
module genesia.classifier.dataset;
import genesia.hash;
import genesia.files;
import std;
namespace genesia::classifier {
    nlohmann::json read_training(const dataset::Concept& source) {
        if (source.type != dataset::ConceptType::classifier) throw std::runtime_error{"Concept is not assigned to Classifier: " + source.key};
        const auto manifest = source.path / ".genesia" / "classifier" / "training.json";
        if (!std::filesystem::exists(manifest)) return nullptr;
        const auto training = files::read_json(manifest);
        if (training.at("version") != 2 || !source.locked) throw std::runtime_error{"Invalid classifier training state: " + source.key};
        return training;
    }
    TrainingData inspect(const dataset::Concept& source, const dataset::Root& root) {
        TrainingData result;
        result.training = read_training(source);
        result.key      = source.key;
        result.root     = source.path;
        if (!root.ready) result.issue = root.error.empty() ? "Root has independent image copies" : root.error;
        for (const auto& entry : std::filesystem::directory_iterator(result.root))
            if (entry.is_directory() && !files::utf8(entry.path().filename()).starts_with('.')) result.classes.push_back(files::utf8(entry.path().filename()));
        std::ranges::sort(result.classes);
        std::vector<std::pair<std::string, std::string>> members;
        std::map<std::string, std::size_t> resources;
        for (const auto& file : root.files) {
            const auto relative = file.path.lexically_relative(result.root);
            if (relative.empty() || *relative.begin() == "..") continue;
            if (std::distance(relative.begin(), relative.end()) < 2) {
                result.issue = "Unclassified image: " + files::utf8(file.path);
                continue;
            }
            const auto name  = files::utf8(*relative.begin());
            const auto label = static_cast<int>(std::ranges::find(result.classes, name) - result.classes.begin());
            members.push_back({files::utf8(relative), file.sha});
            if (!((file.width == 1024 && file.height == 1536) || (file.width == 1536 && file.height == 1024) || (file.width == 1024 && file.height == 1024))) result.issue = "Unsupported classifier size: " + files::utf8(file.path);
            const auto [found, added] = resources.emplace(file.sha, result.samples.size());
            if (added) result.samples.push_back({file, {file.path}, label});
            else {
                auto& sample = result.samples[found->second];
                if (sample.label != label) result.issue = "Conflicting class labels for " + file.sha + ": " + files::utf8(sample.file.path) + " / " + files::utf8(file.path);
                sample.paths.push_back(file.path);
            }
        }
        std::ranges::sort(result.samples, {}, [](const Sample& sample) { return sample.file.path; });
        std::ranges::sort(members);
        const auto text    = nlohmann::json{{"classes", result.classes}, {"members", members}}.dump();
        result.fingerprint = sha256({reinterpret_cast<const unsigned char*>(text.data()), text.size()});
        return result;
    }
    TrainingData inspect(const std::string_view key) {
        const auto source   = dataset::read_concept(key);
        const auto relative = files::path(source.key);
        const dataset::Lock files_lock{"image-moves"};
        dataset::Index index;
        index.scan(files::utf8(*relative.begin()));
        const auto& root      = index.roots.front();
        const auto collection = std::ranges::find(root.concepts, source.key, &dataset::Collection::key);
        if (collection == root.concepts.end()) throw std::runtime_error{"Concept not found: " + source.key};
        return inspect(source, root);
    }
    Dataset::Dataset(const std::filesystem::path& path) : root(std::filesystem::absolute(path)) {
        std::ifstream f(root / "dataset.json");
        f >> metadata;
        if (metadata.at("version") != 2) throw std::runtime_error{"Unsupported classifier snapshot"};
        classes = metadata["classes"].get<std::vector<std::string>>();
        for (const auto& r : metadata["records"]) {
            int index = int(records.size());
            records.push_back({r["id"], r["path"], r["split"], r.at("group").get<std::string>(), r["cache"], r["label"], r["width"], r["height"], r["offset"]});
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

    void prepare(const TrainingData& source, const std::filesystem::path& output, const std::atomic_bool& interrupted, const std::function<void(std::size_t, std::size_t)>& progress) {
        if (!source.issue.empty()) throw std::runtime_error{source.issue};
        if (source.classes.size() < 2) throw std::runtime_error{"A classifier needs at least two nonempty categories"};
        for (int label = 0; label < static_cast<int>(source.classes.size()); ++label)
            if (!std::ranges::any_of(source.samples, [&](const Sample& sample) { return sample.label == label; })) throw std::runtime_error{"Empty category: " + source.classes[label]};
        if (!std::filesystem::create_directories(output)) throw std::runtime_error{"Snapshot output already exists"};
        nlohmann::json records = nlohmann::json::array();
        const auto& classes    = source.classes;
        std::vector<std::string> hashes;
        std::vector<std::uint64_t> phashes;
        std::map<std::string, std::ofstream> caches;
        for (const auto& sample : source.samples) {
            if (interrupted.load()) throw std::runtime_error{"Snapshot preparation stopped"};
            const auto& path = sample.file.path;
            const auto image = read_image(path);
            if (files::digest(path) != sample.file.sha) throw std::runtime_error{"Image changed while preparing: " + files::utf8(path)};
            const auto hex = sha256(image.pixels);
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
            records.push_back({{"id", sample.file.sha}, {"path", files::utf8(path)}, {"label", sample.label}, {"width", image.width}, {"height", image.height}, {"cache", cache}, {"offset", offset}, {"sha256", hex}, {"file_sha256", sample.file.sha}, {"phash", std::format("{:016x}", phashes.back())}});
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

        for (int label = 0; label < static_cast<int>(classes.size()); ++label) {
            for (const std::string_view partition : {"train", "val"})
                if (!std::ranges::any_of(records, [&](const auto& record) { return record.at("label") == label && record.at("split").template get<std::string>() == partition; })) throw std::runtime_error{"Category needs independent train and validation groups: " + classes[label]};
        }
        files::write_json(output / "dataset.json", {{"version", 2}, {"fingerprint", source.fingerprint}, {"seed", 42}, {"classes", classes}, {"records", records}, {"grouping", "SHA256 RGB and 64-bit DCT pHash <= 5, transitive closure"}});
    }
} // namespace genesia::classifier
