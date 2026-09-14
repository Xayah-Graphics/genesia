module genesia.runtime.catalog;
import genesia.data.transactions;
import std;
namespace genesia::runtime {
    CatalogState Catalog::load(const std::optional<std::string> selected) {
        std::filesystem::create_directories(project::raw);
        std::vector<std::string> names;
        if (selected) names.push_back(*selected);
        else
            for (const auto& entry : std::filesystem::directory_iterator{project::directory})
                if (entry.is_directory() && !files::utf8(entry.path().filename()).starts_with('.')) names.push_back(files::utf8(entry.path().filename()));
        CatalogState result;
        for (const auto& name : names) {
            if (std::ranges::any_of(index.roots, [&](const dataset::Root& root) { return root.all.key == name; })) continue;
            for (const auto& entry : std::filesystem::directory_iterator{project::directory / files::path(name)})
                if (entry.is_directory() && !files::utf8(entry.path().filename()).starts_with('.')) dataset::recover_moves(entry.path() / ".genesia" / "audit-moves.json");
            index.scan(name);
            auto loaded = root(name);
            result.roots.push_back(std::move(loaded.roots.front()));
            result.concepts.merge(loaded.concepts);
            result.classifiers.merge(loaded.classifiers);
            result.concept_errors.merge(loaded.concept_errors);
        }
        result.ready = !selected;
        return result;
    }
    CatalogState Catalog::describe(const std::string_view key, const bool membership) {
        CatalogState result;
        const std::string name{key};
        described.insert(name);
        try {
            const auto assigned   = dataset::read_concept(key);
            result.concepts[name] = assigned;
            if (assigned.type == dataset::ConceptType::classifier) {
                auto& info         = classifiers[name];
                const bool rebuild = membership && info.inspected;
                if (membership) info.inspected = false;
                info.key      = assigned.key;
                info.root     = assigned.path;
                info.training = training::read_state(assigned.path);
                info.model    = models::find(key);
                if (rebuild) inspect(key);
                result.classifiers[name] = info;
            } else classifiers.erase(name);
        } catch (const std::exception& failure) {
            result.concept_errors[name] = failure.what();
        }
        return result;
    }
    training::TrainingData& Catalog::inspect(const std::string_view key) {
        const std::string name{key};
        auto found = classifiers.find(name);
        if (found != classifiers.end() && found->second.inspected) return found->second;
        const auto assigned = dataset::read_concept(key);
        const auto root_key = files::utf8(*files::path(key).begin());
        const auto& source  = *std::ranges::find(index.roots, root_key, [](const dataset::Root& root) { return root.all.key; });
        return classifiers.insert_or_assign(name, training::inspect(assigned, source)).first->second;
    }
    CatalogState Catalog::root(const std::string_view key) {
        CatalogState result;
        const auto& source = *std::ranges::find(index.roots, key, [](const dataset::Root& root) { return root.all.key; });
        result.roots.push_back(source);
        for (const auto& collection : source.concepts) {
            if (described.contains(collection.key)) continue;
            auto info = describe(collection.key);
            result.concepts.merge(info.concepts);
            result.classifiers.merge(info.classifiers);
            result.concept_errors.merge(info.concept_errors);
        }
        return result;
    }
} // namespace genesia::runtime
