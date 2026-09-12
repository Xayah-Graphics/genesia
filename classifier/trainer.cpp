#include <csignal>
#include <nlohmann/json.hpp>
import std;
import classifier.training;
namespace {
static_assert(std::atomic_bool::is_always_lock_free);
std::atomic_bool interrupted=false;
void interrupt_training(int signal) {
    std::signal(signal,interrupt_training);
    interrupted.store(true,std::memory_order_relaxed);
}
}
int main(int argc,char** argv) {
    try {
        classifier::TrainingOptions options;
        const std::map<std::string,std::string> integers{{"--batch","physical_batch"},{"--effective-batch","effective_batch"},{"--freeze-steps","head_only_steps"},{"--warmup-steps","warmup_steps"},{"--eval-interval","eval_interval"},{"--save-interval","save_interval"},{"--log-interval","log_interval"}};
        const std::map<std::string,std::string> decimals{{"--backbone-lr","backbone_lr"},{"--head-lr","head_lr"},{"--frozen-head-lr","head_only_lr"},{"--weight-decay","weight_decay"},{"--clip-norm","clip_norm"}};
        for (int i=1;i<argc;++i) {
            std::string option=argv[i];
            if (option=="--help") {
                std::println("classifier-train --dataset <YES/NO folders root> --steps <cumulative target>\n  --restart  Start a new task and archive existing results\n  --batch 4 --effective-batch 64 --freeze-steps 25 --warmup-steps 10\n  --backbone-lr 3e-5 --head-lr 3e-4 --frozen-head-lr 1e-3\n  --weight-decay 0.01 --clip-norm 1 --seed 42\n  --eval-interval 25 --save-interval 25 --log-interval 5\nCtrl+C saves after the current complete update."); return 0;
            }
            if (option=="--restart") { options.restart=true; continue; }
            if (i+1==argc) throw std::runtime_error("Missing value for "+option);
            std::string value=argv[++i];
            if (option=="--dataset") options.dataset=std::filesystem::path(std::u8string(value.begin(),value.end()));
            else if (option=="--steps") options.steps=std::stoi(value);
            else if (option=="--seed") options.overrides["seed"]=std::stoull(value);
            else if (integers.contains(option)) options.overrides[integers.at(option)]=std::stoi(value);
            else if (decimals.contains(option)) options.overrides[decimals.at(option)]=std::stof(value);
            else throw std::runtime_error("Unknown option: "+option);
        }
        if (options.dataset.empty()) throw std::runtime_error("Specify --dataset; use --help for options");
        std::signal(SIGINT,interrupt_training);
        std::signal(SIGTERM,interrupt_training);
#ifdef SIGBREAK
        std::signal(SIGBREAK,interrupt_training);
#endif
        classifier::train(options,interrupted); return 0;
    } catch (const std::exception& e) { std::println(stderr,"{}",e.what()); return 1; }
}
