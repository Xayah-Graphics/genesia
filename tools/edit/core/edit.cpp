module;
#if defined(_WIN32)
#include <windows.h>
#else
#include <cerrno>
#include <csignal>
#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
#endif
#include <nlohmann/json.hpp>
module edit.processing;
import tools.files;
import std;

namespace edit {
    namespace {
        struct Stopped final {};
#if !defined(_WIN32)
        struct Descriptor final {
            int value{-1};
            ~Descriptor() { if (value >= 0) close(value); }
        };
#endif
        struct Worker final {
#if defined(_WIN32)
            std::unique_ptr<void, decltype(&CloseHandle)> job{nullptr, CloseHandle};
            std::unique_ptr<void, decltype(&CloseHandle)> process{nullptr, CloseHandle};
            std::unique_ptr<void, decltype(&CloseHandle)> input{nullptr, CloseHandle};
            std::unique_ptr<void, decltype(&CloseHandle)> output{nullptr, CloseHandle};
#else
            pid_t process{-1};
            Descriptor input, output;
#endif
            std::optional<int> exit_code;
            ~Worker();
            void start(const std::filesystem::path& python, const std::filesystem::path& directory);
            bool send(std::string_view message);
            std::string receive();
            void poll();
        };

        Worker::~Worker() {
#if defined(_WIN32)
            job.reset();
            if (process) {
                TerminateProcess(process.get(), 1);
                WaitForSingleObject(process.get(), INFINITE);
            }
#else
            if (process > 0) {
                kill(-process, SIGKILL);
                int status{};
                while (waitpid(process, &status, 0) < 0 && errno == EINTR) {}
            }
#endif
        }
        void Worker::start(const std::filesystem::path& python, const std::filesystem::path& directory) {
#if defined(_WIN32)
            SECURITY_ATTRIBUTES security{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
            HANDLE read{}, write{};
            if (!CreatePipe(&read, &write, &security, 0)) throw std::system_error{int(GetLastError()), std::system_category(), "Create Python input pipe"};
            std::unique_ptr<void, decltype(&CloseHandle)> child_input{read, CloseHandle};
            input.reset(write);
            if (!SetHandleInformation(input.get(), HANDLE_FLAG_INHERIT, 0)) throw std::system_error{int(GetLastError()), std::system_category(), "Set Python input handle"};
            if (!CreatePipe(&read, &write, &security, 0)) throw std::system_error{int(GetLastError()), std::system_category(), "Create Python output pipe"};
            output.reset(read);
            std::unique_ptr<void, decltype(&CloseHandle)> child_output{write, CloseHandle};
            if (!SetHandleInformation(output.get(), HANDLE_FLAG_INHERIT, 0)) throw std::system_error{int(GetLastError()), std::system_category(), "Set Python output handle"};
            const auto log = CreateFileW((directory / "stderr.log").c_str(), GENERIC_WRITE, FILE_SHARE_READ, &security, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
            if (log == INVALID_HANDLE_VALUE) throw std::system_error{int(GetLastError()), std::system_category(), "Create Python log"};
            std::unique_ptr<void, decltype(&CloseHandle)> child_error{log, CloseHandle};
            job.reset(CreateJobObjectW(nullptr, nullptr));
            if (!job) throw std::system_error{int(GetLastError()), std::system_category(), "Create Python job"};
            JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
            limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            if (!SetInformationJobObject(job.get(), JobObjectExtendedLimitInformation, &limits, sizeof(limits))) throw std::system_error{int(GetLastError()), std::system_category(), "Configure Python job"};
            STARTUPINFOW startup{sizeof(STARTUPINFOW)};
            startup.dwFlags    = STARTF_USESTDHANDLES;
            startup.hStdInput  = child_input.get();
            startup.hStdOutput = child_output.get();
            startup.hStdError  = child_error.get();
            auto command = std::format(L"\"{}\" -X utf8 -B -u \"{}\"", python.native(), std::filesystem::path{EDIT_WORKER_SCRIPT}.native());
            PROCESS_INFORMATION created{};
            if (!CreateProcessW(python.c_str(), command.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW | CREATE_SUSPENDED, nullptr, directory.c_str(), &startup, &created)) throw std::system_error{int(GetLastError()), std::system_category(), "Start Edit Python worker"};
            process.reset(created.hProcess);
            std::unique_ptr<void, decltype(&CloseHandle)> thread{created.hThread, CloseHandle};
            if (!AssignProcessToJobObject(job.get(), process.get())) throw std::system_error{int(GetLastError()), std::system_category(), "Attach Python worker to job"};
            if (ResumeThread(thread.get()) == DWORD(-1)) throw std::system_error{int(GetLastError()), std::system_category(), "Resume Python worker"};
#else
            std::signal(SIGPIPE, SIG_IGN);
            std::array<int, 2> pipe{};
            if (pipe2(pipe.data(), O_CLOEXEC) < 0) throw std::system_error{errno, std::generic_category(), "Create Python input pipe"};
            Descriptor child_input{pipe[0]};
            input.value = pipe[1];
            if (pipe2(pipe.data(), O_CLOEXEC) < 0) throw std::system_error{errno, std::generic_category(), "Create Python output pipe"};
            output.value = pipe[0];
            Descriptor child_output{pipe[1]};
            Descriptor child_error{open((directory / "stderr.log").c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600)};
            if (child_error.value < 0) throw std::system_error{errno, std::generic_category(), "Create Python log"};
            if (fcntl(output.value, F_SETFL, O_NONBLOCK) < 0) throw std::system_error{errno, std::generic_category(), "Configure Python output pipe"};
            posix_spawn_file_actions_t actions{};
            if (const int error = posix_spawn_file_actions_init(&actions)) throw std::system_error{error, std::generic_category(), "Create Python file actions"};
            const std::unique_ptr<posix_spawn_file_actions_t, decltype(&posix_spawn_file_actions_destroy)> release_actions{&actions, posix_spawn_file_actions_destroy};
            for (const auto [source, target] : {std::pair{child_input.value, STDIN_FILENO}, std::pair{child_output.value, STDOUT_FILENO}, std::pair{child_error.value, STDERR_FILENO}})
                if (const int error = posix_spawn_file_actions_adddup2(&actions, source, target)) throw std::system_error{error, std::generic_category(), "Redirect Python stream"};
            posix_spawnattr_t attributes{};
            if (const int error = posix_spawnattr_init(&attributes)) throw std::system_error{error, std::generic_category(), "Create Python spawn attributes"};
            const std::unique_ptr<posix_spawnattr_t, decltype(&posix_spawnattr_destroy)> release_attributes{&attributes, posix_spawnattr_destroy};
            if (const int error = posix_spawnattr_setflags(&attributes, POSIX_SPAWN_SETPGROUP)) throw std::system_error{error, std::generic_category(), "Configure Python process group"};
            if (const int error = posix_spawnattr_setpgroup(&attributes, 0)) throw std::system_error{error, std::generic_category(), "Set Python process group"};
            const std::string script{EDIT_WORKER_SCRIPT};
            const std::array arguments{const_cast<char*>(python.c_str()), const_cast<char*>("-X"), const_cast<char*>("utf8"), const_cast<char*>("-B"), const_cast<char*>("-u"), const_cast<char*>(script.c_str()), static_cast<char*>(nullptr)};
            if (const int error = posix_spawn(&process, python.c_str(), &actions, &attributes, arguments.data(), environ)) throw std::system_error{error, std::generic_category(), "Start Edit Python worker"};
#endif
        }
        bool Worker::send(std::string_view message) {
            while (!message.empty()) {
#if defined(_WIN32)
                DWORD written{};
                if (!WriteFile(input.get(), message.data(), DWORD(std::min(message.size(), std::size_t(MAXDWORD))), &written, nullptr)) {
                    const auto error = GetLastError();
                    if (error == ERROR_BROKEN_PIPE || error == ERROR_NO_DATA) return false;
                    throw std::system_error{int(error), std::system_category(), "Write to Python worker"};
                }
#else
                const auto written = write(input.value, message.data(), message.size());
                if (written < 0) {
                    if (errno == EINTR) continue;
                    if (errno == EPIPE) return false;
                    throw std::system_error{errno, std::generic_category(), "Write to Python worker"};
                }
#endif
                message.remove_prefix(std::size_t(written));
            }
            return true;
        }
        std::string Worker::receive() {
            std::array<char, 65536> bytes;
#if defined(_WIN32)
            DWORD available{}, read{};
            if (!PeekNamedPipe(output.get(), nullptr, 0, nullptr, &available, nullptr)) {
                const auto error = GetLastError();
                if (error == ERROR_BROKEN_PIPE) return {};
                throw std::system_error{int(error), std::system_category(), "Poll Python output"};
            }
            if (!available) return {};
            if (!ReadFile(output.get(), bytes.data(), DWORD(std::min(bytes.size(), std::size_t(available))), &read, nullptr)) throw std::system_error{int(GetLastError()), std::system_category(), "Read Python output"};
#else
            const auto read = ::read(output.value, bytes.data(), bytes.size());
            if (read < 0) {
                if (errno == EAGAIN || errno == EINTR) return {};
                throw std::system_error{errno, std::generic_category(), "Read Python output"};
            }
#endif
            return {bytes.data(), std::size_t(read)};
        }
        void Worker::poll() {
            if (exit_code) return;
#if defined(_WIN32)
            const auto status = WaitForSingleObject(process.get(), 0);
            if (status == WAIT_TIMEOUT) return;
            if (status == WAIT_FAILED) throw std::system_error{int(GetLastError()), std::system_category(), "Wait for Python worker"};
            DWORD code{};
            if (!GetExitCodeProcess(process.get(), &code)) throw std::system_error{int(GetLastError()), std::system_category(), "Read Python exit code"};
            exit_code = int(code);
#else
            int status{};
            const auto finished = waitpid(process, &status, WNOHANG);
            if (!finished) return;
            if (finished < 0) {
                if (errno == EINTR) return;
                throw std::system_error{errno, std::generic_category(), "Wait for Python worker"};
            }
            exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
            kill(-process, SIGKILL);
            process = -1;
#endif
        }
    } // namespace

    Result process(const Request& options, const std::atomic_bool& interrupted, const std::function<void(const Progress&)>& progress) {
        Result result;
        std::filesystem::path working, pending_output, current_file;
        bool output_created{};
        try {
            progress({Stage::preparing});
            if (interrupted) throw Stopped{};
            if (options.prompt.find_first_not_of(" \t\r\n") == std::string::npos) throw std::runtime_error{"Enter a positive prompt before editing images"};
            const auto input     = std::filesystem::canonical(options.input);
            const bool directory = std::filesystem::is_directory(input);
            result.output        = options.output ? std::filesystem::absolute(*options.output).lexically_normal() : (directory ? input : input.parent_path()) / "fix";
            std::vector<std::filesystem::path> images;
            if (directory) {
                for (const auto& entry : std::filesystem::directory_iterator{input}) {
                    if (interrupted) throw Stopped{};
                    if (!entry.is_regular_file()) continue;
                    auto extension = tools::files::utf8(entry.path().extension());
                    std::ranges::transform(extension, extension.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
                    if (extension == ".png") images.push_back(entry.path());
                }
            } else {
                auto extension = tools::files::utf8(input.extension());
                std::ranges::transform(extension, extension.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
                if (extension != ".png") throw std::runtime_error{"Edit accepts PNG images"};
                images.push_back(input);
            }
            if (images.empty()) throw std::runtime_error{"No PNG images directly inside " + tools::files::utf8(input)};
            std::ranges::sort(images);
            result.total = images.size();
            progress({Stage::preparing, 0, result.total});
            const auto runtime = tools::files::read_json(std::filesystem::path{EDIT_ASSET_DIRECTORY} / "edit" / "runtime.json");
            std::random_device random;
            const auto batch = std::format("edit-{:016x}-{:016x}", std::chrono::system_clock::now().time_since_epoch().count(), std::uniform_int_distribution<std::uint64_t>{}(random));
            std::filesystem::create_directories(EDIT_RUNTIME_DIRECTORY);
            const auto directory_path = std::filesystem::path{EDIT_RUNTIME_DIRECTORY} / batch;
            if (!std::filesystem::create_directory(directory_path)) throw std::runtime_error{"Worker directory already exists: " + tools::files::utf8(directory_path)};
            working = directory_path;
            auto inputs = nlohmann::json::array(), references = nlohmann::json::array();
            for (const auto& image : images) inputs.push_back(tools::files::utf8(image));
            for (const auto& reference : options.references) references.push_back(tools::files::utf8(std::filesystem::canonical(reference)));
            const nlohmann::json request{{"runtime", runtime}, {"working", tools::files::utf8(working)}, {"batch", batch}, {"images", std::move(inputs)}, {"references", std::move(references)}, {"prompt", options.prompt}};
            if (interrupted) throw Stopped{};
            Worker worker;
            worker.start(tools::files::path(runtime.at("python").get<std::string>()), working);
            if (!worker.send(request.dump() + "\n")) throw std::runtime_error{"Python worker closed its input before reading the batch"};
            std::string buffered;
            bool finished{};
            std::optional<std::chrono::steady_clock::time_point> stop_deadline;
            for (;;) {
                const auto data = worker.receive();
                buffered += data;
                for (auto end = buffered.find('\n'); end != std::string::npos; end = buffered.find('\n')) {
                    const auto event = nlohmann::json::parse(buffered.substr(0, end));
                    buffered.erase(0, end + 1);
                    const auto type = event.at("type").get<std::string>();
                    if (type == "progress") {
                        current_file = tools::files::path(event.at("file").get<std::string>());
                        progress({stop_deadline ? Stage::stopping : event.at("stage") == "preparing" ? Stage::preparing : Stage::editing, result.completed, result.total, current_file});
                    } else if (type == "image") {
                        current_file = tools::files::path(event.at("file").get<std::string>());
                        output_created |= std::filesystem::create_directories(result.output);
                        pending_output = result.output / tools::files::path(batch + ".part");
                        std::filesystem::copy_file(tools::files::path(event.at("generated").get<std::string>()), pending_output);
                        const auto filename = current_file.stem().native() + tools::files::path(options.suffix).native() + current_file.extension().native();
                        tools::files::publish(pending_output, result.output / filename);
                        pending_output.clear();
                        ++result.completed;
                        progress({stop_deadline ? Stage::stopping : Stage::editing, result.completed, result.total, current_file});
                    } else if (type == "complete") finished = true;
                    else if (type == "stopped") {
                        finished       = true;
                        result.stopped = true;
                    } else if (type == "error") {
                        finished     = true;
                        result.error = event.at("error").get<std::string>();
                    } else throw std::runtime_error{"Unknown Python worker event: " + type};
                }
                worker.poll();
                if (interrupted && !stop_deadline && !worker.exit_code) {
                    progress({Stage::stopping, result.completed, result.total, current_file});
                    worker.send("stop\n");
                    stop_deadline = std::chrono::steady_clock::now() + std::chrono::seconds{60};
                }
                if (worker.exit_code && data.empty()) break;
                if (stop_deadline && !worker.exit_code && std::chrono::steady_clock::now() >= *stop_deadline) throw std::runtime_error{"Python worker did not stop within 60 seconds"};
                if (!worker.exit_code) std::this_thread::sleep_for(std::chrono::milliseconds{20});
            }
            if ((!finished || *worker.exit_code != 0) && result.error.empty()) throw std::runtime_error{std::format("Python worker exited with code {}\n{}", *worker.exit_code, tools::files::read_text(working / "stderr.log"))};
            result.stopped = result.stopped && result.completed != result.total;
        } catch (const Stopped&) {
            result.stopped = true;
        } catch (const std::exception& failure) {
            result.error = current_file.empty() ? failure.what() : std::format("{}\n{}", tools::files::utf8(current_file), failure.what());
        }
        for (const auto& path : {pending_output, working}) {
            if (path.empty()) continue;
            try {
                std::filesystem::remove_all(path);
            } catch (const std::exception& failure) {
                if (!result.error.empty()) result.error += '\n';
                result.error += failure.what();
            }
        }
        if (output_created && !result.completed) {
            try {
                std::filesystem::remove(result.output);
            } catch (const std::exception& failure) {
                if (!result.error.empty()) result.error += '\n';
                result.error += failure.what();
            }
        }
        if (!working.empty()) {
            try {
                if (std::filesystem::is_empty(EDIT_RUNTIME_DIRECTORY)) std::filesystem::remove(EDIT_RUNTIME_DIRECTORY);
            } catch (const std::exception& failure) {
                if (!result.error.empty()) result.error += '\n';
                result.error += failure.what();
            }
        }
        return result;
    }
} // namespace edit
