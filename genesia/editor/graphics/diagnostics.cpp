module;
#include <Windows.h>
#include <DbgHelp.h>
#include <GFSDK_Aftermath_GpuCrashDump.h>
module genesia.editor.graphics.diagnostics;
import genesia.runtime.diagnostics;
import std;

namespace genesia::editor::graphics {
    Diagnostics::Diagnostics() : directory{diagnostics::log_path.parent_path() / diagnostics::log_path.stem()} {
        diagnostics::record("gpu.diagnostics.enable", "Aftermath 2026.3; checkpoints; resources; shader debug information");
        const auto result = GFSDK_Aftermath_EnableGpuCrashDumps(
            GFSDK_Aftermath_Version_API, GFSDK_Aftermath_GpuCrashDumpWatchedApiFlags_Vulkan, GFSDK_Aftermath_GpuCrashDumpFeatureFlags_DeferDebugInfoCallbacks, [](const void* data, std::uint32_t size, void* context) { static_cast<Diagnostics*>(context)->save("nv-gpudmp", data, size); }, [](const void* data, std::uint32_t size, void* context) { static_cast<Diagnostics*>(context)->save("nvdbg", data, size); },
            [](PFN_GFSDK_Aftermath_AddGpuCrashDumpDescription add, void*) {
                add(GFSDK_Aftermath_GpuCrashDumpDescriptionKey_ApplicationName, "Genesia");
                add(GFSDK_Aftermath_GpuCrashDumpDescriptionKey_ApplicationVersion, GENESIA_VERSION);
            },
            [](const void* marker, std::uint32_t, void*, PFN_GFSDK_Aftermath_ResolveMarker resolve) {
                // Every application checkpoint refers to a process-lifetime string literal.
                const auto text = static_cast<const char*>(marker);
                resolve(text, static_cast<std::uint32_t>(std::strlen(text) + 1));
            },
            this);
        if (result != GFSDK_Aftermath_Result_Success) throw std::runtime_error{std::format("Enable Aftermath GPU crash dumps failed: 0x{:x}", static_cast<unsigned int>(result))};
    }

    Diagnostics::~Diagnostics() {
        // Device members have already been destroyed; callbacks cannot outlive this object.
        const auto result = GFSDK_Aftermath_DisableGpuCrashDumps();
        if (result != GFSDK_Aftermath_Result_Success) diagnostics::record("gpu.diagnostics.disable.error", std::format("result=0x{:x}", static_cast<unsigned int>(result)));
    }

    void Diagnostics::capture(const std::string_view reason) {
        if (lost.exchange(true)) return;
        diagnostics::record("gpu.device_lost", reason);
        try {
            std::filesystem::create_directories(directory);
            const auto path   = directory / "first-device-lost.dmp";
            const auto handle = CreateFileW(path.c_str(), GENERIC_WRITE, FILE_SHARE_READ, nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
            if (handle == INVALID_HANDLE_VALUE) throw std::system_error{static_cast<int>(GetLastError()), std::system_category(), "Create device-lost dump"};
            const std::unique_ptr<void, decltype(&CloseHandle)> dump{handle, CloseHandle};
            CONTEXT context{};
            RtlCaptureContext(&context);
            // This synthetic exception identifies the capture site, not an original CPU fault.
            EXCEPTION_RECORD fault{};
            fault.ExceptionCode    = 0xe0000001;
            fault.ExceptionAddress = reinterpret_cast<void*>(context.Rip);
            EXCEPTION_POINTERS pointers{&fault, &context};
            MINIDUMP_EXCEPTION_INFORMATION exception{GetCurrentThreadId(), &pointers, false};
            const auto type = static_cast<MINIDUMP_TYPE>(MiniDumpWithFullMemory | MiniDumpWithFullMemoryInfo | MiniDumpWithThreadInfo | MiniDumpWithUnloadedModules);
            if (!MiniDumpWriteDump(GetCurrentProcess(), GetCurrentProcessId(), dump.get(), type, &exception, nullptr, nullptr)) throw std::system_error{static_cast<int>(GetLastError()), std::system_category(), "Write device-lost dump"};
            diagnostics::record("gpu.cpu_dump", path.string());

            std::array<wchar_t, 32768> executable{};
            const auto length = GetModuleFileNameW(nullptr, executable.data(), static_cast<DWORD>(executable.size()));
            if (!length || length == executable.size()) throw std::system_error{static_cast<int>(GetLastError()), std::system_category(), "Locate running executable"};
            const std::filesystem::path binary{std::wstring_view{executable.data(), length}};
            std::filesystem::copy_file(binary, directory / binary.filename());
            auto symbols = binary;
            symbols.replace_extension(".pdb");
            std::filesystem::copy_file(symbols, directory / symbols.filename());
            const std::array names{"imgui_vertex.spv", "imgui_fragment.spv"};
            for (std::size_t i = 0; i < shaders.size(); ++i) {
                std::ofstream file;
                file.exceptions(std::ios::badbit | std::ios::failbit);
                file.open(directory / names[i], std::ios::binary);
                file.write(reinterpret_cast<const char*>(shaders[i].data()), static_cast<std::streamsize>(shaders[i].size() * sizeof(std::uint32_t)));
                file.close();
            }
            diagnostics::record("gpu.artifacts", directory.string());
        } catch (const std::exception& error) {
            diagnostics::record("gpu.capture.error", error.what());
        }

        // NVIDIA requires keeping the device alive until crash-dump callbacks finish.
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{10};
        GFSDK_Aftermath_CrashDump_Status status{GFSDK_Aftermath_CrashDump_Status_NotStarted};
        for (;;) {
            const auto result = GFSDK_Aftermath_GetCrashDumpStatus(&status);
            if (result != GFSDK_Aftermath_Result_Success) {
                diagnostics::record("gpu.aftermath.status.error", std::format("result=0x{:x}", static_cast<unsigned int>(result)));
                break;
            }
            if (status == GFSDK_Aftermath_CrashDump_Status_Finished || status == GFSDK_Aftermath_CrashDump_Status_CollectingDataFailed) break;
            if (std::chrono::steady_clock::now() >= deadline) {
                diagnostics::record("gpu.aftermath.timeout", std::format("status={}", static_cast<unsigned int>(status)));
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds{50});
        }
        diagnostics::record("gpu.aftermath.status", std::format("status={}", static_cast<unsigned int>(status)));
        const std::lock_guard lock{mutex};
        if (callback_error) {
            try {
                std::rethrow_exception(callback_error);
            } catch (const std::exception& error) {
                diagnostics::record("gpu.aftermath.write.error", error.what());
            }
        }
    }

    void Diagnostics::save(const std::string_view extension, const void* data, const std::uint32_t size) noexcept {
        const std::lock_guard lock{mutex};
        try {
            std::filesystem::create_directories(directory);
            const auto path = directory / std::format("gpu-{}.{}", ++files, extension);
            std::ofstream file;
            file.exceptions(std::ios::badbit | std::ios::failbit);
            file.open(path, std::ios::binary);
            file.write(static_cast<const char*>(data), size);
            file.close();
            diagnostics::record("gpu.aftermath.file", path.string());
        } catch (...) {
            // Never unwind through an NVIDIA callback; report the failure in capture().
            if (!callback_error) callback_error = std::current_exception();
        }
    }
} // namespace genesia::editor::graphics
