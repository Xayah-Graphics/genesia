module;
#define STRICT_TYPED_ITEMIDS
#include <Windows.h>
#include <shellapi.h>
#include <shlobj.h>
#include <wrl/client.h>
#include <wrl/implements.h>
module tools.editor.platform.files;
import tools.files;
import std;
namespace tools::editor::shell {
    namespace {
        void check(const HRESULT result) {
            if (FAILED(result)) throw std::runtime_error{std::format("Windows file operation: 0x{:08X}", static_cast<unsigned long>(result))};
        }
        struct Apartment final {
            Apartment() {
                check(CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE));
            }
            ~Apartment() {
                CoUninitialize();
            }
        };
        struct RecycleSink final : Microsoft::WRL::RuntimeClass<Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::ClassicCom>, IFileOperationProgressSink> {
            bool recycled{};
            HRESULT result{E_ABORT};
            HRESULT STDMETHODCALLTYPE StartOperations() override {
                return S_OK;
            }
            HRESULT STDMETHODCALLTYPE FinishOperations(HRESULT) override {
                return S_OK;
            }
            HRESULT STDMETHODCALLTYPE PreRenameItem(DWORD, IShellItem*, LPCWSTR) override {
                return S_OK;
            }
            HRESULT STDMETHODCALLTYPE PostRenameItem(DWORD, IShellItem*, LPCWSTR, HRESULT, IShellItem*) override {
                return S_OK;
            }
            HRESULT STDMETHODCALLTYPE PreMoveItem(DWORD, IShellItem*, IShellItem*, LPCWSTR) override {
                return S_OK;
            }
            HRESULT STDMETHODCALLTYPE PostMoveItem(DWORD, IShellItem*, IShellItem*, LPCWSTR, HRESULT, IShellItem*) override {
                return S_OK;
            }
            HRESULT STDMETHODCALLTYPE PreCopyItem(DWORD, IShellItem*, IShellItem*, LPCWSTR) override {
                return S_OK;
            }
            HRESULT STDMETHODCALLTYPE PostCopyItem(DWORD, IShellItem*, IShellItem*, LPCWSTR, HRESULT, IShellItem*) override {
                return S_OK;
            }
            HRESULT STDMETHODCALLTYPE PreDeleteItem(DWORD flags, IShellItem*) override {
                return flags & TSF_DELETE_RECYCLE_IF_POSSIBLE ? S_OK : E_ABORT;
            }
            HRESULT STDMETHODCALLTYPE PostDeleteItem(DWORD, IShellItem*, HRESULT status, IShellItem* destination) override {
                result   = status;
                recycled = SUCCEEDED(status) && destination != nullptr;
                return S_OK;
            }
            HRESULT STDMETHODCALLTYPE PreNewItem(DWORD, IShellItem*, LPCWSTR) override {
                return S_OK;
            }
            HRESULT STDMETHODCALLTYPE PostNewItem(DWORD, IShellItem*, LPCWSTR, LPCWSTR, DWORD, HRESULT, IShellItem*) override {
                return S_OK;
            }
            HRESULT STDMETHODCALLTYPE UpdateProgress(UINT, UINT) override {
                return S_OK;
            }
            HRESULT STDMETHODCALLTYPE ResetTimer() override {
                return S_OK;
            }
            HRESULT STDMETHODCALLTYPE PauseTimer() override {
                return S_OK;
            }
            HRESULT STDMETHODCALLTYPE ResumeTimer() override {
                return S_OK;
            }
        };
    } // namespace
    void reveal(const std::filesystem::path& path) {
        PIDLIST_ABSOLUTE item{};
        check(SHParseDisplayName(path.c_str(), nullptr, &item, 0, nullptr));
        const std::unique_ptr<ITEMIDLIST_ABSOLUTE, decltype(&CoTaskMemFree)> location{item, &CoTaskMemFree};
        check(SHOpenFolderAndSelectItems(location.get(), 0, nullptr, 0));
    }
    void recycle(const std::filesystem::path& path) {
        const Apartment apartment;
        Microsoft::WRL::ComPtr<IShellItem> item;
        check(SHCreateItemFromParsingName(path.c_str(), nullptr, IID_PPV_ARGS(item.GetAddressOf())));
        Microsoft::WRL::ComPtr<IFileOperation> operation;
        check(CoCreateInstance(CLSID_FileOperation, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(operation.GetAddressOf())));
        check(operation->SetOperationFlags(FOFX_RECYCLEONDELETE | FOFX_ADDUNDORECORD | FOFX_EARLYFAILURE | FOF_NOERRORUI | FOF_SILENT | FOF_NOCONFIRMATION | FOF_NO_CONNECTED_ELEMENTS));
        const auto sink = Microsoft::WRL::Make<RecycleSink>();
        check(operation->DeleteItem(item.Get(), sink.Get()));
        const auto result = operation->PerformOperations();
        if (sink->recycled) return;
        const auto failure = FAILED(result) ? result : FAILED(sink->result) ? sink->result : E_ABORT;
        throw std::runtime_error{std::format("Could not recycle {}: 0x{:08X}", files::utf8(path), static_cast<unsigned long>(failure))};
    }
} // namespace tools::editor::shell
