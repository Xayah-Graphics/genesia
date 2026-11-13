include_guard(GLOBAL)
FetchContent_Declare(aftermath
        URL "https://developer.nvidia.com/downloads/assets/tools/secure/nsight-aftermath-sdk/2026_3_0/windows_x64/NVIDIA_Nsight_Aftermath_SDK_2026.3.0.26197-windows_x64.zip"
        URL_HASH SHA256=e38136a60110199559b7365d3ea4ec0cb5588dc2b0f593877d864e0299659a3f
        SOURCE_SUBDIR genesia-unused
        SYSTEM EXCLUDE_FROM_ALL)
FetchContent_MakeAvailable(aftermath)
add_library(genesia::aftermath SHARED IMPORTED GLOBAL)
set_target_properties(genesia::aftermath PROPERTIES
        IMPORTED_IMPLIB "${aftermath_SOURCE_DIR}/lib/x64/GFSDK_Aftermath_Lib.x64.lib"
        IMPORTED_LOCATION "${aftermath_SOURCE_DIR}/lib/x64/GFSDK_Aftermath_Lib.x64.dll"
        INTERFACE_INCLUDE_DIRECTORIES "${aftermath_SOURCE_DIR}/include")
