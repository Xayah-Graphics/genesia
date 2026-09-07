include_guard(GLOBAL)

FetchContent_MakeAvailable(stb)

add_library(genesia-stb INTERFACE)
add_library(genesia::stb ALIAS genesia-stb)
target_include_directories(genesia-stb SYSTEM INTERFACE "${stb_SOURCE_DIR}")
