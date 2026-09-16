include_guard(GLOBAL)

FetchContent_MakeAvailable(httplib)

add_library(genesia-httplib INTERFACE)
add_library(genesia::httplib ALIAS genesia-httplib)
target_include_directories(genesia-httplib SYSTEM INTERFACE "${httplib_SOURCE_DIR}")
target_link_libraries(genesia-httplib INTERFACE ws2_32)
