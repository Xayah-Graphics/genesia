include_guard(GLOBAL)
genesia_require_dependency(glfw)
FetchContent_MakeAvailable(imgui)
add_library(genesia_imgui STATIC
        "${imgui_SOURCE_DIR}/imgui.cpp"
        "${imgui_SOURCE_DIR}/imgui_draw.cpp"
        "${imgui_SOURCE_DIR}/imgui_tables.cpp"
        "${imgui_SOURCE_DIR}/imgui_widgets.cpp"
        "${imgui_SOURCE_DIR}/misc/cpp/imgui_stdlib.cpp"
        "${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp")
add_library(genesia::imgui ALIAS genesia_imgui)
target_include_directories(genesia_imgui PUBLIC "${imgui_SOURCE_DIR}" "${imgui_SOURCE_DIR}/backends")
target_compile_definitions(genesia_imgui PRIVATE GLFW_INCLUDE_NONE)
target_link_libraries(genesia_imgui PUBLIC glfw)
