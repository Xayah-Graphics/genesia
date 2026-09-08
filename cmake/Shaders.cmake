set(GENESIA_SHADER_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/generated/editor-shaders")
set(GENESIA_SHADER_BINARIES)
foreach (stage IN ITEMS vertex fragment)
    set(output "${GENESIA_SHADER_OUTPUT_DIRECTORY}/imgui_${stage}.spv")
    add_custom_command(OUTPUT "${output}"
            COMMAND "${CMAKE_COMMAND}" -E make_directory "${GENESIA_SHADER_OUTPUT_DIRECTORY}"
            COMMAND "${GENESIA_SLANG_COMPILER}" "${PROJECT_SOURCE_DIR}/genesia/editor/shaders/imgui.slang"
            -target spirv -profile spirv_1_6 -entry imgui_${stage} -stage ${stage} -emit-spirv-directly
            -capability SPIRV_1_6+spvDescriptorHeapEXT -spirv-unified-descriptor-heap-stride -fvk-use-entrypoint-name -fvk-use-c-layout -matrix-layout-row-major -O2
            -o "${output}"
            DEPENDS "${PROJECT_SOURCE_DIR}/genesia/editor/shaders/imgui.slang" VERBATIM)
    list(APPEND GENESIA_SHADER_BINARIES "${output}")
endforeach ()
add_custom_target(genesia_shaders DEPENDS ${GENESIA_SHADER_BINARIES})
