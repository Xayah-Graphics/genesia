option(CLASSIFIER_BUILD_EDITOR "Build the Classifier Editor" ${GENESIA_BUILD_UI})
option(EDIT_BUILD_EDITOR "Build the Edit Editor" ${GENESIA_BUILD_UI})
option(FOREGROUND_BUILD_EDITOR "Build the Foreground Editor" ${GENESIA_BUILD_UI})
if ((CLASSIFIER_BUILD_EDITOR OR EDIT_BUILD_EDITOR OR FOREGROUND_BUILD_EDITOR) AND NOT WIN32)
    message(FATAL_ERROR "Tool Editors support Windows only")
endif ()

genesia_require_dependency(httplib)

# Shared startup, file, image and model I/O for Headless and Editor builds.
add_library(tools_common STATIC)
target_sources(tools_common PRIVATE
        common/platform/application.cpp
        common/files.cpp
        common/hash.cpp
        common/images.cpp
        common/safetensors.cpp
        PUBLIC FILE_SET cxx_modules TYPE CXX_MODULES FILES
        common/platform/application.ixx
        common/files.ixx
        common/hash.ixx
        common/images.ixx
        common/safetensors.ixx)
target_link_libraries(tools_common PUBLIC nlohmann_json::nlohmann_json PRIVATE genesia::stb)
if (WIN32)
    target_link_libraries(tools_common PRIVATE user32)
endif ()

add_library(tools_compute STATIC)
target_sources(tools_compute PRIVATE
        common/compute/device.cpp
        common/compute/plans.cpp
        common/compute/inference.cpp
        common/compute/inference-kernels.cu
        PUBLIC FILE_SET cxx_modules TYPE CXX_MODULES FILES
        common/compute/device.ixx
        common/compute/plans.ixx
        common/compute/inference.ixx)
target_include_directories(tools_compute PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/common" "${GENESIA_CLION_CUDA_INCLUDE_DIRECTORY}")
target_compile_definitions(tools_compute PRIVATE
        TOOLS_CXX_COMPILER="${CMAKE_CXX_COMPILER_ID}-${CMAKE_CXX_COMPILER_VERSION}"
        TOOLS_PLATFORM="${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")
target_compile_options(tools_compute PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;--extended-lambda>")
target_link_libraries(tools_compute PUBLIC tools_common CCCL::libcudacxx CUDA::toolkit cuDNN::cuDNN CUDA::cublasLt CUDA::cudart
        PRIVATE genesia::cudnn-frontend CUDA::cuda_driver CUDA::nvrtc)

if (CLASSIFIER_BUILD_EDITOR OR EDIT_BUILD_EDITOR OR FOREGROUND_BUILD_EDITOR)
    genesia_require_dependency(vulkan)
    genesia_require_dependency(slang)
    genesia_require_dependency(imgui)

    set(TOOLS_SHADER_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated/editor-shaders")
    set(TOOLS_SHADERS)
    foreach (stage IN ITEMS vertex fragment)
        set(output "${TOOLS_SHADER_DIRECTORY}/imgui_${stage}.spv")
        add_custom_command(OUTPUT "${output}"
                COMMAND "${CMAKE_COMMAND}" -E make_directory "${TOOLS_SHADER_DIRECTORY}"
                COMMAND "${GENESIA_SLANG_COMPILER}" "${CMAKE_CURRENT_SOURCE_DIR}/common/editor/shaders/imgui.slang"
                -target spirv -profile spirv_1_6 -entry imgui_${stage} -stage ${stage} -emit-spirv-directly
                -capability SPIRV_1_6+spvDescriptorHeapEXT -spirv-unified-descriptor-heap-stride
                -fvk-use-entrypoint-name -fvk-use-c-layout -matrix-layout-row-major -O2 -o "${output}"
                DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/common/editor/shaders/imgui.slang" VERBATIM)
        list(APPEND TOOLS_SHADERS "${output}")
    endforeach ()
    add_custom_target(tools_shaders DEPENDS ${TOOLS_SHADERS})

    add_library(tools_editor STATIC)
    target_sources(tools_editor PRIVATE
            common/editor/platform/window.cpp
            common/editor/platform/files.cpp
            common/editor/graphics/device.cpp
            common/editor/graphics/resources.cpp
            common/editor/graphics/renderer.cpp
            common/editor/style.cpp
            PUBLIC FILE_SET cxx_modules TYPE CXX_MODULES FILES
            common/editor/platform/window.ixx
            common/editor/platform/files.ixx
            common/editor/graphics/device.ixx
            common/editor/graphics/resources.ixx
            common/editor/graphics/renderer.ixx
            common/editor/style.ixx)
    target_compile_definitions(tools_editor PUBLIC GLFW_INCLUDE_NONE GLFW_EXPOSE_NATIVE_WIN32
            PRIVATE TOOLS_SHADER_DIRECTORY="${TOOLS_SHADER_DIRECTORY}")
    target_link_libraries(tools_editor PUBLIC genesia::vulkan genesia::imgui
            PRIVATE tools_common dwmapi imm32 shell32 ole32)
    add_dependencies(tools_editor tools_shaders)
endif ()

add_executable(classifier classifier/main.cpp)
target_sources(classifier PRIVATE
        classifier/core/models/convnext/network.cpp
        classifier/core/models/convnext/elementwise.cu
        classifier/core/models/convnext/normalization.cu
        classifier/core/models/convnext/optimizer.cu
        classifier/core/models/convnext/preprocess.cu
        classifier/core/models/convnext/spatial.cu
        classifier/core/data/directory.cpp
        classifier/core/data/models.cpp
        classifier/core/training/state.cpp
        classifier/core/training/samples.cpp
        classifier/core/training/training.cpp
        classifier/core/inference.cpp
        classifier/core/audit.cpp
        classifier/core/classify.cpp
        classifier/core/session.cpp
        classifier/core/headless.cpp
        PRIVATE FILE_SET cxx_modules TYPE CXX_MODULES FILES
        classifier/core/project.ixx
        classifier/core/models/convnext/network.ixx
        classifier/core/data/directory.ixx
        classifier/core/data/models.ixx
        classifier/core/training/state.ixx
        classifier/core/training/samples.ixx
        classifier/core/training/training.ixx
        classifier/core/progress.ixx
        classifier/core/inference.ixx
        classifier/core/audit.ixx
        classifier/core/classify.ixx
        classifier/core/session.ixx
        classifier/core/headless.ixx)
set(CLASSIFIER_ASSET_DIRECTORY "${GENESIA_ASSET_DIRECTORY}" CACHE PATH "Classifier pretrained weights")
target_compile_definitions(classifier PRIVATE
        CLASSIFIER_VERSION="${PROJECT_VERSION}"
        CLASSIFIER_ASSET_DIRECTORY="${CLASSIFIER_ASSET_DIRECTORY}"
        CLASSIFIER_CACHE_DIRECTORY="${CMAKE_CURRENT_BINARY_DIR}/classifier-cache")
target_compile_options(classifier PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;--extended-lambda>")
target_link_libraries(classifier PRIVATE
        tools_compute Threads::Threads
        genesia::cudnn-frontend CUDA::cuda_driver CUDA::nvrtc)

add_executable(edit edit/main.cpp)
target_sources(edit PRIVATE
        edit/core/edit.cpp
        edit/core/session.cpp
        edit/core/headless.cpp
        PRIVATE FILE_SET cxx_modules TYPE CXX_MODULES FILES
        edit/core/edit.ixx
        edit/core/session.ixx
        edit/core/headless.ixx)
target_compile_definitions(edit PRIVATE EDIT_VERSION="${PROJECT_VERSION}")
target_link_libraries(edit PRIVATE Threads::Threads genesia::httplib)

add_executable(foreground foreground/main.cpp)
target_sources(foreground PRIVATE
        foreground/core/foreground.cpp
        foreground/core/models/birefnet/network.cpp
        foreground/core/models/birefnet/kernels.cu
        foreground/core/session.cpp
        foreground/core/headless.cpp
        PRIVATE FILE_SET cxx_modules TYPE CXX_MODULES FILES
        foreground/core/foreground.ixx
        foreground/core/models/birefnet/network.ixx
        foreground/core/session.ixx
        foreground/core/headless.ixx)
set(FOREGROUND_ASSET_DIRECTORY "${GENESIA_ASSET_DIRECTORY}" CACHE PATH "Foreground pretrained weights")
target_compile_definitions(foreground PRIVATE
        FOREGROUND_VERSION="${PROJECT_VERSION}"
        FOREGROUND_ASSET_DIRECTORY="${FOREGROUND_ASSET_DIRECTORY}"
        FOREGROUND_CACHE_DIRECTORY="${CMAKE_CURRENT_BINARY_DIR}/foreground-cache")
target_compile_options(foreground PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;--extended-lambda>")
target_link_libraries(foreground PRIVATE tools_compute Threads::Threads genesia::stb)

foreach (tool IN ITEMS classifier edit foreground)
    string(TOUPPER "${tool}" prefix)
    target_link_libraries(${tool} PRIVATE tools_common)
    if (${prefix}_BUILD_EDITOR)
        target_sources(${tool} PRIVATE
                ${tool}/editor/editor.cpp
                ${tool}/editor/workspace.cpp
                PRIVATE FILE_SET cxx_modules TYPE CXX_MODULES FILES
                ${tool}/editor/editor.ixx
                ${tool}/editor/workspace.ixx)
        set_target_properties(${tool} PROPERTIES WIN32_EXECUTABLE TRUE)
        target_compile_definitions(${tool} PRIVATE ${prefix}_HAS_EDITOR)
        target_link_libraries(${tool} PRIVATE tools_editor)
    endif ()
    if (WIN32)
        target_sources(${tool} PRIVATE common/Windows.manifest)
    endif ()
endforeach ()

if (WIN32)
    add_custom_target(tools_cudnn_runtime
            COMMAND "${CMAKE_COMMAND}" -E make_directory "$<TARGET_FILE_DIR:classifier>"
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different ${GENESIA_CUDNN_DLLS} "$<TARGET_FILE_DIR:classifier>"
            COMMAND_EXPAND_LISTS VERBATIM)
    add_dependencies(classifier tools_cudnn_runtime)
    add_dependencies(foreground tools_cudnn_runtime)
endif ()
