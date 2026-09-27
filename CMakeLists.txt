# ============================================================================
# Project
# ============================================================================
cmake_minimum_required(VERSION 4.4)
set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "f35a9ac6-8463-4d38-8eec-5d6008153e7d")
file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/VERSION" GENESIA_VERSION LIMIT_COUNT 1)
project(genesia VERSION "${GENESIA_VERSION}" LANGUAGES C CXX CUDA)

# ============================================================================
# Project Configuration
# ============================================================================
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_MODULE_STD ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
set(CMAKE_CUDA_ARCHITECTURES 120a-real)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT WIN32 AND NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "Genesia supports only Windows and Linux")
endif ()

option(GENESIA_BUILD_UI "Build the Windows Editors for Genesia and all tools" ${WIN32})
if (GENESIA_BUILD_UI AND NOT WIN32)
    message(FATAL_ERROR "GENESIA_BUILD_UI is supported only on Windows")
endif ()

# ============================================================================
# Compiler Configuration
# ============================================================================
if (WIN32)
    add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN)
endif ()

if (MSVC)
    # Module providers and consumers must use the same C++ compile options.
    add_compile_options(
            $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
            $<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus>
            $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>
            $<$<COMPILE_LANGUAGE:CXX>:/bigobj>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
    )
endif ()

# ============================================================================
# Dependencies
# ============================================================================
include(cmake/Dependencies.cmake)
genesia_require_dependency(cuda)
genesia_require_dependency(cccl)
genesia_require_dependency(threads)
genesia_require_dependency(nlohmann_json)
genesia_require_dependency(cudnn)
genesia_require_dependency(cudnn_frontend)
genesia_require_dependency(stb)
genesia_require_dependency(httplib)
include(cmake/ClionCudaFacade.cmake)
if (GENESIA_BUILD_UI)
    genesia_require_dependency(vulkan)
    genesia_require_dependency(slang)
    genesia_require_dependency(imgui)
    include(cmake/Shaders.cmake)
endif ()

# Genesia owns its runtime, compute and file modules independently of the tools.
add_library(genesia_core STATIC)
target_sources(genesia_core
        PRIVATE
        genesia/core/compute/device.cpp
        genesia/core/compute/inference-attention.cu
        genesia/core/compute/inference-kernels.cu
        genesia/core/compute/inference.cpp
        genesia/core/compute/plans.cpp
        genesia/core/data/images.cpp
        genesia/core/generation/engine.cpp
        genesia/core/generation/settings.cpp
        genesia/core/generation/output.cpp
        genesia/core/headless.cpp
        genesia/core/io/files.cpp
        genesia/core/io/safetensors.cpp
        genesia/core/models/sdxl/kernels.cu
        genesia/core/models/sdxl/preview.cpp
        genesia/core/models/sdxl/sdxl.cpp
        genesia/core/models/sdxl/text.cpp
        genesia/core/models/sdxl/tokenizer.cpp
        genesia/core/models/sdxl/unet.cpp
        genesia/core/models/sdxl/vae.cpp
        genesia/core/models/sdxl/weights.cpp
        genesia/core/models/sdxl/workspace.cpp
        genesia/core/prompt/catalog.cpp
        genesia/core/prompt/presets.cpp
        genesia/core/prompt/prompt.cpp
        genesia/core/prompt/search.cpp
        genesia/core/runtime/session.cpp
        PUBLIC FILE_SET cxx_modules TYPE CXX_MODULES FILES
        genesia/core/compute/device.ixx
        genesia/core/compute/inference.ixx
        genesia/core/compute/plans.ixx
        genesia/core/data/images.ixx
        genesia/core/generation/engine.ixx
        genesia/core/generation/output.ixx
        genesia/core/generation/settings.ixx
        genesia/core/headless.ixx
        genesia/core/io/files.ixx
        genesia/core/io/safetensors.ixx
        genesia/core/models/sdxl/preview.ixx
        genesia/core/models/sdxl/sdxl.ixx
        genesia/core/models/sdxl/text.ixx
        genesia/core/models/sdxl/tokenizer.ixx
        genesia/core/models/sdxl/unet.ixx
        genesia/core/models/sdxl/vae.ixx
        genesia/core/models/sdxl/weights.ixx
        genesia/core/models/sdxl/workspace.ixx
        genesia/core/project.ixx
        genesia/core/prompt/catalog.ixx
        genesia/core/prompt/presets.ixx
        genesia/core/prompt/prompt.ixx
        genesia/core/prompt/search.ixx
        genesia/core/runtime/session.ixx
        genesia/core/runtime/tasks.ixx
)
target_compile_definitions(genesia_core PUBLIC GENESIA_CACHE_DIRECTORY="${PROJECT_BINARY_DIR}/genesia-cache")
target_compile_definitions(genesia_core PRIVATE
        GENESIA_CXX_COMPILER="${CMAKE_CXX_COMPILER_ID}-${CMAKE_CXX_COMPILER_VERSION}"
        GENESIA_PLATFORM="${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")
target_include_directories(genesia_core PUBLIC "${GENESIA_CLION_CUDA_INCLUDE_DIRECTORY}")
target_compile_options(genesia_core PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;--extended-lambda>")
target_link_libraries(genesia_core PUBLIC
        CCCL::libcudacxx CUDA::toolkit cuDNN::cuDNN CUDA::cublasLt CUDA::cudart
        nlohmann_json::nlohmann_json Threads::Threads
        PRIVATE genesia::cudnn-frontend genesia::stb CUDA::cuda_driver CUDA::nvrtc)

# ============================================================================
# Editor
# ============================================================================
if (GENESIA_BUILD_UI)
    add_library(genesia_editor STATIC)
    target_sources(genesia_editor
            PRIVATE
            genesia/editor/editor.cpp
            genesia/editor/panels/loras.cpp
            genesia/editor/graphics/device.cpp
            genesia/editor/graphics/interop.cpp
            genesia/editor/graphics/interop.cu
            genesia/editor/graphics/bridge.cpp
            genesia/editor/graphics/renderer.cpp
            genesia/editor/graphics/resources.cpp
            genesia/editor/panels/application.cpp
            genesia/editor/panels/sidebars.cpp
            genesia/editor/platform/window.cpp
            genesia/editor/viewing/camera.cpp
            genesia/editor/viewing/canvas.cpp
            genesia/editor/viewing/textures.cpp
            genesia/editor/widgets/controls.cpp
            genesia/editor/widgets/tag-editor.cpp
            genesia/editor/workspace.cpp
            genesia/editor/web.cpp
            PUBLIC FILE_SET cxx_modules TYPE CXX_MODULES FILES
            genesia/editor/editor.ixx
            PRIVATE FILE_SET editor_modules TYPE CXX_MODULES FILES
            genesia/editor/panels/loras.ixx
            genesia/editor/graphics/device.ixx
            genesia/editor/graphics/interop.ixx
            genesia/editor/graphics/bridge.ixx
            genesia/editor/graphics/renderer.ixx
            genesia/editor/graphics/resources.ixx
            genesia/editor/panels/application.ixx
            genesia/editor/panels/sidebars.ixx
            genesia/editor/platform/window.ixx
            genesia/editor/viewing/camera.ixx
            genesia/editor/viewing/canvas.ixx
            genesia/editor/viewing/textures.ixx
            genesia/editor/widgets/controls.ixx
            genesia/editor/widgets/tag-editor.ixx
            genesia/editor/workspace.ixx
            genesia/editor/web.ixx
    )
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/genesia/editor/web.html")
    file(READ "${PROJECT_SOURCE_DIR}/genesia/editor/web.html" GENESIA_WEB_PAGE)
    file(GENERATE OUTPUT "${PROJECT_BINARY_DIR}/genesia-web-page.h"
            CONTENT "constexpr char web_page[] = R\"GENESIA_WEB(${GENESIA_WEB_PAGE})GENESIA_WEB\";\n")
    target_compile_options(genesia_editor PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;--extended-lambda>")
    target_include_directories(genesia_editor PRIVATE "${PROJECT_BINARY_DIR}")
    target_compile_definitions(
            genesia_editor
            PRIVATE
            GLFW_INCLUDE_NONE
            GLFW_EXPOSE_NATIVE_WIN32
            GENESIA_SHADER_DIRECTORY="${GENESIA_SHADER_OUTPUT_DIRECTORY}"
    )
    target_link_libraries(
            genesia_editor
            PRIVATE
            genesia_core
            genesia::vulkan
            genesia::imgui
            genesia::stb
            genesia::httplib
            CUDA::cudart
            dwmapi
            imm32
            shell32
            iphlpapi
    )
    add_dependencies(genesia_editor genesia_shaders)
endif ()

# ============================================================================
# Executable
# ============================================================================
add_executable(genesia main.cpp)
target_compile_definitions(genesia_core PRIVATE GENESIA_VERSION="${PROJECT_VERSION}")
target_link_libraries(genesia PRIVATE genesia_core)
if (GENESIA_BUILD_UI)
    set_target_properties(genesia PROPERTIES WIN32_EXECUTABLE TRUE)
    target_compile_definitions(genesia PRIVATE GENESIA_HAS_EDITOR)
    target_link_libraries(genesia PRIVATE genesia_editor)
endif ()

# ============================================================================
# Runtime Resources
# ============================================================================
set(GENESIA_ASSET_DIRECTORY "${PROJECT_SOURCE_DIR}/assets" CACHE PATH "Runtime assets and saved prompts")
target_compile_definitions(genesia_core PUBLIC GENESIA_ASSET_DIRECTORY="${GENESIA_ASSET_DIRECTORY}")

if (WIN32)
    add_custom_target(genesia_cudnn_runtime
            COMMAND "${CMAKE_COMMAND}" -E make_directory "$<TARGET_FILE_DIR:genesia>"
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different ${GENESIA_CUDNN_DLLS}
            "$<TARGET_FILE_DIR:genesia>"
            COMMAND_EXPAND_LISTS
            VERBATIM
    )
    target_sources(genesia PRIVATE cmake/WindowsUtf8.manifest)
    target_link_libraries(genesia PRIVATE user32)
    add_dependencies(genesia genesia_cudnn_runtime)
endif ()

# MSVC interprets a bare @cmake_cxx_std.lib as a response file.
if (MSVC)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/cxx-modules")
endif ()

# ============================================================================
# Tools
# ============================================================================
# Shared startup, file, image and model I/O for Headless and Editor builds.
add_library(tools_common STATIC)
target_sources(tools_common PRIVATE
        tools/common/platform/application.cpp
        tools/common/files.cpp
        tools/common/hash.cpp
        tools/common/images.cpp
        tools/common/safetensors.cpp
        PUBLIC FILE_SET cxx_modules TYPE CXX_MODULES FILES
        tools/common/platform/application.ixx
        tools/common/files.ixx
        tools/common/hash.ixx
        tools/common/images.ixx
        tools/common/safetensors.ixx)
target_link_libraries(tools_common PUBLIC nlohmann_json::nlohmann_json PRIVATE genesia::stb)
if (WIN32)
    target_link_libraries(tools_common PRIVATE user32)
endif ()

add_library(tools_compute STATIC)
target_sources(tools_compute PRIVATE
        tools/common/compute/device.cpp
        tools/common/compute/plans.cpp
        tools/common/compute/inference.cpp
        tools/common/compute/inference-kernels.cu
        PUBLIC FILE_SET cxx_modules TYPE CXX_MODULES FILES
        tools/common/compute/device.ixx
        tools/common/compute/plans.ixx
        tools/common/compute/inference.ixx)
target_include_directories(tools_compute PUBLIC "${PROJECT_SOURCE_DIR}/tools/common" "${GENESIA_CLION_CUDA_INCLUDE_DIRECTORY}")
target_compile_definitions(tools_compute PRIVATE
        TOOLS_CXX_COMPILER="${CMAKE_CXX_COMPILER_ID}-${CMAKE_CXX_COMPILER_VERSION}"
        TOOLS_PLATFORM="${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")
target_compile_options(tools_compute PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;--extended-lambda>")
target_link_libraries(tools_compute PUBLIC tools_common CCCL::libcudacxx CUDA::toolkit cuDNN::cuDNN CUDA::cublasLt CUDA::cudart
        PRIVATE genesia::cudnn-frontend CUDA::cuda_driver CUDA::nvrtc)

if (GENESIA_BUILD_UI)
    set(TOOLS_SHADER_DIRECTORY "${PROJECT_BINARY_DIR}/tools/generated/editor-shaders")
    set(TOOLS_SHADERS)
    foreach (stage IN ITEMS vertex fragment)
        set(output "${TOOLS_SHADER_DIRECTORY}/imgui_${stage}.spv")
        add_custom_command(OUTPUT "${output}"
                COMMAND "${CMAKE_COMMAND}" -E make_directory "${TOOLS_SHADER_DIRECTORY}"
                COMMAND "${GENESIA_SLANG_COMPILER}" "${PROJECT_SOURCE_DIR}/tools/common/editor/shaders/imgui.slang"
                -target spirv -profile spirv_1_6 -entry imgui_${stage} -stage ${stage} -emit-spirv-directly
                -capability SPIRV_1_6+spvDescriptorHeapEXT -spirv-unified-descriptor-heap-stride
                -fvk-use-entrypoint-name -fvk-use-c-layout -matrix-layout-row-major -O2 -o "${output}"
                DEPENDS "${PROJECT_SOURCE_DIR}/tools/common/editor/shaders/imgui.slang" VERBATIM)
        list(APPEND TOOLS_SHADERS "${output}")
    endforeach ()
    add_custom_target(tools_shaders DEPENDS ${TOOLS_SHADERS})

    add_library(tools_editor STATIC)
    target_sources(tools_editor PRIVATE
            tools/common/editor/platform/window.cpp
            tools/common/editor/platform/files.cpp
            tools/common/editor/graphics/device.cpp
            tools/common/editor/graphics/resources.cpp
            tools/common/editor/graphics/renderer.cpp
            tools/common/editor/style.cpp
            PUBLIC FILE_SET cxx_modules TYPE CXX_MODULES FILES
            tools/common/editor/platform/window.ixx
            tools/common/editor/platform/files.ixx
            tools/common/editor/graphics/device.ixx
            tools/common/editor/graphics/resources.ixx
            tools/common/editor/graphics/renderer.ixx
            tools/common/editor/style.ixx)
    target_compile_definitions(tools_editor PUBLIC GLFW_INCLUDE_NONE GLFW_EXPOSE_NATIVE_WIN32
            PRIVATE TOOLS_SHADER_DIRECTORY="${TOOLS_SHADER_DIRECTORY}")
    target_link_libraries(tools_editor PUBLIC genesia::vulkan genesia::imgui
            PRIVATE tools_common dwmapi imm32 shell32 ole32)
    add_dependencies(tools_editor tools_shaders)
endif ()

add_executable(classifier tools/classifier/main.cpp)
target_sources(classifier PRIVATE
        tools/classifier/core/models/convnext/network.cpp
        tools/classifier/core/models/convnext/elementwise.cu
        tools/classifier/core/models/convnext/normalization.cu
        tools/classifier/core/models/convnext/optimizer.cu
        tools/classifier/core/models/convnext/preprocess.cu
        tools/classifier/core/models/convnext/spatial.cu
        tools/classifier/core/data/directory.cpp
        tools/classifier/core/data/models.cpp
        tools/classifier/core/training/state.cpp
        tools/classifier/core/training/samples.cpp
        tools/classifier/core/training/training.cpp
        tools/classifier/core/inference.cpp
        tools/classifier/core/audit.cpp
        tools/classifier/core/classify.cpp
        tools/classifier/core/session.cpp
        tools/classifier/core/headless.cpp
        PRIVATE FILE_SET cxx_modules TYPE CXX_MODULES FILES
        tools/classifier/core/project.ixx
        tools/classifier/core/models/convnext/network.ixx
        tools/classifier/core/data/directory.ixx
        tools/classifier/core/data/models.ixx
        tools/classifier/core/training/state.ixx
        tools/classifier/core/training/samples.ixx
        tools/classifier/core/training/training.ixx
        tools/classifier/core/progress.ixx
        tools/classifier/core/inference.ixx
        tools/classifier/core/audit.ixx
        tools/classifier/core/classify.ixx
        tools/classifier/core/session.ixx
        tools/classifier/core/headless.ixx)
set(CLASSIFIER_ASSET_DIRECTORY "${GENESIA_ASSET_DIRECTORY}" CACHE PATH "Classifier pretrained weights")
target_compile_definitions(classifier PRIVATE
        CLASSIFIER_VERSION="${PROJECT_VERSION}"
        CLASSIFIER_ASSET_DIRECTORY="${CLASSIFIER_ASSET_DIRECTORY}"
        CLASSIFIER_CACHE_DIRECTORY="${PROJECT_BINARY_DIR}/tools/classifier-cache")
target_compile_options(classifier PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;--extended-lambda>")
target_link_libraries(classifier PRIVATE
        tools_compute Threads::Threads
        genesia::cudnn-frontend CUDA::cuda_driver CUDA::nvrtc)

add_executable(edit tools/edit/main.cpp)
target_sources(edit PRIVATE
        tools/edit/core/edit.cpp
        tools/edit/core/session.cpp
        tools/edit/core/headless.cpp
        PRIVATE FILE_SET cxx_modules TYPE CXX_MODULES FILES
        tools/edit/core/edit.ixx
        tools/edit/core/session.ixx
        tools/edit/core/headless.ixx)
target_compile_definitions(edit PRIVATE EDIT_VERSION="${PROJECT_VERSION}")
target_link_libraries(edit PRIVATE Threads::Threads genesia::httplib)

add_executable(foreground tools/foreground/main.cpp)
target_sources(foreground PRIVATE
        tools/foreground/core/foreground.cpp
        tools/foreground/core/models/birefnet/network.cpp
        tools/foreground/core/models/birefnet/kernels.cu
        tools/foreground/core/session.cpp
        tools/foreground/core/headless.cpp
        PRIVATE FILE_SET cxx_modules TYPE CXX_MODULES FILES
        tools/foreground/core/foreground.ixx
        tools/foreground/core/models/birefnet/network.ixx
        tools/foreground/core/session.ixx
        tools/foreground/core/headless.ixx)
set(FOREGROUND_ASSET_DIRECTORY "${GENESIA_ASSET_DIRECTORY}" CACHE PATH "Foreground pretrained weights")
target_compile_definitions(foreground PRIVATE
        FOREGROUND_VERSION="${PROJECT_VERSION}"
        FOREGROUND_ASSET_DIRECTORY="${FOREGROUND_ASSET_DIRECTORY}"
        FOREGROUND_CACHE_DIRECTORY="${PROJECT_BINARY_DIR}/tools/foreground-cache")
target_compile_options(foreground PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;--extended-lambda>")
target_link_libraries(foreground PRIVATE tools_compute Threads::Threads genesia::stb)

add_executable(qwen tools/qwen/main.cpp)
target_sources(qwen PRIVATE
        tools/qwen/core/qwen.cpp
        tools/qwen/core/session.cpp
        tools/qwen/core/headless.cpp
        PRIVATE FILE_SET cxx_modules TYPE CXX_MODULES FILES
        tools/qwen/core/qwen.ixx
        tools/qwen/core/session.ixx
        tools/qwen/core/headless.ixx)
target_compile_definitions(qwen PRIVATE QWEN_VERSION="${PROJECT_VERSION}")
target_link_libraries(qwen PRIVATE Threads::Threads genesia::httplib)

foreach (tool IN ITEMS classifier edit foreground qwen)
    set_target_properties(${tool} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/tools")
    target_link_libraries(${tool} PRIVATE tools_common)
    if (GENESIA_BUILD_UI)
        target_sources(${tool} PRIVATE
                tools/${tool}/editor/editor.cpp
                tools/${tool}/editor/workspace.cpp
                PRIVATE FILE_SET cxx_modules TYPE CXX_MODULES FILES
                tools/${tool}/editor/editor.ixx
                tools/${tool}/editor/workspace.ixx)
        set_target_properties(${tool} PROPERTIES WIN32_EXECUTABLE TRUE)
        string(TOUPPER "${tool}" prefix)
        target_compile_definitions(${tool} PRIVATE ${prefix}_HAS_EDITOR)
        target_link_libraries(${tool} PRIVATE tools_editor)
    endif ()
    if (WIN32)
        target_sources(${tool} PRIVATE tools/common/Windows.manifest)
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
