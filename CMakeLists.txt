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

if (WIN32)
    set(GENESIA_BUILD_UI_DEFAULT ON)
else ()
    set(GENESIA_BUILD_UI_DEFAULT OFF)
endif ()
option(GENESIA_BUILD_UI "Build the Windows Editor into the genesia executable" ${GENESIA_BUILD_UI_DEFAULT})
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
    add_compile_options(
            $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
            $<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus>
            $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>
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
include(cmake/ClionCudaFacade.cmake)
if (GENESIA_BUILD_UI)
    genesia_require_dependency(vulkan)
    genesia_require_dependency(slang)
    genesia_require_dependency(imgui)
    genesia_require_dependency(httplib)
    include(cmake/Shaders.cmake)
endif ()

# Core owns all headless business and compute modules.
add_library(genesia_core STATIC)
target_sources(genesia_core
        PRIVATE
        genesia/core/classification/audit.cpp
        genesia/core/classification/classify.cpp
        genesia/core/classification/inference.cpp
        genesia/core/compute/device.cpp
        genesia/core/compute/inference-attention.cu
        genesia/core/compute/inference-kernels.cu
        genesia/core/compute/inference.cpp
        genesia/core/compute/plans.cpp
        genesia/core/data/concepts.cpp
        genesia/core/data/captions.cpp
        genesia/core/data/datasets.cpp
        genesia/core/data/images.cpp
        genesia/core/data/operations.cpp
        genesia/core/generation/engine.cpp
        genesia/core/generation/output.cpp
        genesia/core/headless.cpp
        genesia/core/io/files.cpp
        genesia/core/io/hash.cpp
        genesia/core/io/safetensors.cpp
        genesia/core/models/birefnet/kernels.cu
        genesia/core/models/birefnet/network.cpp
        genesia/core/segmentation/foreground.cpp
        genesia/core/models/convnext/elementwise.cu
        genesia/core/models/convnext/network.cpp
        genesia/core/models/convnext/normalization.cu
        genesia/core/models/convnext/optimizer.cu
        genesia/core/models/convnext/preprocess.cu
        genesia/core/models/convnext/spatial.cu
        genesia/core/models/registry.cpp
        genesia/core/models/sdxl/kernels.cu
        genesia/core/models/sdxl/lora.cpp
        genesia/core/models/sdxl/preview.cpp
        genesia/core/models/sdxl/sdxl.cpp
        genesia/core/models/sdxl/text.cpp
        genesia/core/models/sdxl/tokenizer.cpp
        genesia/core/models/sdxl/unet.cpp
        genesia/core/models/sdxl/vae.cpp
        genesia/core/models/sdxl/weights.cpp
        genesia/core/models/sdxl/workspace.cpp
        genesia/core/prompt/catalog.cpp
        genesia/core/prompt/library.cpp
        genesia/core/prompt/prompt.cpp
        genesia/core/prompt/search.cpp
        genesia/core/runtime/catalog.cpp
        genesia/core/runtime/session.cpp
        genesia/core/training/samples.cpp
        genesia/core/training/state.cpp
        genesia/core/training/training.cpp
        PUBLIC FILE_SET cxx_modules TYPE CXX_MODULES FILES
        genesia/core/classification/audit.ixx
        genesia/core/classification/classify.ixx
        genesia/core/classification/inference.ixx
        genesia/core/compute/device.ixx
        genesia/core/compute/inference.ixx
        genesia/core/compute/plans.ixx
        genesia/core/data/concepts.ixx
        genesia/core/data/captions.ixx
        genesia/core/data/datasets.ixx
        genesia/core/data/images.ixx
        genesia/core/data/operations.ixx
        genesia/core/generation/defaults.ixx
        genesia/core/generation/engine.ixx
        genesia/core/generation/output.ixx
        genesia/core/generation/settings.ixx
        genesia/core/headless.ixx
        genesia/core/io/files.ixx
        genesia/core/io/hash.ixx
        genesia/core/io/safetensors.ixx
        genesia/core/models/birefnet/network.ixx
        genesia/core/segmentation/foreground.ixx
        genesia/core/models/convnext/network.ixx
        genesia/core/models/registry.ixx
        genesia/core/models/sdxl/lora.ixx
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
        genesia/core/prompt/library.ixx
        genesia/core/prompt/prompt.ixx
        genesia/core/prompt/search.ixx
        genesia/core/runtime/catalog.ixx
        genesia/core/runtime/progress.ixx
        genesia/core/runtime/session.ixx
        genesia/core/runtime/tasks.ixx
        genesia/core/training/samples.ixx
        genesia/core/training/state.ixx
        genesia/core/training/training.ixx
)
target_compile_definitions(genesia_core PUBLIC GENESIA_DATA_DIRECTORY="${PROJECT_SOURCE_DIR}/data" GENESIA_CACHE_DIRECTORY="${PROJECT_BINARY_DIR}/genesia-cache")
target_include_directories(genesia_core PUBLIC "${GENESIA_CLION_CUDA_INCLUDE_DIRECTORY}")
target_compile_options(genesia_core PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;--extended-lambda>")
if (MSVC)
    target_compile_options(genesia_core PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:/bigobj>")
endif ()
target_compile_definitions(
        genesia_core
        PRIVATE
        GENESIA_CXX_COMPILER="${CMAKE_CXX_COMPILER_ID}-${CMAKE_CXX_COMPILER_VERSION}"
        GENESIA_PLATFORM="${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}"
)
target_link_libraries(
        genesia_core
        PUBLIC
        CCCL::libcudacxx
        CUDA::toolkit
        cuDNN::cuDNN
        CUDA::cublasLt
        nlohmann_json::nlohmann_json
        Threads::Threads
        CUDA::cudart
        PRIVATE
        genesia::cudnn-frontend
        CUDA::cuda_driver
        CUDA::nvrtc
        genesia::stb
)

# ============================================================================
# Editor
# ============================================================================
if (GENESIA_BUILD_UI)
    add_library(genesia_editor STATIC)
    target_sources(genesia_editor
            PRIVATE
            genesia/editor/editor.cpp
            genesia/editor/graphics/device.cpp
            genesia/editor/graphics/interop.cpp
            genesia/editor/graphics/interop.cu
            genesia/editor/graphics/bridge.cpp
            genesia/editor/graphics/renderer.cpp
            genesia/editor/graphics/resources.cpp
            genesia/editor/panels/application.cpp
            genesia/editor/panels/datasets.cpp
            genesia/editor/panels/prompts.cpp
            genesia/editor/panels/sidebars.cpp
            genesia/editor/prompt/previews.cpp
            genesia/editor/platform/window.cpp
            genesia/editor/viewing/camera.cpp
            genesia/editor/viewing/canvas.cpp
            genesia/editor/viewing/textures.cpp
            genesia/editor/widgets/controls.cpp
            genesia/editor/widgets/tag-editor.cpp
            genesia/editor/widgets/captions.cpp
            genesia/editor/workspace.cpp
            genesia/editor/web.cpp
            PUBLIC FILE_SET cxx_modules TYPE CXX_MODULES FILES
            genesia/editor/editor.ixx
            PRIVATE FILE_SET editor_modules TYPE CXX_MODULES FILES
            genesia/editor/graphics/device.ixx
            genesia/editor/graphics/interop.ixx
            genesia/editor/graphics/bridge.ixx
            genesia/editor/graphics/renderer.ixx
            genesia/editor/graphics/resources.ixx
            genesia/editor/panels/application.ixx
            genesia/editor/panels/datasets.ixx
            genesia/editor/panels/prompts.ixx
            genesia/editor/panels/sidebars.ixx
            genesia/editor/prompt/previews.ixx
            genesia/editor/platform/window.ixx
            genesia/editor/viewing/camera.ixx
            genesia/editor/viewing/canvas.ixx
            genesia/editor/viewing/textures.ixx
            genesia/editor/widgets/controls.ixx
            genesia/editor/widgets/tag-editor.ixx
            genesia/editor/widgets/captions.ixx
            genesia/editor/workspace.ixx
            genesia/editor/web.ixx
    )
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/genesia/editor/web.html")
    file(READ "${PROJECT_SOURCE_DIR}/genesia/editor/web.html" GENESIA_WEB_PAGE)
    file(GENERATE OUTPUT "${PROJECT_BINARY_DIR}/genesia-web-page.h"
            CONTENT "constexpr char web_page[] = R\"GENESIA_WEB(${GENESIA_WEB_PAGE})GENESIA_WEB\";\n")
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
            ole32
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
    target_compile_definitions(genesia PRIVATE GENESIA_HAS_EDITOR)
    target_link_libraries(genesia PRIVATE genesia_editor)
endif ()

# ============================================================================
# Runtime Resources
# ============================================================================
set(GENESIA_ASSET_DIRECTORY "${PROJECT_SOURCE_DIR}/assets" CACHE PATH "Runtime assets and editable prompt presets")
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
    add_dependencies(genesia genesia_cudnn_runtime)
endif ()

# MSVC interprets a bare @cmake_cxx_std.lib as a response file.
if (MSVC)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/cxx-modules")
endif ()
