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
    include(cmake/Shaders.cmake)
endif ()

# ============================================================================
# Classifier
# ============================================================================
add_library(classifier_runtime STATIC
        classifier/storage.cpp
        classifier/network.cpp
        classifier/inference.cpp
        classifier/preprocess.cu
        classifier/spatial.cu
        classifier/normalization.cu
        classifier/elementwise.cu
        classifier/optimizer.cu
)
add_library(classifier::runtime ALIAS classifier_runtime)
target_sources(classifier_runtime PUBLIC FILE_SET cxx_modules TYPE CXX_MODULES FILES
        classifier/storage.ixx
        classifier/network.ixx
        classifier/inference.ixx
)
target_link_libraries(classifier_runtime
        PUBLIC CUDA::cudart CUDA::cublasLt cuDNN::cuDNN nlohmann_json::nlohmann_json Threads::Threads
        PRIVATE genesia::cudnn-frontend CUDA::nvrtc
)
target_compile_options(classifier_runtime PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;--extended-lambda>")

add_library(classifier_training STATIC classifier/dataset.cpp classifier/training.cpp)
add_library(classifier::training ALIAS classifier_training)
target_sources(classifier_training PUBLIC FILE_SET cxx_modules TYPE CXX_MODULES FILES
        classifier/dataset.ixx
        classifier/training.ixx
)
target_link_libraries(classifier_training PUBLIC classifier::runtime PRIVATE genesia::stb)
add_executable(classifier-train classifier/trainer.cpp)
add_executable(classifier::train ALIAS classifier-train)
target_link_libraries(classifier-train PRIVATE classifier::training)
if (MSVC)
    foreach (target IN ITEMS classifier_runtime classifier_training classifier-train)
        target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:/utf-8;/bigobj>")
    endforeach ()
    target_compile_options(classifier_runtime PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:__cplusplus>")
endif ()

# ============================================================================
# Core
# ============================================================================
add_library(genesia_core STATIC)
target_sources(
        genesia_core
        PRIVATE
        genesia/core/neural/inference-runtime.cpp
        genesia/core/neural/inference-kernels.cu
        genesia/core/neural/inference-attention.cu
        genesia/core/sdxl/tokenizer.cpp
        genesia/core/sdxl/workspace.cpp
        genesia/core/sdxl/weights.cpp
        genesia/core/sdxl/kernels.cu
        genesia/core/sdxl/text.cpp
        genesia/core/sdxl/unet.cpp
        genesia/core/sdxl/preview.cpp
        genesia/core/sdxl/vae.cpp
        genesia/core/sdxl/sdxl.cpp
        genesia/core/headless.cpp
        genesia/core/generation/output.cpp
        genesia/core/generation/images.cpp
        genesia/core/prompt/catalog.cpp
        genesia/core/prompt/prompt.cpp
        genesia/core/prompt/preset.cpp
        PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
        genesia/core/neural/inference-runtime.ixx
        genesia/core/sdxl/tokenizer.ixx
        genesia/core/sdxl/workspace.ixx
        genesia/core/sdxl/weights.ixx
        genesia/core/sdxl/text.ixx
        genesia/core/sdxl/unet.ixx
        genesia/core/sdxl/preview.ixx
        genesia/core/sdxl/vae.ixx
        genesia/core/sdxl/sdxl.ixx
        genesia/core/headless.ixx
        genesia/core/generation/defaults.ixx
        genesia/core/generation/output.ixx
        genesia/core/generation/images.ixx
        genesia/core/prompt/catalog.ixx
        genesia/core/prompt/prompt.ixx
        genesia/core/prompt/preset.ixx
)
target_include_directories(genesia_core PUBLIC "${GENESIA_CLION_CUDA_INCLUDE_DIRECTORY}")
target_compile_options(genesia_core PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>")
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
        classifier::runtime
        CUDA::toolkit
        cuDNN::cuDNN
        CUDA::cublasLt
        PRIVATE
        genesia::cudnn-frontend
        CUDA::cudart
        CUDA::cuda_driver
        CUDA::nvrtc
        nlohmann_json::nlohmann_json
        genesia::stb
        Threads::Threads
)

# ============================================================================
# Editor
# ============================================================================
if (GENESIA_BUILD_UI)
    add_library(genesia_editor STATIC)
    target_sources(
            genesia_editor
            PRIVATE
            genesia/editor/runtime/device.cpp
            genesia/editor/runtime/resources.cpp
            genesia/editor/runtime/images.cpp
            genesia/editor/platform/window.cpp
            genesia/editor/ui/renderer.cpp
            genesia/editor/platform/interop.cpp
            genesia/editor/platform/interop.cu
            genesia/editor/session/session.cpp
            genesia/editor/gallery/gallery.cpp
            genesia/editor/ui/ui.cpp
            genesia/editor/ui/tag-editor.cpp
            genesia/editor/editor.cpp
            PUBLIC
            FILE_SET cxx_modules TYPE CXX_MODULES FILES
            genesia/editor/editor.ixx
            PRIVATE
            FILE_SET editor_modules TYPE CXX_MODULES FILES
            genesia/editor/runtime/device.ixx
            genesia/editor/runtime/resources.ixx
            genesia/editor/runtime/images.ixx
            genesia/editor/platform/window.ixx
            genesia/editor/ui/renderer.ixx
            genesia/editor/platform/interop.ixx
            genesia/editor/session/session.ixx
            genesia/editor/gallery/gallery.ixx
            genesia/editor/ui/ui.ixx
            genesia/editor/ui/tag-editor.ixx
    )
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
            CUDA::cudart
            dwmapi
            shell32
    )
    add_dependencies(genesia_editor genesia_shaders)
endif ()

# ============================================================================
# Executable
# ============================================================================
add_executable(genesia main.cpp)
target_compile_definitions(genesia PRIVATE GENESIA_VERSION="${PROJECT_VERSION}")
target_link_libraries(genesia PRIVATE genesia_core)
if (GENESIA_BUILD_UI)
    target_compile_definitions(genesia PRIVATE GENESIA_HAS_EDITOR)
    target_link_libraries(genesia PRIVATE genesia_editor)
endif ()

# ============================================================================
# Runtime Resources
# ============================================================================
set(GENESIA_ASSET_DIRECTORY "${PROJECT_SOURCE_DIR}/assets" CACHE PATH "Runtime assets and editable prompt presets")
genesia_require_dependency(classifier_weights)
target_compile_definitions(genesia_core PUBLIC GENESIA_ASSET_DIRECTORY="${GENESIA_ASSET_DIRECTORY}")
target_compile_definitions(classifier_runtime PRIVATE CLASSIFIER_MODEL_ROOT="${GENESIA_ASSET_DIRECTORY}/classifier/models")
target_compile_definitions(classifier_training PRIVATE CLASSIFIER_INITIAL_MODEL="${GENESIA_CLASSIFIER_INITIAL_MODEL}")

if (WIN32)
    add_custom_target(genesia_cudnn_runtime
            COMMAND "${CMAKE_COMMAND}" -E make_directory "$<TARGET_FILE_DIR:genesia>"
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different ${GENESIA_CUDNN_DLLS}
                "$<TARGET_FILE_DIR:genesia>"
            COMMAND_EXPAND_LISTS
            VERBATIM
    )
    foreach (target IN ITEMS genesia classifier-train)
        target_sources(${target} PRIVATE cmake/WindowsUtf8.manifest)
        add_dependencies(${target} genesia_cudnn_runtime)
    endforeach ()
endif ()

# MSVC interprets a bare @cmake_cxx_std.lib as a response file.
if (MSVC)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/cxx-modules")
endif ()
