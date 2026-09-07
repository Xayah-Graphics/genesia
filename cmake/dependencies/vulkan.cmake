include_guard(GLOBAL)
find_package(Vulkan 1.4 REQUIRED GLOBAL)
list(GET Vulkan_INCLUDE_DIRS 0 GENESIA_VULKAN_INCLUDE_DIRECTORY)
cmake_path(GET GENESIA_VULKAN_INCLUDE_DIRECTORY PARENT_PATH GENESIA_VULKAN_SDK_DIRECTORY)
add_library(genesia_vulkan STATIC)
add_library(genesia::vulkan ALIAS genesia_vulkan)
target_sources(genesia_vulkan PUBLIC FILE_SET cxx_modules TYPE CXX_MODULES
        BASE_DIRS "${GENESIA_VULKAN_INCLUDE_DIRECTORY}"
        FILES "${GENESIA_VULKAN_INCLUDE_DIRECTORY}/vulkan/vulkan.cppm")
target_link_libraries(genesia_vulkan PUBLIC Vulkan::Vulkan)
target_compile_definitions(genesia_vulkan PUBLIC VK_USE_PLATFORM_WIN32_KHR NOMINMAX)
