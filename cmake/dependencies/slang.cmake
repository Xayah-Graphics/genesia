include_guard(GLOBAL)
genesia_require_dependency(vulkan)
find_program(GENESIA_SLANG_COMPILER NAMES slangc HINTS "${GENESIA_VULKAN_SDK_DIRECTORY}/Bin" REQUIRED NO_DEFAULT_PATH)
