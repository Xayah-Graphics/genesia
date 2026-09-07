include_guard(GLOBAL)

genesia_require_dependency(cuda)

find_package(
        CCCL REQUIRED CONFIG GLOBAL
        COMPONENTS libcudacxx
        PATHS "${CUDAToolkit_LIBRARY_ROOT}/lib/cmake/cccl"
)
