include_guard(GLOBAL)
include(FetchContent)
set(FETCHCONTENT_TRY_FIND_PACKAGE_MODE NEVER)
FetchContent_Declare(glfw
        URL "https://codeload.github.com/glfw/glfw/zip/refs/tags/3.4"
        URL_HASH SHA256=A133DDC3D3C66143EBA9035621DB8E0BCF34DBA1EE9514A9E23E96AFD39FD57A
        SYSTEM EXCLUDE_FROM_ALL)
FetchContent_Declare(imgui
        URL "https://codeload.github.com/ocornut/imgui/zip/b334d19b667958ed970000073644d911fae17e57"
        URL_HASH SHA256=504BC8171B80B8C92F035EBC899F6B3086C9CFA56EFADEE4962753DEB38626A2)
FetchContent_Declare(nlohmann_json
        URL "https://github.com/nlohmann/json/archive/refs/tags/v3.12.0.tar.gz"
        URL_HASH SHA256=4B92EB0C06D10683F7447CE9406CB97CD4B453BE18D7279320F7B2F025C10187
        SYSTEM EXCLUDE_FROM_ALL)
FetchContent_Declare(
        stb
        URL "https://github.com/nothings/stb/archive/28d546d5eb77d4585506a20480f4de2e706dff4c.tar.gz"
        URL_HASH SHA256=4EF16A0E174BC33887FEC582B01CA239155466E0B48081CC27304298556BED47
        SYSTEM
        EXCLUDE_FROM_ALL
)

FetchContent_Declare(
        cudnn_frontend
        URL "https://codeload.github.com/NVIDIA/cudnn-frontend/tar.gz/91159779637b672a3f768738f5b30eee33f2180d"
        URL_HASH SHA256=7ff261059996a163830007d8502664243b539e57ec72593dcb7e659d4604ce94
        SOURCE_SUBDIR genesia-unused
        SYSTEM
        EXCLUDE_FROM_ALL
)

set(GENESIA_DEPENDENCIES_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/dependencies")
macro(genesia_require_dependency dependency)
    include("${GENESIA_DEPENDENCIES_DIRECTORY}/${dependency}.cmake")
endmacro()
