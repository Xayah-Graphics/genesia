include_guard(GLOBAL)

FetchContent_MakeAvailable(cudnn_frontend)
add_library(genesia-cudnn-frontend INTERFACE)
add_library(genesia::cudnn-frontend ALIAS genesia-cudnn-frontend)
target_include_directories(genesia-cudnn-frontend SYSTEM INTERFACE "${cudnn_frontend_SOURCE_DIR}/include")
target_link_libraries(genesia-cudnn-frontend INTERFACE cuDNN::cuDNN nlohmann_json::nlohmann_json)
