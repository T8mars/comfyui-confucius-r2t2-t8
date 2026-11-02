cmake_minimum_required(VERSION 3.26)
project(confucius4_r2t2_native LANGUAGES CXX)

option(R2T2_WHEEL_BUILD "Build an installable, self-contained Python wheel" OFF)
set(LLAMA_CPP_DIR "" CACHE PATH "Pinned llama.cpp source checkout")
if(NOT EXISTS "${LLAMA_CPP_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "Pass -DLLAMA_CPP_DIR=<pinned llama.cpp source>")
endif()
find_package(Git REQUIRED)
execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${LLAMA_CPP_DIR}" rev-parse HEAD
    OUTPUT_VARIABLE R2T2_LLAMA_COMMIT OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE R2T2_GIT_STATUS)
if(NOT R2T2_GIT_STATUS EQUAL 0 OR NOT R2T2_LLAMA_COMMIT STREQUAL "ad6c66839af3c5646fba8c6c2e2087a1e4e38948")
    message(FATAL_ERROR "Expected the pinned llama.cpp commit ad6c66839af3c5646fba8c6c2e2087a1e4e38948")
endif()

if(R2T2_WHEEL_BUILD)
    # Embed llama/mtmd/ggml into the Python extension, with no CLI executable
    # and no dependency on a neighboring Windows build directory.
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
    set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
    set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
    set(GGML_CUDA_NO_VMM ON CACHE BOOL "" FORCE)
    set(GGML_CUDA_NCCL OFF CACHE BOOL "" FORCE)
else()
    set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
endif()
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_MTMD ON CACHE BOOL "" FORCE)
option(GGML_CUDA "Enable NVIDIA CUDA (OFF for CPU wheels)" ON)
set(LLAMA_SUBPROCESS OFF CACHE BOOL "" FORCE)
set(MTMD_VIDEO OFF CACHE BOOL "" FORCE)

add_subdirectory("${LLAMA_CPP_DIR}" llama_cpp EXCLUDE_FROM_ALL)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(qwen3asr_native native_ext.cpp)
target_compile_features(qwen3asr_native PRIVATE cxx_std_17)
target_include_directories(qwen3asr_native PRIVATE
    "${LLAMA_CPP_DIR}/include"
    "${LLAMA_CPP_DIR}/tools/mtmd"
)
target_link_libraries(qwen3asr_native PRIVATE mtmd llama)
if(R2T2_WHEEL_BUILD AND UNIX AND NOT APPLE)
    # Avoid requiring the build machine's newer libstdc++ on user systems.
    target_link_options(qwen3asr_native PRIVATE -static-libstdc++ -static-libgcc)
endif()

# Visual Studio uses RUNTIME_OUTPUT_DIRECTORY for .pyd and adds a configuration
# subdirectory unless the configuration-specific property is also defined.
set_target_properties(qwen3asr_native PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/python"
)

if(R2T2_WHEEL_BUILD)
    if(GGML_CUDA)
        set(R2T2_CUDA_JSON true)
    else()
        set(R2T2_CUDA_JSON false)
    endif()
    file(WRITE "${CMAKE_BINARY_DIR}/build-info.json"
        "{\"api_version\":1,\"llama_commit\":\"ad6c66839af3c5646fba8c6c2e2087a1e4e38948\",\"cuda\":${R2T2_CUDA_JSON}}\n")
    install(TARGETS qwen3asr_native LIBRARY DESTINATION r2t2_native COMPONENT r2t2-native
        RUNTIME DESTINATION r2t2_native COMPONENT r2t2-native)
    install(FILES "${CMAKE_BINARY_DIR}/build-info.json" DESTINATION r2t2_native
        COMPONENT r2t2-native)
    install(FILES "${LLAMA_CPP_DIR}/LICENSE" DESTINATION r2t2_native/licenses
        RENAME llama.cpp-LICENSE COMPONENT r2t2-native)
    install(FILES LICENSE NOTICE.md DESTINATION r2t2_native/licenses COMPONENT r2t2-native)
endif()
