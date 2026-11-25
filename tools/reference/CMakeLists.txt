# Standalone project, not part of the repo build: build.sh configures it against a llama.cpp checkout.
cmake_minimum_required(VERSION 3.28)
project(megakernel_reference LANGUAGES C CXX)

set(LLAMA_DIR "" CACHE PATH "llama.cpp checkout")
set(LLAMA_TAG "" CACHE STRING "llama.cpp release tag, recorded in manifest.json")
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
set(GGML_CUDA OFF CACHE BOOL "" FORCE)
add_subdirectory(${LLAMA_DIR} llama.cpp EXCLUDE_FROM_ALL)

add_executable(ref-dump dump.cpp)
target_link_libraries(ref-dump PRIVATE llama)
target_compile_features(ref-dump PRIVATE cxx_std_20)
target_compile_definitions(ref-dump PRIVATE LLAMA_TAG="${LLAMA_TAG}")
