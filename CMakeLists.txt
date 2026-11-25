cmake_minimum_required(VERSION 3.28)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89 120)
endif()

project(comptime_megakernel LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

# Warnings: interface target so they hit our code only, not googletest.
option(WARNINGS_AS_ERRORS "Treat warnings as errors" OFF)
add_library(project_warnings INTERFACE)
set(WARNING_FLAGS
  -Wall -Wextra -Wpedantic
  -Wshadow -Wconversion -Wsign-conversion
  -Wold-style-cast -Wcast-align -Wnull-dereference
  -Wdouble-promotion -Wformat=2 -Wimplicit-fallthrough
  -Wnon-virtual-dtor -Woverloaded-virtual
)
if(WARNINGS_AS_ERRORS)
  list(APPEND WARNING_FLAGS -Werror)
endif()
# Quoted so the list stays inside the genex.
target_compile_options(project_warnings INTERFACE "$<$<COMPILE_LANGUAGE:CXX>:${WARNING_FLAGS}>")

# libstdc++ bounds/precondition checks in Debug.
add_compile_definitions($<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:_GLIBCXX_ASSERTIONS>)

# Sanitizers: global so googletest is instrumented too (mixing causes false positives).
# Host code only; use compute-sanitizer for kernels.
option(ENABLE_SANITIZERS "Build with ASan + UBSan" OFF)
if(ENABLE_SANITIZERS)
  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address,undefined;-fno-omit-frame-pointer;-fno-sanitize-recover=all>")
  add_link_options(-fsanitize=address,undefined)
endif()

set(GEN_DIR ${CMAKE_BINARY_DIR}/generated)
# A generated header has no compile command, so clangd guesses one and cannot find
# model_types.hpp. This tells it where to look when such a header is opened in an editor.
file(WRITE ${GEN_DIR}/.clangd
  "CompileFlags:\n  Add: [-std=c++${CMAKE_CXX_STANDARD}, -I${CMAKE_SOURCE_DIR}/src]\n")

# xxHash, header-only: the repo root has no CMakeLists.txt, so this only downloads it.
include(FetchContent)
FetchContent_Declare(
  xxhash
  GIT_REPOSITORY https://github.com/Cyan4973/xxHash.git
  GIT_TAG        v0.8.4
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(xxhash)
add_library(xxhash INTERFACE)
# SYSTEM: our warning flags are not meant for a C header.
target_include_directories(xxhash SYSTEM INTERFACE ${xxhash_SOURCE_DIR})
target_compile_definitions(xxhash INTERFACE XXH_INLINE_ALL)

add_library(gguf_header
  tools/gguf_header/reader.cpp
  tools/gguf_header/emitter.cpp
)
target_include_directories(gguf_header PUBLIC tools/gguf_header)
target_link_libraries(gguf_header PRIVATE project_warnings xxhash)

add_executable(gguf2hpp tools/gguf_header/main.cpp)
target_link_libraries(gguf2hpp PRIVATE gguf_header project_warnings)

# --- engine ---
# main.cpp includes the generated model.hpp, so there is no engine without a model.
set(MODEL_GGUF "" CACHE FILEPATH "GGUF to compile the engine for")
if(MODEL_GGUF)
  add_custom_command(
    OUTPUT  ${GEN_DIR}/model.hpp
    COMMAND ${CMAKE_COMMAND} -E make_directory ${GEN_DIR}
    COMMAND gguf2hpp ${MODEL_GGUF} ${GEN_DIR}/model.hpp
    DEPENDS gguf2hpp ${MODEL_GGUF}
    COMMENT "Generating model.hpp from ${MODEL_GGUF}"
  )
  add_executable(engine src/main.cpp ${GEN_DIR}/model.hpp)
  target_link_libraries(engine PRIVATE project_warnings)
  # src: the generated header includes model_types.hpp from there.
  target_include_directories(engine PRIVATE ${GEN_DIR} ${CMAKE_SOURCE_DIR}/src)
else()
  message(WARNING "MODEL_GGUF not set: the engine target is not built")
endif()

# --- tests ---
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.17.0
)
FetchContent_MakeAvailable(googletest)

enable_testing()
include(GoogleTest)

add_executable(smoke_test tests/smoke_test.cpp)
target_link_libraries(smoke_test PRIVATE GTest::gtest_main project_warnings)
gtest_discover_tests(smoke_test)

set(TOY_GGUF ${CMAKE_SOURCE_DIR}/tests/data/toy.gguf)
set(QWEN_GGUF ${CMAKE_SOURCE_DIR}/models/Qwen_Qwen3-0.6B-bf16.gguf)

add_executable(reader_test tests/reader_test.cpp tests/emitter_test.cpp)
target_link_libraries(reader_test PRIVATE gguf_header GTest::gtest_main project_warnings)
target_compile_definitions(reader_test PRIVATE
  TOY_GGUF="${TOY_GGUF}"
  QWEN_GGUF="${QWEN_GGUF}")
gtest_discover_tests(reader_test)

# Milestone 7: emitter_test includes the header gguf2hpp generates from toy.gguf.
add_custom_command(
  OUTPUT  ${GEN_DIR}/toy.hpp
  COMMAND ${CMAKE_COMMAND} -E make_directory ${GEN_DIR}
  COMMAND gguf2hpp ${TOY_GGUF} ${GEN_DIR}/toy.hpp
  DEPENDS gguf2hpp ${TOY_GGUF}
)
target_sources(reader_test PRIVATE ${GEN_DIR}/toy.hpp)
# src: the generated header includes model_types.hpp from there.
target_include_directories(reader_test PRIVATE ${GEN_DIR} ${CMAKE_SOURCE_DIR}/src)

# The real model is not in the repo. When it is there at configure time, emitter_test also
# checks its header, which has layers; toy.gguf has none. The test looks for qwen.hpp itself.
if(EXISTS ${QWEN_GGUF})
  add_custom_command(
    OUTPUT  ${GEN_DIR}/qwen.hpp
    COMMAND ${CMAKE_COMMAND} -E make_directory ${GEN_DIR}
    COMMAND gguf2hpp ${QWEN_GGUF} ${GEN_DIR}/qwen.hpp
    DEPENDS gguf2hpp ${QWEN_GGUF}
  )
  target_sources(reader_test PRIVATE ${GEN_DIR}/qwen.hpp)
endif()
