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

# One CMakeLists.txt: CMake tracks a target's file-level dependency on the generated
# model.hpp only within the directory that generates it.
# src/ is the one include root: every project include is written relative to it.
add_library(gguf_header
  src/gguf/reader.cpp
  src/gguf/emitter.cpp
)
target_include_directories(gguf_header PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(gguf_header PRIVATE project_warnings xxhash)

add_executable(gguf2hpp apps/gguf2hpp/main.cpp)
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
  # The device side: upload and kernels. nvcc compiles it as C++20.
  add_library(weights STATIC src/weights/weights.cu ${GEN_DIR}/model.hpp)
  # src: the generated header includes model/model_types.hpp from there.
  target_include_directories(weights PUBLIC ${GEN_DIR} ${CMAKE_SOURCE_DIR}/src)

  add_executable(engine apps/engine/main.cpp ${GEN_DIR}/model.hpp)
  # gguf_header: the engine checks its file with the reader's own parse() and hash.
  target_link_libraries(engine PRIVATE weights gguf_header project_warnings)
  # The file `engine` loads when run without an argument.
  target_compile_definitions(engine PRIVATE MODEL_GGUF="${MODEL_GGUF}")

  # The forward pass is written for one architecture, and model.hpp does not exist yet at
  # configure time. So read general.architecture here: it is the first key of a GGUF, and
  # file(STRINGS) returns the key and its value as two consecutive strings. A wrong guess
  # cannot build the wrong thing: forward.hpp static_asserts the architecture.
  file(STRINGS ${MODEL_GGUF} model_strings LIMIT_INPUT 4096 LIMIT_COUNT 8)
  list(FIND model_strings "general.architecture" arch_key)
  set(MODEL_ARCH "")
  if(arch_key GREATER -1)
    math(EXPR arch_value "${arch_key} + 1")
    list(GET model_strings ${arch_value} MODEL_ARCH)
  endif()
  message(STATUS "MODEL_GGUF architecture: '${MODEL_ARCH}'")

  if(MODEL_ARCH STREQUAL "qwen3")
    # The reference forward pass, and the tool that checks it against llama.cpp's dumps.
    add_library(forward_pass STATIC src/reference/forward.cu src/megakernel/megakernel.cu
      ${GEN_DIR}/model.hpp)
    target_link_libraries(forward_pass PUBLIC weights)

    # The engine decodes through the megakernel when its model has a forward pass.
    target_link_libraries(engine PRIVATE forward_pass)
    target_compile_definitions(engine PRIVATE MK_ENGINE_DECODES)

    add_executable(forward apps/forward/main.cpp)
    target_link_libraries(forward PRIVATE forward_pass gguf_header project_warnings)
    target_compile_definitions(forward PRIVATE MODEL_GGUF="${MODEL_GGUF}")
  else()
    message(STATUS "No forward pass for this architecture: only the engine is built")
  endif()
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

# The engine's host-only code: the checksum, BF16, the comparison metrics, the position
# count and the megakernel's schedule proofs. No GPU and no model.
add_executable(host_test tests/weights/checksum_test.cpp tests/base/numeric_test.cpp
  tests/reference/compare_test.cpp tests/decode/positions_test.cpp
  tests/megakernel/schedule_test.cpp)
target_include_directories(host_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(host_test PRIVATE GTest::gtest_main project_warnings)
gtest_discover_tests(host_test)

# The device side, on the model the engine is built for. Needs a GPU.
if(MODEL_GGUF)
  add_executable(weights_test tests/weights/weights_test.cpp)
  target_link_libraries(weights_test PRIVATE weights gguf_header GTest::gtest_main project_warnings)
  target_compile_definitions(weights_test PRIVATE MODEL_GGUF="${MODEL_GGUF}")
  gtest_discover_tests(weights_test)
endif()

# The megakernel against the reference pass, phase by phase. Needs the model and a GPU.
if(TARGET forward)
  # The engine itself generates llama.cpp's first tokens for the committed prompt.
  add_test(NAME engine.generates_the_reference_tokens
    COMMAND engine --tokens ${CMAKE_SOURCE_DIR}/tests/data/reference/planets/tokens.txt --generate 8)
  set_tests_properties(engine.generates_the_reference_tokens PROPERTIES
    PASS_REGULAR_EXPRESSION "generated: \\[25, 362, 8, 43562, 11, 48976, 11, 9237\\]")
  # A count whose positions wrap in size_t is refused, and one past SIZE_MAX does not parse.
  add_test(NAME engine.refuses_a_count_that_wraps
    COMMAND engine --tokens ${CMAKE_SOURCE_DIR}/tests/data/reference/planets/tokens.txt
            --generate 18446744073709551613)
  set_tests_properties(engine.refuses_a_count_that_wraps PROPERTIES PASS_REGULAR_EXPRESSION
    "positions are more than the [0-9]+ this build's KV cache holds")
  add_test(NAME engine.refuses_a_count_past_size_max
    COMMAND engine --tokens ${CMAKE_SOURCE_DIR}/tests/data/reference/planets/tokens.txt
            --generate 99999999999999999999999)
  set_tests_properties(engine.refuses_a_count_past_size_max PROPERTIES PASS_REGULAR_EXPRESSION
    "^usage: engine")
  add_test(NAME megakernel.each_phase_matches_reference
    COMMAND forward --megakernel --check ${CMAKE_SOURCE_DIR}/tests/data/reference/planets/tokens.txt)
  # Enough positions for an attention warp to sum more than one: with the default grid, any
  # count above 128.
  add_test(NAME megakernel.matches_reference_at_333_positions
    COMMAND forward --megakernel --check --generate 320
            ${CMAKE_SOURCE_DIR}/tests/data/reference/planets/tokens.txt)
endif()

# The forward pass against llama.cpp's dump of the committed prompt. The dump is not in the
# repo: tools/reference/run.sh writes it to a cache, and without it there is no test.
if(TARGET forward)
  file(STRINGS ${CMAKE_SOURCE_DIR}/tools/reference/build.sh llama_tag REGEX "^LLAMA_TAG=")
  string(REPLACE "LLAMA_TAG=" "" llama_tag "${llama_tag}")
  if(DEFINED ENV{XDG_CACHE_HOME})
    set(cache_home $ENV{XDG_CACHE_HOME})
  else()
    set(cache_home $ENV{HOME}/.cache)
  endif()
  set(REFERENCE_DUMP ${cache_home}/megakernel/reference/${llama_tag}/planets)
  set(planets ${CMAKE_SOURCE_DIR}/tests/data/reference/planets)
  if(EXISTS ${REFERENCE_DUMP}/manifest.json)
    add_test(NAME forward.each_op_matches_llama
      COMMAND forward --compare ${REFERENCE_DUMP} --per-op ${planets}/tokens.txt)
    add_test(NAME forward.exact_stays_close_to_llama
      COMMAND forward --compare ${REFERENCE_DUMP} ${planets}/tokens.txt)
  else()
    message(STATUS "No llama.cpp dump at ${REFERENCE_DUMP}: the forward tests are not added")
  endif()
endif()

set(TOY_GGUF ${CMAKE_SOURCE_DIR}/tests/data/toy.gguf)
set(QWEN_GGUF ${CMAKE_SOURCE_DIR}/models/Qwen_Qwen3-0.6B-bf16.gguf)

add_executable(reader_test tests/gguf/reader_test.cpp tests/gguf/emitter_test.cpp)
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
# src: the generated header includes model/model_types.hpp from there.
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
