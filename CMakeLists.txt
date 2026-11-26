cmake_minimum_required(VERSION 3.28)

# The GPU the build is for, named once: a type in src/config/gpu.hpp, and here its CUDA
# architecture. CUDA compiles for that one, and gpu.hpp static_asserts that the two agree.
set(MK_GPU Rtx4070TiSuper CACHE STRING "The GPU to compile for: a type in src/config/gpu.hpp")
set(MK_GPU_ARCH_Rtx4070TiSuper 89)
set(MK_GPU_ARCH_Rtx5070 120)
if(NOT DEFINED MK_GPU_ARCH_${MK_GPU})
  message(FATAL_ERROR "MK_GPU=${MK_GPU} is not a GPU this build knows: add its type to "
    "src/config/gpu.hpp and its architecture here")
endif()
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES ${MK_GPU_ARCH_${MK_GPU}})
endif()
message(STATUS "MK_GPU: ${MK_GPU}, CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

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

# What the configure runs to read general.architecture. A target too, so the build compiles
# it with the warnings and ctest runs it.
add_executable(gguf_architecture apps/gguf_architecture/main.cpp)
target_link_libraries(gguf_architecture PRIVATE gguf_header project_warnings)

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
  # gpu.hpp's BuildGpu: everything that includes it links weights.
  target_compile_definitions(weights PUBLIC MK_GPU=${MK_GPU})

  add_executable(engine apps/engine/main.cpp ${GEN_DIR}/model.hpp)
  # gguf_header: the engine checks its file with the reader's own parse() and hash.
  target_link_libraries(engine PRIVATE weights gguf_header project_warnings)
  # The file `engine` loads when run without an argument.
  target_compile_definitions(engine PRIVATE MODEL_GGUF="${MODEL_GGUF}")

  # The decoder is written per architecture, and model.hpp does not exist yet at configure
  # time. So the project's own reader reads general.architecture here: gguf_architecture,
  # compiled from the reader's sources at the first configure and again when they change.
  set(arch_probe ${CMAKE_BINARY_DIR}/CMakeFiles/gguf_architecture)
  set(arch_probe_sources ${CMAKE_SOURCE_DIR}/apps/gguf_architecture/main.cpp
    ${CMAKE_SOURCE_DIR}/src/gguf/reader.cpp)
  foreach(source ${arch_probe_sources} ${CMAKE_SOURCE_DIR}/src/gguf/reader.hpp)
    # Also true when the probe does not exist yet.
    if(${source} IS_NEWER_THAN ${arch_probe})
      try_compile(arch_probe_built SOURCES ${arch_probe_sources}
        CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${CMAKE_SOURCE_DIR}/src;${xxhash_SOURCE_DIR}"
        COMPILE_DEFINITIONS -DXXH_INLINE_ALL
        CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON NO_CACHE
        OUTPUT_VARIABLE arch_probe_log COPY_FILE ${arch_probe})
      if(NOT arch_probe_built)
        message(FATAL_ERROR "Cannot build gguf_architecture:\n${arch_probe_log}")
      endif()
      break()
    endif()
  endforeach()
  execute_process(COMMAND ${arch_probe} ${MODEL_GGUF} RESULT_VARIABLE arch_exit
    OUTPUT_VARIABLE MODEL_ARCH ERROR_VARIABLE arch_error)
  if(NOT arch_exit EQUAL 0)
    message(FATAL_ERROR "MODEL_GGUF: ${arch_error}")
  endif()
  message(STATUS "MODEL_GGUF architecture: '${MODEL_ARCH}'")

  # The architectures with a decoder: the model type, defined in src/arch/<architecture>.hpp,
  # and the decoder's sources. Another size of one of them is another MODEL_GGUF, no edit.
  set(MK_MODEL_qwen3 Qwen3)
  set(MK_DECODER_qwen3 src/reference/forward.cu src/megakernel/megakernel.cu)
  # The value comes from the file: it names a variable only when it is a plain name.
  set(MK_MODEL "")
  if(MODEL_ARCH MATCHES "^[a-z0-9_]+$")
    set(MK_MODEL "${MK_MODEL_${MODEL_ARCH}}")
  endif()

  if(MK_MODEL)
    # The reference forward pass and the megakernel, and the tool that checks the first
    # against llama.cpp's dumps and the second against the first.
    add_library(forward_pass STATIC ${MK_DECODER_${MODEL_ARCH}} ${GEN_DIR}/model.hpp)
    target_link_libraries(forward_pass PUBLIC weights)
    # config/build.hpp's BuildModel.
    target_compile_definitions(forward_pass PUBLIC MK_MODEL=${MK_MODEL}
      MK_MODEL_HEADER="arch/${MODEL_ARCH}.hpp")

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
# count, the megakernel's schedule proofs and the model concept. No GPU and no model.
add_executable(host_test tests/weights/checksum_test.cpp tests/base/numeric_test.cpp
  tests/reference/compare_test.cpp tests/decode/positions_test.cpp
  tests/megakernel/schedule_test.cpp tests/config/model_spec_test.cpp)
target_include_directories(host_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(host_test PRIVATE GTest::gtest_main project_warnings)
# config.hpp names a BuildGpu: model_spec_test derives configurations from test models.
target_compile_definitions(host_test PRIVATE MK_GPU=${MK_GPU})
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

# The probe the configure runs prints the architecture and nothing else. ctest ends the
# output it matches with a newline.
add_test(NAME gguf_architecture.prints_toy COMMAND gguf_architecture ${TOY_GGUF})
set_tests_properties(gguf_architecture.prints_toy PROPERTIES PASS_REGULAR_EXPRESSION "^toy\n?$")

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
