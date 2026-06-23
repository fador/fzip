cmake_minimum_required(VERSION 3.20)

project(fzip
    VERSION 0.1.0
    DESCRIPTION "State-of-the-art ZIP compressor in C++20"
    LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Warnings as errors (MSVC and Clang/GCC-compatible).
if(MSVC)
    add_compile_options(/W4 /WX /permissive- /Zc:__cplusplus /Zc:preprocessor
        /EHsc /utf-8 /nologo)
    # Debug info in release for better crash diagnostics.
    add_compile_options($<$<CONFIG:Release>:/Zi>)
    add_link_options($<$<CONFIG:Release>:/DEBUG>)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -Werror)
endif()

if(NOT CMAKE_RUNTIME_LIBRARY_DIRECTORY)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endif()

# --- zstd (vendored submodule) ----------------------------------------------
# Build only the libzstd static library from the submodule; skip CLI/programs.
set(ZSTD_BUILD_PROGRAMS OFF CACHE BOOL "" FORCE)
set(ZSTD_BUILD_STATIC ON CACHE BOOL "" FORCE)
set(ZSTD_BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(ZSTD_BUILD_CONTRIB OFF CACHE BOOL "" FORCE)
set(ZSTD_BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/zstd/build/cmake EXCLUDE_FROM_ALL)

# --- fzip -------------------------------------------------------------------
add_executable(fzip
    src/main.cpp
    src/zip_writer.cpp
    src/crc32.cpp
    src/deflate.cpp
    src/zstd_codec.cpp
    src/codec.cpp
    src/file_type.cpp)
target_include_directories(fzip PRIVATE src)
target_link_libraries(fzip PRIVATE libzstd_static)
target_compile_features(fzip PRIVATE cxx_std_20)

# --- Tests ------------------------------------------------------------------
enable_testing()
find_package(Python3 COMPONENTS Interpreter REQUIRED)
add_test(NAME stage0_smoke
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/verify.py
            --stage 0 --binary $<TARGET_FILE:fzip>
            --workdir ${CMAKE_BINARY_DIR}/test_tmp)
add_test(NAME stage1_store
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/verify.py
            --stage 1 --binary $<TARGET_FILE:fzip>
            --workdir ${CMAKE_BINARY_DIR}/test_tmp)
add_test(NAME stage2_zip64
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/verify.py
            --stage 2 --binary $<TARGET_FILE:fzip>
            --workdir ${CMAKE_BINARY_DIR}/test_tmp2)
set_tests_properties(stage2_zip64 PROPERTIES TIMEOUT 300)
