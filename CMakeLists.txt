cmake_minimum_required(VERSION 3.20)

project(fzip
    VERSION 0.1.0
    DESCRIPTION "State-of-the-art ZIP compressor in C++20"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Warnings as errors (MSVC and Clang/GCC-compatible).
if(MSVC)
    add_compile_options(/W4 /WX /permissive- /Zc:__cplusplus /Zc:preprocessor
        /EHsc /utf-8 /nologo)
    add_compile_options($<$<CONFIG:Release>:/Zi>)
    add_link_options($<$<CONFIG:Release>:/DEBUG>)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -Werror)
endif()

if(NOT CMAKE_RUNTIME_LIBRARY_DIRECTORY)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endif()

# --- fzip -------------------------------------------------------------------
add_executable(fzip
    src/main.cpp
    src/zip_writer.cpp
    src/zip_reader.cpp
    src/crc32.cpp
    src/deflate.cpp
    src/zstd_codec.cpp
    src/zstd_decompress.cpp
    src/zstd_compress.cpp
    src/zstd_fse.cpp
    src/xxhash.cpp
    src/codec.cpp
    src/file_type.cpp)
target_include_directories(fzip PRIVATE src)
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
add_test(NAME stage3_deflate
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/verify.py
            --stage 3 --binary $<TARGET_FILE:fzip>
            --workdir ${CMAKE_BINARY_DIR}/test_tmp3)
set_tests_properties(stage3_deflate PROPERTIES TIMEOUT 60)
