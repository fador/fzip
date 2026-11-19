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

option(FZIP_BUILD_GUI "Build the optional fzip-gui application" OFF)

# Warnings as errors (MSVC and Clang/GCC-compatible).
if(MSVC)
    add_compile_options(/W4 /WX /permissive- /Zc:__cplusplus /Zc:preprocessor
        /EHsc /utf-8 /nologo)
    add_compile_options(
        $<$<CONFIG:Release>:/Zi>
        $<$<CONFIG:Release>:/Oi>
        $<$<CONFIG:Release>:/Ot>
        $<$<CONFIG:Release>:/GL>)
    add_link_options(
        $<$<CONFIG:Release>:/DEBUG>
        $<$<CONFIG:Release>:/LTCG>)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -Werror)
    add_compile_options($<$<CONFIG:Release>:-O3>)
endif()

if(NOT CMAKE_RUNTIME_LIBRARY_DIRECTORY)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endif()

# Thread pool support
find_package(Threads REQUIRED)

# --- fzip_core ---------------------------------------------------------------
add_library(fzip_core STATIC
    src/zip_writer.cpp
    src/zip_reader.cpp
    src/crc32.cpp
    src/deflate.cpp
    src/zstd_codec.cpp
    src/zstd_decompress.cpp
    src/zstd_compress.cpp
    src/zstd_fse.cpp
    src/zstd_huffman.cpp
    src/zstd_sequence.cpp
    src/xxhash.cpp
    src/codec.cpp
    src/file_type.cpp)
target_include_directories(fzip_core PUBLIC src)
target_compile_features(fzip_core PUBLIC cxx_std_20)
target_link_libraries(fzip_core PUBLIC Threads::Threads)

# --- fzip CLI ----------------------------------------------------------------
add_executable(fzip src/main.cpp)
target_link_libraries(fzip PRIVATE fzip_core)
target_compile_features(fzip PRIVATE cxx_std_20)

# --- Optional fzip-gui -------------------------------------------------------
if(FZIP_BUILD_GUI)
    include(FetchContent)
    FetchContent_Declare(
        imgui
        GIT_REPOSITORY https://github.com/ocornut/imgui.git
        GIT_TAG v1.91.8
    )
    FetchContent_MakeAvailable(imgui)

    set(IMGUI_SRCS
        ${imgui_SOURCE_DIR}/imgui.cpp
        ${imgui_SOURCE_DIR}/imgui_demo.cpp
        ${imgui_SOURCE_DIR}/imgui_draw.cpp
        ${imgui_SOURCE_DIR}/imgui_tables.cpp
        ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    )

    if(WIN32)
        list(APPEND IMGUI_SRCS
            ${imgui_SOURCE_DIR}/backends/imgui_impl_win32.cpp
            ${imgui_SOURCE_DIR}/backends/imgui_impl_dx11.cpp
        )
    endif()

    add_library(imgui_lib STATIC ${IMGUI_SRCS})
    target_include_directories(imgui_lib PUBLIC
        ${imgui_SOURCE_DIR}
        ${imgui_SOURCE_DIR}/backends
    )
    # Suppress warnings in third-party ImGui
    if(MSVC)
        target_compile_options(imgui_lib PRIVATE /W0)
    else()
        target_compile_options(imgui_lib PRIVATE -w)
    endif()

    add_executable(fzip-gui
        src/gui/gui_main.cpp
        src/gui/app_state.cpp
        src/gui/native_dialogs.cpp
        src/gui/context_menu.cpp
    )
    target_include_directories(fzip-gui PRIVATE src src/gui)
    target_link_libraries(fzip-gui PRIVATE fzip_core imgui_lib)

    if(WIN32)
        target_link_libraries(fzip-gui PRIVATE d3d11 dxgi d3dcompiler ole32 uuid shell32 advapi32 comdlg32)
    endif()
endif()

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
add_test(NAME stage4_zstd
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/verify.py
            --stage 4 --binary $<TARGET_FILE:fzip>
            --workdir ${CMAKE_BINARY_DIR}/test_tmp4)
set_tests_properties(stage4_zstd PROPERTIES TIMEOUT 60)
add_test(NAME stage5_auto
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/verify.py
            --stage 5 --binary $<TARGET_FILE:fzip>
            --workdir ${CMAKE_BINARY_DIR}/test_tmp5)
set_tests_properties(stage5_auto PROPERTIES TIMEOUT 60)
