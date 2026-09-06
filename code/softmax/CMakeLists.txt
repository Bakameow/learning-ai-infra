cmake_minimum_required(VERSION 3.18)

project(softmax LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 不指定具体 GPU 架构，保持最简单快速；需要时可在 cmake 时传入：
# cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=80
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES OFF)
endif()

file(GLOB SOFTMAX_CU_FILES CONFIGURE_DEPENDS "*.cu")

set(SOFTMAX_TARGETS)
set(SOFTMAX_RUN_TARGETS)
foreach(cu_file IN LISTS SOFTMAX_CU_FILES)
    get_filename_component(name "${cu_file}" NAME_WE)
    set(target "${name}")

    add_executable(${target} "${cu_file}")
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O2>
    )

    add_custom_target(run_${name}
        COMMAND $<TARGET_FILE:${target}>
        DEPENDS ${target}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        USES_TERMINAL
    )

    list(APPEND SOFTMAX_TARGETS ${target})
    list(APPEND SOFTMAX_RUN_TARGETS run_${name})
endforeach()

add_custom_target(build_all DEPENDS ${SOFTMAX_TARGETS})
add_custom_target(run_all DEPENDS ${SOFTMAX_RUN_TARGETS})
