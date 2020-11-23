cmake_minimum_required(VERSION 3.17)
project(06 C)

set(CMAKE_C_STANDARD 11)

add_executable(${PROJECT_NAME} gemv.c)
target_link_libraries(${PROJECT_NAME} mkl_intel_lp64 mkl_sequential mkl_core pthread m)
target_compile_options(${PROJECT_NAME} PRIVATE -Ofast -mtune=native -march=native)

