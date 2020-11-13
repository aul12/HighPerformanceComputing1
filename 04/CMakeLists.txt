cmake_minimum_required(VERSION 3.17)
project(04 C)

set(CMAKE_C_STANDARD 11)

add_executable(${PROJECT_NAME} main.c)
target_compile_options(${PROJECT_NAME} PRIVATE -O3 -mtune=native -march=native)
