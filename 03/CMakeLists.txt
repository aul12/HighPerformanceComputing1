cmake_minimum_required(VERSION 3.17)
project(03 C)

set(CMAKE_C_FLAGS "-Wall -Wextra -Wpedantic -mtune=native -march=native")
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address -fno-omit-frame-pointer")
else()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
endif ()

set(CMAKE_C_STANDARD 11)

add_executable(03 main.c)

add_executable(bench_initmatrix bench_initmatrix.c)

add_executable(bench_gmm bench_gmm.c)
