cmake_minimum_required(VERSION 3.17)
project(10)

set(CMAKE_CXX_STANDARD 20)
include_directories(lib/fmt)

add_executable(main01 main01.cpp)
#add_definitions(-DDIM_M=4 -DDIM_N=4 -DDIM_K=4)
add_executable(main02 main02.cpp)
add_executable(main03 main03.cpp gemm.hpp)
