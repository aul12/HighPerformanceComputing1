cmake_minimum_required(VERSION 3.17)
project(10)

set(CMAKE_CXX_STANDARD 20)
include_directories(lib/fmt)

add_executable(main01 main01.cpp)
add_definitions(-DDIM_M=1 -DDIM_N=13 -DDIM_K=15 -DUGEMM_MR_DEFAUL=2 -DUGEMM_NR_DEFAULT=3
        -DUGEMM_MC_DEFAULT=8 -DUGEMM_NC_DEFAULT=6 -DUGEMM_KC_DEFAULT=4 -DBETA=0)

add_executable(main02 main02.cpp)
add_executable(main03 main03.cpp gemm.hpp)
