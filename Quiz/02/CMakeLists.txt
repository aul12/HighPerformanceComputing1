cmake_minimum_required(VERSION 3.17)
project(02)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 20)

add_executable(${PROJECT_NAME} main.cpp dot.cpp gemv.cpp)
target_compile_definitions(${PROJECT_NAME} PRIVATE USE_AS_LIB)
target_include_directories(${PROJECT_NAME} PRIVATE lib/fmt)
