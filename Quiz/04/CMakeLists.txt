cmake_minimum_required(VERSION 3.17)
project(04)

set(CMAKE_CXX_STANDARD 20)

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} pthread)
target_include_directories(${PROJECT_NAME} PRIVATE SYSTEM .)
