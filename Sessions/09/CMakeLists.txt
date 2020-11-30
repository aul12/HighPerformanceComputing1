cmake_minimum_required(VERSION 3.17)

set(CMAKE_CXX_STANDARD 20)

project(mainA)
add_executable(${PROJECT_NAME} ${PROJECT_NAME}.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE lib/fmt)

project(mainB)
add_executable(${PROJECT_NAME} ${PROJECT_NAME}.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE lib/fmt)
