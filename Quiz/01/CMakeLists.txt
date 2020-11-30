cmake_minimum_required(VERSION 3.17)
project(01 C)

set(CMAKE_C_STANDARD 99)

add_executable(${PROJECT_NAME} main.c quiz01.c)
target_compile_definitions(${PROJECT_NAME} PRIVATE USE_AS_LIB)
