cmake_minimum_required(VERSION 3.23)

project(sportprog)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.1")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse2")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512vl")

#set(CMAKE_CXX_FLAGS "-Wall -Wextra")

#set(CMAKE_CXX_FLAGS "-O3")

add_executable(sportprog main.cpp)
