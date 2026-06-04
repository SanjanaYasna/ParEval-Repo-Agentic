cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(heap CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Use Legion_DIR from environment if set
if(DEFINED ENV{Legion_DIR})
    set(Legion_DIR "$ENV{Legion_DIR}")
    message(STATUS "Using Legion_DIR from environment: $ENV{Legion_DIR}")
endif()

find_package(Legion REQUIRED)
message(STATUS "Found Legion: ${Legion_DIR}")

# Optionally find Boost (used e.g. for program_options replacement or other utilities)
if(DEFINED ENV{BOOST_ROOT})
    set(BOOST_ROOT "$ENV{BOOST_ROOT}")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w -O3")

add_executable(heaps heaps.cpp)
target_link_libraries(heaps Legion::Legion)
