cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(heap CXX)

# Require C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w -O3")

# Find Legion
if(DEFINED ENV{Legion_DIR})
    set(Legion_DIR "$ENV{Legion_DIR}")
    message(STATUS "Using Legion_DIR from environment: $ENV{Legion_DIR}")
endif()

find_package(Legion REQUIRED)
message(STATUS "Found Legion: ${Legion_DIR}")

# Find Boost (if needed for any utility usage)
if(DEFINED ENV{BOOST_ROOT})
    set(BOOST_ROOT "$ENV{BOOST_ROOT}")
endif()

# Build the executable
add_executable(heaps heaps.cpp)
target_link_libraries(heaps Legion::Legion)
