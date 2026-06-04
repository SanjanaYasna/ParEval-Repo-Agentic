cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(heap CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Legion
if(DEFINED ENV{Legion_DIR})
    set(Legion_DIR "$ENV{Legion_DIR}")
    message(STATUS "Using Legion_DIR from environment: ${Legion_DIR}")
endif()

find_package(Legion REQUIRED)
message(STATUS "Found Legion: ${Legion_DIR}")

# Find Boost (used for program_options-like argument parsing if needed,
# but Legion uses its own runtime init; keep for potential utility)
if(DEFINED ENV{BOOST_ROOT})
    set(BOOST_ROOT "$ENV{BOOST_ROOT}")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w -O3")

add_executable(heaps heaps.cpp)
target_link_libraries(heaps Legion::Legion)
