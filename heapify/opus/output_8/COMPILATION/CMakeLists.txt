cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(heap CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags; suppress warnings like the original
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w -O3")

# Find Legion using the Legion_DIR environment variable or CMake cache
if(DEFINED ENV{Legion_DIR})
    set(Legion_DIR "$ENV{Legion_DIR}" CACHE PATH "Path to Legion CMake config")
    message(STATUS "Using Legion_DIR from environment: $ENV{Legion_DIR}")
endif()

find_package(Legion REQUIRED)
message(STATUS "Found Legion: ${Legion_DIR}")

# Find Boost (used for program_options-like argument parsing if needed,
# but primarily here because the original project references BOOST_ROOT)
if(DEFINED ENV{BOOST_ROOT})
    set(BOOST_ROOT "$ENV{BOOST_ROOT}")
endif()

# Build the executable
add_executable(heaps heaps.cpp)
target_link_libraries(heaps Legion::Legion)
target_include_directories(heaps PRIVATE ${CMAKE_SOURCE_DIR})
