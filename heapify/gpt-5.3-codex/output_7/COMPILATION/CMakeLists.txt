cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(heaps LANGUAGES CXX)

# C++17 as required by the codebase
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Optional: honor Legion_DIR from environment if provided
if(DEFINED ENV{Legion_DIR} AND NOT Legion_DIR)
  set(Legion_DIR "$ENV{Legion_DIR}" CACHE PATH "Path to Legion CMake package config")
endif()

# Dependencies
find_package(Legion REQUIRED CONFIG)
find_package(Boost REQUIRED COMPONENTS program_options)

# Build executable (must be named 'heaps')
add_executable(heaps
  heaps.cpp
  sift.hpp
)

# Boost linkage
target_link_libraries(heaps PRIVATE Boost::program_options)

# Legion linkage (support both modern imported target and variable-based exports)
if(TARGET Legion::Legion)
  target_link_libraries(heaps PRIVATE Legion::Legion)
elseif(DEFINED Legion_LIBRARIES)
  target_include_directories(heaps PRIVATE ${Legion_INCLUDE_DIRS})
  target_link_libraries(heaps PRIVATE ${Legion_LIBRARIES})
else()
  message(FATAL_ERROR "Legion package found, but no Legion::Legion target or Legion_LIBRARIES were exported.")
endif()
