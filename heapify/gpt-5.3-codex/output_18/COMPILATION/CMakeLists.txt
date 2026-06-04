cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(heaps LANGUAGES CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Honor Legion_DIR from environment (expected to point to Legion CMake config dir)
if(DEFINED ENV{Legion_DIR} AND NOT Legion_DIR)
  set(Legion_DIR "$ENV{Legion_DIR}" CACHE PATH "Path to Legion CMake configuration")
  message(STATUS "Using Legion_DIR from environment: $ENV{Legion_DIR}")
endif()

# Dependencies
find_package(Legion REQUIRED CONFIG)
find_package(Boost REQUIRED COMPONENTS program_options)

# Executable
add_executable(heaps
  heaps.cpp
  sift.hpp
)

target_include_directories(heaps PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link Legion target (handle common exported target names + fallback variables)
if(TARGET Legion::Legion)
  target_link_libraries(heaps PRIVATE Legion::Legion Boost::program_options)
elseif(TARGET Legion)
  target_link_libraries(heaps PRIVATE Legion Boost::program_options)
else()
  message(STATUS "Using Legion include/library variables fallback")
  target_include_directories(heaps PRIVATE ${Legion_INCLUDE_DIRS})
  target_link_libraries(heaps PRIVATE ${Legion_LIBRARIES} Boost::program_options)
endif()
