cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(heaps LANGUAGES CXX)

# Honor Legion_DIR from environment if provided
if(DEFINED ENV{Legion_DIR} AND NOT Legion_DIR)
  set(Legion_DIR "$ENV{Legion_DIR}" CACHE PATH "Path to Legion CMake package config")
  message(STATUS "Using Legion_DIR from environment: $ENV{Legion_DIR}")
endif()

# Honor BOOST_ROOT from environment if provided
if(DEFINED ENV{BOOST_ROOT} AND NOT BOOST_ROOT)
  set(BOOST_ROOT "$ENV{BOOST_ROOT}" CACHE PATH "Boost root")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Dependencies
find_package(Legion REQUIRED)
find_package(Boost QUIET COMPONENTS program_options)

# Executable name should be "heaps"
add_executable(heaps
  heaps.cpp
  sift.hpp
)

# Compile options analogous to original build flags
target_compile_options(heaps PRIVATE -O3 -w)

# Link Legion (support both modern imported target and variable-based packages)
if(TARGET Legion::Legion)
  target_link_libraries(heaps PRIVATE Legion::Legion)
elseif(TARGET legion)
  target_link_libraries(heaps PRIVATE legion)
else()
  target_include_directories(heaps PRIVATE ${Legion_INCLUDE_DIRS})
  target_link_libraries(heaps PRIVATE ${Legion_LIBRARIES})
endif()

# Link Boost program_options if available
if(Boost_FOUND)
  if(TARGET Boost::program_options)
    target_link_libraries(heaps PRIVATE Boost::program_options)
  else()
    target_include_directories(heaps PRIVATE ${Boost_INCLUDE_DIRS})
    target_link_libraries(heaps PRIVATE ${Boost_LIBRARIES})
  endif()
endif()
