cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(heaps LANGUAGES CXX)

if(POLICY CMP0144)
  cmake_policy(SET CMP0144 NEW)
endif()
if(POLICY CMP0167)
  cmake_policy(SET CMP0167 OLD)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Use Legion_DIR from environment if provided
if(DEFINED ENV{Legion_DIR} AND NOT DEFINED Legion_DIR)
  set(Legion_DIR "$ENV{Legion_DIR}" CACHE PATH "Path to Legion CMake config")
endif()

# Also support LEGION_ROOT by deriving Legion_DIR
if(DEFINED ENV{LEGION_ROOT} AND NOT DEFINED Legion_DIR)
  if(EXISTS "$ENV{LEGION_ROOT}/lib/cmake/Legion")
    set(Legion_DIR "$ENV{LEGION_ROOT}/lib/cmake/Legion" CACHE PATH "Path to Legion CMake config")
  elseif(EXISTS "$ENV{LEGION_ROOT}/share/Legion/cmake")
    set(Legion_DIR "$ENV{LEGION_ROOT}/share/Legion/cmake" CACHE PATH "Path to Legion CMake config")
  endif()
endif()

find_package(Legion REQUIRED CONFIG)
find_package(Boost REQUIRED COMPONENTS program_options)

add_executable(heaps
  heaps.cpp
  sift.hpp
)

# Keep optimization behavior similar to the original project
target_compile_options(heaps PRIVATE -O3 -w)

# Link Legion (handle common package export styles)
if(TARGET Legion::Legion)
  target_link_libraries(heaps PRIVATE Legion::Legion)
elseif(TARGET legion)
  target_link_libraries(heaps PRIVATE legion)
elseif(DEFINED Legion_LIBRARIES)
  target_link_libraries(heaps PRIVATE ${Legion_LIBRARIES})
else()
  message(FATAL_ERROR "Legion found, but no known Legion target/library was exported.")
endif()

# Include dirs if provided by package variables
if(DEFINED Legion_INCLUDE_DIRS)
  target_include_directories(heaps PRIVATE ${Legion_INCLUDE_DIRS})
endif()

# Boost program_options
if(TARGET Boost::program_options)
  target_link_libraries(heaps PRIVATE Boost::program_options)
else()
  target_include_directories(heaps PRIVATE ${Boost_INCLUDE_DIRS})
  target_link_libraries(heaps PRIVATE ${Boost_LIBRARIES})
endif()
