cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(heap_legion LANGUAGES CXX)

if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()
if(POLICY CMP0144)
  cmake_policy(SET CMP0144 NEW)
endif()

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Use Legion_DIR from environment if not already provided to CMake
if(NOT Legion_DIR AND DEFINED ENV{Legion_DIR})
  set(Legion_DIR "$ENV{Legion_DIR}")
endif()
if(NOT Legion_ROOT AND DEFINED ENV{LEGION_ROOT})
  set(Legion_ROOT "$ENV{LEGION_ROOT}")
endif()
if(NOT Boost_ROOT AND DEFINED ENV{BOOST_ROOT})
  set(Boost_ROOT "$ENV{BOOST_ROOT}")
endif()

# Dependencies
find_package(Legion REQUIRED CONFIG)
find_package(Boost REQUIRED COMPONENTS program_options)

# Executable name should be: heaps
add_executable(heaps heaps.cpp)

# Local headers (e.g., sift.hpp)
target_include_directories(heaps PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Link Boost
target_link_libraries(heaps PRIVATE Boost::program_options)

# Link Legion (handle common package target/variable layouts)
if(TARGET Legion::Legion)
  target_link_libraries(heaps PRIVATE Legion::Legion)
elseif(TARGET Legion)
  target_link_libraries(heaps PRIVATE Legion)
elseif(DEFINED Legion_LIBRARIES)
  target_include_directories(heaps PRIVATE ${Legion_INCLUDE_DIRS})
  target_link_libraries(heaps PRIVATE ${Legion_LIBRARIES})
elseif(DEFINED LEGION_LIBRARIES)
  target_include_directories(heaps PRIVATE ${LEGION_INCLUDE_DIRS})
  target_link_libraries(heaps PRIVATE ${LEGION_LIBRARIES})
else()
  message(FATAL_ERROR "Legion was found, but no known Legion target/library variables were exported.")
endif()
