cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(heaps_legion LANGUAGES CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Optional default build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Respect environment-provided package hints
if(NOT DEFINED Legion_DIR AND DEFINED ENV{Legion_DIR})
  set(Legion_DIR "$ENV{Legion_DIR}")
endif()

if(NOT DEFINED BOOST_ROOT AND DEFINED ENV{BOOST_ROOT})
  set(BOOST_ROOT "$ENV{BOOST_ROOT}")
endif()

# Dependencies
find_package(Legion REQUIRED CONFIG)
find_package(Boost REQUIRED COMPONENTS program_options)

# Executable
add_executable(heaps
  heaps.cpp
  sift.hpp
)

target_include_directories(heaps
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link Legion (supports common exported target names)
if(TARGET Legion::Legion)
  target_link_libraries(heaps PRIVATE Legion::Legion)
elseif(TARGET Legion)
  target_link_libraries(heaps PRIVATE Legion)
elseif(DEFINED Legion_LIBRARIES)
  target_include_directories(heaps PRIVATE ${Legion_INCLUDE_DIRS})
  target_link_libraries(heaps PRIVATE ${Legion_LIBRARIES})
else()
  message(FATAL_ERROR "Found Legion package, but no recognized Legion target/library to link.")
endif()

target_link_libraries(heaps PRIVATE Boost::program_options)

# Match prior optimization intent
target_compile_options(heaps PRIVATE -O3)
