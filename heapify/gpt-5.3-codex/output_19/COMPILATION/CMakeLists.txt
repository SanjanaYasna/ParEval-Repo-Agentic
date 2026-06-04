cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(heap_legion LANGUAGES CXX)

# Honor environment-provided package locations
if(DEFINED ENV{Legion_DIR} AND NOT Legion_DIR)
  set(Legion_DIR "$ENV{Legion_DIR}")
endif()

if(DEFINED ENV{BOOST_ROOT} AND NOT BOOST_ROOT)
  set(BOOST_ROOT "$ENV{BOOST_ROOT}")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(Legion REQUIRED CONFIG)
find_package(Boost REQUIRED COMPONENTS program_options)

add_executable(heaps
  heaps.cpp
  sift.hpp
)

target_compile_options(heaps PRIVATE -O3)

# Link against Legion (support common exported target names)
if(TARGET Legion::Legion)
  target_link_libraries(heaps PRIVATE Legion::Legion Boost::program_options)
elseif(TARGET legion)
  target_link_libraries(heaps PRIVATE legion Boost::program_options)
elseif(TARGET Legion)
  target_link_libraries(heaps PRIVATE Legion Boost::program_options)
else()
  message(FATAL_ERROR "Could not find a known Legion CMake target (Legion::Legion / legion / Legion).")
endif()
