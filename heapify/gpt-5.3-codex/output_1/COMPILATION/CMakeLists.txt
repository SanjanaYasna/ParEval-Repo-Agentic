cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

project(heap LANGUAGES CXX)

# Use Legion_DIR from environment if provided
if(DEFINED ENV{Legion_DIR} AND NOT Legion_DIR)
  set(Legion_DIR "$ENV{Legion_DIR}" CACHE PATH "Path to Legion CMake config directory")
  message(STATUS "Using Legion_DIR from environment: $ENV{Legion_DIR}")
endif()

# Use BOOST_ROOT from environment if provided
if(DEFINED ENV{BOOST_ROOT} AND NOT BOOST_ROOT)
  set(BOOST_ROOT "$ENV{BOOST_ROOT}" CACHE PATH "Path to Boost root")
  message(STATUS "Using BOOST_ROOT from environment: $ENV{BOOST_ROOT}")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Dependencies
find_package(Legion REQUIRED CONFIG)
find_package(Boost REQUIRED COMPONENTS program_options)

# Executable
add_executable(heaps heaps.cpp)

target_include_directories(heaps PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_options(heaps PRIVATE -O3 -w)

# Link Legion (support common package target/variable layouts)
if(TARGET Legion::Legion)
  target_link_libraries(heaps PRIVATE Legion::Legion)
elseif(DEFINED Legion_LIBRARIES)
  target_link_libraries(heaps PRIVATE ${Legion_LIBRARIES})
else()
  message(FATAL_ERROR "Could not find a Legion link target. Expected Legion::Legion or Legion_LIBRARIES.")
endif()

target_link_libraries(heaps PRIVATE Boost::program_options)
