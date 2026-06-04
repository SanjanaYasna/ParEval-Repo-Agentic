cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(heap LANGUAGES CXX)

# Pick up Legion package config from environment, if provided
if(DEFINED ENV{Legion_DIR} AND NOT Legion_DIR)
  set(Legion_DIR "$ENV{Legion_DIR}")
  message(STATUS "Using Legion_DIR from environment: ${Legion_DIR}")
endif()

# Pick up Boost root from environment, if provided
if(DEFINED ENV{BOOST_ROOT} AND NOT BOOST_ROOT)
  set(BOOST_ROOT "$ENV{BOOST_ROOT}")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Legion and Boost
find_package(Legion REQUIRED CONFIG)
find_package(Boost REQUIRED COMPONENTS program_options)

add_executable(heaps
  heaps.cpp
  sift.hpp
)

target_include_directories(heaps PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link against Legion (with fallbacks for different package exports)
if(TARGET Legion::Legion)
  target_link_libraries(heaps PRIVATE Legion::Legion Boost::program_options)
elseif(TARGET Legion)
  target_link_libraries(heaps PRIVATE Legion Boost::program_options)
else()
  target_include_directories(heaps PRIVATE ${Legion_INCLUDE_DIRS})
  target_link_libraries(heaps PRIVATE ${Legion_LIBRARIES} Boost::program_options)
endif()

# Match original optimization intent
target_compile_options(heaps PRIVATE -O3)
