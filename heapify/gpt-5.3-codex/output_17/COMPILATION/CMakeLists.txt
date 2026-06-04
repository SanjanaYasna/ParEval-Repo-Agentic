cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(heap_legion CXX)

# Honor environment-provided package roots
if(DEFINED ENV{Legion_DIR} AND NOT Legion_DIR)
  set(Legion_DIR "$ENV{Legion_DIR}")
endif()

if(DEFINED ENV{BOOST_ROOT} AND NOT BOOST_ROOT)
  set(BOOST_ROOT "$ENV{BOOST_ROOT}")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Legion REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options)

add_executable(heaps
  heaps.cpp
  sift.hpp
)

target_compile_options(heaps PRIVATE -O3)

# Include directories (if exported by package config)
if(DEFINED Legion_INCLUDE_DIRS)
  target_include_directories(heaps PRIVATE ${Legion_INCLUDE_DIRS})
elseif(DEFINED LEGION_INCLUDE_DIRS)
  target_include_directories(heaps PRIVATE ${LEGION_INCLUDE_DIRS})
endif()

target_include_directories(heaps PRIVATE ${Boost_INCLUDE_DIRS})

# Link Legion (support common package export styles) + Boost program_options
if(TARGET Legion::Legion)
  target_link_libraries(heaps PRIVATE Legion::Legion)
elseif(TARGET legion)
  target_link_libraries(heaps PRIVATE legion)
elseif(DEFINED Legion_LIBRARIES)
  target_link_libraries(heaps PRIVATE ${Legion_LIBRARIES})
elseif(DEFINED LEGION_LIBRARIES)
  target_link_libraries(heaps PRIVATE ${LEGION_LIBRARIES})
else()
  message(FATAL_ERROR "Legion was found, but no known Legion target/library variables were exported.")
endif()

if(TARGET Boost::program_options)
  target_link_libraries(heaps PRIVATE Boost::program_options)
else()
  target_link_libraries(heaps PRIVATE ${Boost_LIBRARIES})
endif()
