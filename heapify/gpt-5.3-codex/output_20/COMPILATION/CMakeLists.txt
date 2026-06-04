cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(heap_legion LANGUAGES CXX)

# Use Legion_DIR from environment if provided
if(DEFINED ENV{Legion_DIR} AND NOT Legion_DIR)
  set(Legion_DIR "$ENV{Legion_DIR}")
  message(STATUS "Using Legion_DIR from environment: ${Legion_DIR}")
endif()

# Use BOOST_ROOT from environment if provided
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

# Optional optimization/warning behavior similar to original
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(heaps PRIVATE -O3 -w)
endif()

# Boost link target fallback
if(TARGET Boost::program_options)
  set(BOOST_PO_LIB Boost::program_options)
else()
  set(BOOST_PO_LIB ${Boost_LIBRARIES})
endif()

# Legion link target fallback for different package exports
if(TARGET Legion::Legion)
  target_link_libraries(heaps PRIVATE Legion::Legion ${BOOST_PO_LIB})
elseif(TARGET legion)
  target_link_libraries(heaps PRIVATE legion ${BOOST_PO_LIB})
elseif(DEFINED Legion_LIBRARIES)
  target_link_libraries(heaps PRIVATE ${Legion_LIBRARIES} ${BOOST_PO_LIB})
else()
  message(FATAL_ERROR "Could not find a usable Legion target/library from find_package(Legion).")
endif()

# Legion include dirs fallback (if exported as variables)
if(DEFINED Legion_INCLUDE_DIRS)
  target_include_directories(heaps PRIVATE ${Legion_INCLUDE_DIRS})
elseif(DEFINED LEGION_INCLUDE_DIRS)
  target_include_directories(heaps PRIVATE ${LEGION_INCLUDE_DIRS})
endif()

target_include_directories(heaps PRIVATE ${Boost_INCLUDE_DIRS})
