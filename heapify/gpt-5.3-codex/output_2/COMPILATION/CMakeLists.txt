cmake_minimum_required(VERSION 3.15)

if(POLICY CMP0144)
  cmake_policy(SET CMP0144 NEW)
endif()

project(heaps LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default to Release if no build type is specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Legion package location
if(NOT Legion_DIR AND DEFINED ENV{Legion_DIR})
  set(Legion_DIR "$ENV{Legion_DIR}" CACHE PATH "Path to Legion CMake config directory")
endif()

if(DEFINED ENV{LEGION_ROOT})
  list(PREPEND CMAKE_PREFIX_PATH "$ENV{LEGION_ROOT}")
  if(EXISTS "$ENV{LEGION_ROOT}/lib/cmake/Legion")
    list(PREPEND CMAKE_PREFIX_PATH "$ENV{LEGION_ROOT}/lib/cmake/Legion")
  endif()
endif()

find_package(Legion REQUIRED CONFIG)

# Boost location (assumes BOOST_ROOT is set)
if(NOT BOOST_ROOT AND DEFINED ENV{BOOST_ROOT})
  set(BOOST_ROOT "$ENV{BOOST_ROOT}" CACHE PATH "Boost root directory")
endif()
find_package(Boost REQUIRED COMPONENTS program_options)

add_executable(heaps
  heaps.cpp
  sift.hpp
)

target_include_directories(heaps PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${Boost_INCLUDE_DIRS}
)

# Add Legion include directories if exported as variables
if(DEFINED Legion_INCLUDE_DIRS)
  target_include_directories(heaps PRIVATE ${Legion_INCLUDE_DIRS})
elseif(DEFINED Legion_INCLUDE_DIR)
  target_include_directories(heaps PRIVATE ${Legion_INCLUDE_DIR})
endif()

# Link against Legion (handle common exported target/variable styles)
if(TARGET Legion::Legion)
  target_link_libraries(heaps PRIVATE Legion::Legion ${Boost_LIBRARIES})
elseif(TARGET Legion)
  target_link_libraries(heaps PRIVATE Legion ${Boost_LIBRARIES})
elseif(DEFINED Legion_LIBRARIES)
  target_link_libraries(heaps PRIVATE ${Legion_LIBRARIES} ${Boost_LIBRARIES})
else()
  message(FATAL_ERROR "Legion CMake package found, but no known Legion link target/library was exported.")
endif()
