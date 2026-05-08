cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(osc_1d_legion LANGUAGES CXX)

# Honor *_ROOT variables in find_package
if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

# Use Legion_DIR from environment if not explicitly provided
if(NOT Legion_DIR AND DEFINED ENV{Legion_DIR})
  set(Legion_DIR "$ENV{Legion_DIR}" CACHE PATH "Path to Legion CMake config directory")
  message(STATUS "Using Legion_DIR from environment: ${Legion_DIR}")
endif()

# Use BOOST_ROOT from environment if not explicitly provided
if(NOT BOOST_ROOT AND DEFINED ENV{BOOST_ROOT})
  set(BOOST_ROOT "$ENV{BOOST_ROOT}" CACHE PATH "Path to Boost root")
  message(STATUS "Using BOOST_ROOT from environment: ${BOOST_ROOT}")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Legion REQUIRED CONFIG)
find_package(Boost REQUIRED COMPONENTS program_options)

add_executable(odeint
  odeint.cpp
  shared_resize.hpp
  algebra.hpp
  shared_operations.hpp
  system.hpp
)

# Optimization similar to original build
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  target_compile_options(odeint PRIVATE -O3 -w)
elseif(MSVC)
  target_compile_options(odeint PRIVATE /O2 /W0)
endif()

target_include_directories(odeint PRIVATE ${Boost_INCLUDE_DIRS})
target_link_libraries(odeint PRIVATE Boost::program_options)

# Link Legion (handle common exported target/variable names)
if(TARGET Legion::Legion)
  target_link_libraries(odeint PRIVATE Legion::Legion)
elseif(TARGET legion)
  target_link_libraries(odeint PRIVATE legion)
elseif(DEFINED Legion_LIBRARIES)
  target_link_libraries(odeint PRIVATE ${Legion_LIBRARIES})
  if(DEFINED Legion_INCLUDE_DIRS)
    target_include_directories(odeint PRIVATE ${Legion_INCLUDE_DIRS})
  endif()
else()
  message(FATAL_ERROR "Legion found, but no known Legion target/variables were exported.")
endif()
