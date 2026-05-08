cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(osc_1d_legion LANGUAGES CXX)

# Use C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default to Release builds
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Honor environment-provided package roots
if(DEFINED ENV{Legion_DIR} AND NOT DEFINED Legion_DIR)
  set(Legion_DIR "$ENV{Legion_DIR}" CACHE PATH "Path to LegionConfig.cmake")
endif()

if(DEFINED ENV{BOOST_ROOT} AND NOT DEFINED BOOST_ROOT)
  set(BOOST_ROOT "$ENV{BOOST_ROOT}" CACHE PATH "Boost root directory")
endif()

# Dependencies
find_package(Legion REQUIRED CONFIG)
find_package(Boost REQUIRED COMPONENTS program_options)

# Executable
add_executable(odeint
  odeint.cpp
  shared_resize.hpp
  algebra.hpp
  shared_operations.hpp
  system.hpp
)

# Include directories (fallbacks for non-imported-package setups)
target_include_directories(odeint PRIVATE ${Boost_INCLUDE_DIRS})
if(DEFINED Legion_INCLUDE_DIRS)
  target_include_directories(odeint PRIVATE ${Legion_INCLUDE_DIRS})
elseif(DEFINED Legion_INCLUDE_DIR)
  target_include_directories(odeint PRIVATE ${Legion_INCLUDE_DIR})
endif()

# Link Legion (supports common package export styles)
if(TARGET Legion::Legion)
  target_link_libraries(odeint PRIVATE Legion::Legion)
elseif(TARGET Legion)
  target_link_libraries(odeint PRIVATE Legion)
elseif(DEFINED Legion_LIBRARIES)
  target_link_libraries(odeint PRIVATE ${Legion_LIBRARIES})
else()
  message(FATAL_ERROR "Legion package found, but no usable Legion target/library was exported.")
endif()

# Link Boost
target_link_libraries(odeint PRIVATE ${Boost_LIBRARIES})

# Match original optimization/silencing behavior for GCC/Clang
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(odeint PRIVATE -O3 -w)
endif()
