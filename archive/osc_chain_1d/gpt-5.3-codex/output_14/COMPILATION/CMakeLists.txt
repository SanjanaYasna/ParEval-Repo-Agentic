cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(osc_1d_legion LANGUAGES CXX)

# Respect Legion_DIR provided from environment (points to Legion CMake config path)
if(DEFINED ENV{Legion_DIR} AND NOT Legion_DIR)
  set(Legion_DIR "$ENV{Legion_DIR}")
endif()

# Respect BOOST_ROOT from environment if provided
if(DEFINED ENV{BOOST_ROOT} AND NOT BOOST_ROOT)
  set(BOOST_ROOT "$ENV{BOOST_ROOT}")
endif()

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default to Release if not specified by user
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Dependencies
find_package(Legion REQUIRED CONFIG)
find_package(Boost REQUIRED COMPONENTS program_options thread)
find_package(Threads REQUIRED)

# Legion target name can vary by package export; pick whichever exists.
if(TARGET Legion::Legion)
  set(LEGION_TARGET Legion::Legion)
elseif(TARGET Legion::legion)
  set(LEGION_TARGET Legion::legion)
elseif(TARGET legion)
  set(LEGION_TARGET legion)
else()
  message(FATAL_ERROR "Legion CMake package found, but no known Legion target was exported.")
endif()

add_executable(odeint
  odeint.cpp
  shared_resize.hpp
  algebra.hpp
  shared_operations.hpp
  system.hpp
)

target_include_directories(odeint PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_options(odeint PRIVATE
  -O3
)

target_link_libraries(odeint PRIVATE
  ${LEGION_TARGET}
  Boost::program_options
  Boost::thread
  Threads::Threads
)
