cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(osc_1d CXX)

# Respect environment-provided package roots
if(DEFINED ENV{Legion_DIR} AND NOT Legion_DIR)
  set(Legion_DIR "$ENV{Legion_DIR}" CACHE PATH "Path to Legion CMake config" FORCE)
  message(STATUS "Using Legion_DIR from environment: $ENV{Legion_DIR}")
endif()

if(DEFINED ENV{BOOST_ROOT} AND NOT BOOST_ROOT)
  set(BOOST_ROOT "$ENV{BOOST_ROOT}" CACHE PATH "Path to Boost root" FORCE)
  message(STATUS "Using BOOST_ROOT from environment: $ENV{BOOST_ROOT}")
endif()

# Build settings (equivalent intent to original HPX CMake)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_compile_options(-O3 -w)

# Dependencies
find_package(Legion REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options)

# Executable
add_executable(odeint
  odeint.cpp
  shared_resize.hpp
  shared_operations.hpp
  algebra.hpp
  system.hpp
)

# Link Boost (supports both modern and legacy FindBoost styles)
if(TARGET Boost::program_options)
  target_link_libraries(odeint PRIVATE Boost::program_options)
else()
  target_include_directories(odeint PRIVATE ${Boost_INCLUDE_DIRS})
  target_link_libraries(odeint PRIVATE ${Boost_LIBRARIES})
endif()

# Link Legion (supports common exported target names and variable-based configs)
if(TARGET Legion::Legion)
  target_link_libraries(odeint PRIVATE Legion::Legion)
elseif(TARGET legion)
  target_link_libraries(odeint PRIVATE legion)
elseif(DEFINED Legion_LIBRARIES)
  target_include_directories(odeint PRIVATE ${Legion_INCLUDE_DIRS})
  target_link_libraries(odeint PRIVATE ${Legion_LIBRARIES})
else()
  message(FATAL_ERROR "Legion was found, but no known Legion target/variables were exported.")
endif()
