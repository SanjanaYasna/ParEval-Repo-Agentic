cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(osc_1d_legion LANGUAGES CXX)

# Use environment-provided package roots when available
if(DEFINED ENV{Legion_DIR} AND NOT Legion_DIR)
  set(Legion_DIR "$ENV{Legion_DIR}")
  message(STATUS "Using Legion_DIR from environment: ${Legion_DIR}")
endif()

if(DEFINED ENV{BOOST_ROOT} AND NOT BOOST_ROOT)
  set(BOOST_ROOT "$ENV{BOOST_ROOT}")
  message(STATUS "Using BOOST_ROOT from environment: ${BOOST_ROOT}")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Legion (CONFIG mode, via Legion_DIR)
find_package(Legion REQUIRED CONFIG)

# Boost used by the codebase
find_package(Boost REQUIRED COMPONENTS program_options serialization)

add_executable(odeint
  odeint.cpp
  shared_resize.hpp
  algebra.hpp
  shared_operations.hpp
  system.hpp
)

target_include_directories(odeint PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${Boost_INCLUDE_DIRS}
)

# Link Legion (handle common exported target names/fallback variables)
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
  message(FATAL_ERROR "Legion found, but no known Legion target/library variable is available to link.")
endif()

target_link_libraries(odeint PRIVATE ${Boost_LIBRARIES})

# Often needed transitively; harmless if already included by Legion package
find_package(Threads REQUIRED)
target_link_libraries(odeint PRIVATE Threads::Threads)

set_target_properties(odeint PROPERTIES OUTPUT_NAME odeint)
