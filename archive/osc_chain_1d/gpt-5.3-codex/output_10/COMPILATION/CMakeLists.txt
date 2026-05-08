cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(osc_1d CXX)

# Respect environment-provided package roots.
if(DEFINED ENV{Legion_DIR} AND NOT Legion_DIR)
  set(Legion_DIR "$ENV{Legion_DIR}")
endif()

if(DEFINED ENV{BOOST_ROOT} AND NOT BOOST_ROOT)
  set(BOOST_ROOT "$ENV{BOOST_ROOT}")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default to Release if not specified.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(Legion REQUIRED CONFIG)
find_package(Boost REQUIRED COMPONENTS program_options serialization)

add_executable(odeint
  odeint.cpp
  shared_resize.hpp
  algebra.hpp
  shared_operations.hpp
  system.hpp
)

target_compile_options(odeint PRIVATE -O3)
target_include_directories(odeint PRIVATE ${Boost_INCLUDE_DIRS})

# Link Legion (support common exported target names/fallback variables).
if(TARGET Legion::Legion)
  target_link_libraries(odeint PRIVATE Legion::Legion)
elseif(TARGET Legion::legion)
  target_link_libraries(odeint PRIVATE Legion::legion)
elseif(TARGET legion)
  target_link_libraries(odeint PRIVATE legion)
elseif(DEFINED Legion_LIBRARIES)
  target_include_directories(odeint PRIVATE ${Legion_INCLUDE_DIRS})
  target_link_libraries(odeint PRIVATE ${Legion_LIBRARIES})
else()
  message(FATAL_ERROR "Legion was found, but no usable target/library was exposed.")
endif()

# Link Boost (prefer imported targets when available).
if(TARGET Boost::program_options AND TARGET Boost::serialization)
  target_link_libraries(odeint PRIVATE Boost::program_options Boost::serialization)
else()
  target_link_libraries(odeint PRIVATE ${Boost_LIBRARIES})
endif()
