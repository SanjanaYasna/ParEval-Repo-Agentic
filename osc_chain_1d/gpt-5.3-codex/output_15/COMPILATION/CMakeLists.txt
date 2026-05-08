cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(osc_1d_legion LANGUAGES CXX)

# Honor environment-provided package roots
if(DEFINED ENV{Legion_DIR} AND NOT DEFINED Legion_DIR)
  set(Legion_DIR "$ENV{Legion_DIR}")
endif()

if(DEFINED ENV{BOOST_ROOT} AND NOT DEFINED BOOST_ROOT)
  set(BOOST_ROOT "$ENV{BOOST_ROOT}")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(Legion REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options serialization thread)

add_executable(odeint
  odeint.cpp
  shared_resize.hpp
  algebra.hpp
  shared_operations.hpp
  system.hpp
)

target_include_directories(odeint PRIVATE ${Boost_INCLUDE_DIRS})
if(DEFINED Legion_INCLUDE_DIRS)
  target_include_directories(odeint PRIVATE ${Legion_INCLUDE_DIRS})
endif()

# Link Legion (support common exported target names + variable fallback)
if(TARGET Legion::Legion)
  target_link_libraries(odeint PRIVATE Legion::Legion)
elseif(TARGET Legion::legion)
  target_link_libraries(odeint PRIVATE Legion::legion)
elseif(DEFINED Legion_LIBRARIES)
  target_link_libraries(odeint PRIVATE ${Legion_LIBRARIES})
else()
  message(FATAL_ERROR "Legion found, but no known Legion target/library variable is available.")
endif()

# Link Boost (imported targets if available, otherwise fallback variables)
if(TARGET Boost::program_options)
  target_link_libraries(odeint PRIVATE Boost::program_options)
  if(TARGET Boost::serialization)
    target_link_libraries(odeint PRIVATE Boost::serialization)
  endif()
  if(TARGET Boost::thread)
    target_link_libraries(odeint PRIVATE Boost::thread)
  endif()
else()
  target_link_libraries(odeint PRIVATE ${Boost_LIBRARIES})
endif()
