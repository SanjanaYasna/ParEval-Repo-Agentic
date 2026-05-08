cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(osc_1d LANGUAGES CXX)

# Respect environment-provided package locations.
if(NOT Legion_DIR AND DEFINED ENV{Legion_DIR})
  set(Legion_DIR "$ENV{Legion_DIR}" CACHE PATH "Path to LegionConfig.cmake")
endif()

if(NOT BOOST_ROOT AND DEFINED ENV{BOOST_ROOT})
  set(BOOST_ROOT "$ENV{BOOST_ROOT}" CACHE PATH "Boost root directory")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default to optimized build for performance-sensitive code.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(Legion REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options serialization thread)

add_executable(odeint
  odeint.cpp
)

target_compile_options(odeint PRIVATE -O3)

# Legion include dirs (if exported as variables by the package).
if(DEFINED Legion_INCLUDE_DIRS)
  target_include_directories(odeint PRIVATE ${Legion_INCLUDE_DIRS})
elseif(DEFINED LEGION_INCLUDE_DIRS)
  target_include_directories(odeint PRIVATE ${LEGION_INCLUDE_DIRS})
endif()

# Boost linkage (prefer imported targets when available).
if(TARGET Boost::program_options)
  target_link_libraries(odeint PRIVATE
    Boost::program_options
    Boost::serialization
    Boost::thread
  )
else()
  target_include_directories(odeint PRIVATE ${Boost_INCLUDE_DIRS})
  target_link_libraries(odeint PRIVATE ${Boost_LIBRARIES})
endif()

# Legion linkage (support common exported target/variable names).
if(TARGET Legion::Legion)
  target_link_libraries(odeint PRIVATE Legion::Legion)
elseif(TARGET legion)
  target_link_libraries(odeint PRIVATE legion)
elseif(TARGET Legion)
  target_link_libraries(odeint PRIVATE Legion)
elseif(DEFINED Legion_LIBRARIES)
  target_link_libraries(odeint PRIVATE ${Legion_LIBRARIES})
elseif(DEFINED LEGION_LIBRARIES)
  target_link_libraries(odeint PRIVATE ${LEGION_LIBRARIES})
else()
  message(FATAL_ERROR "Legion was found, but no known Legion target/library variable is available.")
endif()
