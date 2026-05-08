cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(osc_1d_legion LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Use Legion_DIR from environment if not provided on the CMake command line.
if(NOT Legion_DIR AND DEFINED ENV{Legion_DIR})
  set(Legion_DIR "$ENV{Legion_DIR}")
  message(STATUS "Using Legion_DIR from environment: $ENV{Legion_DIR}")
endif()

# Use BOOST_ROOT from environment if not provided.
if(NOT BOOST_ROOT AND DEFINED ENV{BOOST_ROOT})
  set(BOOST_ROOT "$ENV{BOOST_ROOT}")
endif()

# Legion
find_package(Legion REQUIRED CONFIG)

# Boost (program_options is typically needed for CLI parsing)
find_package(Boost REQUIRED COMPONENTS program_options)

add_executable(odeint
  odeint.cpp
)

target_include_directories(odeint PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_options(odeint PRIVATE -O3)

# Link Boost
if(TARGET Boost::program_options)
  target_link_libraries(odeint PRIVATE Boost::program_options)
else()
  target_include_directories(odeint PRIVATE ${Boost_INCLUDE_DIRS})
  target_link_libraries(odeint PRIVATE ${Boost_LIBRARIES})
endif()

# Link Legion (support common exported target names + variable fallback)
if(TARGET Legion::Legion)
  target_link_libraries(odeint PRIVATE Legion::Legion)
elseif(TARGET Legion)
  target_link_libraries(odeint PRIVATE Legion)
elseif(DEFINED Legion_LIBRARIES)
  if(DEFINED Legion_INCLUDE_DIRS)
    target_include_directories(odeint PRIVATE ${Legion_INCLUDE_DIRS})
  endif()
  target_link_libraries(odeint PRIVATE ${Legion_LIBRARIES})
else()
  message(FATAL_ERROR "Legion was found, but no known Legion target or library variables are available.")
endif()
