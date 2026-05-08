cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(osc_1d LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Optional: default to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Respect environment-provided package roots
if(DEFINED ENV{Legion_DIR} AND NOT DEFINED Legion_DIR)
  set(Legion_DIR "$ENV{Legion_DIR}" CACHE PATH "Path to LegionConfig.cmake")
endif()

if(DEFINED ENV{BOOST_ROOT} AND NOT DEFINED BOOST_ROOT)
  set(BOOST_ROOT "$ENV{BOOST_ROOT}" CACHE PATH "Path to Boost root")
endif()

# Find Legion and Boost
find_package(Legion REQUIRED CONFIG)
find_package(Boost REQUIRED COMPONENTS program_options)

add_executable(odeint
  odeint.cpp
)

# Boost include path (for headers like odeint, random, etc.)
target_include_directories(odeint PRIVATE ${Boost_INCLUDE_DIRS})

# Link Legion (target name can vary by Legion version/package)
if(TARGET Legion::Legion)
  target_link_libraries(odeint PRIVATE Legion::Legion)
elseif(TARGET Legion)
  target_link_libraries(odeint PRIVATE Legion)
elseif(TARGET legion)
  target_link_libraries(odeint PRIVATE legion)
elseif(DEFINED Legion_LIBRARIES)
  target_include_directories(odeint PRIVATE ${Legion_INCLUDE_DIRS})
  target_link_libraries(odeint PRIVATE ${Legion_LIBRARIES})
else()
  message(FATAL_ERROR "Could not determine Legion link target from installed Legion package.")
endif()

# Link Boost.ProgramOptions (used by command-line parsing)
if(TARGET Boost::program_options)
  target_link_libraries(odeint PRIVATE Boost::program_options)
else()
  target_link_libraries(odeint PRIVATE ${Boost_PROGRAM_OPTIONS_LIBRARY})
endif()
