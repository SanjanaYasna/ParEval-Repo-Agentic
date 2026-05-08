cmake_minimum_required(VERSION 3.14)
project(osc_1d_legion LANGUAGES CXX)

if(POLICY CMP0144)
  cmake_policy(SET CMP0144 NEW)
endif()
if(POLICY CMP0167)
  cmake_policy(SET CMP0167 OLD)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default to Release if not specified by the user
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Respect Legion_DIR from environment (points to Legion *.cmake config location)
if(DEFINED ENV{Legion_DIR} AND NOT DEFINED Legion_DIR)
  set(Legion_DIR "$ENV{Legion_DIR}" CACHE PATH "Path to LegionConfig.cmake")
endif()

# Fallback for setups that export LEGION_ROOT
if(NOT DEFINED Legion_DIR AND DEFINED ENV{LEGION_ROOT})
  set(Legion_DIR "$ENV{LEGION_ROOT}/lib/cmake/Legion" CACHE PATH "Path to LegionConfig.cmake")
endif()

# Respect BOOST_ROOT from environment
if(DEFINED ENV{BOOST_ROOT} AND NOT DEFINED BOOST_ROOT)
  set(BOOST_ROOT "$ENV{BOOST_ROOT}" CACHE PATH "Path to Boost root")
endif()

find_package(Legion REQUIRED CONFIG)
find_package(Boost REQUIRED COMPONENTS program_options)

add_executable(odeint
  odeint.cpp
)

target_include_directories(odeint
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link Legion (prefer imported target, with fallback to variables)
if(TARGET Legion::Legion)
  target_link_libraries(odeint PRIVATE Legion::Legion)
else()
  target_include_directories(odeint PRIVATE ${Legion_INCLUDE_DIRS})
  target_link_libraries(odeint PRIVATE ${Legion_LIBRARIES})
endif()

# Link Boost.Program_options (imported target when available)
if(TARGET Boost::program_options)
  target_link_libraries(odeint PRIVATE Boost::program_options)
else()
  target_include_directories(odeint PRIVATE ${Boost_INCLUDE_DIRS})
  target_link_libraries(odeint PRIVATE ${Boost_LIBRARIES})
endif()

target_compile_options(odeint PRIVATE -O3)
