cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(osc_1d_legion CXX)

# Respect environment-provided package roots.
if(DEFINED ENV{Legion_DIR} AND NOT Legion_DIR)
  set(Legion_DIR "$ENV{Legion_DIR}" CACHE PATH "Path to Legion CMake config directory")
endif()

if(DEFINED ENV{BOOST_ROOT} AND NOT BOOST_ROOT)
  set(BOOST_ROOT "$ENV{BOOST_ROOT}" CACHE PATH "Path to Boost installation")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Optional: default to Release if not specified.
if(NOT CMAKE_BUILD_TYPE)
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

target_compile_options(odeint PRIVATE -O3)

# Boost
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

# Legion (support common exported target/variable names)
if(TARGET Legion::Legion)
  target_link_libraries(odeint PRIVATE Legion::Legion)
elseif(TARGET Legion::Runtime)
  target_link_libraries(odeint PRIVATE Legion::Runtime)
elseif(TARGET legion)
  target_link_libraries(odeint PRIVATE legion)
elseif(DEFINED Legion_LIBRARIES)
  target_link_libraries(odeint PRIVATE ${Legion_LIBRARIES})
elseif(DEFINED LEGION_LIBRARIES)
  target_link_libraries(odeint PRIVATE ${LEGION_LIBRARIES})
else()
  message(FATAL_ERROR "find_package(Legion) succeeded, but no known Legion target/library was found.")
endif()

if(DEFINED Legion_INCLUDE_DIRS)
  target_include_directories(odeint PRIVATE ${Legion_INCLUDE_DIRS})
elseif(DEFINED LEGION_INCLUDE_DIRS)
  target_include_directories(odeint PRIVATE ${LEGION_INCLUDE_DIRS})
endif()
