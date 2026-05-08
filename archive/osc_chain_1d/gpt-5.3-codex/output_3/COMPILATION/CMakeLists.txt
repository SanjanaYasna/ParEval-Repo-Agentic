cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(osc_1d_legion LANGUAGES CXX)

# Prefer Legion_DIR from environment if provided
if(DEFINED ENV{Legion_DIR} AND NOT DEFINED Legion_DIR)
  set(Legion_DIR "$ENV{Legion_DIR}" CACHE PATH "Path to Legion CMake config directory")
endif()

# Prefer BOOST_ROOT from environment if provided
if(DEFINED ENV{BOOST_ROOT} AND NOT DEFINED BOOST_ROOT)
  set(BOOST_ROOT "$ENV{BOOST_ROOT}" CACHE PATH "Path to Boost root")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Dependencies
find_package(Legion REQUIRED CONFIG)
find_package(Boost REQUIRED COMPONENTS program_options thread)

add_executable(odeint
  odeint.cpp
  shared_resize.hpp
  algebra.hpp
  shared_operations.hpp
  system.hpp
)

# Boost link targets (modern or fallback)
set(BOOST_LINK_LIBS "")
if(TARGET Boost::program_options)
  list(APPEND BOOST_LINK_LIBS Boost::program_options)
  if(TARGET Boost::thread)
    list(APPEND BOOST_LINK_LIBS Boost::thread)
  endif()
else()
  list(APPEND BOOST_LINK_LIBS ${Boost_LIBRARIES})
  target_include_directories(odeint PRIVATE ${Boost_INCLUDE_DIRS})
endif()

# Legion link target (handle common exported target names)
if(TARGET Legion::Legion)
  target_link_libraries(odeint PRIVATE Legion::Legion ${BOOST_LINK_LIBS})
elseif(TARGET Legion::legion)
  target_link_libraries(odeint PRIVATE Legion::legion ${BOOST_LINK_LIBS})
elseif(TARGET Legion)
  target_link_libraries(odeint PRIVATE Legion ${BOOST_LINK_LIBS})
elseif(DEFINED Legion_LIBRARIES)
  target_link_libraries(odeint PRIVATE ${Legion_LIBRARIES} ${BOOST_LINK_LIBS})
  if(DEFINED Legion_INCLUDE_DIRS)
    target_include_directories(odeint PRIVATE ${Legion_INCLUDE_DIRS})
  endif()
else()
  message(FATAL_ERROR "Legion was found, but no known Legion CMake target/library variable is available.")
endif()
