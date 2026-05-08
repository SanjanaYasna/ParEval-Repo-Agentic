cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(osc_1d_legion LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Prefer package roots from environment when provided
if(NOT Legion_DIR AND DEFINED ENV{Legion_DIR})
  set(Legion_DIR "$ENV{Legion_DIR}")
endif()

if(NOT Boost_ROOT AND DEFINED ENV{BOOST_ROOT})
  set(Boost_ROOT "$ENV{BOOST_ROOT}")
endif()

# Legion runtime (default mapper; no custom mapper setup here)
find_package(Legion REQUIRED CONFIG)

# Boost used by the codebase (program_options + common runtime deps)
find_package(Boost REQUIRED COMPONENTS program_options thread system serialization)

add_executable(odeint
  odeint.cpp
  algebra.hpp
  shared_resize.hpp
  shared_operations.hpp
  system.hpp
)

target_include_directories(odeint PRIVATE ${Boost_INCLUDE_DIRS})

# Link Legion (handle common exported target names)
if(TARGET Legion::Legion)
  target_link_libraries(odeint PRIVATE Legion::Legion)
elseif(TARGET Legion)
  target_link_libraries(odeint PRIVATE Legion)
elseif(DEFINED LEGION_LIBRARIES)
  target_link_libraries(odeint PRIVATE ${LEGION_LIBRARIES})
else()
  message(FATAL_ERROR "Legion was found, but no known Legion CMake target is available.")
endif()

target_link_libraries(odeint PRIVATE ${Boost_LIBRARIES})

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(odeint PRIVATE -O3 -w)
endif()
