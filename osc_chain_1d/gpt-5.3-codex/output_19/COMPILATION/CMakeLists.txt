cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(osc_1d_legion LANGUAGES CXX)

# Respect environment variables if provided
if(DEFINED ENV{Legion_DIR} AND NOT Legion_DIR)
  set(Legion_DIR "$ENV{Legion_DIR}")
endif()

if(DEFINED ENV{BOOST_ROOT} AND NOT BOOST_ROOT)
  set(BOOST_ROOT "$ENV{BOOST_ROOT}")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Legion runtime (assumes Legion_DIR points to Legion CMake config files)
find_package(Legion REQUIRED)

# Boost used by odeint + serialization/program options in this codebase
find_package(Boost REQUIRED COMPONENTS program_options serialization)

add_executable(odeint
  odeint.cpp
  algebra.hpp
  shared_operations.hpp
  shared_resize.hpp
  system.hpp
)

target_include_directories(odeint PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link Legion (handle common exported target/variable styles)
if(TARGET Legion::Legion)
  target_link_libraries(odeint PRIVATE Legion::Legion)
elseif(TARGET legion)
  target_link_libraries(odeint PRIVATE legion)
elseif(DEFINED LEGION_LIBRARIES)
  target_link_libraries(odeint PRIVATE ${LEGION_LIBRARIES})
  if(DEFINED LEGION_INCLUDE_DIRS)
    target_include_directories(odeint PRIVATE ${LEGION_INCLUDE_DIRS})
  endif()
else()
  message(FATAL_ERROR "Legion was found, but no known Legion target/library variable was exported.")
endif()

# Link Boost (modern targets if available, otherwise fallback)
if(TARGET Boost::program_options AND TARGET Boost::serialization)
  target_link_libraries(odeint PRIVATE Boost::program_options Boost::serialization)
else()
  target_include_directories(odeint PRIVATE ${Boost_INCLUDE_DIRS})
  target_link_libraries(odeint PRIVATE ${Boost_LIBRARIES})
endif()

target_compile_options(odeint PRIVATE -O3)
