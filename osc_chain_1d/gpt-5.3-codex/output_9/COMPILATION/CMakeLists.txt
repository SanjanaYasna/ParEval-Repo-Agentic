cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(osc_1d_legion LANGUAGES CXX)

# C++ settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Optional: pick up package roots from environment
if(DEFINED ENV{Legion_DIR} AND NOT Legion_DIR)
  set(Legion_DIR "$ENV{Legion_DIR}")
  message(STATUS "Using Legion_DIR from environment: ${Legion_DIR}")
endif()

if(DEFINED ENV{BOOST_ROOT} AND NOT BOOST_ROOT)
  set(BOOST_ROOT "$ENV{BOOST_ROOT}")
  message(STATUS "Using BOOST_ROOT from environment: ${BOOST_ROOT}")
endif()

# Dependencies
find_package(Legion REQUIRED CONFIG)
find_package(Boost REQUIRED COMPONENTS program_options)

# Executable
add_executable(odeint
  odeint.cpp
)

target_include_directories(odeint PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

# Optimization level similar to original setup
target_compile_options(odeint PRIVATE -O3)

# Link Legion (support common export styles)
if(TARGET Legion::Legion)
  target_link_libraries(odeint PRIVATE Legion::Legion)
elseif(DEFINED Legion_LIBRARIES)
  target_link_libraries(odeint PRIVATE ${Legion_LIBRARIES})
  if(DEFINED Legion_INCLUDE_DIRS)
    target_include_directories(odeint PRIVATE ${Legion_INCLUDE_DIRS})
  endif()
else()
  message(FATAL_ERROR "Legion package found, but no usable Legion target/libraries were exported.")
endif()

# Link Boost.Program_options
if(TARGET Boost::program_options)
  target_link_libraries(odeint PRIVATE Boost::program_options)
else()
  target_include_directories(odeint PRIVATE ${Boost_INCLUDE_DIRS})
  target_link_libraries(odeint PRIVATE ${Boost_PROGRAM_OPTIONS_LIBRARY})
endif()
