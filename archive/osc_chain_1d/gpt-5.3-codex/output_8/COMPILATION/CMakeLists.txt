cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(osc_1d_legion LANGUAGES CXX)

# Use Legion_DIR from environment if provided
if(DEFINED ENV{Legion_DIR} AND NOT Legion_DIR)
  set(Legion_DIR "$ENV{Legion_DIR}")
endif()

# Use BOOST_ROOT from environment if provided
if(DEFINED ENV{BOOST_ROOT} AND NOT BOOST_ROOT)
  set(BOOST_ROOT "$ENV{BOOST_ROOT}")
endif()

find_package(Legion REQUIRED CONFIG)
find_package(Boost REQUIRED COMPONENTS program_options serialization thread)

add_executable(odeint
  odeint.cpp
  algebra.hpp
  shared_resize.hpp
  shared_operations.hpp
  system.hpp
)

target_compile_features(odeint PRIVATE cxx_std_17)
target_compile_options(odeint PRIVATE -O3 -w)

# Link Legion (handle common exported target names)
if(TARGET Legion::Legion)
  target_link_libraries(odeint PRIVATE Legion::Legion)
elseif(TARGET legion)
  target_link_libraries(odeint PRIVATE legion)
elseif(DEFINED Legion_LIBRARIES)
  target_link_libraries(odeint PRIVATE ${Legion_LIBRARIES})
  if(DEFINED Legion_INCLUDE_DIRS)
    target_include_directories(odeint PRIVATE ${Legion_INCLUDE_DIRS})
  endif()
else()
  message(FATAL_ERROR "Legion library target not found.")
endif()

# Link Boost
if(TARGET Boost::program_options)
  target_link_libraries(odeint PRIVATE Boost::program_options Boost::serialization Boost::thread)
else()
  target_include_directories(odeint PRIVATE ${Boost_INCLUDE_DIRS})
  target_link_libraries(odeint PRIVATE ${Boost_LIBRARIES})
endif()
