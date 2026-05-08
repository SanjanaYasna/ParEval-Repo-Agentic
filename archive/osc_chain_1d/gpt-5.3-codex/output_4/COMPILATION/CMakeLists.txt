cmake_minimum_required(VERSION 3.15 FATAL_ERROR)
project(osc_1d_legion LANGUAGES CXX)

# Use Legion_DIR from environment if provided
if(DEFINED ENV{Legion_DIR} AND NOT DEFINED Legion_DIR)
    set(Legion_DIR "$ENV{Legion_DIR}")
    message(STATUS "Using Legion_DIR from environment: ${Legion_DIR}")
endif()

# Use BOOST_ROOT from environment if provided
if(DEFINED ENV{BOOST_ROOT} AND NOT DEFINED BOOST_ROOT)
    set(BOOST_ROOT "$ENV{BOOST_ROOT}")
    message(STATUS "Using BOOST_ROOT from environment: ${BOOST_ROOT}")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(Legion REQUIRED CONFIG)
find_package(Boost REQUIRED COMPONENTS program_options thread serialization)

add_executable(odeint
    odeint.cpp
)

target_include_directories(odeint PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link Legion (supports both modern and legacy exported targets/variables)
if(TARGET Legion::Legion)
    target_link_libraries(odeint PRIVATE Legion::Legion)
elseif(TARGET Legion)
    target_link_libraries(odeint PRIVATE Legion)
else()
    target_include_directories(odeint PRIVATE ${Legion_INCLUDE_DIRS})
    target_link_libraries(odeint PRIVATE ${Legion_LIBRARIES})
endif()

target_link_libraries(odeint PRIVATE
    Boost::program_options
    Boost::thread
    Boost::serialization
)
