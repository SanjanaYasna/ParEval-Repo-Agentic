cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(osc_1d CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Legion
if(DEFINED ENV{Legion_DIR})
    set(Legion_DIR "$ENV{Legion_DIR}")
    message(STATUS "Using Legion_DIR from environment: $ENV{Legion_DIR}")
endif()

find_package(Legion REQUIRED)

# Find Boost
if(DEFINED ENV{BOOST_ROOT})
    set(BOOST_ROOT "$ENV{BOOST_ROOT}")
endif()

find_package(Boost REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w -O3")

add_executable(odeint odeint.cpp)

target_include_directories(odeint PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${Boost_INCLUDE_DIRS}
)

target_link_libraries(odeint
    Legion::Legion
    ${Boost_LIBRARIES}
)
