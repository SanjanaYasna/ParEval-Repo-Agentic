cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(osc_1d_legion LANGUAGES CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Respect environment-provided package roots
if(NOT DEFINED Legion_DIR AND DEFINED ENV{Legion_DIR})
    set(Legion_DIR "$ENV{Legion_DIR}" CACHE PATH "Path to Legion CMake config")
endif()

if(NOT DEFINED BOOST_ROOT AND DEFINED ENV{BOOST_ROOT})
    set(BOOST_ROOT "$ENV{BOOST_ROOT}" CACHE PATH "Boost root directory")
endif()

# Dependencies
find_package(Legion REQUIRED CONFIG)
find_package(Boost REQUIRED COMPONENTS program_options serialization thread)

# Build target
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
elseif(TARGET Legion::legion)
    target_link_libraries(odeint PRIVATE Legion::legion)
elseif(TARGET legion)
    target_link_libraries(odeint PRIVATE legion)
elseif(DEFINED Legion_LIBRARIES)
    target_link_libraries(odeint PRIVATE ${Legion_LIBRARIES})
else()
    message(FATAL_ERROR "Could not find a valid Legion link target from LegionConfig.cmake")
endif()

# Link Boost components used by the codebase
target_link_libraries(odeint PRIVATE
    Boost::program_options
    Boost::serialization
    Boost::thread
)
