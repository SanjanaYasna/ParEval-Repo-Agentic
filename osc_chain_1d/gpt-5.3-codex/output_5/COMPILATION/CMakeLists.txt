cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(osc_1d LANGUAGES CXX)

# Prefer Legion_DIR from environment if provided
if(DEFINED ENV{Legion_DIR} AND NOT DEFINED Legion_DIR)
    set(Legion_DIR "$ENV{Legion_DIR}")
    message(STATUS "Using Legion_DIR from environment: ${Legion_DIR}")
endif()

# C++ standard and optimization settings (matching original intent)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Dependencies
find_package(Legion REQUIRED CONFIG)
find_package(Boost REQUIRED COMPONENTS program_options)

add_executable(odeint
    odeint.cpp
)

target_include_directories(odeint PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Keep original compile intent (-O3, suppress warnings)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(odeint PRIVATE -O3 -w)
elseif(MSVC)
    target_compile_options(odeint PRIVATE /O2 /W0)
endif()

# Link Legion (support common exported target names + variable fallback)
if(TARGET Legion::Legion)
    target_link_libraries(odeint PRIVATE Legion::Legion)
elseif(TARGET Legion::legion)
    target_link_libraries(odeint PRIVATE Legion::legion)
elseif(TARGET legion)
    target_link_libraries(odeint PRIVATE legion)
elseif(DEFINED Legion_LIBRARIES)
    target_link_libraries(odeint PRIVATE ${Legion_LIBRARIES})
    if(DEFINED Legion_INCLUDE_DIRS)
        target_include_directories(odeint PRIVATE ${Legion_INCLUDE_DIRS})
    endif()
else()
    message(FATAL_ERROR "Could not determine Legion link target from Legion package.")
endif()

# Link Boost.Program_options (imported target if available, otherwise fallback)
if(TARGET Boost::program_options)
    target_link_libraries(odeint PRIVATE Boost::program_options)
else()
    target_include_directories(odeint PRIVATE ${Boost_INCLUDE_DIRS})
    target_link_libraries(odeint PRIVATE ${Boost_LIBRARIES})
endif()
