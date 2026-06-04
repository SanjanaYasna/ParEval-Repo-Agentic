cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(heaps LANGUAGES CXX)

# Respect Legion_DIR from environment if provided
if(DEFINED ENV{Legion_DIR} AND NOT Legion_DIR)
  set(Legion_DIR "$ENV{Legion_DIR}")
  message(STATUS "Using Legion_DIR from environment: $ENV{Legion_DIR}")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Legion (CONFIG mode; assumes Legion_DIR points to LegionConfig.cmake location)
find_package(Legion CONFIG REQUIRED)

# Boost (for program_options)
find_package(Boost REQUIRED COMPONENTS program_options)

add_executable(heaps
  heaps.cpp
  sift.hpp
)

target_include_directories(heaps PRIVATE ${Boost_INCLUDE_DIRS})
target_link_libraries(heaps PRIVATE ${Boost_LIBRARIES})

# Handle common Legion package target/variable variants
if(TARGET Legion::Legion)
  target_link_libraries(heaps PRIVATE Legion::Legion)
elseif(TARGET Legion)
  target_link_libraries(heaps PRIVATE Legion)
elseif(DEFINED Legion_LIBRARIES OR DEFINED LEGION_LIBRARIES)
  target_include_directories(heaps PRIVATE
    ${Legion_INCLUDE_DIRS}
    ${LEGION_INCLUDE_DIRS}
  )
  target_link_libraries(heaps PRIVATE
    ${Legion_LIBRARIES}
    ${LEGION_LIBRARIES}
  )
else()
  message(FATAL_ERROR "Legion found, but no known Legion target/library variables were exported.")
endif()

target_compile_options(heaps PRIVATE -O3 -w)
