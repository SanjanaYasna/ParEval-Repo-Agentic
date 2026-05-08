cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
project(heap CXX)

if(POLICY CMP0144)
cmake_policy(SET CMP0144 NEW)
endif()

# Only search if we're building stand-alone and not as part of Legion
# Make sure Legion_DIR env variable points towards the .cmake configs
if(NOT Legion_SOURCE_DIR)
find_package(Legion REQUIRED)
endif()

# Added because some generated code uses Boost libraries like program options
find_package(Boost REQUIRED COMPONENTS system thread filesystem program_options)

add_executable(heaps heaps.cpp)
target_link_libraries(heaps Legion::Legion Boost::program_options)
target_compile_options(heaps PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${CXX_BUILD_WARNING_FLAGS}>)

if(Legion_ENABLE_TESTING)
add_test(NAME heaps COMMAND ${Legion_TEST_LAUNCHER} $<TARGET_FILE:heaps> ${Legion_TEST_ARGS})
endif()
