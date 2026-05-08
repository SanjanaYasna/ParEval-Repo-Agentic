cmake_minimum_required(VERSION 3.10)
project(osc_1d CXX)

# Only search if were building stand-alone and not as part of Legion
#make sure Legion_DIR env variable points towards the .cmake configs
if(NOT Legion_SOURCE_DIR)
  find_package(Legion REQUIRED)
endif()

find_package(Boost REQUIRED)

add_executable(odeint odeint.cpp)
target_link_libraries(odeint Legion::Legion)
target_include_directories(odeint PRIVATE ${Boost_INCLUDE_DIRS})
target_compile_options(odeint PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${CXX_BUILD_WARNING_FLAGS}>)
if(Legion_ENABLE_TESTING)
    add_test(NAME odeint COMMAND ${Legion_TEST_LAUNCHER} $<TARGET_FILE:odeint> ${Legion_TEST_ARGS})
endif()
