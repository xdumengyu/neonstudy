CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12 FATAL_ERROR)
include_directories(${googletest-distribution_SOURCE_DIR}/googletest/include)
include_directories(${googletest-distribution_SOURCE_DIR}/googlemock/include)

include_directories(${winograd_stu_SOURCE_DIR}/src)
include_directories(${winograd_stu_SOURCE_DIR}/inc)
add_executable(
		winograd_test
		${CMAKE_CURRENT_SOURCE_DIR}/src/winograd_test.cpp
		)
target_link_libraries(
		winograd_test
		gtest_main
		)
