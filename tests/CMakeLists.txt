CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12 FATAL_ERROR)
include_directories(googletest_SOURCE_DIR/googletest/include)
include_directories(googletest_SOURCE_DIR/googlemock/include)
add_executable(
		winograd_test
		${CMAKE_CURRENT_SOURCE_DIR}/src/winograd_test.cpp
		)
target_link_libraries(
		winograd_test
		gtest_main
		)
