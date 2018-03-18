CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12 FATAL_ERROR)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/inc)
#add debug info
add_definitions("-Wall -pg")
PROJECT(winograd_stu)
SET(SRCS
	${CMAKE_SOURCE_DIR}/src/winograd_2x2_3x3.cpp)
ADD_LIBRARY(winograd_f2k3 STATIC ${SRCS})
SET(TESTS
	${CMAKE_SOURCE_DIR}/test/test_2x2_3x3.cpp)
ADD_EXECUTABLE(winograd ${TESTS})
target_link_libraries(winograd winograd_f2k3)

