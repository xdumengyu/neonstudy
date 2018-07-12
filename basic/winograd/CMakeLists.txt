CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12 FATAL_ERROR)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/inc)
#change cross build install dir
#add debug info
add_definitions("-Wall -O3  -std=c++11")
#set OpenCV_DIR env
#set(ANDROID_ABI "arm64-v8a")
#set(OpenCV_DIR "/home/mintmy/Downloads/OpenCV-android-sdk/sdk/native/jni")
set(OpenCV_STATIC ON)
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui)
include_directories(${OpenCV_INCLUDE_DIRS})


set(CMAKE_BUILD_TYPE Release)
PROJECT(winograd_stu)
SET(SRCS
		${CMAKE_CURRENT_SOURCE_DIR}/src/winograd_2x2_3x3.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/src/cvtColor.cpp)
ADD_LIBRARY(winograd_f2k3 STATIC ${SRCS})
SET(TESTS
		${CMAKE_CURRENT_SOURCE_DIR}/test/test_2x2_3x3.cpp)
ADD_EXECUTABLE(winograd ${TESTS})
#link static opencv
target_link_libraries(winograd_f2k3 ${OpenCV_LIBS})
target_link_libraries(winograd 
		winograd_f2k3 
		${OpenCV_LIBS}
		)
