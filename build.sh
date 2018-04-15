#!/bin/bash
rm -rf build
mkdir -p build
pushd build
export OpenCV_DIR="/home/mintmy/Downloads/OpenCV-android-sdk/sdk/native/jni"
export ANDROID_NDK="/home/mintmy/kenzo/android-ndk-r14b"

# def DANDROID_NDK_ABI_NAME for opencv link
cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake -DOpenCV_DIR=${OpenCV_DIR} -DANDROID_ABI="arm64-v8a" -DANDROID_TOOLCHAIN=gcc -DANDROID_NDK_ABI_NAME="arm64-v8a" ..
make 
popd
