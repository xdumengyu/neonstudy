#!/bin/bash
mkdir -p build
pushd build
cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a  -DANDROID_TOOLCHAIN=gcc ..
make 
popd
