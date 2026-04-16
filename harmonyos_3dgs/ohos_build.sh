#!/bin/bash
set -e
NDK=/home/randy/harmonyos/cmdline-tools/sdk/default/openharmony/native
cmake -B build-ohos \
    -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/ohos.toolchain.cmake \
    -DOHOS_ARCH=arm64-v8a \
    -DENABLE_OPENCL=ON \
    -DBUILD_TESTS=OFF \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build-ohos -j$(nproc)
echo "Binary: build-ohos/gs3d_render"
file build-ohos/gs3d_render
