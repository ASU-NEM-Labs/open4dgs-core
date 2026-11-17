#!/usr/bin/env bash
# Build the headless COLMAP 3.12.0 CUDA CLI for the prep workflows.
set -Eeuo pipefail

fail() { printf 'COLMAP setup: %s\n' "$*" >&2; exit 1; }
need_file() { [[ -e "$1" ]] || fail "Missing $1"; }
need_command() { command -v "$1" >/dev/null || fail "Missing command: $1"; }

[[ $(uname -s) == Linux ]] || fail 'This script requires Linux.'
[[ -n ${CONDA_PREFIX:-} ]] || fail 'Activate the trigs Conda environment first.'
need_command cmake
need_command ninja
need_command git
need_command nvidia-smi
need_file /usr/bin/gcc
need_file /usr/bin/g++
need_file /usr/include/glog/logging.h
need_file /usr/include/eigen3/Eigen/Core
need_file /usr/lib/x86_64-linux-gnu/libmetis.so
need_file /usr/lib/x86_64-linux-gnu/libamd.so
need_file /usr/lib/x86_64-linux-gnu/libcamd.so
need_file /usr/lib/x86_64-linux-gnu/libccolamd.so
need_file /usr/lib/x86_64-linux-gnu/libcholmod.so
need_file /usr/lib/x86_64-linux-gnu/libcolamd.so
need_file /usr/lib/x86_64-linux-gnu/libspqr.so
need_file /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.so
need_file /usr/lib/x86_64-linux-gnu/openblas-openmp/libopenblas.so
need_file /usr/lib/x86_64-linux-gnu/cmake/Ceres/CeresConfig.cmake
need_file "$CONDA_PREFIX/bin/nvcc"
printf '#include <glog/logging.h>\nint main() { return 0; }\n' |
  /usr/bin/g++ -std=c++17 -x c++ -fsyntax-only - ||
  fail 'Ubuntu GCC cannot compile against the installed glog headers.'

colmap_src=${COLMAP_SRC:-"$HOME/src/colmap-3.12.0"}
colmap_build=${COLMAP_BUILD:-"$colmap_src/build-system"}
need_file "$colmap_src/CMakeLists.txt"
expected_commit=e7e89eb0c82eccc00b78086e419f4def4e5860ae
actual_commit=$(git -C "$colmap_src" rev-parse HEAD)
[[ $actual_commit == "$expected_commit" ]] ||
  fail "Expected COLMAP commit $expected_commit; found $actual_commit"

cuda_arch=${COLMAP_CUDA_ARCHITECTURES:-}
if [[ -z $cuda_arch ]]; then
  compute_cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sed -n '1p' | tr -d '[:space:]')
  [[ $compute_cap =~ ^[0-9]+\.[0-9]+$ ]] || fail "Cannot parse GPU compute capability: $compute_cap"
  cuda_arch=${compute_cap/./}
fi
[[ $cuda_arch =~ ^[0-9]+$ ]] || fail "Invalid CUDA architecture: $cuda_arch"
if [[ -f "$colmap_build/CMakeCache.txt" ]]; then
  grep -Eq '^CMAKE_CXX_COMPILER:(FILEPATH|STRING)=/usr/bin/g\+\+$' "$colmap_build/CMakeCache.txt" ||
    fail "Build cache at $colmap_build uses a different compiler. Choose a fresh COLMAP_BUILD path."
fi

printf 'Building COLMAP %s for CUDA architecture %s\n' "$expected_commit" "$cuda_arch"
cmake -S "$colmap_src" -B "$colmap_build" -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/gcc \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++ \
  -DCMAKE_FIND_NO_INSTALL_PREFIX=TRUE \
  -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" \
  -DCMAKE_INSTALL_RPATH="$CONDA_PREFIX/lib;$CONDA_PREFIX/lib/colmap" \
  -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON \
  -DCMAKE_CUDA_COMPILER="$CONDA_PREFIX/bin/nvcc" \
  -DCUDAToolkit_ROOT="$CONDA_PREFIX" \
  -DCMAKE_CUDA_ARCHITECTURES="$cuda_arch" \
  -DCMAKE_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu \
  -DBoost_NO_BOOST_CMAKE=ON \
  -DBoost_NO_SYSTEM_PATHS=ON \
  -DBOOST_ROOT=/usr \
  -DBOOST_INCLUDEDIR=/usr/include \
  -DBOOST_LIBRARYDIR=/usr/lib/x86_64-linux-gnu \
  -DOPENGL_opengl_LIBRARY=/usr/lib/x86_64-linux-gnu/libOpenGL.so \
  -DOPENGL_glx_LIBRARY=/usr/lib/x86_64-linux-gnu/libGLX.so \
  -DCeres_DIR=/usr/lib/x86_64-linux-gnu/cmake/Ceres \
  -Dgflags_DIR=/usr/lib/x86_64-linux-gnu/cmake/gflags \
  -Dglog_DIR=/usr/lib/x86_64-linux-gnu/cmake/glog \
  -DSUITESPARSE_LIBRARY_DIR_HINTS=/usr/lib/x86_64-linux-gnu \
  -DSuiteSparse_AMD_LIBRARY:FILEPATH=/usr/lib/x86_64-linux-gnu/libamd.so \
  -DSuiteSparse_CAMD_LIBRARY:FILEPATH=/usr/lib/x86_64-linux-gnu/libcamd.so \
  -DSuiteSparse_CCOLAMD_LIBRARY:FILEPATH=/usr/lib/x86_64-linux-gnu/libccolamd.so \
  -DSuiteSparse_CHOLMOD_LIBRARY:FILEPATH=/usr/lib/x86_64-linux-gnu/libcholmod.so \
  -DSuiteSparse_COLAMD_LIBRARY:FILEPATH=/usr/lib/x86_64-linux-gnu/libcolamd.so \
  -DSuiteSparse_SPQR_LIBRARY:FILEPATH=/usr/lib/x86_64-linux-gnu/libspqr.so \
  -DSuiteSparse_Config_LIBRARY:FILEPATH=/usr/lib/x86_64-linux-gnu/libsuitesparseconfig.so \
  -DMETIS_LIBRARY_RELEASE:FILEPATH=/usr/lib/x86_64-linux-gnu/libmetis.so \
  -DMETIS_LIBRARY:FILEPATH=/usr/lib/x86_64-linux-gnu/libmetis.so \
  -DBLAS_LIBRARIES=/usr/lib/x86_64-linux-gnu/openblas-openmp/libopenblas.so \
  -DLAPACK_LIBRARIES=/usr/lib/x86_64-linux-gnu/openblas-openmp/libopenblas.so \
  -DBLA_VENDOR=OpenBLAS \
  -DCGAL_ENABLED=OFF \
  -DFETCH_POSELIB=ON \
  -DFETCH_FAISS=ON \
  -DCUDA_ENABLED=ON \
  -DGUI_ENABLED=OFF \
  -DTESTS_ENABLED=OFF

cmake --build "$colmap_build" --parallel "${COLMAP_BUILD_JOBS:-2}"
cmake --install "$colmap_build"
help_output=$("$CONDA_PREFIX/bin/colmap" help 2>&1)
printf '%s\n' "$help_output" | grep -F 'COLMAP 3.12.0' >/dev/null || fail 'Installed COLMAP version is not 3.12.0.'
printf '%s\n' "$help_output" | grep -F 'with CUDA' >/dev/null || fail 'Installed COLMAP does not report CUDA.'
printf '%s\n' "$help_output" | head -n 4
