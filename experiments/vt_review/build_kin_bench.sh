#!/usr/bin/env bash
# build_kin_bench.sh <out dir>: builds kin_bench in four flag variants (g++-13, -O3 -march=native)
set -e
HERE=$(cd $(dirname $0) && pwd); SRC=$HERE/../../src/ecmech; O=${1:-.}
B=/home/robert/exaconstit_cuda_builds
INC="-I$SRC -I$B/snls_threads/build/vt_simd/include/ecmech -isystem $B/RAJA/install_cpu_omp/include -isystem $B/camp/install_cpu_omp/include -isystem $B/snls_threads/install/snls_rv_cpu/include/snls -isystem $B/CHAI/install_cpu_omp/include -isystem $B/Umpire/install_cpu_omp/include -DFMT_HEADER_ONLY=1"
LIBS="/home/robert/exaconstit_cuda_builds/CHAI/install_cpu_omp/lib/libchai.a /home/robert/exaconstit_cuda_builds/Umpire/install_cpu_omp/lib/libumpire.a /home/robert/exaconstit_cuda_builds/Umpire/install_cpu_omp/lib/libfmt.a /home/robert/exaconstit_cuda_builds/RAJA/install_cpu_omp/lib/libRAJA.a /home/robert/exaconstit_cuda_builds/camp/install_cpu_omp/lib/libcamp.a -fopenmp -ldl"
CXX=${CXX:-g++-13}
BASE="-std=c++17 -O3 -DNDEBUG -march=native -fopenmp-simd"
$CXX $BASE -DECMECH_SIMD_EXPT $INC $HERE/kin_bench.cxx -o $O/kin_bench -lmvec $LIBS -fopt-info-vec-optimized=$O/vec_default.txt -fopt-info-vec-missed=$O/vecmissed_default.txt
$CXX $BASE -DECMECH_SIMD_EXPT -DKB_DECLARE_SIMD_MATH $INC $HERE/kin_bench.cxx -o $O/kin_bench_declsimd -lmvec $LIBS -fopt-info-vec-optimized=$O/vec_declsimd.txt
$CXX $BASE -ffast-math $INC $HERE/kin_bench.cxx -o $O/kin_bench_fastmath -lmvec $LIBS -fopt-info-vec-optimized=$O/vec_fastmath.txt -fopt-info-vec-missed=$O/vecmissed_fastmath.txt
$CXX $BASE -DECMECH_SIMD_EXPT -ffp-contract=off $INC $HERE/kin_bench.cxx -o $O/kin_bench_nofma -lmvec $LIBS
$CXX $BASE -DECMECH_SIMD_EXPT -DECMECH_SIMD_INHOUSE=1 $INC $HERE/kin_bench.cxx -o $O/kin_bench_inhouse -lmvec $LIBS
