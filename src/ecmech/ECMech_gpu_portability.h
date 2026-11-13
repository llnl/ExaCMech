/**
 * @file ECMech_gpu_portability.h
 * @brief Compiler-portability macros for host/device function annotations across CUDA,
 * HIP, and CPU-only builds.
 *
 * ExaCMech's kernels are written once and compiled either for the CPU or for a GPU back
 * end (CUDA or HIP) selected at build time. Rather than sprinkling `#ifdef __CUDACC__`
 * throughout the codebase, every function that needs a `__host__`/`__device__`
 * annotation uses one of the `__ecmech_*__` macros defined here, which expand to the
 * appropriate CUDA/HIP annotation when compiling for a GPU back end and to nothing when
 * compiling for the CPU.
 *
 * @see ECMech_core.h for the umbrella header that pulls this in
 */

#pragma once

#ifdef __CUDACC__
#include <cuda_runtime_api.h>
#endif

#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#endif

/**
 * @def __ecmech_gpu_active__
 * @brief Defined (with no value) when compiling for a CUDA or HIP GPU target; can be
 * used to conditionally compile GPU-only code paths. Not defined for CPU-only builds.
 */
/**
 * @def __ecmech_host__
 * @brief Marks a function as callable from host code only. Expands to `__host__` when
 * compiling for CUDA/HIP, or to nothing for CPU-only builds.
 */
/**
 * @def __ecmech_device__
 * @brief Marks a function as callable from device (GPU) code only. Expands to
 * `__device__` when compiling for CUDA/HIP, or to nothing for CPU-only builds.
 */
/**
 * @def __ecmech_global__
 * @brief Marks a function as a GPU kernel entry point. Expands to `__global__` when
 * compiling for CUDA/HIP, or to nothing for CPU-only builds.
 */
/**
 * @def __ecmech_hdev__
 * @brief Marks a function as callable from both host and device code — the annotation
 * used by the vast majority of ExaCMech's math/kinetics routines so the same
 * implementation runs unmodified on CPU or GPU. Expands to `__host__ __device__` when
 * compiling for CUDA/HIP, or to nothing for CPU-only builds.
 */
#if defined(__CUDACC__) || defined(__HIPCC__)
#define __ecmech_gpu_active__
#define __ecmech_host__   __host__
#define __ecmech_device__ __device__
#define __ecmech_global__ __global__
#define __ecmech_hdev__   __host__ __device__
#else
#define __ecmech_host__
#define __ecmech_device__
#define __ecmech_global__
#define __ecmech_hdev__
#endif

/**
 * @def ECMECH_GPU_THREADS
 * @brief Default number of threads per block used when launching ExaCMech's GPU
 * kernels. Adjust here if a non-standard block size is needed for a given target
 * architecture, or override it at compile time (-DECMECH_GPU_THREADS=N).
 */
#ifndef ECMECH_GPU_THREADS
#define ECMECH_GPU_THREADS 256
#endif

/**
 * @def __ecmech_device_only__
 * @brief Defined when the current compilation pass is generating device (GPU) code.
 * Use to guard code that can only run/compile on the device.
 *
 * nvcc/hipcc compile each `__host__ __device__` function twice — once for the host and
 * once for the device — so this and #__ecmech_host_only__ let code differentiate
 * between those two passes via `__CUDA_ARCH__` (only defined, and > 0, during the
 * device compilation pass) or `__HIP_DEVICE_COMPILE__`.
 */
/**
 * @def __ecmech_host_only__
 * @brief Defined when the current compilation pass is generating host (CPU) code only —
 * i.e. `__ecmech_device_only__` is not defined. Use to guard code (such as
 * exception-based error handling) that is only valid on the host.
 */
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))  || defined(__HIP_DEVICE_COMPILE__)
#define __ecmech_device_only__
#else
#define __ecmech_host_only__
#endif

/**
 * @def __ecmech_device_noinline__
 * @brief Keeps a function out of line in device (GPU) code only; it is still
 * optimized, and inlined as usual on the host. Use it where inlining a function
 * into a very large GPU kernel leads the device compiler to wrong code (see
 * SlipGeomBCCNonSchmid::NSprojection).
 */
#if defined(__ecmech_device_only__)
#define __ecmech_device_noinline__ __noinline__
#else
#define __ecmech_device_noinline__
#endif

// Trial switches (MI300A bcc_e investigation, kit BCC_E_TRIALS.md): with
// -DECMECH_TRIAL_FN_MASK=<bits>, bit k gives function ECMECH_TRIAL_FN_k (the list in that
// document) device attributes, in the device pass only (HIP). ECMECH_TRIAL_FN_MODE=1:
// noinline; ECMECH_TRIAL_FN_MODE=2 (default): noinline and optnone (that function is
// compiled without optimization, the rest as usual). Off by default.
// ECMECH_TRIAL_FN_HOST applies them in host passes too (to test the switches on a CPU).
#if defined(ECMECH_TRIAL_FN_MASK) && (defined(__HIP_DEVICE_COMPILE__) || defined(ECMECH_TRIAL_FN_HOST))
#if defined(ECMECH_TRIAL_FN_MODE) && (ECMECH_TRIAL_FN_MODE == 1)
#define ECMECH_TRIAL_FN_ATTR __attribute__((noinline))
#else
#define ECMECH_TRIAL_FN_ATTR __attribute__((noinline, optnone))
#endif
#if ((ECMECH_TRIAL_FN_MASK) >> 0) & 1
#define ECMECH_TRIAL_FN_0 ECMECH_TRIAL_FN_ATTR
#endif
#if ((ECMECH_TRIAL_FN_MASK) >> 1) & 1
#define ECMECH_TRIAL_FN_1 ECMECH_TRIAL_FN_ATTR
#endif
#if ((ECMECH_TRIAL_FN_MASK) >> 2) & 1
#define ECMECH_TRIAL_FN_2 ECMECH_TRIAL_FN_ATTR
#endif
#if ((ECMECH_TRIAL_FN_MASK) >> 3) & 1
#define ECMECH_TRIAL_FN_3 ECMECH_TRIAL_FN_ATTR
#endif
#if ((ECMECH_TRIAL_FN_MASK) >> 4) & 1
#define ECMECH_TRIAL_FN_4 ECMECH_TRIAL_FN_ATTR
#endif
#if ((ECMECH_TRIAL_FN_MASK) >> 5) & 1
#define ECMECH_TRIAL_FN_5 ECMECH_TRIAL_FN_ATTR
#endif
#if ((ECMECH_TRIAL_FN_MASK) >> 6) & 1
#define ECMECH_TRIAL_FN_6 ECMECH_TRIAL_FN_ATTR
#endif
#if ((ECMECH_TRIAL_FN_MASK) >> 7) & 1
#define ECMECH_TRIAL_FN_7 ECMECH_TRIAL_FN_ATTR
#endif
#if ((ECMECH_TRIAL_FN_MASK) >> 8) & 1
#define ECMECH_TRIAL_FN_8 ECMECH_TRIAL_FN_ATTR
#endif
#if ((ECMECH_TRIAL_FN_MASK) >> 9) & 1
#define ECMECH_TRIAL_FN_9 ECMECH_TRIAL_FN_ATTR
#endif
#if ((ECMECH_TRIAL_FN_MASK) >> 10) & 1
#define ECMECH_TRIAL_FN_10 ECMECH_TRIAL_FN_ATTR
#endif
#if ((ECMECH_TRIAL_FN_MASK) >> 11) & 1
#define ECMECH_TRIAL_FN_11 ECMECH_TRIAL_FN_ATTR
#endif
#if ((ECMECH_TRIAL_FN_MASK) >> 12) & 1
#define ECMECH_TRIAL_FN_12 ECMECH_TRIAL_FN_ATTR
#endif
#endif
#ifndef ECMECH_TRIAL_FN_0
#define ECMECH_TRIAL_FN_0
#endif
#ifndef ECMECH_TRIAL_FN_1
#define ECMECH_TRIAL_FN_1
#endif
#ifndef ECMECH_TRIAL_FN_2
#define ECMECH_TRIAL_FN_2
#endif
#ifndef ECMECH_TRIAL_FN_3
#define ECMECH_TRIAL_FN_3
#endif
#ifndef ECMECH_TRIAL_FN_4
#define ECMECH_TRIAL_FN_4
#endif
#ifndef ECMECH_TRIAL_FN_5
#define ECMECH_TRIAL_FN_5
#endif
#ifndef ECMECH_TRIAL_FN_6
#define ECMECH_TRIAL_FN_6
#endif
#ifndef ECMECH_TRIAL_FN_7
#define ECMECH_TRIAL_FN_7
#endif
#ifndef ECMECH_TRIAL_FN_8
#define ECMECH_TRIAL_FN_8
#endif
#ifndef ECMECH_TRIAL_FN_9
#define ECMECH_TRIAL_FN_9
#endif
#ifndef ECMECH_TRIAL_FN_10
#define ECMECH_TRIAL_FN_10
#endif
#ifndef ECMECH_TRIAL_FN_11
#define ECMECH_TRIAL_FN_11
#endif
#ifndef ECMECH_TRIAL_FN_12
#define ECMECH_TRIAL_FN_12
#endif
