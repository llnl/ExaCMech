#pragma once
/**
 * @file ECMech_value_type.h
 * @brief The value-type layer (SNLS_value_type.h) in namespace ecmech.
 *
 * The point-wise numerics are templated on a value type R: double for one
 * point (the CPU and GPU paths), or a pack of points processed in lockstep
 * (the CPU SIMD path, ECMECH_SIMD_EXPT). Per-point branches use
 * `if (any(m)) { ... set_where(m, x, value); }`, which for double is the
 * original branch with the original operations.
 */

#include "SNLS_value_type.h"

namespace ecmech {
   using snls::value_traits;
   using snls::mask_t;
   using snls::index_t;
   using snls::lanes_of;
   using snls::select;
   using snls::any;
   using snls::all;
   using snls::set_where;
   using snls::merge_mask;
   using snls::all_lanes;
   using snls::no_lanes;
   using snls::lane;
   using snls::set_lane;
   using snls::mask_lane;
   using snls::set_mask_lane;
} // namespace ecmech

// The pack type, its traits and its vector math; host code of CPU builds
// with ECMECH_SIMD_EXPT only. Included here so that the math overloads are
// declared before any templated code that calls exp/log/pow/sin/cos.
// The in-house pack (ECMECH_SIMD_INHOUSE) is declared in both nvcc/hipcc
// passes (host functions are parsed in the device pass too); it is only used
// by host code.
#if defined(ECMECH_SIMD_EXPT) && (defined(ECMECH_SIMD_INHOUSE) || (!defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)))
#include "simd/ECMech_simd_pack.h"
#endif
