#pragma once
/**
 * @file ECMech_simd_pack.h
 * @brief CPU SIMD path (ECMECH_SIMD_EXPT): the pack value type, n_lanes points in
 * one snls::simd::pack<n_lanes> (SNLS_pack.h: plain arrays that the compiler
 * vectorizes, standard C++17, per-lane math functions).
 *
 * n_lanes defaults to the ISA's double width (8 with AVX-512, 4 with AVX, else 2);
 * -DECMECH_SIMD_LANES=N overrides it.
 */

#include "SNLS_pack.h"

namespace ecmech {
namespace simd {
/// Points per pack on the CPU SIMD path
#if defined(ECMECH_SIMD_LANES)
constexpr int n_lanes = ECMECH_SIMD_LANES;
#else
constexpr int n_lanes = snls::simd::pack_native_lanes();
#endif
/// The pack value type: n_lanes doubles, one per point
using PackDouble = snls::simd::pack<n_lanes>;
/// The mask type of PackDouble: one flag per point
using PackMask = snls::simd::pack_mask<n_lanes>;
} // namespace simd
// (the math functions for packs are found by argument-dependent lookup)
} // namespace ecmech
