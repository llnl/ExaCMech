#pragma once
/**
 * @file ECMech_simd_pack.h
 * @brief CPU SIMD path (ECMECH_SIMD_EXPT): the pack value type, W points in
 * one snls::simd::pack<W> (SNLS_pack.h: plain arrays that the compiler
 * vectorizes, standard C++17, per-lane math functions).
 *
 * W defaults to the ISA's double width (8 with AVX-512, 4 with AVX, else 2);
 * -DECMECH_SIMD_LANES=N overrides it.
 */

#include "SNLS_pack.h"

namespace ecmech {
namespace simd {
#if defined(ECMECH_SIMD_LANES)
constexpr int W = ECMECH_SIMD_LANES;
#else
constexpr int W = snls::simd::pack_native_lanes();
#endif
using vdouble = snls::simd::pack<W>;
using vmask = snls::simd::pack_mask<W>;
} // namespace simd
// (the math functions for packs are found by argument-dependent lookup)
} // namespace ecmech
