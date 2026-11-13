#pragma once
/**
 * @file ECMech_simd_pack.h
 * @brief The CPU SIMD path's value type: n_lanes material points in one
 * snls::simd::pack<n_lanes> (SNLS_pack.h: plain arrays that the compiler
 * vectorizes, standard C++17, per-lane math functions).
 *
 * n_lanes is the CMake option ECMECH_SIMD_LANES (default 4, see ECMech_config.h).
 */

#include "ECMech_config.h"
#include "SNLS_pack.h"

namespace ecmech {
namespace simd {
/// Points per pack on the CPU SIMD path (CMake option ECMECH_SIMD_LANES)
constexpr int n_lanes = ECMECH_SIMD_LANES;
static_assert(n_lanes >= 1, "ECMECH_SIMD_LANES must be at least 1");
/// The pack value type: n_lanes doubles, one per point
using PackDouble = snls::simd::pack<n_lanes>;
} // namespace simd
// (the math functions for packs are found by argument-dependent lookup)
} // namespace ecmech
