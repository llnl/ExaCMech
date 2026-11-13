#pragma once
/**
 * @file ECMech_simd_evptn.h
 * @brief The CPU SIMD path: the CPU/OpenMP launcher that runs
 * evptn::getResponseLanes (the one getResponse driver, ECMech_evptnLanes.h) on
 * packs of n_lanes points (simd/ECMech_simd_pack.h). getResponseECM uses it on
 * the CPU and OpenMP backends for every model it supports (simd_supported).
 */

#include "RAJA/RAJA.hpp"
#include "simd/ECMech_simd_pack.h"
#include "evptn/ECMech_evptnSngl.h"

namespace ecmech {
namespace simd {
using evptn::PackArrays;
using evptn::max_pack_hist;
using evptn::pack_supported;

/**
 * @brief Whether the CPU SIMD path supports a model (evptn::pack_supported for
 * the pack type PackDouble); other models run the one-point path.
 * @tparam SlipGeom, Kinetics, ThermoElastN the model's classes
 */
template<class SlipGeom, class Kinetics, class ThermoElastN>
struct simd_supported {
   static constexpr bool value = pack_supported<SlipGeom, Kinetics, ThermoElastN, PackDouble>::value;
};

/**
 * @brief CPU/OpenMP SIMD path: getResponse for a.nPassed points, n_lanes at a
 * time, with evptn::getResponseLanes on PackDouble. Points that fail to
 * converge are marked as getResponsePerPoint marks them (hist[iHistA_nFEval] *= -1).
 *
 * @tparam SlipGeom, Kinetics, ThermoElastN, EosModel the model's classes
 * @param[in] slipGeom, kinetics, elastN, eos the model
 * @param[in,out] a the arrays of the getResponseECM call
 * @param[in] use_openmp run the packs in an OpenMP parallel loop (RAJA)
 */
template<class SlipGeom, class Kinetics, class ThermoElastN, class EosModel>
void getResponseSimd(const SlipGeom& slipGeom, const Kinetics& kinetics, const ThermoElastN& elastN, const EosModel& eos,
                     const PackArrays& a, const bool use_openmp)
{
   static_assert(simd_supported<SlipGeom, Kinetics, ThermoElastN>::value, "unsupported model for the SIMD path");
   if (a.history_stride > (unsigned)max_pack_hist) { ECMECH_FAIL(__func__, "SIMD path: history too long for the padding scratch"); }
   const int n_packs = (a.nPassed + n_lanes - 1) / n_lanes;

   auto pack_body = [&] (int ip) {
      const auto converged = evptn::getResponseLanes<PackDouble>(ip, slipGeom, kinetics, elastN, eos, a);
      for (int l = 0; l < n_lanes; ++l) {
         const int i = ip * n_lanes + l;
         if (i < a.nPassed && !mask_lane(converged, l)) { a.histV[a.history_stride * i + evptn::iHistA_nFEval] *= -1; } // as getResponsePerPoint marks a failure
      }
   };
   // the packs in parallel through RAJA (no OpenMP in this file: builds without
   // OpenMP run the sequential loop)
   const RAJA::RangeSegment packs(0, n_packs);
#if defined(RAJA_ENABLE_OPENMP) && defined(OPENMP_ENABLE)
   if (use_openmp) { RAJA::forall<RAJA::omp_parallel_for_exec>(packs, pack_body); }
   else
#endif
   { (void) use_openmp; RAJA::forall<RAJA::seq_exec>(packs, pack_body); }
}

} // namespace simd
} // namespace ecmech

