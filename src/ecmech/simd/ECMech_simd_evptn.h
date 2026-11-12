#pragma once
/**
 * @file ECMech_simd_evptn.h
 * @brief CPU SIMD path (ECMECH_SIMD_EXPT, run time ECMECH_SIMD=1): the
 * CPU/OpenMP launcher that runs evptn::getResponseLanes (the one getResponse
 * driver, ECMech_evptnLanes.h) on packs of n_lanes points (simd/ECMech_simd_pack.h).
 */

#if defined(ECMECH_SIMD_EXPT)

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <vector>
#if defined(ECMECH_SIMD_EXPT)
#include "simd/ECMech_simd_pack.h"
#endif
#include "evptn/ECMech_evptnSngl.h"

namespace ecmech {
namespace simd {
using evptn::PackArrays;
using evptn::max_pack_hist;
using evptn::pack_supported;

#if defined(ECMECH_SIMD_EXPT)
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
 * @brief Run-time switch of the CPU SIMD path: the environment variable
 * ECMECH_SIMD is set (read once).
 * @return true if the SIMD path should run
 */
inline bool simd_enabled()
{
   static const bool on = std::getenv("ECMECH_SIMD") != nullptr;
   return on;
}

/**
 * @brief CPU/OpenMP SIMD path: getResponse for a.nPassed points, n_lanes at a
 * time, with evptn::getResponseLanes on PackDouble. Points that fail to
 * converge are marked as getResponseSolo marks them (hist[iHistA_nFEval] *= -1).
 * ECMECH_SIMD_STATS (run time) prints the main solve's lane utilization.
 *
 * @tparam SlipGeom, Kinetics, ThermoElastN, EosModel the model's classes
 * @param[in] slipGeom, kinetics, elastN, eos the model
 * @param[in,out] a the arrays of the getResponseECM call
 * @param[in] use_openmp run the packs in an OpenMP parallel loop
 */
template<class SlipGeom, class Kinetics, class ThermoElastN, class EosModel>
void getResponseSimd(const SlipGeom& slipGeom, const Kinetics& kinetics, const ThermoElastN& elastN, const EosModel& eos,
                     const PackArrays& a, const bool use_openmp)
{
   static_assert(simd_supported<SlipGeom, Kinetics, ThermoElastN>::value, "unsupported model for the SIMD path");
   if (a.history_stride > (unsigned)max_pack_hist) { ECMECH_FAIL(__func__, "SIMD path: history too long for the padding scratch"); }
   const int n_packs = (a.nPassed + n_lanes - 1) / n_lanes;
   // ECMECH_SIMD_STATS=1: lane utilization of the main solve
   static const bool stats_on = std::getenv("ECMECH_SIMD_STATS") != nullptr;
   static long stat_useful = 0, stat_issued = 0, stat_calls = 0;

   #pragma omp parallel for schedule(static) if (use_openmp)
   for (int ip = 0; ip < n_packs; ++ip) {
      int fevals[n_lanes + 1];
      const auto converged = evptn::getResponseLanes<PackDouble>(ip, slipGeom, kinetics, elastN, eos, a, stats_on ? fevals : nullptr);
      for (int l = 0; l < n_lanes; ++l) {
         const int i = ip * n_lanes + l;
         if (i < a.nPassed && !mask_lane(converged, l)) { a.histV[a.history_stride * i + evptn::iHistA_nFEval] *= -1; } // as getResponseSolo marks a failure
      }
      if (stats_on) {
         long sum = 0;
         for (int l = 0; l < n_lanes; ++l) { sum += fevals[l]; }
         #pragma omp atomic
         stat_useful += sum;
         #pragma omp atomic
         stat_issued += long(fevals[n_lanes]) * n_lanes;
      }
   }
   if (stats_on && (++stat_calls % 60 == 0)) {
      std::printf("SIMDSTATS lanes %d main-solve lane utilization %.3f (useful %ld / issued %ld)\n",
                  n_lanes, double(stat_useful) / double(stat_issued), stat_useful, stat_issued);
   }
}
#endif

} // namespace simd
} // namespace ecmech

#endif // ECMECH_SIMD_EXPT
