#pragma once
/**
 * @file ECMech_simd_evptn.h
 * @brief CPU SIMD path (ECMECH_SIMD_EXPT, run time ECMECH_SIMD=1): the
 * CPU/OpenMP launcher that runs evptn::getResponseLanes (the one getResponse
 * driver, ECMech_evptnLanes.h) on packs of W points (simd/ECMech_simd_pack.h).
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
using evptn::maxPackHist;
using evptn::pack_supported;

#if defined(ECMECH_SIMD_EXPT)
template<class SlipGeom, class Kinetics, class ThermoElastN>
struct simd_supported {
   static constexpr bool value = pack_supported<SlipGeom, Kinetics, ThermoElastN, vdouble>::value;
};

/// Run-time switch ECMECH_SIMD=1
inline bool simdEnabled()
{
   static const bool on = std::getenv("ECMECH_SIMD") != nullptr;
   return on;
}

/// CPU/OpenMP: getResponse for nPassed points, W at a time.
template<class SlipGeom, class Kinetics, class ThermoElastN, class EosModel>
void getResponseSimd(const SlipGeom& slipGeom, const Kinetics& kinetics, const ThermoElastN& elastN, const EosModel& eos,
                     const PackArrays& a, const bool useOpenMP)
{
   static_assert(simd_supported<SlipGeom, Kinetics, ThermoElastN>::value, "unsupported model for the SIMD path");
   if (a.history_stride > (unsigned)maxPackHist) { ECMECH_FAIL(__func__, "SIMD path: history too long for the padding scratch"); }
   const int nPacks = (a.nPassed + W - 1) / W;
   // ECMECH_SIMD_STATS=1: lane utilization of the main solve
   static const bool statsOn = std::getenv("ECMECH_SIMD_STATS") != nullptr;
   static long statUseful = 0, statIssued = 0, statCalls = 0;

   #pragma omp parallel for schedule(static) if (useOpenMP)
   for (int ip = 0; ip < nPacks; ++ip) {
      int fe[W + 1];
      const auto conv = evptn::getResponseLanes<vdouble>(ip, slipGeom, kinetics, elastN, eos, a, statsOn ? fe : nullptr);
      for (int l = 0; l < W; ++l) {
         const int i = ip * W + l;
         if (i < a.nPassed && !mask_lane(conv, l)) { a.histV[a.history_stride * i + evptn::iHistA_nFEval] *= -1; } // as getResponseSolo marks a failure
      }
      if (statsOn) {
         long sum = 0;
         for (int l = 0; l < W; ++l) { sum += fe[l]; }
         #pragma omp atomic
         statUseful += sum;
         #pragma omp atomic
         statIssued += long(fe[W]) * W;
      }
   }
   if (statsOn && (++statCalls % 60 == 0)) {
      std::printf("SIMDSTATS lanes %d main-solve lane utilization %.3f (useful %ld / issued %ld)\n",
                  W, double(statUseful) / double(statIssued), statUseful, statIssued);
   }
}
#endif

} // namespace simd
} // namespace ecmech

#endif // ECMECH_SIMD_EXPT
