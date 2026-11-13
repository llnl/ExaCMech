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

/**
 * @brief Whether the CPU SIMD path supports a model: its kinetics' slip rates
 * (evalGdots) are templated on the value type (evptn::has_pack_slip_rates for
 * PackDouble). Other models run the one-point path.
 * @tparam Kinetics the model's kinetics class
 */
template<class Kinetics>
struct simd_supported {
    static constexpr bool value = evptn::has_pack_slip_rates<Kinetics, PackDouble>::value;
};

/**
 * @brief CPU/OpenMP SIMD path: getResponse for arrays.nPassed points, n_lanes at
 * a time, with evptn::getResponseLanes on PackDouble. Points that fail to
 * converge are marked as getResponsePerPoint marks them (hist[iHistA_nFEval] *= -1).
 *
 * @tparam SlipGeom, Kinetics, ThermoElastN, EosModel the model's classes
 * @param[in] slipGeom, kinetics, elastN, eos the model
 * @param[in,out] arrays the arrays of the getResponseECM call
 * @param[in] use_openmp run the packs in an OpenMP parallel loop (RAJA)
 */
template<class SlipGeom, class Kinetics, class ThermoElastN, class EosModel>
void getResponseSimd(const SlipGeom& slipGeom, const Kinetics& kinetics, const ThermoElastN& elastN, const EosModel& eos,
                     const evptn::ResponseArrays& arrays, const bool use_openmp)
{
    static_assert(simd_supported<Kinetics>::value, "unsupported model for the SIMD path");
    if (arrays.history_stride > (unsigned)evptn::max_pack_hist) {
        ECMECH_FAIL(__func__, "SIMD path: history too long for the padding scratch");
    }
    const int n_packs = (arrays.nPassed + n_lanes - 1) / n_lanes;

    auto pack_body = [&] (int i_pack) {
        const auto converged = evptn::getResponseLanes<PackDouble>(i_pack, slipGeom, kinetics, elastN, eos, arrays);
        for (int l = 0; l < n_lanes; ++l) {
            const int i = i_pack * n_lanes + l;
            // as getResponsePerPoint marks a failure
            if (i < arrays.nPassed && !mask_lane(converged, l)) {
                arrays.histV[arrays.history_stride * i + evptn::iHistA_nFEval] *= -1;
            }
        }
    };
    // the packs in parallel (OpenMP through RAJA) when requested and available,
    // else in sequence
    const RAJA::RangeSegment packs(0, n_packs);
#if defined(RAJA_ENABLE_OPENMP) && defined(OPENMP_ENABLE)
    if (use_openmp) {
        RAJA::forall<RAJA::omp_parallel_for_exec>(packs, pack_body);
        return;
    }
#endif
    (void) use_openmp;
    RAJA::forall<RAJA::seq_exec>(packs, pack_body);
}

} // namespace simd
} // namespace ecmech
