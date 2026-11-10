#pragma once
/**
 * @file ECMech_simd_evptn.h
 * @brief Experiment (ECMECH_SIMD_EXPT, CPU only): getResponse for W points
 * at a time with the pack value type (simd/ECMech_simd_pack.h).
 *
 * There is no SIMD copy of the model here: the problem is the original
 * evptn::EvptnUpdstProblem instantiated on a pack problem state, its
 * kinetics/elasticity/slip geometry are the original classes (templated on
 * the value type), and the Newton solve is SNLS's SNLSTrDlDenseG_Lanes,
 * built from the same kernels as SNLSTrDlDenseG. What is here is the
 * per-pack driver: gather the points into a pack, solve, scatter back.
 * Preprocess (EOS, and the hardness update unless the kinetics can do it on
 * packs) and postprocess run per point with the original code.
 */

#if defined(ECMECH_SIMD_EXPT)

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <vector>
#include "simd/ECMech_simd_pack.h"
#include "evptn/ECMech_evptnSngl.h"
#include "SNLS_TrDLDenseG_Lanes.h"

namespace ecmech {
namespace simd {

/// The fields of evptn::ProblemState that EvptnUpdstProblem reads, for a
/// pack of points
template<class Kinetics>
struct ProblemStatePack {
   using value_type = vdouble;
   double dt;
   vdouble rel_vol_new, energy_new, pressure_EOS, tkelv;
   vdouble elast_d5_n[ecmech::ntvec];
   vdouble quat_n[ecmech::qdim];
   vdouble def_rate_d5_sample[ecmech::ntvec];
   vdouble spin_vec_sample[ecmech::nwvec];
   vdouble h_state_u[(Kinetics::nH > 0) ? Kinetics::nH : 1];
};

template<class SG, class = void> struct has_rss_from_stress : std::false_type {};
template<class SG> struct has_rss_from_stress<SG, std::void_t<decltype(SG::rssFromStress)>> : std::true_type {};

/// A dynamic (stress-dependent) slip geometry for a pack of points: each
/// point's P, Q and extra values (as its own geometry computed them in
/// preprocess) gathered into packs, behind the interface the templated
/// problem uses (nslip, nSlipExtra, dynamic, getP, getQ, evalRSS,
/// getExtras).
template<class SG>
struct SlipGeomPack {
   static constexpr int nslip = SG::nslip;
   static constexpr int nSlipExtra = SG::nSlipExtra;
   static constexpr bool dynamic = SG::dynamic;
   const SG* m_g0 = nullptr; ///< one point's geometry (for projections that depend on the stress only)
   vdouble m_P[ecmech::ntvec * nslip];
   vdouble m_Q[ecmech::nwvec * nslip];
   vdouble m_extras[(nSlipExtra > 0) ? nSlipExtra : 1];

   /// gather point l's geometry
   void setLane(int l, const SG& g) {
      if (!m_g0) { m_g0 = &g; }
      const double* const P = g.getP();
      const double* const Q = g.getQ();
      for (int k = 0; k < ecmech::ntvec * nslip; ++k) { m_P[k][l] = P[k]; }
      for (int k = 0; k < ecmech::nwvec * nslip; ++k) { m_Q[k][l] = Q[k]; }
      if constexpr (nSlipExtra > 0) {
         double e[nSlipExtra];
         g.getExtras(e);
         for (int k = 0; k < nSlipExtra; ++k) { m_extras[k][l] = e[k]; }
      }
   }
   const vdouble* getP() const { return m_P; }
   const vdouble* getQ() const { return m_Q; }
   template<class T>
   void evalRSS(vdouble* taua, const T* const kirchoff, const vdouble* P_vec) const {
      if constexpr (has_rss_from_stress<SG>::value) {
         m_g0->evalRSS(taua, kirchoff, (const double*)nullptr);
      }
      else {
         vecsVaTM<ecmech::ntvec, nslip>(taua, kirchoff, P_vec);
      }
   }
   void getExtras(vdouble* const e) const {
      for (int k = 0; k < nSlipExtra; ++k) { e[k] = m_extras[k]; }
   }
};

/// kinetics whose slip rates are templated on the value type
template<class K, class = void> struct has_pack_slip_rates : std::false_type {};
template<class K>
struct has_pack_slip_rates<K, std::void_t<decltype(std::declval<const K&>().evalGdots(
   (vdouble*)nullptr, (vdouble*)nullptr, (const vdouble*)nullptr, (const vdouble*)nullptr))>> : std::true_type {};
/// kinetics whose hardness update can run on packs
template<class K, class = void> struct has_pack_hardening : std::false_type {};
template<class K>
struct has_pack_hardening<K, std::void_t<decltype(std::declval<const K&>().updateHLanes(
   (vdouble*)nullptr, (const vdouble*)nullptr, 0.0, (const vdouble*)nullptr))>> : std::true_type {};

/// Models the pack path supports (others use the scalar path)
template<class SlipGeom, class Kinetics, class ThermoElastN>
struct simd_supported {
   static constexpr bool value = has_pack_slip_rates<Kinetics>::value;
};

/// Run-time switch ECMECH_SIMD=1
inline bool simdEnabled()
{
   static const bool on = std::getenv("ECMECH_SIMD") != nullptr;
   return on;
}

/// getResponse for nPassed points, W at a time.
template<class SlipGeom, class Kinetics, class ThermoElastN, class EosModel>
void getResponseSimd(const SlipGeom& slipGeom, const Kinetics& kinetics, const ThermoElastN& elastN, const EosModel& eos,
                     const double dt, const double tolerance,
                     const double* def_rate_d6vV, unsigned int def_rate_stride,
                     const double* spin_vecV, unsigned int spin_v_stride,
                     const double* rel_vol_ratiosV, unsigned int vol_ratio_stride,
                     double* internal_energyV, unsigned int int_eng_stride,
                     double* cauchy_stress_d6pV, unsigned int stress_stride,
                     double* histV, unsigned int history_stride,
                     double* tkelvV, unsigned int tkelv_stride,
                     double* sddV, unsigned int sdd_stride,
                     double* mtanSDV,
                     const int nPassed, const bool useOpenMP)
{
   using PS = evptn::ProblemState<SlipGeom, Kinetics, ThermoElastN, EosModel>;
   using PSV = ProblemStatePack<Kinetics>;
   constexpr bool dynamicGeom = SlipGeom::dynamic;
   using GeomV = std::conditional_t<dynamicGeom, SlipGeomPack<SlipGeom>, SlipGeom>;
   using Prob = evptn::EvptnUpdstProblem<GeomV, Kinetics, ThermoElastN, PSV>;
   using Solver = snls::SNLSTrDlDenseG_Lanes<Prob, vdouble>;
   static_assert(simd_supported<SlipGeom, Kinetics, ThermoElastN>::value, "unsupported model for the SIMD path");
   constexpr bool hardeningOnPacks = has_pack_hardening<Kinetics>::value;
   constexpr int nH = Kinetics::nH;
   constexpr int nslip = SlipGeom::nslip;
   constexpr int n = Prob::nDimSys;

   snls::TrDeltaControl deltaControl;
   deltaControl._deltaInit = 1e0;
   const int nPacks = (nPassed + W - 1) / W;
   const unsigned int nHist = history_stride;
   if (nHist > 128) { ECMECH_FAIL(__func__, "SIMD path: history too long for the padding scratch"); }
   // ECMECH_SIMD_STATS=1: lane utilization of the main solve
   static const bool statsOn = std::getenv("ECMECH_SIMD_STATS") != nullptr;
   static long statUseful = 0, statIssued = 0, statCalls = 0;

   #pragma omp parallel for schedule(static) if (useOpenMP)
   for (int ip = 0; ip < nPacks; ++ip) {
      // point index per lane; lanes past the end repeat the last point on copies
      int idx[W];
      bool real[W];
      for (int l = 0; l < W; ++l) {
         const int i = ip * W + l;
         real[l] = (i < nPassed);
         idx[l] = real[l] ? i : (nPassed - 1);
      }
      double padHist[W][128], padStress[W][8], padEnergy[W][8], padSdd[W][8], padT[W];
      alignas(PS) unsigned char psMem[W][sizeof(PS)];
      // dynamic slip geometries: one copy per point (as getResponseSolo makes per thread)
      alignas(SlipGeom) unsigned char geomMem[dynamicGeom ? W : 1][sizeof(SlipGeom)];
      const SlipGeom* geom[W];
      PS* ps[W];
      bool ok[W];
      double halfVMidDt[W], devE[W];
      double* histL[W]; double* stressL[W]; double* energyL[W]; double* sddL[W]; double* tL[W];

      // per point: preprocess (original code)
      for (int l = 0; l < W; ++l) {
         const int i = idx[l];
         histL[l] = &histV[history_stride * i];
         stressL[l] = &cauchy_stress_d6pV[stress_stride * i];
         energyL[l] = &internal_energyV[int_eng_stride * i];
         sddL[l] = &sddV[sdd_stride * i];
         tL[l] = &tkelvV[tkelv_stride * i];
         if (!real[l]) {
            for (unsigned int k = 0; k < nHist; ++k) { padHist[l][k] = histL[l][k]; }
            for (int k = 0; k < 8 && k < (int)stress_stride; ++k) { padStress[l][k] = stressL[l][k]; }
            for (int k = 0; k < 8 && k < (int)int_eng_stride; ++k) { padEnergy[l][k] = energyL[l][k]; }
            for (int k = 0; k < 8 && k < (int)sdd_stride; ++k) { padSdd[l][k] = sddL[l][k]; }
            padT[l] = *tL[l];
            histL[l] = padHist[l]; stressL[l] = padStress[l]; energyL[l] = padEnergy[l]; sddL[l] = padSdd[l]; tL[l] = &padT[l];
         }
         ps[l] = new (psMem[l]) PS(histL[l], stressL[l], *tL[l], &def_rate_d6vV[def_rate_stride * i],
                                   &spin_vecV[spin_v_stride * i], &rel_vol_ratiosV[vol_ratio_stride * i], dt);
         if constexpr (dynamicGeom) { geom[l] = new (geomMem[l]) SlipGeom(slipGeom); }
         else { geom[l] = &slipGeom; }
         ok[l] = evptn::preprocess<SlipGeom, Kinetics, EosModel, ThermoElastN, PS, false, hardeningOnPacks>
                    (*geom[l], kinetics, eos, elastN, &rel_vol_ratiosV[vol_ratio_stride * i],
                     energyL[l], &def_rate_d6vV[def_rate_stride * i], *ps[l], halfVMidDt[l], devE[l]);
      }
      int lref = 0;
      while (lref < W - 1 && !ok[lref]) { ++lref; }

      // hardness update on the pack (KineticsOrowanD's reduced solve)
      if constexpr (hardeningOnPacks) {
         vdouble hs_o[nH], gdot_o[nslip], hs_u[nH];
         for (int l = 0; l < W; ++l) {
            const PS& p = *ps[ok[l] ? l : lref];
            for (int k = 0; k < nH; ++k) { hs_o[k][l] = p.h_state[k]; }
            for (int k = 0; k < nslip; ++k) { gdot_o[k][l] = p.gdot[k]; }
         }
         const vmask hok = kinetics.updateHLanes(hs_u, hs_o, dt, gdot_o);
         for (int l = 0; l < W; ++l) {
            if (!ok[l]) { continue; }
            if (!hok[l]) { ok[l] = false; continue; } // as preprocess: "Hardening failed to converge"
            for (int k = 0; k < nH; ++k) { ps[l]->h_state_u[k] = hs_u[k][l]; }
         }
      }

      // gather the pack problem state (failed points ride along on a valid point's data)
      PSV pv;
      pv.dt = dt;
      vmask active0 = snls::all_lanes<vdouble>();
      for (int l = 0; l < W; ++l) {
         if (!ok[l]) { active0[l] = false; }
         const PS& p = *ps[ok[l] ? l : lref];
         pv.rel_vol_new[l] = p.rel_vol_new;
         pv.energy_new[l] = p.energy_new;
         pv.pressure_EOS[l] = p.pressure_EOS;
         pv.tkelv[l] = p.tkelv;
         for (int k = 0; k < ecmech::ntvec; ++k) { pv.elast_d5_n[k][l] = p.elast_d5_n[k]; pv.def_rate_d5_sample[k][l] = p.def_rate_d5_sample[k]; }
         for (int k = 0; k < ecmech::qdim; ++k) { pv.quat_n[k][l] = p.quat_n[k]; }
         for (int k = 0; k < ecmech::nwvec; ++k) { pv.spin_vec_sample[k][l] = p.spin_vec_sample[k]; }
         for (int k = 0; k < nH; ++k) { pv.h_state_u[k][l] = p.h_state_u[k]; }
      }

      // the slip geometry of the pack
      const GeomV* geomV = nullptr;
      SlipGeomPack<SlipGeom> geomPack;
      if constexpr (dynamicGeom) {
         for (int l = 0; l < W; ++l) { geomPack.setLane(l, *geom[ok[l] ? l : lref]); }
         geomV = &geomPack;
      }
      else { geomV = &slipGeom; }

      // the Newton solve on the pack (main_problem's settings)
      Prob prob(*geomV, kinetics, elastN, pv);
      Solver solver(prob);
      solver.setupSolver(200, tolerance, &deltaControl);
      for (int k = 0; k < n; ++k) { solver._x[k] = 0.0; }
      solver.solve(active0);
      if (statsOn) {
         long sum = 0, mx = 0;
         for (int l = 0; l < W; ++l) { if (real[l] && ok[l]) { sum += solver.fevals[l]; mx = std::max<long>(mx, solver.fevals[l]); } }
         #pragma omp atomic
         statUseful += sum;
         #pragma omp atomic
         statIssued += mx * W;
      }

      // tangent (computeTangentStiffness's evaluation), state update and the
      // values postprocessing needs, on the pack
      vdouble mtan_vecds[ecmech::nsvec2];
      if (mtanSDV) {
         for (int k = 0; k < ecmech::nsvec2; ++k) { mtan_vecds[k] = 0.0; }
         prob.provideMTan(mtan_vecds);
         vdouble r[n], J[n * n];
         prob.computeRJ(r, J, solver._x);
         prob.clearMTan();
      }
      vdouble elast_d5_u[ecmech::ntvec], quat_u[ecmech::qdim];
      prob.stateFromX(elast_d5_u, quat_u, solver._x);
      vdouble pl_disipation_rate, effective_shear_rate, gdot[nslip];
      prob.get_slip_contribution(pl_disipation_rate, effective_shear_rate, gdot, elast_d5_u);
      vdouble cauchy_xtal[ecmech::nsvec];
      prob.elastNEtoC(cauchy_xtal, elast_d5_u);
      const vdouble hdnScale = prob.getHdnScale();

      // per point: computeTangentStiffness's scaling, postprocess_prob's
      // bookkeeping, postprocess (original code)
      for (int l = 0; l < W; ++l) {
         if (!real[l]) { ps[l]->~PS(); continue; }
         if (!ok[l] || !solver.converged[l]) {
            histL[l][evptn::iHistA_nFEval] *= -1; // as getResponseSolo marks a failure
            ps[l]->~PS();
            continue;
         }
         const int i = idx[l];
         PS& p = *ps[l];
         if (mtanSDV) {
            double m6[ecmech::nsvec2];
            for (int k = 0; k < ecmech::nsvec2; ++k) { m6[k] = mtan_vecds[k][l] * prob.getDtRi(); }
            m6[ECMECH_NN_INDX(iSvecS, iSvecS, ecmech::nsvec)] += three * p.bulk_modulus_new;
            mtan_conv_sd_svec<true>(&mtanSDV[ecmech::nsvec2 * i], m6);
         }
         for (int k = 0; k < ecmech::ntvec; ++k) { p.elast_d5_u[k] = elast_d5_u[k][l]; }
         for (int k = 0; k < ecmech::qdim; ++k) { p.quat_u[k] = quat_u[k][l]; }
         histL[l][evptn::iHistA_nFEval] = solver.fevals[l] + (mtanSDV ? 1 : 0);
         for (int k = 0; k < nH; ++k) { p.h_state[k] = p.h_state_u[k]; }
         for (int k = 0; k < nslip; ++k) { p.gdot[k] = gdot[k][l]; }
         p.eps_dot = effective_shear_rate[l];
         p.eps += p.eps_dot * p.dt;
         {
            const double dEff = vecd_Deff(p.def_rate_d5_sample);
            double flow_strength = hdnScale[l];
            if (dEff > idp_tiny_sqrt) { flow_strength = pl_disipation_rate[l] / dEff; }
            p.flow_strength = flow_strength;
         }
         double cauchy_stress_d5p_xtal[ecmech::nsvec];
         for (int k = 0; k < ecmech::nsvec; ++k) { cauchy_stress_d5p_xtal[k] = cauchy_xtal[k][l]; }
         evptn::postprocess(p, elastN, &def_rate_d6vV[def_rate_stride * i], sddL[l], energyL[l],
                            cauchy_stress_d5p_xtal, devE[l], halfVMidDt[l]);
         p.~PS();
      }
      if constexpr (dynamicGeom) { for (int l = 0; l < W; ++l) { geom[l]->~SlipGeom(); } }
   }
   if (statsOn && (++statCalls % 60 == 0)) {
      std::printf("SIMDSTATS lanes %d main-solve lane utilization %.3f (useful %ld / issued %ld)\n",
                  W, double(statUseful) / double(statIssued), statUseful, statIssued);
   }
}

} // namespace simd
} // namespace ecmech

#endif // ECMECH_SIMD_EXPT
