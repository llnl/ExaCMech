#pragma once
/**
 * @file ECMech_evptnLanes.h
 * @brief The one getResponse driver: getResponseLanes<R> for one point
 * (R = double) or a pack of points, and getResponseSngl on top of it.
 */

#include <new>
#include <type_traits>
#include "SNLS_TrDLDenseG.h"

namespace ecmech {
namespace evptn {

/// The fields of evptn::ProblemState that EvptnUpdstProblem reads, for a
/// pack of points (value type R)
template<class Kinetics, class R>
struct ProblemStatePack {
   using value_type = R;
   double dt;
   R rel_vol_new, energy_new, pressure_EOS, tkelv;
   R elast_d5_n[ecmech::ntvec];
   R quat_n[ecmech::qdim];
   R def_rate_d5_sample[ecmech::ntvec];
   R spin_vec_sample[ecmech::nwvec];
   R h_state_u[(Kinetics::nH > 0) ? Kinetics::nH : 1];
};

template<class SG, class = void> struct has_rss_from_stress : std::false_type {};
template<class SG> struct has_rss_from_stress<SG, std::void_t<decltype(SG::rssFromStress)>> : std::true_type {};

/// A dynamic (stress-dependent) slip geometry for a pack of points: each
/// point's P, Q and extra values (as its own geometry computed them in
/// preprocess) gathered into packs, behind the interface the templated
/// problem uses (nslip, nSlipExtra, dynamic, getP, getQ, evalRSS,
/// getExtras).
template<class SG, class R>
struct SlipGeomPack {
   static constexpr int nslip = SG::nslip;
   static constexpr int nSlipExtra = SG::nSlipExtra;
   static constexpr bool dynamic = SG::dynamic;
   const SG* m_g0 = nullptr; ///< one point's geometry (for projections that depend on the stress only)
   R m_P[ecmech::ntvec * nslip];
   R m_Q[ecmech::nwvec * nslip];
   R m_extras[(nSlipExtra > 0) ? nSlipExtra : 1];

   /// gather point l's geometry
   __ecmech_hdev__ void setLane(int l, const SG& g) {
      if (!m_g0) { m_g0 = &g; }
      const double* const P = g.getP();
      const double* const Q = g.getQ();
      for (int k = 0; k < ecmech::ntvec * nslip; ++k) { set_lane(m_P[k], l, P[k]); }
      for (int k = 0; k < ecmech::nwvec * nslip; ++k) { set_lane(m_Q[k], l, Q[k]); }
      if constexpr (nSlipExtra > 0) {
         double e[nSlipExtra];
         g.getExtras(e);
         for (int k = 0; k < nSlipExtra; ++k) { set_lane(m_extras[k], l, e[k]); }
      }
   }
   __ecmech_hdev__ const R* getP() const { return m_P; }
   __ecmech_hdev__ const R* getQ() const { return m_Q; }
   template<class T>
   __ecmech_hdev__ void evalRSS(R* taua, const T* const kirchoff, const R* P_vec) const {
      if constexpr (has_rss_from_stress<SG>::value) {
         m_g0->evalRSS(taua, kirchoff, (const double*)nullptr);
      }
      else {
         vecsVaTM<ecmech::ntvec, nslip>(taua, kirchoff, P_vec);
      }
   }
   __ecmech_hdev__ void getExtras(R* const e) const {
      for (int k = 0; k < nSlipExtra; ++k) { e[k] = m_extras[k]; }
   }
};

/// kinetics whose slip rates are templated on the value type
template<class K, class R, class = void> struct has_pack_slip_rates : std::false_type {};
template<class K, class R>
struct has_pack_slip_rates<K, R, std::void_t<decltype(std::declval<const K&>().evalGdots(
   (R*)nullptr, (R*)nullptr, (const R*)nullptr, (const R*)nullptr))>> : std::true_type {};
/// kinetics whose hardness update can run on packs
template<class K, class R, class = void> struct has_pack_hardening : std::false_type {};
template<class K, class R>
struct has_pack_hardening<K, R, std::void_t<decltype(std::declval<const K&>().updateHLanes(
   (R*)nullptr, (const R*)nullptr, 0.0, (const R*)nullptr))>> : std::true_type {};

/// Models the pack path supports for value type R (others use the scalar path)
template<class SlipGeom, class Kinetics, class ThermoElastN, class R>
struct pack_supported {
   static constexpr bool value = has_pack_slip_rates<Kinetics, R>::value;
};

/// The point-wise arrays and strides of one getResponseECM call
struct PackArrays {
   double dt, tolerance;
   const double* def_rate_d6vV; unsigned int def_rate_stride;
   const double* spin_vecV;     unsigned int spin_v_stride;
   const double* rel_vol_ratiosV; unsigned int vol_ratio_stride;
   double* internal_energyV;    unsigned int int_eng_stride;
   double* cauchy_stress_d6pV;  unsigned int stress_stride;
   double* histV;               unsigned int history_stride;
   double* tkelvV;              unsigned int tkelv_stride;
   double* sddV;                unsigned int sdd_stride;
   double* mtanSDV;
   int nPassed;
};
/// longest history the padding scratch holds (checked before the launch)
constexpr int maxPackHist = 128;

/**
 * @brief getResponse for points ip*W .. ip*W+W-1, W = lanes of the value
 * type R: one point for R = double (the scalar CPU path and one GPU thread
 * per point), W points in lockstep for a pack (the CPU SIMD path).
 *
 * The steps are those of the original getResponseSngl: per point,
 * ProblemState and preprocess (EOS, hardness update); then the Newton solve
 * of EvptnUpdstProblem with SNLS's trust-region dogleg on R; the tangent;
 * the state update; per point, postprocess. For a pack the per-point inputs
 * are gathered into a pack problem state and the results scattered back;
 * for R = double the problem works on the point's own ProblemState and the
 * gather/scatter steps compile away (`if constexpr`). A pack whose last
 * lanes run past nPassed repeats the last point on scratch copies.
 *
 * Returns the points that converged (for R = double: a bool). The caller
 * marks failures (hist[iHistA_nFEval] *= -1), as for getResponseSngl.
 * fevalsOut (optional): the evaluations per lane and the most of any lane.
 */
template<class R, class SlipGeom, class Kinetics, class ThermoElastN, class EosModel>
__ecmech_hdev__
inline mask_t<R> getResponseLanes(const int ip, const SlipGeom& slipGeom, const Kinetics& kinetics,
                                  const ThermoElastN& elastN, const EosModel& eos, const PackArrays& a,
                                  int* const fevalsOut = nullptr)
{
   constexpr int W = lanes_of<R>();
   constexpr bool one = snls::is_scalar_value<R>();
   using PS = ProblemState<SlipGeom, Kinetics, ThermoElastN, EosModel>;
   using PSV = std::conditional_t<one, PS, ProblemStatePack<Kinetics, R>>;
   constexpr bool dynamicGeom = SlipGeom::dynamic;
   using GeomV = std::conditional_t<dynamicGeom && !one, SlipGeomPack<SlipGeom, R>, SlipGeom>;
   using Prob = EvptnUpdstProblem<GeomV, Kinetics, ThermoElastN, PSV>;
   using Solver = snls::SNLSTrDlDenseG<Prob, Prob::nDimSys, R>;
   constexpr bool hardeningOnPacks = !one && has_pack_hardening<Kinetics, R>::value;
   constexpr int nH = Kinetics::nH;
   constexpr int nslip = SlipGeom::nslip;
   constexpr int n = Prob::nDimSys;
   constexpr int WP = one ? 1 : W; // scratch for padding lanes (none for one point)
   const double dt = a.dt;

   // point index per lane; lanes past the end repeat the last point on copies
   int idx[W];
   bool real[W];
   for (int l = 0; l < W; ++l) {
      const int i = ip * W + l;
      real[l] = one || (i < a.nPassed);
      idx[l] = real[l] ? i : (a.nPassed - 1);
   }
   double* histL[W]; double* stressL[W]; double* energyL[W]; double* sddL[W]; double* tL[W];
   [[maybe_unused]] double padHist[WP][one ? 1 : maxPackHist], padStress[WP][8], padEnergy[WP][8], padSdd[WP][8], padT[WP];
   alignas(PS) unsigned char psMem[W][sizeof(PS)];
   PS* ps[W];
   alignas(SlipGeom) unsigned char geomMem[(dynamicGeom && !one) ? W : 1][(dynamicGeom && !one) ? sizeof(SlipGeom) : 1];
   const SlipGeom* geom[W];
   bool ok[W];
   double halfVMidDt[W], devE[W];

   // per point: problem state and preprocess
   for (int l = 0; l < W; ++l) {
      const int i = idx[l];
      histL[l] = &a.histV[a.history_stride * i];
      stressL[l] = &a.cauchy_stress_d6pV[a.stress_stride * i];
      energyL[l] = &a.internal_energyV[a.int_eng_stride * i];
      sddL[l] = &a.sddV[a.sdd_stride * i];
      tL[l] = &a.tkelvV[a.tkelv_stride * i];
      if constexpr (!one) {
         if (!real[l]) {
            const unsigned int nHist = a.history_stride;
            for (unsigned int k = 0; k < nHist; ++k) { padHist[l][k] = histL[l][k]; }
            for (int k = 0; k < 8 && k < (int)a.stress_stride; ++k) { padStress[l][k] = stressL[l][k]; }
            for (int k = 0; k < 8 && k < (int)a.int_eng_stride; ++k) { padEnergy[l][k] = energyL[l][k]; }
            for (int k = 0; k < 8 && k < (int)a.sdd_stride; ++k) { padSdd[l][k] = sddL[l][k]; }
            padT[l] = *tL[l];
            histL[l] = padHist[l]; stressL[l] = padStress[l]; energyL[l] = padEnergy[l]; sddL[l] = padSdd[l]; tL[l] = &padT[l];
         }
      }
      ps[l] = new (psMem[l]) PS(histL[l], stressL[l], *tL[l], &a.def_rate_d6vV[a.def_rate_stride * i],
                                &a.spin_vecV[a.spin_v_stride * i], &a.rel_vol_ratiosV[a.vol_ratio_stride * i], dt);
      // dynamic slip geometries: a copy per lane of a pack; one point uses the
      // caller's geometry, as getResponseSngl always did
      if constexpr (dynamicGeom && !one) { geom[l] = new (geomMem[l]) SlipGeom(slipGeom); }
      else { geom[l] = &slipGeom; }
      ok[l] = preprocess<SlipGeom, Kinetics, EosModel, ThermoElastN, PS, false, hardeningOnPacks>
                 (*geom[l], kinetics, eos, elastN, &a.rel_vol_ratiosV[a.vol_ratio_stride * i],
                  energyL[l], &a.def_rate_d6vV[a.def_rate_stride * i], *ps[l], halfVMidDt[l], devE[l]);
   }
   int lref = 0;
   while (lref < W - 1 && !ok[lref]) { ++lref; }

   mask_t<R> converged = no_lanes<R>();
   if constexpr (one) {
      if (!ok[0]) { ps[0]->~PS(); return converged; }
   }

   // hardness update on the pack (KineticsOrowanD's reduced solve)
   if constexpr (hardeningOnPacks) {
      R hs_o[nH], gdot_o[nslip], hs_u[nH];
      for (int l = 0; l < W; ++l) {
         const PS& p = *ps[ok[l] ? l : lref];
         for (int k = 0; k < nH; ++k) { set_lane(hs_o[k], l, p.h_state[k]); }
         for (int k = 0; k < nslip; ++k) { set_lane(gdot_o[k], l, p.gdot[k]); }
      }
      const mask_t<R> hok = kinetics.updateHLanes(hs_u, hs_o, dt, gdot_o);
      for (int l = 0; l < W; ++l) {
         if (!ok[l]) { continue; }
         if (!mask_lane(hok, l)) { ok[l] = false; continue; } // as preprocess: "Hardening failed to converge"
         for (int k = 0; k < nH; ++k) { ps[l]->h_state_u[k] = lane(hs_u[k], l); }
      }
   }

   // the problem state of the pack: a point's own state, or the lanes gathered
   // (failed points ride along on a valid point's data)
   mask_t<R> active0 = all_lanes<R>();
   [[maybe_unused]] ProblemStatePack<Kinetics, R> pvMem;
   PSV* pv = nullptr;
   if constexpr (one) { pv = ps[0]; }
   else {
      pv = &pvMem;
      pv->dt = dt;
      for (int l = 0; l < W; ++l) {
         if (!ok[l]) { set_mask_lane(active0, l, false); }
         const PS& p = *ps[ok[l] ? l : lref];
         set_lane(pv->rel_vol_new, l, p.rel_vol_new);
         set_lane(pv->energy_new, l, p.energy_new);
         set_lane(pv->pressure_EOS, l, p.pressure_EOS);
         set_lane(pv->tkelv, l, p.tkelv);
         for (int k = 0; k < ecmech::ntvec; ++k) { set_lane(pv->elast_d5_n[k], l, p.elast_d5_n[k]); set_lane(pv->def_rate_d5_sample[k], l, p.def_rate_d5_sample[k]); }
         for (int k = 0; k < ecmech::qdim; ++k) { set_lane(pv->quat_n[k], l, p.quat_n[k]); }
         for (int k = 0; k < ecmech::nwvec; ++k) { set_lane(pv->spin_vec_sample[k], l, p.spin_vec_sample[k]); }
         for (int k = 0; k < nH; ++k) { set_lane(pv->h_state_u[k], l, p.h_state_u[k]); }
      }
   }

   // the slip geometry of the pack
   const GeomV* geomV = nullptr;
   [[maybe_unused]] SlipGeomPack<SlipGeom, R> geomPack;
   if constexpr (dynamicGeom && !one) {
      for (int l = 0; l < W; ++l) { geomPack.setLane(l, *geom[ok[l] ? l : lref]); }
      geomV = &geomPack;
   }
   else { geomV = geom[0]; }

   // the Newton solve (main_problem's settings)
   snls::TrDeltaControl deltaControl;
   deltaControl._deltaInit = 1e0;
   Prob prob(*geomV, kinetics, elastN, *pv);
   Solver solver(prob);
   solver.setupSolver(200, a.tolerance, &deltaControl);
   for (int k = 0; k < n; ++k) { solver._x[k] = 0.0; }
   solver.solve(active0);
   converged = solver.getConverged();
   if (fevalsOut) {
      int mx = 0;
      for (int l = 0; l < W; ++l) {
         const int f = (real[l] && ok[l]) ? solver.getNFEvals(l) : 0;
         fevalsOut[l] = f;
         mx = (f > mx) ? f : mx;
      }
      fevalsOut[W] = mx;
   }

   // tangent (computeTangentStiffness's evaluation), state update and the
   // values postprocessing needs
   R mtan_vecds[ecmech::nsvec2];
   if (a.mtanSDV) {
      for (int k = 0; k < ecmech::nsvec2; ++k) { mtan_vecds[k] = 0.0; }
      prob.provideMTan(mtan_vecds);
      R r[n], J[n * n];
      prob.computeRJ(r, J, solver._x);
      prob.clearMTan();
   }
   R elast_d5_u[ecmech::ntvec], quat_u[ecmech::qdim];
   prob.stateFromX(elast_d5_u, quat_u, solver._x);
   R pl_disipation_rate, effective_shear_rate, gdot[nslip];
   prob.get_slip_contribution(pl_disipation_rate, effective_shear_rate, gdot, elast_d5_u);
   R cauchy_xtal[ecmech::nsvec];
   prob.elastNEtoC(cauchy_xtal, elast_d5_u);
   const R hdnScale = prob.getHdnScale();

   // per point: computeTangentStiffness's scaling, postprocess_prob's
   // bookkeeping, postprocess
   for (int l = 0; l < W; ++l) {
      if (!real[l] || !ok[l] || !mask_lane(converged, l)) {
         if (real[l] && ok[l]) { set_mask_lane(converged, l, false); }
         ps[l]->~PS();
         continue;
      }
      const int i = idx[l];
      PS& p = *ps[l];
      if (a.mtanSDV) {
         double m6[ecmech::nsvec2];
         for (int k = 0; k < ecmech::nsvec2; ++k) { m6[k] = lane(mtan_vecds[k], l) * prob.getDtRi(); }
         m6[ECMECH_NN_INDX(iSvecS, iSvecS, ecmech::nsvec)] += three * p.bulk_modulus_new;
         mtan_conv_sd_svec<true>(&a.mtanSDV[ecmech::nsvec2 * i], m6);
      }
      for (int k = 0; k < ecmech::ntvec; ++k) { p.elast_d5_u[k] = lane(elast_d5_u[k], l); }
      for (int k = 0; k < ecmech::qdim; ++k) { p.quat_u[k] = lane(quat_u[k], l); }
      histL[l][iHistA_nFEval] = solver.getNFEvals(l) + (a.mtanSDV ? 1 : 0);
      for (int k = 0; k < nH; ++k) { p.h_state[k] = p.h_state_u[k]; }
      for (int k = 0; k < nslip; ++k) { p.gdot[k] = lane(gdot[k], l); }
      p.eps_dot = lane(effective_shear_rate, l);
      p.eps += p.eps_dot * p.dt;
      {
         const double dEff = vecd_Deff(p.def_rate_d5_sample);
         double flow_strength = lane(hdnScale, l);
         if (dEff > idp_tiny_sqrt) { flow_strength = lane(pl_disipation_rate, l) / dEff; }
         p.flow_strength = flow_strength;
      }
      double cauchy_stress_d5p_xtal[ecmech::nsvec];
      for (int k = 0; k < ecmech::nsvec; ++k) { cauchy_stress_d5p_xtal[k] = lane(cauchy_xtal[k], l); }
      postprocess(p, elastN, &a.def_rate_d6vV[a.def_rate_stride * i], sddL[l], energyL[l],
                  cauchy_stress_d5p_xtal, devE[l], halfVMidDt[l]);
      p.~PS();
   }
   if constexpr (dynamicGeom && !one) { for (int l = 0; l < W; ++l) { geom[l]->~SlipGeom(); } }
   return converged;
}

/**
 * @brief Complete time step for one material point (see the documentation
 * above the declaration in ECMech_evptnSngl.h): getResponseLanes with
 * R = double on this point's arrays.
 */
template<class SlipGeom, class Kinetics, class ThermoElastN, class EosModel>
__ecmech_hdev__
inline
bool getResponseSngl(const SlipGeom& slipGeom,
                     const Kinetics& kinetics,
                     const ThermoElastN& elastN,
                     const EosModel& eos,
                     const double dt,
                     const double tolerance,
                     const double* const def_rate_d6v_sample,
                     const double* const spin_vec_sample,
                     const double* const rel_vol_ratios,
                     double* const internal_energy,
                     double* const cauchy_stress_d6p,
                     double* const hist,
                     double& tkelv,
                     double* const sdd,
                     double* const mtanSD,
                     int = 0)
{
   const PackArrays a{dt, tolerance, def_rate_d6v_sample, 0, spin_vec_sample, 0, rel_vol_ratios, 0,
                      internal_energy, 0, cauchy_stress_d6p, 0, hist, 0, &tkelv, 0, sdd, 0, mtanSD, 1};
   return getResponseLanes<double>(0, slipGeom, kinetics, elastN, eos, a);
}

} // namespace evptn
} // namespace ecmech
