#pragma once
/**
 * @file ECMech_simd_evptn.h
 * @brief Experiment (ECMECH_SIMD_EXPT, CPU only): the evptn elastic-strain +
 * lattice-rotation Newton solve for a pack of W points at once.
 *
 * Per pack: preprocess (EOS, hardening update) runs per point with the
 * original scalar code; the 8x8 trust-region Newton solve runs on the pack
 * (EvptnUpdstProblemSimd::computeRJ, a masked LU and a masked copy of
 * SNLSTrDlDenseG's dogleg/trust-region logic); the tangent, state update
 * and postprocessing run per point with the original scalar code at the
 * converged x. Lanes that have converged or failed stop changing but still
 * ride along until every lane of the pack is done (SIMT with W lanes).
 */

#if defined(ECMECH_SIMD_EXPT)

#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include "simd/ECMech_simd_core.h"
#include "simd/ECMech_simd_kinetics.h"
#include "evptn/ECMech_evptnSngl.h"
#include "SNLS_TrDelta.h"

namespace ecmech {
namespace simd {

// ---------------------------------------------------------------------
// base functions (evptn/ECMech_base_fcns.h) on packs
// ---------------------------------------------------------------------
template<class R>
inline void get_xtal_frame_vel_grad_terms(R* const def_rate_d5_xtal, R* const spin_vec_xtal,
                                          const R* const def_rate_d5_sample, const R* const spin_vec_sample,
                                          const R* const xtal_rmat, const R* const xtal_rot_mat5)
{
   vecsVMTa<ecmech::ntvec>(def_rate_d5_xtal, xtal_rot_mat5, def_rate_d5_sample);
   vecsVMTa<ecmech::ndim>(spin_vec_xtal, xtal_rmat, spin_vec_sample);
}

template<class R>
inline void elasticity_higher_order_terms(R* const A_e_M35, R* const ee_spin_vec, R& ee_fac,
                                          const R& inv_a_vol, const R* const elast_d5, const R* const elast_dt_d5)
{
   M35_d_AAoB_dA(A_e_M35, elast_d5);
   vecsVMa<nwvec, ntvec>(ee_spin_vec, A_e_M35, elast_dt_d5);
   ee_fac = onehalf * inv_a_vol * inv_a_vol;
}

/// Plastic deformation rate and spin from the slip rates (get_slip_rate_terms)
template<int nslip, class R, class KinS>
inline void get_slip_rate_terms(R* const dgdot_dtau, R* const plastic_def_rate_d5, R* const plastic_spin_vec,
                                const R* const kirchoff, const R* const kinetic_values,
                                const double* const P, const double* const Q, const KinS& slip_kinetics,
                                R* const gdot_out = nullptr, R* const rss_out = nullptr)
{
   R rss[nslip];
   R gdot[nslip];
   vecsVaTM<ecmech::ntvec, nslip>(rss, kirchoff, P); // slip_geom.evalRSS
   slip_kinetics.evalGdots(gdot, dgdot_dtau, rss, kinetic_values);
   vecsVMa<ntvec, nslip>(plastic_def_rate_d5, P, gdot);
   vecsVMa<nwvec, nslip>(plastic_spin_vec, Q, gdot);
   if (gdot_out) { for (int i = 0; i < nslip; ++i) { gdot_out[i] = gdot[i]; } }
   if (rss_out) { for (int i = 0; i < nslip; ++i) { rss_out[i] = rss[i]; } }
}

/// get_slip_rate_deriv_terms, diagonal (cubic) elasticity path
template<int ns, class R>
inline void get_slip_rate_deriv_terms(R* const dDp_hat_delast_strain, R* const dWp_hat_delast_strain,
                                      const R* const dgdot_dtau, const R& inv_a_vol,
                                      const double* const P, const double* const Q, const double* const K_diag)
{
   R d[ecmech::ntvec];
   for (int iT = 0; iT < ecmech::ntvec; ++iT) { d[iT] = K_diag[iT] * inv_a_vol; } // dTdepsDiag
   for (int iT = 0; iT < ecmech::ntvec; ++iT) {
      R pw[ns];
      for (int s = 0; s < ns; ++s) { pw[s] = P[ECMECH_NM_INDX(iT, s, ecmech::ntvec, ns)] * dgdot_dtau[s]; }
      for (int jT = iT; jT < ecmech::ntvec; ++jT) {
         R m = 0.0;
         for (int s = 0; s < ns; ++s) { m += pw[s] * P[ECMECH_NM_INDX(jT, s, ecmech::ntvec, ns)]; }
         dDp_hat_delast_strain[ECMECH_NM_INDX(iT, jT, ecmech::ntvec, ecmech::ntvec)] = d[jT] * m;
         dDp_hat_delast_strain[ECMECH_NM_INDX(jT, iT, ecmech::ntvec, ecmech::ntvec)] = d[iT] * m;
      }
   }
   for (int iW = 0; iW < ecmech::nwvec; ++iW) {
      R qw[ns];
      for (int s = 0; s < ns; ++s) { qw[s] = Q[ECMECH_NM_INDX(iW, s, ecmech::nwvec, ns)] * dgdot_dtau[s]; }
      for (int jT = 0; jT < ecmech::ntvec; ++jT) {
         R m = 0.0;
         for (int s = 0; s < ns; ++s) { m += qw[s] * P[ECMECH_NM_INDX(jT, s, ecmech::ntvec, ns)]; }
         dWp_hat_delast_strain[ECMECH_NM_INDX(iW, jT, ecmech::nwvec, ecmech::ntvec)] = d[jT] * m;
      }
   }
}

// ---------------------------------------------------------------------
// dense LU with partial pivoting on a pack (SNLS_LUP_Solve): every lane
// pivots on its own rows, done with masked row swaps
// ---------------------------------------------------------------------
/// Solves a x_r = b_r for nRHS right-hand sides stored one after the other
/// in xb (n values each, overwritten with the solutions), as SNLS_LUP_SolveX;
/// a is overwritten. Returns the lanes whose pivots were all >= tol.
template<int n, class R>
inline mask_t<R> lup_solveX(R* const a, R* const xb, const int nRHS, const double tol = 1e-50)
{
   mask_t<R> ok = (a[0] == a[0]) || (a[0] != a[0]); // all true
   for (int i = 0; i < n; ++i) {
      // pivot search: first row with the largest |a(k,i)|, k >= i
      R maxa = fabs(a[i * n + i]);
      R imax = i;
      for (int k = i + 1; k < n; ++k) {
         const R absa = fabs(a[k * n + i]);
         const auto bigger = absa > maxa;
         maxa = select(bigger, absa, maxa);
         imax = select(bigger, R(k), imax);
      }
      ok = ok && !(maxa < tol);
      // swap rows i and imax (lane-wise) of a and of the right-hand side
      for (int k = i + 1; k < n; ++k) {
         const auto sw = (imax == k);
         if (!any(sw)) { continue; }
         for (int c = 0; c < n; ++c) {
            const R ti = a[i * n + c];
            const R tk = a[k * n + c];
            a[i * n + c] = select(sw, tk, ti);
            a[k * n + c] = select(sw, ti, tk);
         }
         for (int r = 0; r < nRHS; ++r) {
            R* const y = &xb[r * n];
            const R yi = y[i];
            y[i] = select(sw, y[k], yi);
            y[k] = select(sw, yi, y[k]);
         }
      }
      const R rd = one / select(ok, a[i * n + i], R(one));
      for (int j = i + 1; j < n; ++j) {
         const R l = a[j * n + i] * rd;
         a[j * n + i] = l;
         for (int k = i + 1; k < n; ++k) { a[j * n + k] -= l * a[i * n + k]; }
      }
   }
   // forward (unit L) and back substitution, each right-hand side
   for (int r = 0; r < nRHS; ++r) {
      R* const x = &xb[r * n];
      for (int i = 0; i < n; ++i) {
         R s = x[i];
         for (int k = 0; k < i; ++k) { s -= a[i * n + k] * x[k]; }
         x[i] = s;
      }
      for (int i = n - 1; i >= 0; i--) {
         R s = x[i];
         for (int k = i + 1; k < n; ++k) { s -= a[i * n + k] * x[k]; }
         x[i] = s * (one / select(ok, a[i * n + i], R(one)));
      }
   }
   return ok;
}

/// Solves a x = b (SNLS_LUP_Solve); a is overwritten
template<int n, class R>
inline mask_t<R> lup_solve(R* const a, R* const x, const R* const b, const double tol = 1e-50)
{
   for (int i = 0; i < n; ++i) { x[i] = b[i]; }
   return lup_solveX<n>(a, x, 1, tol);
}

// ---------------------------------------------------------------------
// EvptnUpdstProblem on a pack (cubic elasticity, fixed slip geometry)
// ---------------------------------------------------------------------
template<class SlipGeom, class Kinetics, class R>
class EvptnUpdstProblemSimd
{
   public:
      static constexpr int nDimSys = ecmech::ntvec + ecmech::nwvec;
      static constexpr int nslip = SlipGeom::nslip;
      static constexpr int m_i_sub_e = 0;
      static constexpr int m_i_sub_r = ecmech::ntvec;
      using KinS = SimdKinetics<Kinetics>;

      /// What the tangent and postprocessing need at the converged x
      /// (computeRJ fills it when passed; as provideMTan() + computeRJ() +
      /// get_slip_contribution() + elastNEtoC() + stateFromX() do).
      struct Final {
         bool doTangent;
         R mtan_vecds[ecmech::nsvec2];     ///< before computeTangentStiffness's scaling
         R gdot[nslip];
         R pl_disipation_rate, effective_shear_rate;
         R cauchy_xtal[ecmech::nsvec];
         R elast_d5[ecmech::ntvec];
         R quat[ecmech::qdim];
      };

      /// Shared (per model) data
      const double* m_P;
      const double* m_Q;
      double m_K_diag[ecmech::ntvec];
      const KinS& m_kinetics;
      double m_dt, m_inv_dt;
      /// Per point (one lane each)
      R m_elast_d5_n[ecmech::ntvec];
      R m_xtal_ori_quat_n[ecmech::qdim];
      R m_def_rate_d5_sample[ecmech::ntvec];
      R m_spin_vec_sample[ecmech::nwvec];
      R m_pressure_EOS, m_a_vol, m_inv_a_vol, m_inv_det_v_e;
      R m_kin_vals[Kinetics::nVals];
      R m_epsdot_scale_inv, m_rotincr_scale_inv;

      EvptnUpdstProblemSimd(const double* P, const double* Q, const double* K_diag, const KinS& kin, double dt)
         : m_P(P), m_Q(Q), m_kinetics(kin), m_dt(dt), m_inv_dt(1.0 / dt)
      {
         for (int i = 0; i < ecmech::ntvec; ++i) { m_K_diag[i] = K_diag[i]; }
      }

      /// elast_strain_to_kirchoff_stress for ThermoElastNCubic
      void kirchoff_stress(R* const kirchoff, const R* const elast_d5) const
      {
         R elast_d5v[ecmech::nsvec];
         vecsVxa<ntvec>(elast_d5v, m_inv_a_vol, elast_d5);
         elast_d5v[iSvecS] = sqr3 * log(m_a_vol);
         const R ln_J = sqr3 * elast_d5v[iSvecS];
         const R J = exp(ln_J);
         const R kirchoff_pressure = -sqr3 * J * m_pressure_EOS;
         vecsVAdiagB<ntvec>(kirchoff, m_K_diag, elast_d5v);
         kirchoff[iSvecS] = kirchoff_pressure;
      }

      /// EvptnUpdstProblem::computeRJ without the tangent and the kinetics cache
      void computeRJ(R* const resid, R* const Jacobian, const R* const x, Final* const fin = nullptr) const
      {
         for (int ijJ = 0; ijJ < nDimSys * nDimSys; ++ijJ) { Jacobian[ijJ] = 0.0; }
         for (int iR = 0; iR < nDimSys; ++iR) { resid[iR] = 0.0; }

         R elast_dt_d5[ecmech::ntvec];
         vecsVxa<ntvec>(elast_dt_d5, ecmech::e_scale, &(x[m_i_sub_e]));
         R elast_d5_f[ntvec];
         vecsVapb<ntvec>(elast_d5_f, elast_dt_d5, m_elast_d5_n);
         vecsVsa<ntvec>(elast_dt_d5, m_inv_dt);

         R xi_f[nwvec];
         vecsVxa<nwvec>(xi_f, ecmech::r_scale, &(x[m_i_sub_r]));
         R A_quat[ecmech::qdim];
         emap_to_quat(A_quat, xi_f);
         R xtal_ori_quat[ecmech::qdim];
         get_c_quat(xtal_ori_quat, A_quat, m_xtal_ori_quat_n);
         R xtal_rmat[ecmech::ndim * ecmech::ndim];
         quat_to_tensor(xtal_rmat, xtal_ori_quat);
         R rmat_5x5_sample2xtal[ecmech::ntvec * ecmech::ntvec];
         get_rot_mat_vecd(rmat_5x5_sample2xtal, xtal_rmat);
         R def_rate_d5_xtal[ecmech::ntvec];
         R spin_vec_xtal[ecmech::nwvec];
         get_xtal_frame_vel_grad_terms(def_rate_d5_xtal, spin_vec_xtal, m_def_rate_d5_sample, m_spin_vec_sample,
                                       xtal_rmat, rmat_5x5_sample2xtal);

         R kirchoff[ecmech::nsvec];
         kirchoff_stress(kirchoff, elast_d5_f);

         R dgdot_dtau[nslip];
         R plastic_def_rate_d5[ecmech::ntvec];
         R plastic_spin_vec[ecmech::nwvec];
         R rss[nslip];
         get_slip_rate_terms<nslip>(dgdot_dtau, plastic_def_rate_d5, plastic_spin_vec, kirchoff, m_kin_vals,
                                    m_P, m_Q, m_kinetics, fin ? fin->gdot : nullptr, fin ? rss : nullptr);

         R A_e_M35[ecmech::nwvec * ecmech::ntvec];
         R ee_wvec[ecmech::nwvec];
         R ee_fac;
         elasticity_higher_order_terms(A_e_M35, ee_wvec, ee_fac, m_inv_a_vol, elast_d5_f, elast_dt_d5);

         // residuals (get_elast_strain_residual, get_omega_residual)
         for (int iT = 0; iT < ecmech::ntvec; ++iT) {
            resid[iT] = m_epsdot_scale_inv * (m_inv_a_vol * elast_dt_d5[iT] + plastic_def_rate_d5[iT] - def_rate_d5_xtal[iT]);
         }
         for (int iW = 0; iW < ecmech::nwvec; ++iW) {
            resid[m_i_sub_r + iW] = m_rotincr_scale_inv *
               (xi_f[iW] - m_dt * (spin_vec_xtal[iW] - plastic_spin_vec[iW] + ee_fac * ee_wvec[iW]));
         }

         // Jacobian
         R dpl_deps_symm[ ecmech::ntvec * ecmech::ntvec ];
         R dpl_deps_skew[ ecmech::nwvec * ecmech::ntvec ];
         get_slip_rate_deriv_terms<nslip>(dpl_deps_symm, dpl_deps_skew, dgdot_dtau, m_inv_a_vol, m_P, m_Q, m_K_diag);

         R dxtal_ori_quat_dxi_T[ ecmech::nwvec * ecmech::qdim ];
         R dDsm_dxi[ ecmech::ntvec * ecmech::nwvec ];
         R dWsm_dxi[ ecmech::nwvec * ecmech::nwvec ];
         eval_d_dxi_impl_quat(dxtal_ori_quat_dxi_T, dDsm_dxi, dWsm_dxi,
                              m_def_rate_d5_sample, m_spin_vec_sample, xi_f,
                              m_xtal_ori_quat_n, xtal_rmat, xtal_ori_quat);

         auto J = [&](int i, int j) -> R& { return Jacobian[ECMECH_NN_INDX(i, j, nDimSys)]; };
         // d(B_S)/d(elast_d5_f): get_deriv_elast_strain_wrt_elast_strain
         for (int jT = 0; jT < ecmech::ntvec; ++jT) {
            for (int iT = 0; iT < ecmech::ntvec; ++iT) {
               J(iT, jT) = dpl_deps_symm[ECMECH_NN_INDX(iT, jT, ecmech::ntvec)];
            }
         }
         {
            const R adti = m_inv_a_vol * m_inv_dt;
            for (int iT = 0; iT < ecmech::ntvec; ++iT) { J(iT, iT) += adti; }
         }
         // d(B_S)/d(xi_f): get_deriv_elast_strain_wrt_omega
         for (int jW = 0; jW < ecmech::nwvec; ++jW) {
            for (int iT = 0; iT < ecmech::ntvec; ++iT) {
               J(iT, jW + m_i_sub_r) = -dDsm_dxi[ECMECH_NM_INDX(iT, jW, ecmech::ntvec, ecmech::nwvec)];
            }
         }
         // d(B_xi)/d(elast_d5_f): get_deriv_omega_wrt_elast_strain
         {
            R A_edot_M35[ecmech::nwvec * ecmech::ntvec];
            M35_d_AAoB_dA(A_edot_M35, elast_dt_d5);
            const R dt_ee_fac = m_dt * ee_fac;
            for (int iW = 0; iW < ecmech::nwvec; ++iW) {
               for (int jT = 0; jT < ecmech::ntvec; ++jT) {
                  const int ijWT = ECMECH_NM_INDX(iW, jT, ecmech::nwvec, ecmech::ntvec);
                  J(iW + m_i_sub_r, jT) = m_dt * dpl_deps_skew[ijWT] - dt_ee_fac * (A_e_M35[ijWT] * m_inv_dt - A_edot_M35[ijWT]);
               }
            }
         }
         // d(B_xi)/d(xi_f): get_deriv_omega_wrt_omega
         for (int iW = 0; iW < ecmech::nwvec; ++iW) {
            for (int jW = 0; jW < ecmech::nwvec; ++jW) {
               J(iW + m_i_sub_r, jW + m_i_sub_r) = -m_dt * dWsm_dxi[ECMECH_NN_INDX(iW, jW, ecmech::nwvec)];
            }
            J(iW + m_i_sub_r, iW + m_i_sub_r) += one;
         }
         if (fin) { finalOutputs(*fin, Jacobian, kirchoff, elast_d5_f, elast_dt_d5, dgdot_dtau, rss,
                                 plastic_def_rate_d5, dxtal_ori_quat_dxi_T, rmat_5x5_sample2xtal, xtal_ori_quat, xtal_rmat); }
         // scaling
         for (int iJ = 0; iJ < m_i_sub_r; ++iJ) {
            R s = m_epsdot_scale_inv * ecmech::e_scale;
            for (int jJ = 0; jJ < m_i_sub_r; ++jJ) { J(iJ, jJ) *= s; }
            s = m_epsdot_scale_inv * ecmech::r_scale;
            for (int jJ = m_i_sub_r; jJ < nDimSys; ++jJ) { J(iJ, jJ) *= s; }
         }
         for (int iJ = m_i_sub_r; iJ < nDimSys; ++iJ) {
            R s = m_rotincr_scale_inv * ecmech::e_scale;
            for (int jJ = 0; jJ < m_i_sub_r; ++jJ) { J(iJ, jJ) *= s; }
            s = m_rotincr_scale_inv * ecmech::r_scale;
            for (int jJ = m_i_sub_r; jJ < nDimSys; ++jJ) { J(iJ, jJ) *= s; }
         }
      }

      /// The converged-state outputs (see Final); the tangent part is
      /// get_material_tangent_stiffness() for cubic elasticity, on the
      /// unscaled Jacobian (which it overwrites, as the original does).
      void finalOutputs(Final& f, R* const Jacobian, const R* const kirchoff, const R* const elast_d5_f,
                        const R* const elast_dt_d5, const R* const dgdot_dtau, const R* const rss,
                        const R* const plastic_def_rate_d5, const R* const dquat_domega_t,
                        const R* const rmat_5x5_sample2xtal, const R* const quat, const R* const rmat) const;
};

template<class SlipGeom, class Kinetics, class R>
void EvptnUpdstProblemSimd<SlipGeom, Kinetics, R>::finalOutputs(
   Final& f, R* const Jacobian, const R* const kirchoff, const R* const elast_d5_f,
   const R* const elast_dt_d5, const R* const dgdot_dtau, const R* const rss,
   const R* const plastic_def_rate_d5, const R* const dquat_domega_t,
   const R* const rmat_5x5_sample2xtal, const R* const quat, const R* const rmat) const
{
   // stress and state at x (elastNEtoC, stateFromX)
   R cauchy_stress[ecmech::nsvec];
   for (int iS = 0; iS < ecmech::nsvec; ++iS) { cauchy_stress[iS] = m_inv_det_v_e * kirchoff[iS]; } // getCauchy
   for (int iS = 0; iS < ecmech::nsvec; ++iS) { f.cauchy_xtal[iS] = cauchy_stress[iS]; }
   for (int iT = 0; iT < ecmech::ntvec; ++iT) { f.elast_d5[iT] = elast_d5_f[iT]; }
   for (int iQ = 0; iQ < ecmech::qdim; ++iQ) { f.quat[iQ] = quat[iQ]; }
   // get_slip_contributions
   {
#if defined(ECMECH_USE_DPEFF)
      f.effective_shear_rate = sqr2b3 * vecNorm<ecmech::ntvec>(plastic_def_rate_d5);
#else
      R sa = 0.0;
      for (int i = 0; i < nslip; ++i) { sa += fabs(f.gdot[i]); }
      f.effective_shear_rate = sa;
#endif
      R d = 0.0;
      for (int i = 0; i < nslip; ++i) { d += rss[i] * f.gdot[i]; }
      f.pl_disipation_rate = m_inv_det_v_e * d;
   }
   if (!f.doTangent) { return; }

   R elast_dev_press_vec[ecmech::nsvec];               // get_scaled_elast_strain_vec
   vecsVxa<ntvec>(elast_dev_press_vec, m_inv_a_vol, elast_d5_f);
   elast_dev_press_vec[iSvecS] = sqr3 * log(m_a_vol);
   // get_slip_rate_deriv_sph_terms (ThermoElastNCubic::getDTDepsSph)
   R dDp_hat_deps_sph[ecmech::ntvec], dWp_hat_deps_sph[ecmech::nwvec];
   {
      R dT_deps_sph[ecmech::ntvec];
      for (int iT = 0; iT < ecmech::ntvec; ++iT) { dT_deps_sph[iT] = -sqr3i * m_K_diag[iT] * elast_dev_press_vec[iT]; }
      R dgdot_deps_sph[nslip];
      for (int iSlip = 0; iSlip < nslip; ++iSlip) {
         R dtaua_deps_sph = 0.0;
         for (int iT = 0; iT < ecmech::ntvec; ++iT) {
            dtaua_deps_sph += m_P[ECMECH_NM_INDX(iT, iSlip, ecmech::ntvec, nslip)] * dT_deps_sph[iT];
         }
         dgdot_deps_sph[iSlip] = dgdot_dtau[iSlip] * dtaua_deps_sph;
      }
      vecsVMa<ntvec, nslip>(dDp_hat_deps_sph, m_P, dgdot_deps_sph);
      vecsVMa<nwvec, nslip>(dWp_hat_deps_sph, m_Q, dgdot_deps_sph);
   }
   // get_material_tangent_stiffness<ThermoElastNCubic, nDimSys, m_i_sub_r>
   constexpr int JAC_SIZE = nDimSys;
   constexpr int ind_sub_omega = m_i_sub_r;
   constexpr int nRHS = ecmech::nsvec;
   const double dt = m_dt;
   R* const material_tangent = f.mtan_vecds;
   R dstrainomega_ddef_rate_t[ nRHS * JAC_SIZE ];
   for (int k = 0; k < nRHS * JAC_SIZE; ++k) { dstrainomega_ddef_rate_t[k] = 0.0; }
   {
      for (int jE = 0; jE < ecmech::ntvec; ++jE) {
         for (int iE = 0; iE < ntvec; ++iE) {
            dstrainomega_ddef_rate_t[ECMECH_NM_INDX(jE, iE, nRHS, JAC_SIZE)] = rmat_5x5_sample2xtal[ECMECH_NN_INDX(jE, iE, ntvec)];
         }
      }
      for (int iE = 0; iE < ecmech::ntvec; ++iE) {
         dstrainomega_ddef_rate_t[ECMECH_NM_INDX(iSvecS, iE, nRHS, JAC_SIZE)] =
            -dt * (dDp_hat_deps_sph[iE] - ecmech::sqr3i * m_inv_a_vol * elast_dt_d5[iE]);
      }
      for (int iW = 0; iW < ecmech::nwvec; ++iW) {
         dstrainomega_ddef_rate_t[ECMECH_NM_INDX(iSvecS, ind_sub_omega + iW, nRHS, JAC_SIZE)] = -dt * dt * dWp_hat_deps_sph[iW];
      }
      lup_solveX<JAC_SIZE>(Jacobian, dstrainomega_ddef_rate_t, nRHS);
   }
   {
      R dstrain_ddef_rate[ ecmech::ntvec * ecmech::nsvec ];
      for (int iTvec = 0; iTvec < ecmech::ntvec; ++iTvec) {
         for (int jSvec = 0; jSvec < ecmech::nsvec; ++jSvec) {
            dstrain_ddef_rate[ECMECH_NM_INDX(iTvec, jSvec, ecmech::ntvec, ecmech::nsvec)] =
               dstrainomega_ddef_rate_t[ECMECH_NM_INDX(jSvec, iTvec, nRHS, JAC_SIZE)];
         }
      }
      // ThermoElastNCubic::multCauchyDif<ntvec, nsvec>(temp_M6, dstrain_ddef_rate, elast_dev_press_vec, dt, inv_det_v_e, inv_a_vol)
      R temp_M6[ ecmech::nsvec2 ];
      for (int iTvec = 0; iTvec < ecmech::ntvec; ++iTvec) {
         const R vFact = m_inv_det_v_e * m_inv_a_vol * m_K_diag[iTvec];
         for (int jSvec = 0; jSvec < ecmech::nsvec; ++jSvec) {
            temp_M6[ECMECH_NN_INDX(iTvec, jSvec, ecmech::nsvec)] = vFact * dstrain_ddef_rate[ECMECH_NM_INDX(iTvec, jSvec, ecmech::ntvec, ecmech::nsvec)];
         }
      }
      for (int jSvec = 0; jSvec < ecmech::nsvec; ++jSvec) { temp_M6[ECMECH_NN_INDX(iSvecS, jSvec, ecmech::nsvec)] = 0.0; }
      {
         const R vFact = -fourthirds * sqr3 * dt * m_inv_det_v_e;
         for (int iTvec = 0; iTvec < ecmech::ntvec; ++iTvec) {
            temp_M6[ECMECH_NN_INDX(iTvec, iSvecS, ecmech::nsvec)] += vFact * m_K_diag[iTvec] * elast_dev_press_vec[iTvec];
         }
      }
      qr6x6_pre_mul<ecmech::nsvec, false>(material_tangent, temp_M6, rmat_5x5_sample2xtal);
   }
   {
      R dcauchy_dquat[ ecmech::ntvec * ecmech::qdim ];
      {
         R drmat_dquat[ ecmech::ndim * ecmech::ndim * ecmech::qdim ];
         d_quat_to_tensor(drmat_dquat, quat);
         R dcauchy_drmat[ ecmech::ntvec * ecmech::ndim * ecmech::ndim ];
         d_rot_mat_vecd_smop(dcauchy_drmat, rmat, cauchy_stress);
         vecsMAB<ntvec, qdim, ndim*ndim>(dcauchy_dquat, dcauchy_drmat, drmat_dquat);
      }
      R dquat_ddef_rate[ ecmech::qdim * nRHS ];
      for (int ii_I = 0; ii_I < nRHS; ++ii_I) {
         for (int ii_Q = 0; ii_Q < ecmech::qdim; ++ii_Q) {
            const int iiQI = ECMECH_NM_INDX(ii_Q, ii_I, ecmech::qdim, nRHS);
            dquat_ddef_rate[iiQI] = 0.0;
            for (int ii_W = 0; ii_W < ecmech::nwvec; ++ii_W) {
               dquat_ddef_rate[iiQI] +=
                  dquat_domega_t[ECMECH_NM_INDX(ii_W, ii_Q, ecmech::nwvec, ecmech::qdim)] *
                  dstrainomega_ddef_rate_t[ECMECH_NM_INDX(ii_I, ind_sub_omega + ii_W, nRHS, JAC_SIZE)];
            }
         }
      }
      R dqcauchy_ddefrate[ ecmech::ntvec * nRHS ];
      vecsMAB<ecmech::ntvec, nRHS, ecmech::qdim>(dqcauchy_ddefrate, dcauchy_dquat, dquat_ddef_rate);
      for (int ii_T = 0; ii_T < ecmech::ntvec; ++ii_T) {
         for (int ii_I = 0; ii_I < nRHS; ++ii_I) {
            material_tangent[ECMECH_NN_INDX(ii_T, ii_I, ecmech::nsvec)] += dqcauchy_ddefrate[ECMECH_NM_INDX(ii_T, ii_I, ecmech::ntvec, nRHS)];
         }
      }
   }
}

// ---------------------------------------------------------------------
// SNLSTrDlDenseG (dogleg trust region) on a pack, lane-masked
// ---------------------------------------------------------------------
template<class Problem, class R>
struct TrDlDenseGSimd
{
   static constexpr int n = Problem::nDimSys;
   using M = mask_t<R>;
   const Problem& m_prob;
   const snls::TrDeltaControl& m_dc;
   R x[n];
   M converged;          ///< lanes that converged
   int fevals[W > 0 ? W : 1]; ///< computeRJ evaluations per lane while active

   TrDlDenseGSimd(const Problem& prob, const snls::TrDeltaControl& dc) : m_prob(prob), m_dc(dc) {}

   // TrDeltaControl::decrDelta on the lanes in m (returns the lanes that stayed above deltaMin)
   M decrDelta(const M& m, R& delta, const R& normfull, const M& took_full) const
   {
      const R tempa = delta * m_dc._xiDecDelta;
      const R tempb = normfull * m_dc._xiDecDelta;
      R d = select(took_full, sqrt(tempa * tempb), delta * m_dc._xiDecDelta);
      const M tooSmall = d < m_dc._deltaMin;
      d = select(tooSmall, R(m_dc._deltaMin), d);
      delta = select(m, d, delta);
      return !tooSmall;
   }
   // TrDeltaControl::decrDeltaPastFullStep
   M decrDeltaPastFullStep(const M& m, R& delta, const R& normfull, const M& took_full) const
   {
      M success = m;            // lanes still fine
      M loop = m && took_full;
      while (true) {
         loop = loop && success && (delta >= normfull);
         if (!any(loop)) { break; }
         success = select_mask(loop, decrDelta(loop, delta, normfull, took_full), success);
      }
      return success;
   }
   static M select_mask(const M& m, const M& a, const M& b) { return (m && a) || (!m && b); }

   /// solve from x = 0; delta starts at deltaInit; maxIter iterations
   void solve(const double tolerance, const int maxIter, const M& active0)
   {
      R residual[n], Jacobian[n * n];
      R grad[n], nrStep[n], delx[n];
      for (int i = 0; i < n; ++i) { x[i] = 0.0; grad[i] = 0.0; nrStep[i] = 0.0; }
      M active = active0;
      converged = !active0 && active0; // all false
      for (int l = 0; l < W; ++l) { fevals[l] = 0; }
      auto countEval = [&](const M& m) {
         if constexpr (std::is_same<R, double>::value) { fevals[0] += m ? 1 : 0; }
         else { for (int l = 0; l < W; ++l) { fevals[l] += m[l] ? 1 : 0; } }
      };

      R delta = m_dc.getDeltaInit();
      m_prob.computeRJ(residual, Jacobian, x);
      countEval(active);
      R res = vecNorm<n>(residual);
      R res_0 = res;
      R Jg_2 = 0.0;
      R rhoLast = 0.0;
      M reject_prev = !active && active; // all false

      for (int iter = 0; iter < maxIter && any(active); ++iter) {
         const M fresh = active && !reject_prev;
         if (any(fresh)) {
            // gradient and |J g|^2
            R g[n];
            for (int i = 0; i < n; ++i) {
               g[i] = 0.0;
               for (int k = 0; k < n; ++k) { g[i] += Jacobian[k * n + i] * residual[k]; }
            }
            R jg2 = 0.0;
            for (int i = 0; i < n; ++i) {
               R t = 0.0;
               for (int k = 0; k < n; ++k) { t += Jacobian[i * n + k] * g[k]; }
               jg2 += t * t;
            }
            // Newton step
            R step[n];
            const M lu_ok = lup_solve<n>(Jacobian, step, residual);
            for (int i = 0; i < n; ++i) {
               grad[i] = select(fresh, g[i], grad[i]);
               nrStep[i] = select(fresh, -step[i], nrStep[i]);
            }
            Jg_2 = select(fresh, jg2, Jg_2);
            active = active && !(fresh && !lu_ok); // linearSolveFailure
         }

         // dogleg
         const R nr_norm = vecNorm<n>(nrStep);
         R pred_resid;
         M use_nr;
         {
            use_nr = nr_norm <= delta;
            R norm2_grad = grad[0] * grad[0];
            for (int i = 1; i < n; ++i) { norm2_grad += grad[i] * grad[i]; }
            const R norm_grad = sqrt(norm2_grad);
            const R alpha = select(Jg_2 > 0.0, norm2_grad / select(Jg_2 > 0.0, Jg_2, R(one)), R(one));
            const R norm_grad_inv = select(norm_grad > 0.0, one / select(norm_grad > 0.0, norm_grad, R(one)), R(one));
            const R norm_s_sd_opt = alpha * norm_grad;
            const M leg1 = !use_nr && (norm_s_sd_opt >= delta);
            const M leg2 = !use_nr && !leg1;
            // first leg
            const R factor = -delta * norm_grad_inv;
            const R val = -(delta * norm_grad) + 0.5 * delta * delta * Jg_2 * (norm_grad_inv * norm_grad_inv);
            const R pred1 = sqrt(fmax(2.0 * val + res_0 * res_0, R(0.0)));
            // second leg
            R qb = 0.0, qa = 0.0;
            for (int i = 0; i < n; ++i) {
               const R p_i = nrStep[i] + alpha * grad[i];
               qa += p_i * p_i;
               qb += p_i * grad[i];
            }
            qb *= alpha;
            const R qc = norm_s_sd_opt * norm_s_sd_opt - delta * delta;
            const R qa_s = select(leg2, qa, R(one));
            R beta = (qb + sqrt(fmax(qb * qb - qa * qc, R(0.0)))) / qa_s;
            beta = fmax(R(0.0), fmin(R(1.0), beta));
            const R omb = 1.0 - beta;
            const R omba = omb * alpha;
            const R res_cauchy = select(Jg_2 > 0.0, sqrt(fmax(res_0 * res_0 - alpha * norm2_grad, R(0.0))), res_0);
            const R pred2 = omb * res_cauchy;
            for (int i = 0; i < n; ++i) {
               const R d1 = factor * grad[i];
               const R d2 = beta * nrStep[i] - omba * grad[i];
               delx[i] = select(use_nr, nrStep[i], select(leg1, d1, d2));
               delx[i] = select(active, delx[i], R(0.0));
               x[i] += delx[i];
            }
            pred_resid = select(use_nr, R(0.0), select(leg1, pred1, pred2));
         }
         reject_prev = !active && active; // false

         m_prob.computeRJ(residual, Jacobian, x);
         countEval(active);
         // updateDelta<nDim, IdenticalRetry = true> (computeRJ always succeeds here)
         {
            res = select(active, vecNorm<n>(residual), res);
            const M conv = active && (res < tolerance);
            converged = converged || conv;
            M upd = active && !conv;
            // TrDeltaControl::updateDelta
            const R actual_change = res - res_0;
            const R pred_change = pred_resid - res_0;
            const M zeroPred = upd && (pred_change == 0.0);
            M success = upd;
            // predicted change zero
            {
               const M atMax = zeroPred && (delta >= m_dc._deltaMax);
               success = success && !atMax;
               delta = select(zeroPred && !atMax, fmin(delta * m_dc._xiForcedIncDelta, R(m_dc._deltaMax)), delta);
            }
            const M nz = upd && !zeroPred;
            const R rho = actual_change / select(nz, pred_change, R(one));
            rhoLast = select(nz, rho, rhoLast);
            const M incr = nz && (rho > m_dc._xiLG) && (actual_change < 0.0) && (rho < m_dc._xiUG) && !use_nr;
            delta = select(incr, fmin(delta * m_dc._xiIncDelta, R(m_dc._deltaMax)), delta);
            const M decr = nz && !((rho > m_dc._xiLG) && (actual_change < 0.0) && (rho < m_dc._xiUG))
                           && ((rho < m_dc._xiLO) || (rho > m_dc._xiUO));
            if (any(decr)) {
               const M s = decrDelta(decr, delta, nr_norm, use_nr);
               success = select_mask(decr, s && success, success);
            }
            const M rej = upd && (actual_change > 0.0);
            // IdenticalRetry
            const M rejOk = rej && success;
            if (any(rejOk)) {
               const M s = decrDeltaPastFullStep(rejOk, delta, nr_norm, use_nr);
               success = select_mask(rejOk, s, success);
            }
            const M deltaFail = upd && !success;
            reject_prev = rej && !deltaFail;
            active = upd && !deltaFail;
            // rejected step: x back, residual norm back
            for (int i = 0; i < n; ++i) { x[i] = select(reject_prev, x[i] - delx[i], x[i]); }
            res = select(reject_prev, res_0, res);
            res_0 = select(active, res, res_0);
         }
      }
   }
};

/// Supported combinations (others use the scalar path)
template<class SlipGeom, class Kinetics, class ThermoElastN>
struct simd_supported {
   template<class K, class = void> struct has_simd : std::false_type {};
   template<class K> struct has_simd<K, std::void_t<decltype(SimdKinetics<K>::supported)>>
      : std::integral_constant<bool, SimdKinetics<K>::supported> {};
   static constexpr bool value = !SlipGeom::dynamic && std::is_same<ThermoElastN, evptn::ThermoElastNCubic>::value
                                 && has_simd<Kinetics>::value;
};

template<class KinS, class = void> struct has_simd_hardening : std::false_type {};
template<class KinS> struct has_simd_hardening<KinS, std::void_t<decltype(KinS::simdHardening)>>
   : std::integral_constant<bool, KinS::simdHardening> {};

/// The EOS part of evptn::preprocess (fixed slip geometry), per point
template<class EosModel, class PS>
inline void preprocess_eos(const EosModel& eos, const double* const rel_vol_ratios, const double* const internal_energy,
                           const double* const def_rate_d6v_sample, PS& prob_state,
                           double& halfVMidDt, double& dev_strain_energy_total)
{
   halfVMidDt = oneqrtr * (rel_vol_ratios[0] + rel_vol_ratios[1]) * prob_state.dt;
   dev_strain_energy_total = halfVMidDt * vecsInnerSvecDev(prob_state.cauchy_stress_d6p, def_rate_d6v_sample);
   const double energy_old = internal_energy[ecmech::i_ne_total];
   {
      double pressure_BOS;
      const double rel_vol_old = rel_vol_ratios[0];
      eos.evalPT(pressure_BOS, prob_state.tkelv, rel_vol_old, energy_old);
   }
   {
      const double pressure_old = prob_state.cauchy_stress_d6p[6];
      double tkelv_new, dpde, dpdv, dtde;
      updateSimple(eos, prob_state.pressure_EOS, tkelv_new, prob_state.energy_new, prob_state.bulk_modulus_new,
                   dpde, dpdv, dtde, rel_vol_ratios[1], rel_vol_ratios[3], energy_old, pressure_old);
   }
}

/// KineticsOrowanD::updateH + updateHOrowanReduced (LOGFORM) on a pack:
/// the reduced dislocation-density solve of all W points at once
template<class KinS, class PS>
inline void updateHOrowanReducedSimd(const KinS& kinS, PS* const* ps, bool* const ok, const double dt)
{
   constexpr int n = KinS::nS;
   const double hdn_min = kinS.hdnMin();
   const double berg = kinS.bergMag();
   using HP = typename KinS::template HProblem<vdouble>;
   HP prob(kinS, dt);
   vmask active = (vdouble(0.0) == 0.0);
   int lref = 0;
   while (lref < W - 1 && !ok[lref]) { ++lref; }
   for (int l = 0; l < W; ++l) {
      const int s = ok[l] ? l : lref;
      if (!ok[l]) { active[l] = false; }
      const double* const hs_o = ps[s]->h_state;
      const double* const gdot = ps[s]->gdot;
      for (int i = 0; i < n; i++) {
         const double div = fmax(hs_o[i], hdn_min) * berg;
         prob.nu[i][l] = std::abs(gdot[i]) / div;
         prob.y_m_o[i][l] = log(fmax(hs_o[i], hdn_min));
         prob.y_t_o[i][l] = log(fmax(hs_o[n + i], hdn_min));
      }
   }
   prob.setScales();
   snls::TrDeltaControl deltaControl;
   deltaControl._deltaInit = 1e0;
   TrDlDenseGSimd<HP, vdouble> solver(prob, deltaControl);
   solver.solve(1e-10, 100, active);
   vdouble h[2 * n];
   prob.getHn(h, solver.x);
   for (int l = 0; l < W; ++l) {
      if (!ok[l]) { continue; }
      if (!solver.converged[l]) {
         ok[l] = false; // as preprocess: "Hardening failed to converge"
         continue;
      }
      for (int i = 0; i < 2 * n; i++) { ps[l]->h_state_u[i] = std::exp(static_cast<double>(h[i][l])); }
   }
}

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
   using ProbS = evptn::EvptnUpdstProblem<SlipGeom, Kinetics, ThermoElastN, PS>;
   using ProbV = EvptnUpdstProblemSimd<SlipGeom, Kinetics, vdouble>;
   using KinS = SimdKinetics<Kinetics>;
   static_assert(simd_supported<SlipGeom, Kinetics, ThermoElastN>::value, "unsupported model for the SIMD path");
   constexpr bool hardeningOnPacks = has_simd_hardening<KinS>::value;

   const KinS kinS(kinetics, SlipGeom::nslip);
   double K_diag[ecmech::ntvec];
   {
      std::vector<double> p; elastN.getParams(p);
      const double c11 = p[0], c12 = p[1], c44 = p[2];
      K_diag[0] = c11 - c12; K_diag[1] = c11 - c12;
      K_diag[2] = two * c44; K_diag[3] = two * c44; K_diag[4] = two * c44;
   }
   snls::TrDeltaControl deltaControl;
   deltaControl._deltaInit = 1e0;
   const int nPacks = (nPassed + W - 1) / W;
   // ECMECH_SIMD_STATS=1: lane utilization of the main solve
   // (useful evaluations / (W x evaluations the packs ran)), printed every 60 calls
   static const bool statsOn = std::getenv("ECMECH_SIMD_STATS") != nullptr;
   static long statUseful = 0, statIssued = 0, statCalls = 0;
   const unsigned int nHist = history_stride;
   if (nHist > 128) { ECMECH_FAIL(__func__, "SIMD path: history too long for the padding scratch"); }

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
      // scratch copies for padding lanes (written but discarded)
      double padHist[W][128], padStress[W][8], padEnergy[W][8], padSdd[W][8], padT[W];
      alignas(PS) unsigned char psMem[W][sizeof(PS)];
      PS* ps[W];
      bool ok[W];
      double halfVMidDt[W], devE[W];
      double* histL[W]; double* stressL[W]; double* energyL[W]; double* sddL[W]; double* tL[W];
      for (int l = 0; l < W; ++l) {
         const int i = idx[l];
         histL[l] = &histV[history_stride * i];
         stressL[l] = &cauchy_stress_d6pV[stress_stride * i];
         energyL[l] = &internal_energyV[int_eng_stride * i];
         sddL[l] = &sddV[sdd_stride * i];
         tL[l] = &tkelvV[tkelv_stride * i];
         if (!real[l]) {
            for (unsigned int k = 0; k < nHist && k < 128; ++k) { padHist[l][k] = histL[l][k]; }
            for (int k = 0; k < 8 && k < (int)stress_stride; ++k) { padStress[l][k] = stressL[l][k]; }
            for (int k = 0; k < 8 && k < (int)int_eng_stride; ++k) { padEnergy[l][k] = energyL[l][k]; }
            for (int k = 0; k < 8 && k < (int)sdd_stride; ++k) { padSdd[l][k] = sddL[l][k]; }
            padT[l] = *tL[l];
            histL[l] = padHist[l]; stressL[l] = padStress[l]; energyL[l] = padEnergy[l]; sddL[l] = padSdd[l]; tL[l] = &padT[l];
         }
         ps[l] = new (psMem[l]) PS(histL[l], stressL[l], *tL[l], &def_rate_d6vV[def_rate_stride * i],
                                   &spin_vecV[spin_v_stride * i], &rel_vol_ratiosV[vol_ratio_stride * i], dt);
         if constexpr (hardeningOnPacks) {
            preprocess_eos(eos, &rel_vol_ratiosV[vol_ratio_stride * i], energyL[l], &def_rate_d6vV[def_rate_stride * i],
                           *ps[l], halfVMidDt[l], devE[l]);
            ok[l] = true;
         }
         else {
            ok[l] = evptn::preprocess(slipGeom, kinetics, eos, elastN, &rel_vol_ratiosV[vol_ratio_stride * i],
                                      energyL[l], &def_rate_d6vV[def_rate_stride * i], *ps[l], halfVMidDt[l], devE[l]);
         }
      }
      if constexpr (hardeningOnPacks) {
         updateHOrowanReducedSimd(kinS, ps, ok, dt);
      }
      int lref = 0;
      while (lref < W - 1 && !ok[lref]) { ++lref; }

      // pack the per-point data of the problem
      ProbV prob(slipGeom.getP(), slipGeom.getQ(), K_diag, kinS, dt);
      double hdnScale[W];
      vmask active0 = (vdouble(0.0) == 0.0);
      for (int l = 0; l < W; ++l) {
         const int s = ok[l] ? l : lref; // failed lanes ride along on a valid lane's data
         if (!ok[l]) { active0[l] = false; }
         const PS& p = *ps[s];
         for (int k = 0; k < ecmech::ntvec; ++k) { prob.m_elast_d5_n[k][l] = p.elast_d5_n[k]; prob.m_def_rate_d5_sample[k][l] = p.def_rate_d5_sample[k]; }
         for (int k = 0; k < ecmech::qdim; ++k) { prob.m_xtal_ori_quat_n[k][l] = p.quat_n[k]; }
         for (int k = 0; k < ecmech::nwvec; ++k) { prob.m_spin_vec_sample[k][l] = p.spin_vec_sample[k]; }
         prob.m_pressure_EOS[l] = p.pressure_EOS;
         prob.m_inv_det_v_e[l] = 1.0 / p.rel_vol_new;
         const double a_vol = pow(p.rel_vol_new, onethird);
         prob.m_a_vol[l] = a_vol;
         prob.m_inv_a_vol[l] = 1.0 / a_vol;
         double vals[Kinetics::nVals];
         hdnScale[l] = kinetics.getVals(vals, p.pressure_EOS, p.tkelv, p.h_state_u);
         for (int k = 0; k < Kinetics::nVals; ++k) { prob.m_kin_vals[k][l] = vals[k]; }
         // EvptnUpdstProblem's residual scalings
         const double adots_ref = kinetics.getFixedRefRate(vals);
         const double eff = ecmech::vecNorm<ntvec>(p.def_rate_d5_sample);
         const double esi = (eff < epsdot_scl_nzeff * adots_ref) ? one / adots_ref : fmin(one / eff, 1e6 * dt);
         prob.m_epsdot_scale_inv[l] = esi;
         prob.m_rotincr_scale_inv[l] = (1.0 / dt) * esi;
      }

      TrDlDenseGSimd<ProbV, vdouble> solver(prob, deltaControl);
      solver.solve(tolerance, 200, active0);
      if (statsOn) {
         long sum = 0, mx = 0;
         for (int l = 0; l < W; ++l) { if (real[l] && ok[l]) { sum += solver.fevals[l]; mx = std::max<long>(mx, solver.fevals[l]); } }
         #pragma omp atomic
         statUseful += sum;
         #pragma omp atomic
         statIssued += mx * W;
      }

      // tangent and the converged-state values, on the pack (one more
      // evaluation, as computeTangentStiffness does)
      typename ProbV::Final fin;
      fin.doTangent = (mtanSDV != nullptr);
      {
         vdouble r[ProbV::nDimSys], J[ProbV::nDimSys * ProbV::nDimSys];
         prob.computeRJ(r, J, solver.x, &fin);
      }

      // per point: computeTangentStiffness's scaling, state update,
      // postprocess_prob and postprocess (the last with the original code)
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
            double mtanSD_vecds[ecmech::nsvec2];
            for (int k = 0; k < ecmech::nsvec2; ++k) { mtanSD_vecds[k] = fin.mtan_vecds[k][l] * (1.0 / dt); }
            mtanSD_vecds[ECMECH_NN_INDX(iSvecS, iSvecS, ecmech::nsvec)] += three * p.bulk_modulus_new;
            mtan_conv_sd_svec<true>(&mtanSDV[ecmech::nsvec2 * i], mtanSD_vecds);
         }
         for (int k = 0; k < ecmech::ntvec; ++k) { p.elast_d5_u[k] = fin.elast_d5[k][l]; }
         for (int k = 0; k < ecmech::qdim; ++k) { p.quat_u[k] = fin.quat[k][l]; }
         histL[l][evptn::iHistA_nFEval] = solver.fevals[l] + (mtanSDV ? 1 : 0);
         // postprocess_prob
         for (int k = 0; k < Kinetics::nH; ++k) { p.h_state[k] = p.h_state_u[k]; }
         for (int k = 0; k < SlipGeom::nslip; ++k) { p.gdot[k] = fin.gdot[k][l]; }
         p.eps_dot = fin.effective_shear_rate[l];
         p.eps += p.eps_dot * p.dt;
         {
            const double dEff = vecd_Deff(p.def_rate_d5_sample);
            double flow_strength = hdnScale[l];
            if (dEff > idp_tiny_sqrt) { flow_strength = fin.pl_disipation_rate[l] / dEff; }
            p.flow_strength = flow_strength;
         }
         double cauchy_stress_d5p_xtal[ecmech::nsvec];
         for (int k = 0; k < ecmech::nsvec; ++k) { cauchy_stress_d5p_xtal[k] = fin.cauchy_xtal[k][l]; }
         evptn::postprocess(p, elastN, &def_rate_d6vV[def_rate_stride * i], sddL[l], energyL[l],
                            cauchy_stress_d5p_xtal, devE[l], halfVMidDt[l]);
         p.~PS();
      }
   }
   if (statsOn && (++statCalls % 60 == 0)) {
      std::printf("SIMDSTATS lanes %d main-solve lane utilization %.3f (useful %ld / issued %ld)\n",
                  W, double(statUseful) / double(statIssued), statUseful, statIssued);
   }
}

} // namespace simd
} // namespace ecmech

#endif // ECMECH_SIMD_EXPT
