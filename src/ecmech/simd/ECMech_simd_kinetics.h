#pragma once
/**
 * @file ECMech_simd_kinetics.h
 * @brief Experiment (ECMECH_SIMD_EXPT): slip-rate laws on packs of points.
 *
 * SimdKinetics<Kinetics> mirrors one kinetics class: it is built from the
 * scalar object's getParams() (so the original classes stay untouched),
 * re-derives the same constants setParams() derives, and provides
 * evalGdots() for any value type R (double or vdouble). The per-point
 * kinetic values (getVals) and the reference rate are cheap and come from
 * the scalar object, one lane at a time.
 *
 * The bodies are the original evalGdot() code with each early return
 * turned into a "done" mask: every lane computes every branch on safe
 * values, and the result of the branch it would have taken is selected.
 */

#include <vector>
#include "simd/ECMech_simd_core.h"
#include "kinetics/ECMech_kinetics.h"

namespace ecmech {
namespace simd {

template<class Kinetics> struct SimdKinetics; // no SIMD version: not supported

/// KineticsVocePL (power law with Voce hardening)
template<bool nonlinear>
struct SimdKinetics<KineticsVocePL<nonlinear>> {
   static constexpr bool supported = true;
   using K = KineticsVocePL<nonlinear>;
   int nslip;
   double m_gam_w, m_xn, m_xnn, m_t_min, m_t_max;

   SimdKinetics(const K& k, int nslip_) : nslip(nslip_) {
      std::vector<double> p; k.getParams(p);
      const double xm = p[1];
      m_gam_w = p[2];
      m_xnn = one / xm;
      m_xn = m_xnn - one;
      m_t_min = pow(ecmech::gam_ratio_min, xm);
      m_t_max = pow(ecmech::gam_ratio_ovf, xm);
   }

   template<class R>
   void evalGdot(R& gdot, R& dgdot_dtau, const R& gIn, const R& tau) const
   {
      const R g_i = one / gIn;
      const R t_frac = tau * g_i;
      const R at = fabs(t_frac);
      const auto act = at > m_t_min;
      const auto sat = at > m_t_max;
      const R gdot_sat = copysign(R(ecmech::gam_ratio_ovffx * m_gam_w), tau);
      const R abslog = log(select(act, at, R(one)));
      const R blog = m_xn * abslog;
      const R temp = m_gam_w * exp(blog);
      const R g = temp * t_frac;
      const R dg = m_xnn * g / select(act, tau, R(one));
      gdot = select(act, select(sat, gdot_sat, g), R(zero));
      dgdot_dtau = select(act && !sat, dg, R(zero));
   }

   template<class R>
   void evalGdots(R* const gdot, R* const dgdot_dtau, const R* const tau, const R* const vals) const
   {
      for (int iSlip = 0; iSlip < nslip; ++iSlip) {
         evalGdot(gdot[iSlip], dgdot_dtau[iSlip], vals[0], tau[iSlip]);
      }
   }
};

/// The thermally activated (MTS) + drag slip-rate law shared by
/// KineticsKMBalD::evalGdot and KineticsOrowanD::evalGdot: same code as
/// those, with the reference rates and strengths passed in.
template<bool withGAthermal, bool pOne, bool qOne>
struct MtsDragLaw {
   double m_tau_a, m_p, m_q, m_wrD;
   double m_xn, m_xnn, m_t_min, m_t_max;

   void setDerived(double c_1, double tkelv_ref, double mu_ref) {
      const double xm = one / (two * ((c_1 / tkelv_ref) * mu_ref * m_p * m_q));
      m_xnn = one / xm;
      m_xn = m_xnn - one;
      m_t_min = pow(ecmech::gam_ratio_min, xm);
      m_t_max = pow(ecmech::gam_ratio_ovf, xm);
   }

   template<class R>
   void get_mts_dG(R& exp_arg, R& mts_dfac, const R& c_e, const R& denom_i, const R& t_frac) const
   {
      mts_dfac = c_e * denom_i;
      R p_func;
      if (pOne) {
         p_func = t_frac;
      }
      else {
         const auto tiny = fabs(t_frac) < idp_tiny_sqrt;
         if (m_p == one) {
            p_func = select(tiny, R(zero), t_frac);
            mts_dfac = select(tiny, mts_dfac * 1e10, mts_dfac);
         }
         else {
            const R t_s = select(tiny, R(one), t_frac);
            const R pf = copysign(pow(fabs(t_s), m_p), t_s);
            p_func = select(tiny, R(zero), pf);
            mts_dfac = select(tiny, mts_dfac * 1e10, mts_dfac * m_p * pf / t_s);
         }
      }
      R q_arg = one - p_func;
      const auto qsmall = q_arg < idp_tiny_sqrt;
      R pq_fac;
      if (qOne || m_q == one) {
         pq_fac = q_arg;
      }
      else {
         const R q_s = select(qsmall, R(one), q_arg);
         const R temp = pow(fabs(q_s), m_q);
         mts_dfac = mts_dfac * m_q * temp / fabs(q_s);
         pq_fac = copysign(temp, q_s);
      }
      mts_dfac = select(qsmall, R(zero), mts_dfac);
      pq_fac = select(qsmall, R(zero), pq_fac);
      exp_arg = -c_e * pq_fac;
   }

   template<class R>
   void evalGdot(R& gdot, R& dgdot_dtau, const R& gam_w, const R& gam_r, const R& gIn, const R& c_t,
                 const R& tau, double mu) const
   {
      static const double gdot_w_pl_scaling = 10.0;
      using M = mask_t<R>;

      R g_i, gAth;
      M zeroOut = (tau != tau); // all false
      if (withGAthermal) {
         gAth = gIn;
         g_i = one / m_tau_a;
      }
      else {
         gAth = m_tau_a;
         zeroOut = zeroOut || (tau == zero);
         g_i = one / gIn;
      }
      const R atau = fabs(tau);
      const R at_0 = fmax(R(zero), atau - gAth) * g_i;
      R gdot_r, dgdot_r;
      {
         const R exp_arg = (atau - gAth) / m_wrD;
         zeroOut = zeroOut || (exp_arg < gam_ratio_min);
         // early return when every lane has returned (most slip systems are inactive)
         if (all(zeroOut)) { gdot = 0.0; dgdot_dtau = 0.0; return; }
         const auto lin = exp_arg < idp_eps_sqrt;
         const R e = exp(-exp_arg);
         const R temp = select(lin, one - exp_arg, e);
         gdot_r = select(lin, gam_r * exp_arg, gam_r * (one - e));
         dgdot_r = gam_r * temp / m_wrD;
      }
      // drag-limited
      const M drag = !zeroOut && (at_0 > m_t_max);
      if (all(zeroOut || drag)) {
         gdot = select(zeroOut, R(zero), copysign(gdot_r, tau));
         dgdot_dtau = select(zeroOut, R(zero), dgdot_r);
         return;
      }

      R gdot_w, dgdot_w;
      {
         const R c_e = c_t * mu;
         const R t_frac = (atau - gAth) * g_i;
         R exp_arg, mts_dfac;
         get_mts_dG(exp_arg, mts_dfac, c_e, g_i, t_frac);
         zeroOut = zeroOut || (!drag && (exp_arg < ln_gam_ratio_min));
         if (all(zeroOut || drag)) {
            gdot = select(zeroOut, R(zero), copysign(gdot_r, tau));
            dgdot_dtau = select(zeroOut, R(zero), dgdot_r);
            return;
         }
         gdot_w = gam_w * exp(exp_arg);
         dgdot_w = mts_dfac * gdot_w;

         const R t_frac_m = (-atau - gAth) * g_i;
         R exp_arg_m, mts_dfac_m;
         get_mts_dG(exp_arg_m, mts_dfac_m, c_e, g_i, t_frac_m);
         const auto back = exp_arg_m > ln_gam_ratio_min;
         if (any(back)) { // (skipped when no lane needs it, as the scalar code does)
            const R gdot_w_m = gam_w * exp(exp_arg_m);
            gdot_w = select(back, gdot_w - gdot_w_m, gdot_w);
            dgdot_w = select(back, dgdot_w - mts_dfac_m * gdot_w_m, dgdot_w);
            zeroOut = zeroOut || (!drag && back && (fabs(gdot_w / gam_w) < gam_ratio_min));
         }
      }
      {
         const auto pl = at_0 > m_t_min;
         if (any(pl)) {
            const R abslog = log(select(pl, at_0, R(one)));
            const R blog = m_xn * abslog;
            const R temp = (gam_w * gdot_w_pl_scaling) * exp(blog);
            gdot_w = select(pl, gdot_w + temp * at_0, gdot_w);
            dgdot_w = select(pl, dgdot_w + temp * m_xnn * g_i, dgdot_w);
         }
      }
      R g, dg;
      {
         g = one / (one / gdot_w + one / gdot_r);
         const R gdrdiv2 = one / (gdot_r * gdot_r);
         const R gdwdiv2 = one / (gdot_w * gdot_w);
         dg = (g * g) * (dgdot_w * gdwdiv2 + dgdot_r * gdrdiv2);
      }
      gdot = copysign(select(drag, gdot_r, g), tau);
      dgdot_dtau = select(drag, dgdot_r, dg);
      gdot = select(zeroOut, R(zero), gdot);
      dgdot_dtau = select(zeroOut, R(zero), dgdot_dtau);
   }
};

/// KineticsKMBalD (MTS-style thermally activated slip with drag), one set
/// of values for all slip systems (perSS = false, nVPer = 1)
template<bool withGAthermal, bool pOne, bool qOne, bool perSS, int nVPer>
struct SimdKinetics<KineticsKMBalD<withGAthermal, pOne, qOne, perSS, nVPer>> {
   // per-slip-system values (perSS) not done: those models use the scalar path
   static constexpr bool supported = !perSS && nVPer == 1;
   using K = KineticsKMBalD<withGAthermal, pOne, qOne, perSS, nVPer>;
   int nslip;
   double m_mu_ref;
   MtsDragLaw<withGAthermal, pOne, qOne> m_law;

   SimdKinetics(const K& k, int nslip_) : nslip(nslip_) {
      std::vector<double> p; k.getParams(p);
      int i = 0;
      m_mu_ref = p[i++];
      const double tkelv_ref = p[i++];
      const double c_1 = p[i++];
      m_law.m_tau_a = p[i++];
      m_law.m_p = p[i++];
      m_law.m_q = p[i++];
      i += 2; // gam_wo, gam_ro
      m_law.m_wrD = p[i++];
      m_law.setDerived(c_1, tkelv_ref, m_mu_ref);
   }

   template<class R>
   void evalGdots(R* const gdot, R* const dgdot_dtau, const R* const tau, const R* const vals) const
   {
      for (int iSlip = 0; iSlip < nslip; ++iSlip) {
         m_law.evalGdot(gdot[iSlip], dgdot_dtau[iSlip], vals[0], vals[1], vals[2], vals[2 + nVPer], tau[iSlip], m_mu_ref);
      }
   }
};

/// KineticsOrowanD (dislocation-density Orowan kinetics), one set of values
/// for all slip systems (perSS = false, nVPer = 1); slip rates only (the
/// hardening update is separate)
template<bool withGAthermal, bool pOne, bool qOne, bool isotropic, bool perSS, int nVPer, class SlipGeomK, bool LOGFORM>
struct SimdKinetics<KineticsOrowanD<withGAthermal, pOne, qOne, isotropic, perSS, nVPer, SlipGeomK, LOGFORM>> {
   static constexpr bool supported = !perSS && nVPer == 1;
   using K = KineticsOrowanD<withGAthermal, pOne, qOne, isotropic, perSS, nVPer, SlipGeomK, LOGFORM>;
   int nslip;
   double m_mu_ref, m_lbar_b, m_fD, m_gam_ro;
   MtsDragLaw<withGAthermal, pOne, qOne> m_law;

   SimdKinetics(const K& k, int nslip_) : nslip(nslip_), m_k(&k) {
      std::vector<double> p; k.getParams(p);
      int i = 0;
      m_mu_ref = p[i++];
      const double tkelv_ref = p[i++];
      i += nVPer; // berg_mag
      m_lbar_b = p[i++];
      m_gam_ro = p[i++];
      m_law.m_wrD = p[i++];
      m_fD = p[i++];
      const double c_1 = p[i++];
      m_law.m_tau_a = p[i++];
      m_law.m_p = p[i++];
      m_law.m_q = p[i++];
      m_law.setDerived(c_1, tkelv_ref, m_mu_ref);
   }

   /// The reduced (log-form) hardening update is done on packs too
#if defined(ECMECH_OROWAN_FULL_HSOLVE)
   static constexpr bool simdHardening = false;
#else
   static constexpr bool simdHardening = LOGFORM;
#endif
   static constexpr int nS = SlipGeomK::nslip;
   const K* m_k; ///< the scalar model (constants read directly, see the friend in KineticsOrowanD)
   double hdnMin() const { return m_k->m_hdn_min; }
   double bergMag() const { return m_k->m_berg_mag[0]; }

   /// KineticsOrowanD::reducedHardening on a pack: mobile densities from the
   /// total ones (per slip system, a 1-D Newton iteration, lane-masked),
   /// the reduced residual S and its derivative dS
   template<class R>
   void reducedHardening(const R* const y_t, const R* const y_m_o, const R* const nu, const double dt,
                         R* const y_m, R* const S, R* const dS) const
   {
      constexpr int n = nS;
      const double* const a_mat = m_k->m_a_mat;
      R Ht[n];
      for (int i = 0; i < n; i++) { Ht[i] = exp(y_t[i]); }
      R forest[n];
      vecsVMa<n, n>(&forest[0], a_mat, &Ht[0]);
      const double alpha = m_k->m_c_mult - m_k->m_c_trap;
      const double beta = m_k->m_c_ann * m_k->m_d_ann;
      for (int i = 0; i < n; i++) {
         const R sf = sqrt(forest[i]);
         const R a_i = dt * nu[i];
         R y = y_m_o[i];
         auto act = (y == y) || (y != y); // all lanes
         for (int k = 0; k < 50; k++) {
            const R e = exp(y);
            const R g = y - y_m_o[i] - a_i * (alpha * sf - beta * e);
            const R dy = g / (one + a_i * beta * e);
            const R ynew = y - dy;
            y = select(act, ynew, y);
            act = act && !(fabs(dy) <= 1e-15 * (one + fabs(ynew)));
            if (!any(act)) { break; }
         }
         y_m[i] = y;
         const R Hm = exp(y);
         S[i] = a_i * Hm * (m_k->m_c_mult * sf - beta * Hm) / Ht[i];
         if (dS) {
            const R gp = one + a_i * beta * Hm;
            for (int j = 0; j < n; j++) {
               const R dsf = a_mat[i * n + j] * Ht[j] / (2.0 * sf);
               const R dym = a_i * alpha * dsf / gp;
               dS[i * n + j] = a_i / Ht[i] * ((m_k->m_c_mult * sf - 2.0 * beta * Hm) * Hm * dym + m_k->m_c_mult * Hm * dsf);
            }
            dS[i * n + i] -= S[i];
         }
      }
   }

   /// OrowanReducedHProblem on a pack
   template<class R>
   struct HProblem {
      static constexpr int nDimSys = nS;
      const SimdKinetics& kin;
      double dt;
      R y_m_o[nS], y_t_o[nS], nu[nS];
      R x_scale[nS], res_scale[nS];

      HProblem(const SimdKinetics& k, double dt_) : kin(k), dt(dt_) {}
      void setScales() {
         for (int i = 0; i < nS; i++) {
            x_scale[i] = fmax(y_t_o[i], R(1.0));
            res_scale[i] = 1.0 / x_scale[i];
         }
      }
      void getHn(R* const h, const R* const x) const {
         R y_t[nS], S[nS];
         for (int i = 0; i < nS; i++) { y_t[i] = y_t_o[i] + x[i] * x_scale[i]; }
         kin.reducedHardening(y_t, y_m_o, nu, dt, &h[0], S, (R*)nullptr);
         for (int i = 0; i < nS; i++) { h[nS + i] = y_t[i]; }
      }
      void computeRJ(R* const resid, R* const Jacobian, const R* const x) const {
         R y_t[nS], y_m[nS], S[nS];
         for (int i = 0; i < nS; i++) { y_t[i] = y_t_o[i] + x[i] * x_scale[i]; }
         kin.reducedHardening(y_t, y_m_o, nu, dt, y_m, S, Jacobian);
         for (int i = 0; i < nS; i++) {
            resid[i] = (x[i] * x_scale[i] - S[i]) * res_scale[i];
         }
         for (int i = 0; i < nS; i++) {
            for (int j = 0; j < nS; j++) {
               Jacobian[i * nS + j] *= -res_scale[i] * x_scale[j];
            }
            Jacobian[i * nS + i] += 1.0;
         }
      }
   };

   template<class R>
   void evalGdots(R* const gdot, R* const dgdot_dtau, const R* const tau, const R* const vals) const
   {
      const R c_t = vals[1 + 2 * nslip];
      for (int iSlip = 0; iSlip < nslip; ++iSlip) {
         const R gIn = vals[1 + iSlip];
         const R qm = vals[1 + nslip + iSlip];
         const R gam_w = m_lbar_b * m_fD / sqrt(qm);
         const R gam_r = m_gam_ro * qm;
         m_law.evalGdot(gdot[iSlip], dgdot_dtau[iSlip], gam_w, gam_r, gIn, c_t, tau[iSlip], m_mu_ref);
      }
   }
};

} // namespace simd
} // namespace ecmech
