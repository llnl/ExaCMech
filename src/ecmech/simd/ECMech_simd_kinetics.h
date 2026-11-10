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

/// KineticsKMBalD (MTS-style thermally activated slip with drag), one set
/// of values for all slip systems (perSS = false, nVPer = 1)
template<bool withGAthermal, bool pOne, bool qOne, bool perSS, int nVPer>
struct SimdKinetics<KineticsKMBalD<withGAthermal, pOne, qOne, perSS, nVPer>> {
   // per-slip-system values (perSS) not done: those models use the scalar path
   static constexpr bool supported = !perSS && nVPer == 1;
   using K = KineticsKMBalD<withGAthermal, pOne, qOne, perSS, nVPer>;
   int nslip;
   double m_mu_ref, m_tau_a, m_p, m_q, m_wrD;
   double m_xn, m_xnn, m_t_min, m_t_max;

   SimdKinetics(const K& k, int nslip_) : nslip(nslip_) {
      std::vector<double> p; k.getParams(p);
      int i = 0;
      m_mu_ref = p[i++];
      const double tkelv_ref = p[i++];
      const double c_1 = p[i++];
      m_tau_a = p[i++];
      m_p = p[i++];
      m_q = p[i++];
      i += 2; // gam_wo, gam_ro
      m_wrD = p[i++];
      const double xm = one / (two * ((c_1 / tkelv_ref) * m_mu_ref * m_p * m_q));
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
   void evalGdot(R& gdot, R& dgdot_dtau, const R* const vals, const R& tau, double mu) const
   {
      static const double gdot_w_pl_scaling = 10.0;
      const R gam_w = vals[0];
      const R gam_r = vals[1];
      const R gIn = vals[2];
      const R c_t = vals[2 + nVPer];
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
         const auto lin = exp_arg < idp_eps_sqrt;
         const R e = exp(-exp_arg);
         const R temp = select(lin, one - exp_arg, e);
         gdot_r = select(lin, gam_r * exp_arg, gam_r * (one - e));
         dgdot_r = gam_r * temp / m_wrD;
      }
      // drag-limited
      const M drag = !zeroOut && (at_0 > m_t_max);

      R gdot_w, dgdot_w;
      {
         const R c_e = c_t * mu;
         const R t_frac = (atau - gAth) * g_i;
         R exp_arg, mts_dfac;
         get_mts_dG(exp_arg, mts_dfac, c_e, g_i, t_frac);
         zeroOut = zeroOut || (!drag && (exp_arg < ln_gam_ratio_min));
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

   template<class R>
   void evalGdots(R* const gdot, R* const dgdot_dtau, const R* const tau, const R* const vals) const
   {
      for (int iSlip = 0; iSlip < nslip; ++iSlip) {
         evalGdot(gdot[iSlip], dgdot_dtau[iSlip], vals, tau[iSlip], m_mu_ref);
      }
   }
};

} // namespace simd
} // namespace ecmech
