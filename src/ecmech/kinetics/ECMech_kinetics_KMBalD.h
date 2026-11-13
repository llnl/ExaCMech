/**
 * @file ECMech_kinetics_KMBalD.h
 * @brief Kocks-Mecking single-dislocation-density hardening law, paired with balanced
 * (bidirectional) thermally-activated MTS-like slip kinetics and phonon drag.
 *
 * Model form follows Barton, Winter, and Reaugh, "Defect evolution and pore collapse in
 * crystalline energetic materials," Modelling Simul. Mater. Sci. Eng. 17, 035003 (2009)
 * @cite hmx.
 *
 * "Balanced" means the slip-rate law explicitly evaluates and subtracts a
 * reverse-direction thermally-activated jump rate from the forward one (see "Slip-rate
 * law" below), so the net rate goes to zero continuously as the resolved shear stress
 * goes to zero, rather than having an artificial threshold/discontinuity there.
 *
 * The single hardening state variable ρ tracked by this model is a **relative/
 * normalized dislocation density** (dimensionless, order-unity), not an absolute
 * density with physical units -- notice there is no Burgers-vector parameter anywhere
 * in this class. This is different from KineticsBCCMD and KineticsOrowanD, which track
 * absolute dislocation densities and use an explicit Burgers-vector-scaled Orowan
 * relation (γ̇ = ρ·b·v) in their slip-rate laws. Consistent with that, `updateH` solves
 * for ρ implicitly in log space with the comment "h treated as a normalized (unitless)
 * dislocation density".
 *
 * **Template parameters**:
 * - `withGAthermal`: selects which of the per-group hardening-dependent stress ĝ and
 *   the reference stress τ_a plays the role of the athermal stress floor vs. the MTS
 *   normalizing stress in the slip-rate law (see below) -- i.e. which one is treated as
 *   "the (possibly Peierls-related) thermally activated part" vs. "the athermal part".
 * - `pOne` / `qOne`: when `true`, the MTS activation-energy exponents p / q (`m_p` /
 *   `m_q`) are assumed to be exactly 1, skipping a `pow()` call.
 * - `perSS`: when `true`, the per-group MTS parameters (`m_c_1`, `m_go`, `m_s` below)
 *   are given one-per-slip-system (`nVPer` must equal the slip system count) rather
 *   than shared across all slip systems (`nVPer` must be 1) -- useful for e.g. HCP
 *   materials where slip families (basal/prismatic/pyramidal) have very different
 *   characteristics.
 * - `nVPer`: number of parameter groups; 1 if `!perSS`, else the slip system count.
 *
 * **Kinetic values** (see KineticsKMBalD::getVals), computed from the current
 * (relative/normalized, dimensionless) dislocation density ρ = `h_state[0]`:
 *
 *   γ̇_w = γ̇_w0 / √ρ
 *   γ̇_r = γ̇_r0 · ρ
 *   ĝᵢ = g0ᵢ + sᵢ · √ρ
 *   c_tᵢ = C1ᵢ / T
 *
 * where γ̇_w = `vals[0]` (reference rate for the thermally-activated branch), γ̇_w0 =
 * `m_gam_wo`, γ̇_r = `vals[1]` (reference rate for the drag-limited branch), γ̇_r0 =
 * `m_gam_ro`, ĝᵢ = `vals[2+i]` (per-group reference stress -- plays the role of either
 * the athermal floor or the MTS-normalizing stress in the slip-rate law below,
 * depending on `withGAthermal`), g0ᵢ = `m_go[i]`, sᵢ = `m_s[i]`, c_tᵢ =
 * `vals[2+nVPer+i]` (thermal energy scale), C1ᵢ = `m_c_1[i]`, and T = `tkelv`
 * (temperature).
 *
 * **MTS thermal-activation energy function** (see KineticsKMBalD::get_mts_dG), a
 * Kocks-Argon-Ashby-style activation-energy profile:
 *
 *   E(t) = -c_e · [1 - sign(t)·|t|^p]^q
 *
 * so that a thermally-activated rate is `(reference rate) · exp(E(t))` -- `E` pegs to 0
 * once `t ≥ 1` (barrier fully overcome by stress) and grows more negative (stronger
 * suppression) as `t` decreases. `t` itself is where the resolved shear stress and CRSS
 * actually enter the thermal-activation rate:
 *
 *   t = (σ - g_ath) / g_MTS
 *
 * for a signed driving-stress term σ (the forward evaluation uses σ = |τ|; the
 * reverse/balancing evaluation uses σ = -|τ|; see "Slip-rate law" below) and the
 * athermal-floor/MTS-normalizing-stress pair g_ath/g_MTS (assigned from ĝ and τ_a
 * depending on `withGAthermal`, also in "Slip-rate law" below). Here E = `exp_arg`,
 * c_e = `c_e` (= c_t·μ, a thermal energy prefactor), p = `m_p`, q = `m_q`, and t =
 * `t_frac`.
 *
 * **Slip-rate law** (see KineticsKMBalD::evalGdot): depending on `withGAthermal`, the
 * athermal stress floor g_ath and MTS-normalizing stress g_MTS are assigned from the
 * per-group reference stress ĝ and τ_a (`m_tau_a`) in one of two ways:
 *
 *   withGAthermal:   g_ath = ĝ,     g_MTS = τ_a
 *   !withGAthermal:  g_ath = τ_a,   g_MTS = ĝ
 *
 * The thermally-activated (balanced/bidirectional) rate and the drag-limited rate are
 *
 *   γ̇_th = γ̇_w · [exp(E(t_fwd)) - exp(E(t_rev))]
 *   t_fwd = (|τ| - g_ath) / g_MTS,   t_rev = -(|τ| + g_ath) / g_MTS
 *
 *   γ̇_drag = γ̇_r · (1 - exp(-(|τ| - g_ath)/w_rD))
 *
 * (the reverse-hop term is dropped once negligible), plus a high-stress power-law tail
 * once the clamped forward argument `max(0, t_fwd)` exceeds an underflow threshold
 * t_min (mirroring #gam_ratio_min/#gam_ratio_ovf, via an effective rate-sensitivity
 * exponent derived below):
 *
 *   γ̇_pl = 10 · γ̇_w · max(0, t_fwd)^xnn
 *
 * and finally the thermal and drag branches are combined by harmonic mean (as
 * resistances combine in series), since either mechanism being much slower than the
 * other dominates the net rate:
 *
 *   γ̇ = 1 / (1/(γ̇_th + γ̇_pl) + 1/γ̇_drag) · sign(τ)
 *
 * where τ = `tau`, w_rD = `m_wrD` (drag stress scale), and τ_a = `m_tau_a`. Once the
 * clamped forward argument exceeds an overflow threshold t_max, the thermally-activated
 * part is treated as having overflowed and the rate is purely drag-limited (γ̇ =
 * γ̇_drag).
 *
 * An effective power-law rate-sensitivity exponent is derived once, at `setParams`
 * time, from the reference MTS parameters, so the power-law tail is asymptotically
 * consistent with the MTS thermal part at high stress ratios (`xnn = 1/xm`, `xn = xnn -
 * 1`, matching the `t_min`/`t_max` convention used by the other kinetics models):
 *
 *   xmᵢ = 1 / (2 · (C1ᵢ/T_ref) · μ_ref · p · q)
 *
 * where T_ref = `m_tkelv_ref` and μ_ref = `m_mu_ref` (reference temperature and shear
 * modulus).
 *
 * **Hardening law** (see KineticsKMBalD::getEvolVals / KineticsKMBalD::getSdot1): a
 * Kocks-Mecking single-dislocation-density law, solved implicitly in log space (so ρ
 * cannot be driven negative by the nonlinear solve; see #updateH1):
 *
 *   d(ln ρ)/dt = (k1/√ρ - k2) · γ̇_eff
 *   k2 = k2_0 · (γ̇_0/γ̇_eff)^(1/n)
 *
 * where k1 = `m_k1` (multiplication-rate coefficient), k2 = the rate-dependent
 * recovery-rate coefficient, k2_0 = `m_k2o`, γ̇_0 = `m_gamma_o` (reference shear rate),
 * 1/n = `m_ninv` (recovery rate-sensitivity exponent), and γ̇_eff = `shrate_eff` (the
 * effective, sum-of-absolute-value, shear rate across all slip systems).
 *
 * @see ECMech_kinetics.h for the kinetics model interface contract this class
 * implements, and KineticsVocePL/KineticsOrowanD/KineticsBCCMD for the other available
 * kinetics models
 */

// -*-c++-*-

#ifndef ECMECH_KINETICS_KMBALD_H
#define ECMECH_KINETICS_KMBALD_H

#include <cassert>
#include <cmath>

#include <string>
#include <vector>

namespace ecmech {
   /**
    * @brief Kocks-Mecking single-dislocation-density hardening law with balanced,
    * thermally-activated MTS-like slip kinetics and phonon drag.
    *
    * See the file-level documentation in ECMech_kinetics_KMBalD.h for the governing
    * equations and template-parameter meanings.
    *
    * @tparam withGAthermal Selects which of the per-group hardening-dependent stress or
    * the reference stress `m_tau_a` is the athermal floor vs. the MTS-normalizing stress.
    * @tparam pOne When `true`, the MTS exponent p is fixed at 1.
    * @tparam qOne When `true`, the MTS exponent q is fixed at 1.
    * @tparam perSS When `true`, the per-group MTS parameters vary per slip system.
    * @tparam nVPer Number of parameter groups (1 if `!perSS`, else the slip system count).
    *
    * @ingroup ECMech_kinetics
    */
   template<bool withGAthermal,
            bool pOne, // l_p_1
            bool qOne, // l_q_1
            bool perSS,
            int  nVPer>
   class KineticsKMBalD
   {
      public:
         /** @brief Number of hardening state variables: 1 (a single dislocation density). */
         static constexpr int nH = 1;
         /** @brief Number of parameters: 8 MTS/power-law + 3 per group + 4 Kocks-Mecking + nH initial-state. */
         static constexpr int nParams = 8 + 3 * nVPer + 4 + nH;
         /** @brief Number of kinetic values precomputed by getVals and reused by evalGdots: γ̇_w, γ̇_r, plus 2 per group (ĝ and c_t). */
         static constexpr int nVals = 2 + nVPer + nVPer;
         /** @brief Number of intermediate values precomputed by getEvolVals and reused by getSdot1: 2 (effective shear rate and recovery coefficient k2). */
         static constexpr int nEvolVals = 2;
         /**
          * @brief Construct with a given number of slip systems; parameters must be set
          * separately via setParams.
          * @param _nslip Number of slip systems; must equal `nVPer` if `perSS`,
          * otherwise `nVPer` must be 1.
          */
         __ecmech_hdev__
         KineticsKMBalD(int _nslip) : nslip(_nslip) {
            if (perSS) {
               assert(nslip == nVPer);
            }
            else {
               assert(nVPer == 1);
            }
         }
         /** @brief Destructor (default; no owned resources). */
         ~KineticsKMBalD() = default;

         /**
          * @brief Construct with a given number of slip systems and immediately set
          * parameters.
          * @param params Parameter array; see setParams for the expected order.
          * @param _nslip Number of slip systems; must equal `nVPer` if `perSS`,
          * otherwise `nVPer` must be 1.
          */
         __ecmech_hdev__
         KineticsKMBalD(const double* const params, int _nslip) :
         nslip(_nslip)
         {
            if (perSS) {
               assert(nslip == nVPer);
            }
            else {
               assert(nVPer == 1);
            }
            setParams(params);
         }

         /**
          * @brief Set parameters from a `std::vector` (host-side convenience wrapper
          * around the array-based overload).
          * @param params Parameter vector; see the array overload for the expected order.
          */
         __ecmech_host__
         inline void setParams(const std::vector<double> & params)
         {
            setParams(params.data());
         }

         /**
          * @brief Set parameters from a flat array.
          *
          * Expected order: `mu_ref` (μ_ref), `tkelv_ref` (T_ref), `c_1[nVPer]` (C1ᵢ),
          * `tau_a` (τ_a), `p`, `q`, `gam_wo` (γ̇_w0), `gam_ro` (γ̇_r0), `wrD` (w_rD),
          * `go[nVPer]` (g0ᵢ), `s[nVPer]` (sᵢ) -- then `k1`, `k2o` (k2_0), `ninv` (1/n),
          * `gamma_o` (γ̇_0) -- then `hdn_init` (initial dislocation density). Also
          * derives, per group, the effective power-law rate-sensitivity exponent and the
          * overflow/underflow stress-ratio thresholds `m_xnn`/`m_xn`/`m_t_min`/`m_t_max`
          * (see the file-level documentation in ECMech_kinetics_KMBalD.h), and the
          * dislocation-density floor `m_hdn_min = 1e-4 * hdn_init`.
          * @param params Flat parameter array of length #nParams.
          */
         __ecmech_hdev__
         inline
         void setParams(const double* const params) {
            const double* parsIt = params;

            //////////////////////////////
            // power-law stuff

            m_mu_ref = *parsIt; ++parsIt;
            m_tkelv_ref = *parsIt; ++parsIt;
            for (int iVal = 0; iVal<nVPer; ++iVal) {
               m_c_1[iVal] = *parsIt; ++parsIt;
            }

            m_tau_a = *parsIt; ++parsIt;
            m_p = *parsIt; ++parsIt;
            m_q = *parsIt; ++parsIt;
            m_gam_wo = *parsIt; ++parsIt;
            m_gam_ro = *parsIt; ++parsIt;
            m_wrD = *parsIt; ++parsIt;
            for (int iVal = 0; iVal<nVPer; ++iVal) {
               m_go[iVal] = *parsIt; ++parsIt;
            }

            for (int iVal = 0; iVal<nVPer; ++iVal) {
               m_s[iVal] = *parsIt; ++parsIt;
            }

            if (pOne) {
               assert(m_p == one);
            }
            if (qOne) {
               assert(m_q == one);
            }

            // plaw_from_elawRef
            //
            for (int iVal = 0; iVal<nVPer; ++iVal) {
               // pl%xm = getMtsxmEffective(pl, mu_ref, T_ref)
               double xm = one / (two * ((m_c_1[iVal] / m_tkelv_ref) * m_mu_ref * m_p * m_q));
               //
               // CALL fill_power_law(pl)
               // xmm  = xm - one ;
               m_xnn[iVal] = one / xm;
               m_xn[iVal] = m_xnn[iVal] - one;
               // xMp1 = xnn + one
               //
               // CALL set_t_min_max(pl)
               m_t_min[iVal] = pow(ecmech::gam_ratio_min, xm);
               m_t_max[iVal] = pow(ecmech::gam_ratio_ovf, xm);
            }

            //////////////////////////////
            // Kocks-Mecking stuff

            m_k1 = *parsIt; ++parsIt;
            m_k2o = *parsIt; ++parsIt;
            m_ninv = *parsIt; ++parsIt;
            m_gamma_o = *parsIt; ++parsIt;

            //////////////////////////////
            // nH

            m_hdn_init = *parsIt; ++parsIt;

            m_hdn_min = 1e-4 * m_hdn_init;

            //////////////////////////////

#if defined(ECMECH_DEBUG)
            int iParam = parsIt - params;
            if (iParam != nParams) {
               ECMECH_FAIL(__func__, "iParam != nParams");
            }
#endif
         }

         /**
          * @brief Append this model's current parameters to `params`, in the same order
          * setParams expects them.
          * @param[in,out] params Parameter vector to append to (not cleared first).
          */
         __ecmech_host__
         void getParams(std::vector<double> & params
                        ) const {
#ifdef ECMECH_DEBUG
            // do not clear params in case adding to an existing set
            int paramsStart = params.size();
#endif

            //////////////////////////////
            // power-law stuff

            params.push_back(m_mu_ref);
            params.push_back(m_tkelv_ref);
            for (int iVal = 0; iVal<nVPer; ++iVal) {
               params.push_back(m_c_1[iVal]);
            }

            params.push_back(m_tau_a);
            params.push_back(m_p);
            params.push_back(m_q);
            params.push_back(m_gam_wo);
            params.push_back(m_gam_ro);
            params.push_back(m_wrD);
            for (int iVal = 0; iVal<nVPer; ++iVal) {
               params.push_back(m_go[iVal]);
            }

            for (int iVal = 0; iVal<nVPer; ++iVal) {
               params.push_back(m_s[iVal]);
            }

            //////////////////////////////
            // Kocks-Mecking stuff

            params.push_back(m_k1);
            params.push_back(m_k2o);
            params.push_back(m_ninv);
            params.push_back(m_gamma_o);

            //////////////////////////////
            // nH

            params.push_back(m_hdn_init);

            //////////////////////////////
#ifdef ECMECH_DEBUG
            assert((params.size() - paramsStart) == nParams);
#endif
         }

         /**
          * @brief Describe the single hardening history variable ("rho_dd", the
          * dislocation density) for history-array bookkeeping.
          * @param[out] names Appended with `"rho_dd"`.
          * @param[out] init Appended with #m_hdn_init.
          * @param[out] plot Appended with `true`.
          * @param[out] state Appended with `true`.
          */
         __ecmech_host__
         void getHistInfo(std::vector<std::string> & names,
                          std::vector<double>       & init,
                          std::vector<bool>        & plot,
                          std::vector<bool>        & state) const {
            names.push_back("rho_dd");
            init.push_back(m_hdn_init);
            plot.push_back(true);
            state.push_back(true);
         }

      private:

         /** @brief Number of slip systems handled by this instance. */
         const int nslip; // could template on this if there were call to do so

         //////////////////////////////
         // MTS-like stuff

         // parameters
         /** @brief Shear modulus at reference conditions, μ_ref [stress units]; not currently varied for current conditions. */
         double m_mu_ref; // may evetually set for current conditions
         /** @brief Reference temperature, T_ref [Kelvin]. */
         double m_tkelv_ref;
         /** @brief Reference stress τ_a; plays the role of either the athermal stress floor or the MTS-normalizing stress in the slip-rate law, depending on `withGAthermal` (see the file-level documentation). */
         double m_tau_a; // if withGAthermal then is Peierls barrier
         /** @brief MTS activation-energy exponent p; only used if `pOne` is false (otherwise p is taken to be exactly 1). */
         double m_p; // only used if pOne is false
         /** @brief MTS activation-energy exponent q; only used if `qOne` is false (otherwise q is taken to be exactly 1). */
         double m_q; // only used if qOne is false
         /** @brief Reference-rate coefficient γ̇_r0 for the drag-limited branch. */
         double m_gam_ro;
         /** @brief Reference-rate coefficient γ̇_w0 for the thermally-activated branch. */
         double m_gam_wo; // adots0
         /** @brief Per-group thermal-activation energy scale numerator C1ᵢ. */
         double m_c_1[nVPer];
         /** @brief Drag stress scale w_rD. */
         double m_wrD;
         /** @brief Per-group base stress g0ᵢ and dislocation-density sensitivity coefficient sᵢ, combining to form the per-group reference stress ĝᵢ = g0ᵢ + sᵢ·√ρ (see getVals). */
         double m_go[nVPer], m_s[nVPer];

         // derived from parameters
         /** @brief Per-group overflow/underflow stress-ratio thresholds (mirroring #gam_ratio_min, #gam_ratio_ovf) and power-law exponent helpers `xnn = 1/xm`, `xn = xnn - 1`, where the effective rate-sensitivity exponent xm is derived from the reference MTS parameters (see the file-level documentation). */
         double m_t_max[nVPer], m_t_min[nVPer], m_xn[nVPer], m_xnn[nVPer];

         //////////////////////////////
         // Kocks-Mecking stuff

         /** @brief Kocks-Mecking hardening parameters: multiplication-rate coefficient k1, reference recovery-rate coefficient k2_0, recovery rate-sensitivity exponent 1/n (`m_ninv`), and reference shear rate γ̇_0 for the recovery term. */
         double m_k1, m_k2o, m_ninv, m_gamma_o;

         //////////////////////////////

         /** @brief Initial dislocation density and its floor (`m_hdn_min = 1e-4 * m_hdn_init`). */
         double m_hdn_init, m_hdn_min;

      public:

         /**
          * @brief Reference slip rate used for scaling elsewhere in the solve (e.g. by
          * the evptn elastic-strain/rotation solver): the harmonic mean of the
          * thermally-activated and drag-limited reference rates, matching the same
          * combination rule used for the actual slip rate in evalGdot.
          * @param vals Kinetic values from getVals; `vals[0]` = γ̇_w, `vals[1]` = γ̇_r.
          * @return `1 / (1/γ̇_w + 1/γ̇_r)`.
          */
         __ecmech_hdev__
         inline
         double
         getFixedRefRate(const double* const vals) const
         {
            return 1.0 / (1.0 / vals[0] + 1.0 / vals[1]);
         }

         /**
          * @brief Precompute the kinetic values used by evalGdots: the thermal and
          * drag-limited reference rates, and the per-group reference stress and thermal
          * energy scale -- see the "Kinetic values" equations in the file-level
          * documentation in ECMech_kinetics_KMBalD.h.
          *
          * Could eventually bring in additional pressure and temperature dependence
          * through the dependence of `m_mu_ref` on such conditions.
          * @param[out] vals Kinetic values, length #nVals: `vals[0]` = γ̇_w, `vals[1]` =
          * γ̇_r, `vals[2+i]` = ĝᵢ, `vals[2+nVPer+i]` = c_tᵢ.
          * @param p Pressure; not currently used by this model.
          * @param tkelv Temperature [Kelvin], T.
          * @param[in] h_state Current hardening state: `h_state[0]` = relative/
          * normalized (dimensionless) dislocation density ρ.
          * @return The average per-group reference stress ĝ across all groups.
          */
         __ecmech_hdev__
         inline
         double
         getVals(double* const vals, // [nVals]
                 double, // p, not used
                 double tkelv,
                 const double* const h_state
                 ) const
         {
            double const nVPerInv = 1.0 / nVPer;

            // double sqrtDDens = exp(onehalf * h_state[0]) ; // this is for h_state[0] storing the log of the dislocation density
            double sqrtDDens = sqrt(h_state[0]);

            vals[0] = m_gam_wo / sqrtDDens; // _gam_w
            vals[1] = m_gam_ro * sqrtDDens * sqrtDDens; // _gam_r

            double hdnScale = 0.;
            for (int iVal = 0; iVal<nVPer; ++iVal) {
               double hdnI = m_go[iVal] + m_s[iVal] * sqrtDDens; // _gAll
               hdnScale += hdnI;
               vals[2 + iVal] = hdnI;
               vals[2 + nVPer + iVal] = m_c_1[iVal] / tkelv; // _c_t
               if (!withGAthermal) {
                  assert(vals[2 + iVal] > zero);
               }
            }

            hdnScale = hdnScale * nVPerInv;

            if (withGAthermal) {
               assert(m_tau_a > 0);
            }

            return hdnScale;
         }

         /**
          * @brief Evaluate the slip rate and its derivative w.r.t. resolved shear stress
          * on every slip system.
          * @param[out] gdot Slip rate on each slip system γ̇ [1/time].
          * @param[out] dgdot_dtau Derivative of slip rate w.r.t. resolved shear stress on
          * each slip system.
          * @param[in] tau Resolved shear stress on each slip system τ [stress units].
          * @param[in] vals Kinetic values from getVals.
          * @tparam R value type: double (one point) or a pack of points (ECMech_value_type.h)
          */
         template<class R>
         __ecmech_hdev__
         inline
         void
         evalGdots(R* const gdot,
                   R* const dgdot_dtau,
                   const R* const tau,
                   const R* const vals
                   ) const
         {
            for (int iSlip = 0; iSlip < this->nslip; ++iSlip) {
               mask_t<R> l_act;
               this->evalGdot(gdot[iSlip], l_act, dgdot_dtau[iSlip],
                              vals, iSlip,
                              tau[iSlip],
                              m_mu_ref // gss%ctrl%mu(islip)
                              );
            }
         }

         /**
          * @brief Evaluate the MTS activation-energy function E(t) and its derivative
          * factor, for a single (signed) dimensionless MTS argument -- see the "MTS
          * thermal-activation energy function" equation in the file-level documentation
          * in ECMech_kinetics_KMBalD.h.
          *
          * Handles three regimes: `t` near zero (linearizes to avoid a 0/0 in the
          * derivative when `pOne` is false), `q_arg = 1 - p_func` at or below zero
          * (barrier fully overcome -- "pegged" to `E = 0`), and the general case.
          * @param[out] exp_arg E(t), the log of the thermal-activation rate factor.
          * @param[out] mts_dfac Derivative helper factor; combined with the caller's own
          * chain-rule terms to get `d(exp(E))/dτ`.
          * @param c_e Thermal energy prefactor c_e (= c_t·μ).
          * @param denom_i 1/g_MTS (see the file-level documentation), the reciprocal of
          * whichever stress is currently playing the MTS-normalizing role.
          * @param t_frac The dimensionless MTS argument t.
          * @param active The points that need the result (true for one point);
          * pow() is evaluated only for them.
          * @tparam R value type: double (one point) or a pack of points (ECMech_value_type.h)
          */
         template<class R>
         __ecmech_hdev__
         inline
         void
         get_mts_dG(R &exp_arg,
                    R &mts_dfac,
                    const R& c_e, const R& denom_i,
                    const R& t_frac,
                    const mask_t<R>& active) const {
            mts_dfac = c_e * denom_i;

            R p_func = t_frac;
            if (!pOne) {
               const auto tiny = (fabs(t_frac) < idp_tiny_sqrt);
               if (any(tiny)) {
                  // !! p_dfac is either zero or blows up
                  // !IF (pl%p > one) THEN ! no longer allowed
                  // !   mts_dfac = zero
                  // !ELSE
                  // ! blows up, but just set big
                  set_where(tiny, p_func, zero);
                  set_where(tiny, mts_dfac, mts_dfac * 1e10);
               }
               if (m_p != one) { // the m_p == one case keeps p_func = t_frac
                  const auto needs_pow = active && !tiny;
                  if (any(needs_pow)) {
                     const R t_frac_safe = select(needs_pow, t_frac, R(one));
                     R t_frac_pow_p = pow_where(needs_pow, fabs(t_frac_safe), m_p);
                     t_frac_pow_p = copysign(t_frac_pow_p, t_frac_safe);
                     set_where(needs_pow, p_func, t_frac_pow_p);
                     set_where(needs_pow, mts_dfac, mts_dfac * m_p * t_frac_pow_p / t_frac_safe); // always positive
                  }
               }
            }

            const R q_arg = one - p_func;
            const auto q_arg_small = (q_arg < idp_tiny_sqrt);
            R pq_fac = q_arg;
            if (!qOne && m_q != one) {
               const auto needs_pow = active && !q_arg_small;
               if (any(needs_pow)) {
                  const R q_arg_safe = select(needs_pow, q_arg, R(one));
                  const R temp = pow_where(needs_pow, fabs(q_arg_safe), m_q);
                  set_where(needs_pow, mts_dfac, mts_dfac * m_q * temp / fabs(q_arg_safe));
                  set_where(needs_pow, pq_fac, copysign(temp, q_arg_safe));
               }
            }
            if (any(q_arg_small)) {
               // peg
               set_where(q_arg_small, mts_dfac, zero);
               set_where(q_arg_small, pq_fac, zero);
            }

            exp_arg = -c_e * pq_fac;
         }

         /**
          * @brief Evaluate the balanced thermally-activated + power-law-tail slip rate,
          * combined by harmonic mean with the drag-limited rate, for a single slip
          * system -- see the "Slip-rate law" equations and symbol table in the
          * file-level documentation in ECMech_kinetics_KMBalD.h.
          *
          * Assigns the athermal floor and MTS-normalizing stress from `withGAthermal`,
          * then: if the drag-limited argument is negligible, the system is inactive; if
          * the (clamped) forward MTS argument exceeds the overflow threshold `t_max`,
          * the rate is purely drag-limited; otherwise the forward (and, unless
          * negligible, reverse) thermally-activated rate is evaluated via get_mts_dG,
          * the power-law tail is added once the forward argument exceeds the underflow
          * threshold `t_min`, and the thermal+power-law rate is combined with the
          * drag-limited rate by harmonic mean.
          * @param[out] gdot Slip rate γ̇ [1/time].
          * @param[out] l_act `true` if the slip system is active.
          * @param[out] dgdot_dtau Derivative of `gdot` w.r.t. resolved shear stress.
          * @param[in] vals Kinetic values from getVals.
          * @param iSlip Slip system index (selects which parameter group to use when
          * `perSS`).
          * @param tau Resolved shear stress τ [stress units].
          * @param mu Shear modulus, used to form the thermal energy prefactor c_e.
          * @note For a pack, each early return of the scalar law is a per-point "done"
          * mask; for double these are the original returns.
          * @tparam R value type: double (one point) or a pack of points (ECMech_value_type.h)
          */
         template<class R>
         __ecmech_hdev__
         inline
         void
         evalGdot(
            R & gdot,
            mask_t<R> & l_act,
            R & dgdot_dtau, // wrt resolved shear stress
            const R* const vals,
            int      iSlip,
            const R& tau,
            double   mu
            ) const
         {
            static const double gdot_w_pl_scaling = 10.0;
            static const double one = 1.0, zero = 0.0;
            const R gam_w = vals[0];
            const R gam_r = vals[1];
            const int iVal = perSS ? iSlip : 0;
            const R gIn = vals[2 + iVal];
            const R c_t = vals[2 + nVPer + iVal];
            const double xn = m_xn[iVal];
            const double xnn = m_xnn[iVal];
            const double t_max = m_t_max[iVal];
            const double t_min = m_t_min[iVal];

            // zero things so that can more easily just return if inactive
            gdot = zero;
            dgdot_dtau = zero;
            l_act = no_lanes<R>();
            // points whose result is final (each "return" of the scalar law)
            mask_t<R> done = no_lanes<R>();

            R g_i;
            R gAth;
            if (withGAthermal) {
               gAth = gIn;
               g_i = one / m_tau_a;
            }
            else {
               gAth = m_tau_a;
               done = (tau == zero);
               if (all(done)) {
                  return;
               }
               g_i = one / select(done, R(one), gIn);
            }

            const R at_0 = fmax(R(zero), fabs(tau) - gAth) * g_i;

            // calculate drag limited kinetics
            //
            R gdot_r, dgdot_r_dtau;
            {
               const R exp_arg = (fabs(tau) - gAth) / m_wrD;
               // note that this should catch tau <= g
               done = done || (exp_arg < gam_ratio_min);
               if (all(done)) {
                  // gdot_r is zero
                  return;
               }
               R temp;
               const auto small_exp_arg = (exp_arg < idp_eps_sqrt);
               if (any(small_exp_arg)) {
                  // linear expansion is cheaper and more accurate
                  set_where(small_exp_arg, gdot_r, gam_r * exp_arg);
                  set_where(small_exp_arg, temp, one - exp_arg);
               }
               if (any(!small_exp_arg)) {
                  const R exp_neg_arg = exp_where(!done && !small_exp_arg, -exp_arg);
                  set_where(!small_exp_arg, temp, exp_neg_arg);
                  set_where(!small_exp_arg, gdot_r, gam_r * (one - exp_neg_arg));
               }
               dgdot_r_dtau = gam_r * temp / m_wrD;
            }
            {
               const auto drag = !done && (at_0 > t_max);
               if (any(drag)) {
                  // have overflow of thermally activated kinetics, purely drag limited
                  set_where(drag, gdot, copysign(gdot_r, tau));
                  set_where(drag, dgdot_dtau, dgdot_r_dtau);
                  l_act = l_act || drag;
                  done = done || drag;
                  if (all(done)) {
                     return;
                  }
               }
            }

            R gdot_w, dgdot_w_dtau;
            //
            // calculate thermally activated kinetics
            {
               const R c_e = c_t * mu;
               const R t_frac = (fabs(tau) - gAth) * g_i;
               R exp_arg, mts_dfac;
               get_mts_dG(exp_arg, mts_dfac, c_e, g_i, t_frac, !done);

               done = done || (exp_arg < ln_gam_ratio_min);
               if (all(done)) {
                  // effectively zero due to thermally activated kinetics
                  return;
               }

               gdot_w = gam_w * exp_where(!done, exp_arg);
               dgdot_w_dtau = mts_dfac * gdot_w;

               const R t_frac_m = (-fabs(tau) - gAth) * g_i;
               R exp_arg_m, mts_dfac_m;
               get_mts_dG(exp_arg_m, mts_dfac_m, c_e, g_i, t_frac_m, !done);

               const auto has_back_term = !done && (exp_arg_m > ln_gam_ratio_min);
               if (any(has_back_term)) {
                  // non-vanishing contribution from balancing MTS-like kinetics
                  const R gdot_w_m = gam_w * exp_where(has_back_term, exp_arg_m);
                  set_where(has_back_term, gdot_w, gdot_w - gdot_w_m);
                  const R contrib = mts_dfac_m * gdot_w_m;
                  // sign used to be the other way, but suspect that was a bug
                  set_where(has_back_term, dgdot_w_dtau, dgdot_w_dtau - contrib);
                  done = done || (has_back_term && (fabs(gdot_w / gam_w) < gam_ratio_min));
                  if (all(done)) {
                     // effectively zero from roundoff
                     return;
                  }
               }
            }

            const auto power_law = !done && (at_0 > t_min);
            if (any(power_law)) {
               // need power-law part
               const R abslog = log_where(power_law, at_0);
               const R blog = xn * abslog;
               const R temp = (gam_w * gdot_w_pl_scaling) * exp_where(power_law, blog);

               const R gdot_w_pl = temp * at_0; // not signed
               set_where(power_law, gdot_w, gdot_w + gdot_w_pl);

               const R contrib = temp * xnn * g_i;
               set_where(power_law, dgdot_w_dtau, dgdot_w_dtau + contrib);
            }

            const auto not_done = !done;
            l_act = l_act || not_done;
            {
               const R gdot_combined = one / (one / gdot_w + one / gdot_r);
               const R gdrdiv2 = one / (gdot_r * gdot_r);
               const R gdwdiv2 = one / (gdot_w * gdot_w);
               set_where(not_done, dgdot_dtau, (gdot_combined * gdot_combined) * (dgdot_w_dtau * gdwdiv2 + dgdot_r_dtau * gdrdiv2));
               set_where(not_done, gdot, copysign(gdot_combined, tau));
            }
         } // evalGdot

         /**
          * @brief Advance the single relative/normalized dislocation-density hardening
          * state one time step by delegating to the shared scalar-hardness SNLS solve,
          * solving in log space so the density cannot go negative.
          * @param[out] hs_u End-of-step relative dislocation density, `hs_u[0]`.
          * @param[in] hs_o Start-of-step dislocation density, `hs_o[0]`; floored at
          * #m_hdn_min before taking the log.
          * @param dt Time step size.
          * @param[in] gdot Slip rates on all slip systems, used to compute the evolution
          * inputs via getEvolVals.
          * @param hvals Unused by this model.
          * @param tkelv Temperature [Kelvin].
          * @param outputLevel Verbosity passed through to the SNLS solver.
          * @return Function-evaluation count from updateH1, or a negative value if the
          * solve failed to converge.
          * @see updateH1 in ECMech_kinetics.h
          */
         __ecmech_hdev__
         inline
         int
         updateH(double* const hs_u,
                 const double* const hs_o,
                 double dt,
                 const double* const gdot,
                 const double* const /*hvals*/,
                 double tkelv,
                 int outputLevel = 0) const
         {
            // do not yet both with l_overdriven and setting-to-saturation machinery as in Fortran coding

            // update is done on log(h) -- h treated as a nomralized (unitless) dislocation density
            double log_hs_u;
            double log_hs_o = log(fmax(hs_o[0], m_hdn_min));
            int nFEvals = updateH1<KineticsKMBalD>(this,
                                                   log_hs_u, log_hs_o, dt, gdot, tkelv,
                                                   outputLevel);
            hs_u[0] = exp(log_hs_u);

            return nFEvals;
         }

         /**
          * @brief Precompute the effective shear rate γ̇_eff and rate-dependent
          * recovery coefficient k2 used by getSdot1 (see the "Hardening law" equations
          * in the file-level documentation in ECMech_kinetics_KMBalD.h).
          * @param[out] evolVals `evolVals[0]` = γ̇_eff (sum of absolute slip rates
          * across all slip systems); `evolVals[1]` = k2 (`m_k2o` if the effective shear
          * rate is negligible).
          * @param[in] gdot Slip rates on all slip systems [1/time].
          */
         __ecmech_hdev__
         inline
         void
         getEvolVals(double* const evolVals,
                     const double* const gdot
                     ) const
         {
            // recompute effective shear rate here versus using a stored value
            double shrate_eff = vecsssumabs_n(gdot, nslip); // could switch to template if template class on nslip

            double k2 = m_k2o;
            if (shrate_eff > ecmech::idp_tiny_sqrt) {
               k2 = m_k2o * pow((m_gamma_o / shrate_eff), m_ninv);
            }

            evolVals[0] = shrate_eff;
            evolVals[1] = k2;
         }

         /**
          * @brief Evaluate the Kocks-Mecking hardening rate (in log space) and its
          * derivative w.r.t. the hardening state, for the scalar-hardness SNLS solve:
          *
          *   d(ln ρ)/dt = (k1/√ρ - k2) · γ̇_eff
          *
          * where ρ = exp(h) (h being the log-space hardening state actually solved
          * for). See the file-level documentation in ECMech_kinetics_KMBalD.h for the
          * full symbol-to-code mapping.
          * @param[out] sdot Hardening rate d(ln ρ)/dt.
          * @param[out] dsdot_ds Derivative of `sdot` w.r.t. `h`.
          * @param h Current hardening state, in log space (ln ρ).
          * @param[in] evolVals Values from getEvolVals: `evolVals[0]` = γ̇_eff
          * (effective shear rate), `evolVals[1]` = k2 (recovery-rate coefficient).
          * @param tkelv Temperature [Kelvin]; not currently used by this model.
          */
         __ecmech_hdev__
         inline
         void
         getSdot1(double &sdot,
                  double &dsdot_ds,
                  double h,
                  const double* const evolVals,
                  double /*tkelv*/
                  ) const
         {
            double shrate_eff = evolVals[0];
            double k2 = evolVals[1];
            double temp_hs_a = exp(-onehalf * h);
            double temp1 = m_k1 * temp_hs_a - k2;
            sdot = temp1 * shrate_eff;
            dsdot_ds = (-m_k1 * onehalf * temp_hs_a) * shrate_eff;
            // }
         }
   }; // class KineticsKMBalD
} // namespace ecmech

#endif // ECMECH_KINETICS_KMBALD_H
