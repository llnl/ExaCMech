#pragma once
/**
 * @file ECMech_evptnLanes.h
 * @brief The one getResponse driver: getResponseLanes<R> for one point
 * (R = double) or a pack of points, and getResponseSngl on top of it.
 *
 * Included at the end of ECMech_evptnSngl.h, whose ProblemState, preprocess and
 * postprocess it uses; include that header, not this one.
 */

#include <new>
#include <type_traits>
#include "SNLS_TrDLDenseG.h"

namespace ecmech {
namespace evptn {

/**
 * @brief The fields of evptn::ProblemState that EvptnUpdstProblem reads, for a
 * pack of points: lane l of each field holds point l's value.
 *
 * @tparam Kinetics kinetics class (for the hardening state size nH)
 * @tparam R value type: a pack of points (see ECMech_value_type.h)
 */
template<class Kinetics, class R>
struct ProblemStatePack {
    /** @brief The value type EvptnUpdstProblem is templated on */
    using value_type = R;
    /** @brief Time step (the same for every point) */
    double dt;
    /** @brief Relative volume, internal energy, EOS pressure and temperature at the end of the step */
    R rel_vol_new, energy_new, pressure_EOS, tkelv;
    /** @brief Deviatoric elastic strain at the start of the step */
    R elast_d5_n[ecmech::ntvec];
    /** @brief Lattice orientation (quaternion) at the start of the step */
    R quat_n[ecmech::qdim];
    /** @brief Deviatoric deformation rate in the sample frame */
    R def_rate_d5_sample[ecmech::ntvec];
    /** @brief Spin vector in the sample frame */
    R spin_vec_sample[ecmech::nwvec];
    /** @brief Hardening state at the end of the step */
    R h_state_u[(Kinetics::nH > 0) ? Kinetics::nH : 1];
};

/// True if the slip geometry SG computes the resolved shear stresses from the
/// stress alone (SG::rss_from_stress), so one point's geometry serves a pack
template<class SG, class = void> struct has_rss_from_stress : std::false_type {};
template<class SG> struct has_rss_from_stress<SG, std::void_t<decltype(SG::rss_from_stress)>> : std::true_type {};

/**
 * @brief A dynamic (stress-dependent) slip geometry for a pack of points.
 *
 * Each point's P and Q matrices and extra values, as its own geometry computed
 * them in preprocess, gathered into packs, behind the interface the templated
 * problem uses (nslip, nSlipExtra, dynamic, getP, getQ, evalRSS, getExtras).
 *
 * @tparam SG the slip geometry class of one point
 * @tparam R value type: a pack of points
 */
template<class SG, class R>
struct SlipGeomPack {
    /** @brief Number of slip systems */
    static constexpr int nslip = SG::nslip;
    /** @brief Number of extra slip quantities (SG::getExtras) */
    static constexpr int nSlipExtra = SG::nSlipExtra;
    /** @brief Stress-dependent geometry (as SG) */
    static constexpr bool dynamic = SG::dynamic;
    /** @brief One point's geometry (for projections that depend on the stress only) */
    const SG* geom_0 = nullptr;
    /** @brief P (symmetric Schmid tensors, ntvec x nslip), per lane */
    R p_vec[ecmech::ntvec * nslip];
    /** @brief Q (skew Schmid tensors, nwvec x nslip), per lane */
    R q_vec[ecmech::nwvec * nslip];
    /** @brief The geometry's extra values (SG::getExtras), per lane */
    R extras[(nSlipExtra > 0) ? nSlipExtra : 1];

    /**
     * @brief Gather point l's geometry into lane l.
     * @param[in] l lane
     * @param[in] g point l's slip geometry (after its preprocess)
     */
    __ecmech_hdev__ void gather_lane(int l, const SG& g) {
        if (!geom_0) { geom_0 = &g; }
        const double* const P = g.getP();
        const double* const Q = g.getQ();
        for (int k = 0; k < ecmech::ntvec * nslip; ++k) { set_lane(p_vec[k], l, P[k]); }
        for (int k = 0; k < ecmech::nwvec * nslip; ++k) { set_lane(q_vec[k], l, Q[k]); }
        if constexpr (nSlipExtra > 0) {
            double e[nSlipExtra];
            g.getExtras(e);
            for (int k = 0; k < nSlipExtra; ++k) { set_lane(extras[k], l, e[k]); }
        }
    }
    /** @brief P per lane, as SG::getP */
    __ecmech_hdev__ const R* getP() const { return p_vec; }
    /** @brief Q per lane, as SG::getQ */
    __ecmech_hdev__ const R* getQ() const { return q_vec; }
    /**
     * @brief Resolved shear stresses, as SG::evalRSS: from the stress alone if
     * the geometry supports it (has_rss_from_stress), else P^T kirchoff per lane.
     * @tparam T value type of kirchoff (double or R)
     */
    template<class T>
    __ecmech_hdev__ void evalRSS(R* taua, const T* const kirchoff, const R* P_vec) const {
        if constexpr (has_rss_from_stress<SG>::value) {
            geom_0->evalRSS(taua, kirchoff, (const double*)nullptr);
        }
        else {
            vecsVaTM<ecmech::ntvec, nslip>(taua, kirchoff, P_vec);
        }
    }
    /** @brief The extra values per lane, as SG::getExtras */
    __ecmech_hdev__ void getExtras(R* const e) const {
        for (int k = 0; k < nSlipExtra; ++k) { e[k] = extras[k]; }
    }
};

/// True if kinetics class K's slip rates (evalGdots) are templated on the value
/// type R: such models run on packs (simd::simd_supported)
template<class K, class R, class = void> struct has_pack_slip_rates : std::false_type {};
template<class K, class R>
struct has_pack_slip_rates<K, R, std::void_t<decltype(std::declval<const K&>().evalGdots(
    (R*)nullptr, (R*)nullptr, (const R*)nullptr, (const R*)nullptr))>> : std::true_type {};
/// True if kinetics class K's hardening update can run on packs (K::updateHLanes)
template<class K, class R, class = void> struct has_pack_hardening : std::false_type {};
template<class K, class R>
struct has_pack_hardening<K, R, std::void_t<decltype(std::declval<const K&>().updateHLanes(
    (R*)nullptr, (const R*)nullptr, 0.0, (const R*)nullptr))>> : std::true_type {};

/**
 * @brief The point-wise arrays and strides of one getResponseECM call (all
 * points, or one point with every stride 0), as one argument for
 * getResponseLanes. Each array and stride is the getResponseECM argument of the
 * same name; point i's data start at array[stride * i].
 */
struct ResponseArrays {
    double dt;                                                     ///< time step
    double tolerance;                                              ///< Newton solve tolerance
    const double* def_rate_d6vV;   unsigned int def_rate_stride;   ///< deformation rate (d6v)
    const double* spin_vecV;       unsigned int spin_v_stride;     ///< spin vector
    const double* rel_vol_ratiosV; unsigned int vol_ratio_stride;  ///< relative volume ratios
    double* internal_energyV;      unsigned int int_eng_stride;    ///< internal energy (in/out)
    double* cauchy_stress_d6pV;    unsigned int stress_stride;     ///< Cauchy stress (d6p, in/out)
    double* histV;                 unsigned int history_stride;    ///< history variables (in/out)
    double* tkelvV;                unsigned int tkelv_stride;      ///< temperature (in/out)
    double* sddV;                  unsigned int sdd_stride;        ///< sdd output
    double* mtanSDV;                                               ///< tangent stiffness (nsvec2 per point), or nullptr
    int nPassed;                                                   ///< number of points
};

/** @brief Longest history the padding scratch of getResponseLanes holds (checked before the launch) */
constexpr int max_pack_hist = 128;

/**
 * @brief The per-point problem states of getResponseLanes.
 *
 * ProblemState has reference members, so it has no default constructor. For
 * one point (the GPU and the one-point CPU path) the state is an ordinary
 * member, constructed directly from the point's arrays: no placement new on the
 * GPU. For a pack (CPU only) the states are built per lane in raw storage (see
 * the specialization below).
 *
 * @tparam ProblemStatePoint one point's evptn::ProblemState
 * @tparam single_point one point (true) or a pack (false)
 * @tparam n_lanes lanes of the pack
 */
template<class ProblemStatePoint, bool single_point, int n_lanes>
struct PointStates {
    /** @brief The point's state */
    ProblemStatePoint state;
    /** @brief Construct the point's state (the ProblemState constructor's arguments) */
    template<class... Args>
    __ecmech_hdev__ explicit PointStates(Args&&... args) : state(args...) {}
    /** @brief The state of lane l (one point: l = 0) */
    __ecmech_hdev__ ProblemStatePoint* operator[](int) { return &state; }
};

/**
 * @brief Pack specialization (CPU only): raw storage per lane, constructed by
 * construct() in the gather loop and destroyed by destroy() after the scatter.
 */
template<class ProblemStatePoint, int n_lanes>
struct PointStates<ProblemStatePoint, false, n_lanes> {
    /** @brief Storage of the lanes' states */
    alignas(ProblemStatePoint) unsigned char mem[n_lanes][sizeof(ProblemStatePoint)];
    /** @brief The lanes' states (in mem) */
    ProblemStatePoint* states[n_lanes];
    PointStates() = default;
    /** @brief Construct lane l's state in place (the ProblemState constructor's arguments) */
    template<class... Args>
    ProblemStatePoint* construct(int l, Args&&... args) { return states[l] = new (mem[l]) ProblemStatePoint(args...); }
    /** @brief Destroy lane l's state */
    void destroy(int l) { states[l]->~ProblemStatePoint(); }
    /** @brief The state of lane l */
    ProblemStatePoint* operator[](int l) { return states[l]; }
};

/**
 * @brief The PointStates of getResponseLanes: for one point, its state built
 * from the arguments (the ProblemState constructor's); for a pack, the empty
 * storage (the arguments are not used). Returned as a prvalue, so the state is
 * constructed in the caller's variable (no copy; C++17).
 */
template<class ProblemStatePoint, bool single_point, int n_lanes, class... Args>
__ecmech_hdev__ inline PointStates<ProblemStatePoint, single_point, n_lanes> make_point_states(Args&&... args)
{
    if constexpr (single_point) { return PointStates<ProblemStatePoint, true, n_lanes>(args...); }
    else { return PointStates<ProblemStatePoint, false, n_lanes>(); }
}

/**
 * @brief getResponse for points i_pack*n_lanes .. i_pack*n_lanes+n_lanes-1, n_lanes = lanes of the value
 * type R: one point for R = double (the one-point CPU path and one GPU thread
 * per point), n_lanes points in lockstep for a pack (the CPU SIMD path).
 *
 * The steps: per point, ProblemState and preprocess (EOS, hardness update);
 * then the Newton solve of EvptnUpdstProblem with SNLS's trust-region dogleg on
 * R; the tangent; the state update; per point, postprocess. For a pack the
 * per-point inputs are gathered into a pack problem state and the results
 * scattered back; for R = double the problem works on the point's own
 * ProblemState and the gather/scatter steps compile away (`if constexpr`). A
 * pack whose last lanes run past nPassed repeats the last point on scratch
 * copies.
 *
 * @tparam R value type: double (one point) or a pack of points
 * @tparam SlipGeom, Kinetics, ThermoElastN, EosModel the model's classes
 * @param[in] i_pack index of the pack: points i_pack * n_lanes .. i_pack * n_lanes + n_lanes - 1
 * @param[in] slipGeom, kinetics, elastN, eos the model
 * @param[in,out] arrays the arrays of the getResponseECM call (ResponseArrays)
 * @return the points that converged (for R = double: a bool). The caller
 * marks failures (hist[iHistA_nFEval] *= -1), as for getResponseSngl.
 */
template<class R, class SlipGeom, class Kinetics, class ThermoElastN, class EosModel>
__ecmech_hdev__
inline mask_t<R> getResponseLanes(const int i_pack, const SlipGeom& slipGeom, const Kinetics& kinetics,
                                  const ThermoElastN& elastN, const EosModel& eos, const ResponseArrays& arrays)
{
    constexpr int n_lanes = lanes_of<R>();
    constexpr bool single_point = snls::is_scalar_value<R>();
    // problem states: one point's (ProblemStatePoint), and the one the solve
    // works on (ProblemStateR: the point's own for one point, else gathered lanes)
    using ProblemStatePoint = ProblemState<SlipGeom, Kinetics, ThermoElastN, EosModel>;
    using ProblemStateR = std::conditional_t<single_point, ProblemStatePoint, ProblemStatePack<Kinetics, R>>;
    constexpr bool dynamic_geom = SlipGeom::dynamic;
    // the slip geometry the solve works on: lanes gathered for a dynamic geometry
    using SlipGeomR = std::conditional_t<dynamic_geom && !single_point, SlipGeomPack<SlipGeom, R>, SlipGeom>;
    using Prob = EvptnUpdstProblem<SlipGeomR, Kinetics, ThermoElastN, ProblemStateR>;
    using Solver = snls::SNLSTrDlDenseG<Prob, Prob::nDimSys, R>;
    constexpr bool hardening_on_packs = !single_point && has_pack_hardening<Kinetics, R>::value;
    constexpr int nH = Kinetics::nH;
    constexpr int nslip = SlipGeom::nslip;
    constexpr int n = Prob::nDimSys;
    constexpr int n_pad_lanes = single_point ? 1 : n_lanes; // scratch for padding lanes (none for one point)
    const double dt = arrays.dt;

    // point index per lane; lanes past the end repeat the last point on copies
    int point_idx[n_lanes];
    bool in_range[n_lanes];
    for (int l = 0; l < n_lanes; ++l) {
        const int i = i_pack * n_lanes + l;
        in_range[l] = single_point || (i < arrays.nPassed);
        point_idx[l] = in_range[l] ? i : (arrays.nPassed - 1);
    }
    double* hist_l[n_lanes];
    double* stress_l[n_lanes];
    double* energy_l[n_lanes];
    double* sdd_l[n_lanes];
    double* tkelv_l[n_lanes];
    // scratch copies of a padding lane's in/out arrays
    [[maybe_unused]] double pad_hist[n_pad_lanes][single_point ? 1 : max_pack_hist];
    [[maybe_unused]] double pad_stress[n_pad_lanes][ecmech::nsvp];
    [[maybe_unused]] double pad_energy[n_pad_lanes][ecmech::ne];
    [[maybe_unused]] double pad_sdd[n_pad_lanes][ecmech::nsdd];
    [[maybe_unused]] double pad_tkelv[n_pad_lanes];
    alignas(SlipGeom) unsigned char geom_mem[(dynamic_geom && !single_point) ? n_lanes : 1][(dynamic_geom && !single_point) ? sizeof(SlipGeom) : 1];
    const SlipGeom* geom[n_lanes];
    bool preprocess_ok[n_lanes];
    double halfVMidDt[n_lanes];
    double dev_strain_energy_total[n_lanes];

    // per point: the arrays (scratch copies for padding lanes)
    for (int l = 0; l < n_lanes; ++l) {
        const int i = point_idx[l];
        hist_l[l] = &arrays.histV[arrays.history_stride * i];
        stress_l[l] = &arrays.cauchy_stress_d6pV[arrays.stress_stride * i];
        energy_l[l] = &arrays.internal_energyV[arrays.int_eng_stride * i];
        sdd_l[l] = &arrays.sddV[arrays.sdd_stride * i];
        tkelv_l[l] = &arrays.tkelvV[arrays.tkelv_stride * i];
        if constexpr (!single_point) {
            if (!in_range[l]) {
                const unsigned int n_hist = arrays.history_stride;
                for (unsigned int k = 0; k < n_hist; ++k) { pad_hist[l][k] = hist_l[l][k]; }
                for (int k = 0; k < ecmech::nsvp && k < (int)arrays.stress_stride; ++k) { pad_stress[l][k] = stress_l[l][k]; }
                for (int k = 0; k < ecmech::ne && k < (int)arrays.int_eng_stride; ++k) { pad_energy[l][k] = energy_l[l][k]; }
                for (int k = 0; k < ecmech::nsdd && k < (int)arrays.sdd_stride; ++k) { pad_sdd[l][k] = sdd_l[l][k]; }
                pad_tkelv[l] = *tkelv_l[l];
                hist_l[l] = pad_hist[l];
                stress_l[l] = pad_stress[l];
                energy_l[l] = pad_energy[l];
                sdd_l[l] = pad_sdd[l];
                tkelv_l[l] = &pad_tkelv[l];
            }
        }
    }
    // the problem states: one point's constructed directly, a pack's per lane below
    PointStates<ProblemStatePoint, single_point, n_lanes> point_states = make_point_states<ProblemStatePoint, single_point, n_lanes>
        (hist_l[0], stress_l[0], *tkelv_l[0], &arrays.def_rate_d6vV[arrays.def_rate_stride * point_idx[0]],
         &arrays.spin_vecV[arrays.spin_v_stride * point_idx[0]], &arrays.rel_vol_ratiosV[arrays.vol_ratio_stride * point_idx[0]], dt);

    // per point: preprocess
    for (int l = 0; l < n_lanes; ++l) {
        const int i = point_idx[l];
        if constexpr (!single_point) {
            point_states.construct(l, hist_l[l], stress_l[l], *tkelv_l[l], &arrays.def_rate_d6vV[arrays.def_rate_stride * i],
                                   &arrays.spin_vecV[arrays.spin_v_stride * i], &arrays.rel_vol_ratiosV[arrays.vol_ratio_stride * i], dt);
        }
        // dynamic slip geometries: a copy per lane of a pack; one point uses the
        // caller's geometry (already a copy per GPU thread, see getResponsePerPoint)
        if constexpr (dynamic_geom && !single_point) { geom[l] = new (geom_mem[l]) SlipGeom(slipGeom); }
        else { geom[l] = &slipGeom; }
        preprocess_ok[l] = preprocess<SlipGeom, Kinetics, EosModel, ThermoElastN, ProblemStatePoint, false, hardening_on_packs>
                               (*geom[l], kinetics, eos, elastN, &arrays.rel_vol_ratiosV[arrays.vol_ratio_stride * i],
                                energy_l[l], &arrays.def_rate_d6vV[arrays.def_rate_stride * i], *point_states[l],
                                halfVMidDt[l], dev_strain_energy_total[l]);
    }
    // the first lane whose preprocess succeeded: failed lanes run on its data
    int ref_lane = 0;
    while (ref_lane < n_lanes - 1 && !preprocess_ok[ref_lane]) { ++ref_lane; }

    mask_t<R> converged = no_lanes<R>();
    if constexpr (single_point) {
        // failed: nFEval >= 1 so that the caller's failure mark (nFEval *= -1)
        // makes it negative (with nFEval still 0, at the first step, it would not)
        if (!preprocess_ok[0]) { hist_l[0][iHistA_nFEval] = 1; return converged; }
    }

    // hardness update on the pack (KineticsOrowanD's reduced solve)
    if constexpr (hardening_on_packs) {
        R hs_o[nH], gdot_o[nslip], hs_u[nH];
        for (int l = 0; l < n_lanes; ++l) {
            const ProblemStatePoint& p = *point_states[preprocess_ok[l] ? l : ref_lane];
            for (int k = 0; k < nH; ++k) { set_lane(hs_o[k], l, p.h_state[k]); }
            for (int k = 0; k < nslip; ++k) { set_lane(gdot_o[k], l, p.gdot[k]); }
        }
        const mask_t<R> hardening_ok = kinetics.updateHLanes(hs_u, hs_o, dt, gdot_o);
        for (int l = 0; l < n_lanes; ++l) {
            if (!preprocess_ok[l]) { continue; }
            // as preprocess: "Hardening failed to converge"
            if (!mask_lane(hardening_ok, l)) {
                preprocess_ok[l] = false;
                continue;
            }
            for (int k = 0; k < nH; ++k) { point_states[l]->h_state_u[k] = lane(hs_u[k], l); }
        }
    }

    // the problem state of the pack: a point's own state, or the lanes gathered
    // (failed points ride along on a valid point's data)
    mask_t<R> active = all_lanes<R>();
    [[maybe_unused]] ProblemStatePack<Kinetics, R> state_pack;
    ProblemStateR* problem_state = nullptr;
    if constexpr (single_point) { problem_state = point_states[0]; }
    else {
        problem_state = &state_pack;
        problem_state->dt = dt;
        for (int l = 0; l < n_lanes; ++l) {
            if (!preprocess_ok[l]) { set_mask_lane(active, l, false); }
            const ProblemStatePoint& p = *point_states[preprocess_ok[l] ? l : ref_lane];
            set_lane(problem_state->rel_vol_new, l, p.rel_vol_new);
            set_lane(problem_state->energy_new, l, p.energy_new);
            set_lane(problem_state->pressure_EOS, l, p.pressure_EOS);
            set_lane(problem_state->tkelv, l, p.tkelv);
            for (int k = 0; k < ecmech::ntvec; ++k) {
                set_lane(problem_state->elast_d5_n[k], l, p.elast_d5_n[k]);
                set_lane(problem_state->def_rate_d5_sample[k], l, p.def_rate_d5_sample[k]);
            }
            for (int k = 0; k < ecmech::qdim; ++k) { set_lane(problem_state->quat_n[k], l, p.quat_n[k]); }
            for (int k = 0; k < ecmech::nwvec; ++k) { set_lane(problem_state->spin_vec_sample[k], l, p.spin_vec_sample[k]); }
            for (int k = 0; k < nH; ++k) { set_lane(problem_state->h_state_u[k], l, p.h_state_u[k]); }
        }
    }

    // the slip geometry of the pack
    const SlipGeomR* slip_geom_r = nullptr;
    [[maybe_unused]] SlipGeomPack<SlipGeom, R> slip_geom_pack;
    if constexpr (dynamic_geom && !single_point) {
        for (int l = 0; l < n_lanes; ++l) { slip_geom_pack.gather_lane(l, *geom[preprocess_ok[l] ? l : ref_lane]); }
        slip_geom_r = &slip_geom_pack;
    }
    else { slip_geom_r = geom[0]; }

    // the Newton solve (max_newton_iterations, initial trust region 1)
    snls::TrDeltaControl deltaControl;
    deltaControl._deltaInit = 1e0;
    Prob prob(*slip_geom_r, kinetics, elastN, *problem_state);
    Solver solver(prob);
    solver.setupSolver(max_newton_iterations, arrays.tolerance, &deltaControl);
    for (int k = 0; k < n; ++k) { solver._x[k] = 0.0; }
    solver.solve(active);
    converged = solver.getConverged();

    // tangent: one more evaluation at the converged x with the tangent output
    // on; then the state update and the values postprocessing needs
    R mtan_vecds[ecmech::nsvec2];
    if (arrays.mtanSDV) {
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
    const R hdn_scale = prob.getHdnScale();

    // per point: scale the tangent to an increment and add the EOS bulk term,
    // store the state and slip quantities, postprocess
    for (int l = 0; l < n_lanes; ++l) {
        if (!in_range[l] || !preprocess_ok[l] || !mask_lane(converged, l)) {
            if (in_range[l]) {
                if (preprocess_ok[l]) { set_mask_lane(converged, l, false); }
                // failed: nFEval >= 1 so that the caller's failure mark (nFEval *= -1)
                // makes it negative (with nFEval still 0, at the first step, it would not)
                const int f = preprocess_ok[l] ? solver.getNFEvals(l) : 0;
                hist_l[l][iHistA_nFEval] = (f > 0) ? f : 1;
            }
            if constexpr (!single_point) { point_states.destroy(l); }
            continue;
        }
        const int i = point_idx[l];
        ProblemStatePoint& p = *point_states[l];
        if (arrays.mtanSDV) {
            double m6[ecmech::nsvec2];
            for (int k = 0; k < ecmech::nsvec2; ++k) { m6[k] = lane(mtan_vecds[k], l) * prob.getDtRi(); }
            m6[ECMECH_NN_INDX(iSvecS, iSvecS, ecmech::nsvec)] += three * p.bulk_modulus_new;
            mtan_conv_sd_svec<true>(&arrays.mtanSDV[ecmech::nsvec2 * i], m6);
        }
        for (int k = 0; k < ecmech::ntvec; ++k) { p.elast_d5_u[k] = lane(elast_d5_u[k], l); }
        for (int k = 0; k < ecmech::qdim; ++k) { p.quat_u[k] = lane(quat_u[k], l); }
        hist_l[l][iHistA_nFEval] = solver.getNFEvals(l) + (arrays.mtanSDV ? 1 : 0);
        for (int k = 0; k < nH; ++k) { p.h_state[k] = p.h_state_u[k]; }
        for (int k = 0; k < nslip; ++k) { p.gdot[k] = lane(gdot[k], l); }
        p.eps_dot = lane(effective_shear_rate, l);
        p.eps += p.eps_dot * p.dt;
        {
            const double dEff = vecd_Deff(p.def_rate_d5_sample);
            double flow_strength = lane(hdn_scale, l);
            if (dEff > idp_tiny_sqrt) { flow_strength = lane(pl_disipation_rate, l) / dEff; }
            p.flow_strength = flow_strength;
        }
        double cauchy_stress_d5p_xtal[ecmech::nsvec];
        for (int k = 0; k < ecmech::nsvec; ++k) { cauchy_stress_d5p_xtal[k] = lane(cauchy_xtal[k], l); }
        postprocess(p, elastN, &arrays.def_rate_d6vV[arrays.def_rate_stride * i], sdd_l[l], energy_l[l],
                    cauchy_stress_d5p_xtal, dev_strain_energy_total[l], halfVMidDt[l]);
        if constexpr (!single_point) { point_states.destroy(l); }
    }
    if constexpr (dynamic_geom && !single_point) {
        for (int l = 0; l < n_lanes; ++l) { geom[l]->~SlipGeom(); }
    }
    return converged;
}

/**
 * @brief Complete time step integration for single material point (main interface).
 *
 * This is the primary entry point for crystal plasticity integration. It runs
 * getResponseLanes<double> on this point's arrays (see there for the steps):
 * preprocessing → nonlinear solve → post-processing.
 *
 * **Function signature summary**:
 * - Input: Beginning-of-step state, prescribed kinematics, material models, time step
 * - Output: End-of-step stress, updated history variables, optional tangent
 * - Return: Success/failure flag
 *
 * **Template parameters**:
 * @tparam SlipGeom Slip geometry class (e.g., SlipGeomFCC, SlipGeomBCC)
 * @tparam Kinetics Kinetics class (e.g., KineticsKMBalD, KineticsVocePL)
 * @tparam ThermoElastN Thermoelastic class (e.g., ThermoElastNCubic)
 * @tparam EosModel Equation of state class (e.g., EosModelConst)
 *
 * @param[in] slipGeom Slip geometry object defining slip systems
 * @param[in] kinetics Kinetics object for rate-dependent strength
 * @param[in] elastN Thermoelastic object for stress-strain relation
 * @param[in] eos Equation of state object for pressure-volume-energy
 * @param[in] dt Time step size [time units]
 * @param[in] tolerance Convergence tolerance for nonlinear solver [dimensionless]
 * @param[in] def_rate_d6v_sample Prescribed deformation rate [1/time, nsvp = 7 components]
 *                                 Only first ntvec = 5 used (deviatoric)
 * @param[in] spin_vec_sample Prescribed spin [1/time, ndim = 3 components]
 * @param[in] rel_vol_ratios Volume ratio array [nvr = 4 components]:
 *                           [0] = J_n, [1] = J_{n+1}, [2] = (J_{n+1}-J_n)/dt, [3] = J_{n+1}-J_n
 * @param[in,out] internal_energy Internal energy [energy/volume, ne = 1 component]
 *                                 Updated with plastic work contribution
 * @param[in,out] cauchy_stress_d6p Cauchy stress [stress units, nsvp = 7 components]
 *                                   Input: σ_n, Output: σ_{n+1}
 *                                   Layout: [σ11, σ22, σ33, σ23, σ13, σ12, -p]
 * @param[in,out] hist History variable array [numHist components]
 *                     Contains and updates: orientation, elastic strain, hardening, etc.
 * @param[in,out] tkelv Temperature [Kelvin]
 *                      May be updated by EOS evaluation
 * @param[out] sdd Auxiliary outputs [nsdd = 2 components]:
 *                 [0] = bulk modulus, [1] = shear modulus
 * @param[out] mtanSD Material tangent stiffness [nsvec2 = 36 components or nullptr]
 *                    If nullptr: Skip tangent computation
 *                    If non-null: Compute and store ∂σ/∂Δε
 * @param outputLevel unused (kept for interface compatibility)
 *
 * @return true if integration successful, false if solver failed to converge
 *
 * **History array layout**:
 * Defined by case-specific indices (iHistLbQ, iHistLbGdot, etc.):
 * - Orientation quaternion [4 components]
 * - Slip rates [nslip components]
 * - Elastic deviatoric strain [ntvec = 5 components]
 * - Hardening state [nH components]
 * - Equivalent plastic strain [1 component]
 * - Equivalent plastic strain rate [1 component]
 * - Flow strength [1 component]
 * - Number of function evaluations [1 component]
 *
 * **Convergence failure**:
 * - Returns false if preprocessing fails (hardening divergence)
 * - Returns false if main nonlinear solve fails to converge
 * - Host code should handle failure (reduce time step, flag element, etc.)
 *
 * @note Thread-safe: No static storage, can be called in parallel for different points
 * @note GPU-compatible: All functions marked __ecmech_hdev__
 * @note Const correctness: Material models passed by const reference
 *
 * @see getResponseECM() in ECMech_evptnWrap.h for vectorized batch interface
 * @see ProblemState constructor for history array interpretation
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
                     int /* outputLevel */ = 0)
{
    const ResponseArrays arrays{dt, tolerance, def_rate_d6v_sample, 0, spin_vec_sample, 0, rel_vol_ratios, 0,
                                internal_energy, 0, cauchy_stress_d6p, 0, hist, 0, &tkelv, 0, sdd, 0, mtanSD, 1};
    return getResponseLanes<double>(0, slipGeom, kinetics, elastN, eos, arrays);
}

} // namespace evptn
} // namespace ecmech
