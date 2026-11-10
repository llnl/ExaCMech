/**
 * @file ECMech_evptnWrap.h
 * @brief Template wrapper class for crystal plasticity material models.
 * 
 * This file provides the matModel template class that wraps specific crystal plasticity
 * constitutive models (slip geometry, kinetics, thermoelastic, EOS) into a unified
 * interface compatible with host finite element codes.
 * 
 * Key features:
 * - **Template-based composition**: Combines SlipGeom, Kinetics, ThermoElastN, EosModel
 * - **RAJA/GPU support**: Execution on CPU, GPU, or OpenMP via CHAI managed pointers
 * - **Factory pattern**: make_class_factory for GPU-compatible object creation
 * - **Batch processing**: Parallel evaluation of material points
 * - **SNLS integration**: Uses SNLS nonlinear solver for implicit integration
 * 
 * @see matModelBase for the base interface
 * @see SlipGeom classes for slip system geometry
 * @see Kinetics classes for rate-dependent strength
 */
// -*-c++-*-

#ifndef ecmech_evptnWrap_include
#define ecmech_evptnWrap_include

#include "ECMech_core.h"
#include "SNLS_config.h"
#include "SNLS_device_forall.h"
#include "RAJA/RAJA.hpp"

#if defined(SNLS_RAJA_PORT_SUITE)
#include "SNLS_memory_manager.h"
#include <chai/managed_ptr.hpp>
#endif

#if defined(__ecmech_host_only__)
#include <sstream>
#include <cstdlib>
#endif

#include "ECMech_matModelBase.h"
#include "evptn/ECMech_evptnSngl.h"
#if defined(ECMECH_SIMD_EXPT) || defined(ECMECH_GPU_PACK_EXPT)
#include "simd/ECMech_simd_evptn.h"
#endif

#include "ECMech_unused.h"

namespace ecmech {
namespace internal {
#if !defined(SNLS_RAJA_PORT_SUITE)
   /**
    * @brief Pseudo-CHAI managed pointer for CPU-only builds.
    * 
    * When SNLS_RAJA_PORT_SUITE is not defined, this provides a compatible
    * interface to chai::managed_ptr without actual GPU memory management.
    * Enables consistent code structure between CPU and GPU builds.
    */
   template<class T>
   class PseudoChaiManagedPtr {
      public:
         PseudoChaiManagedPtr() = default;
         template<typename ...Args>
         PseudoChaiManagedPtr(const double* const params, Args... args) : m_val(new T(params, std::forward<Args...>(args...))) {}
         PseudoChaiManagedPtr(const double* const params) : m_val(new T(params)) {}
         PseudoChaiManagedPtr(const PseudoChaiManagedPtr&) = default;
         ~PseudoChaiManagedPtr() = default;

         __ecmech_hdev__
         inline T& operator*() const { return *m_val; }
         __ecmech_hdev__
         inline void free() const { if (m_val) { delete m_val; } }

      public:
         T* m_val = nullptr;
   };

   template<class T>
   using pcmptr = PseudoChaiManagedPtr<T>;
#endif


   /**
    * @brief Factory function for creating GPU-compatible class instances with variadic arguments.
    * 
    * Creates instances of templated classes (SlipGeom, Kinetics, etc.) with managed memory
    * that is accessible from both CPU and GPU. Uses CHAI managed pointers when GPU support
    * is enabled, or simple pointers for CPU-only builds.
    * 
    * @tparam T Class type to instantiate
    * @tparam Args Variadic template for additional constructor arguments
    * @param[in] params Material parameters vector (copied to managed memory)
    * @param[in] args Additional constructor arguments forwarded to T
    * @return Managed pointer (chai::managed_ptr or pcmptr) to new instance
    */
   template<class T, typename ...Args>
   __ecmech_host__
   auto make_class_factory(const std::vector<double>& params, Args... args) {
#if defined(SNLS_RAJA_PORT_SUITE)
      auto mm = snls::memoryManager::getInstance();
      auto mvec = mm.allocManagedArray<double>(params.size());

      auto mvec_data = mvec.data(chai::ExecutionSpace::CPU);
      for (size_t i = 0; i < params.size(); i++ ) {
         mvec_data[i] = params[i];
      }

      return chai::make_managed<T>(chai::unpack(mvec), std::forward<Args...>(args...));
#else
      return pcmptr<T>(params.data(), std::forward<Args...>(args...));
#endif
   }

   /**
    * @brief Factory function for creating GPU-compatible class instances (params only).
    * 
    * Overload for classes that only require a parameter array in their constructor.
    * 
    * @tparam T Class type to instantiate
    * @param[in] params Material parameters vector
    * @return Managed pointer to new instance
    */
   template<class T>
   __ecmech_host__
   auto make_class_factory(const std::vector<double>& params) {
#if defined(SNLS_RAJA_PORT_SUITE)
      auto mm = snls::memoryManager::getInstance();
      auto mvec = mm.allocManagedArray<double>(params.size());

      auto mvec_data = mvec.data(chai::ExecutionSpace::CPU);
      for (size_t i = 0; i < params.size(); i++ ) {
         mvec_data[i] = params[i];
      }

      return chai::make_managed<T>(chai::unpack(mvec));
#else
      return pcmptr<T>(params.data());
#endif
   }

}
}

namespace ecmech {
   namespace evptn {
      /**
       * @class matModel
       * @brief Template class wrapping crystal plasticity constitutive models.
       * 
       * This template class composes four key components into a complete material model:
       * - **SlipGeom**: Slip system geometry (Schmid tensors, normals, directions)
       * - **Kinetics**: Rate-dependent strength evolution (Voce, KMBalD, etc.)
       * - **ThermoElastN**: Thermoelastic response (cubic, hexagonal symmetry)
       * - **EosModel**: Equation of state for pressure-volume-energy
       * 
       * **Template parameters**:
       * @tparam SlipGeom Slip geometry class (e.g., SlipGeomFCC, SlipGeomBCC)
       * @tparam Kinetics Kinetics class (e.g., Kin_Voce, Kin_KMBalD_FFF)
       * @tparam ThermoElastN Thermoelastic class (e.g., EVPTN_cubic, EVPTN_hex)
       * @tparam EosModel Equation of state class (e.g., EosModelConst<false>)
       * 
       * **Memory management**:
       * - Uses chai::managed_ptr for GPU builds (SNLS_RAJA_PORT_SUITE defined)
       * - Uses simple pointers for CPU-only builds
       * - Component objects created via make_class_factory for GPU compatibility
       * 
       * **Execution strategies**:
       * - CPU: Serial execution
       * - GPU: CUDA/HIP execution (if compiled with support)
       * - OPENMP: Multi-threaded CPU execution (if compiled with support)
       * 
       * **Typical usage**:
       * ```cpp
       * using MyModel = matModel<SlipGeomFCC, Kin_KMBalD_FFF, 
       *                          EVPTN_cubic, EosModelConst<false>>;
       * MyModel* model = new MyModel();
       * model->initFromParams(opts, pars, strs);
       * model->complete();
       * model->setExecutionStrategy(ECM_EXEC_STRAT_GPU);
       * model->getResponseECM(dt, ...);
       * ```
       */
      template<class SlipGeom, class Kinetics, class ThermoElastN, class EosModel>
      class matModel : public matModelBase
      {
         public:
            /**
             * @name Compile-Time Constants
             * @brief Static constants derived from template parameters.
             * @{
             */

            /** @brief Starting index of slip rate variables in history array */
            static constexpr int iHistLbGdot = NumHist<SlipGeom, Kinetics, ThermoElastN, EosModel>::iHistLbGdot;
            /** @brief Total number of history variables per integration point */
            static constexpr int numHist = NumHist<SlipGeom, Kinetics, ThermoElastN, EosModel>::numHist;
            /** @brief Number of hardening state variables from Kinetics component */
            static constexpr int nH = Kinetics::nH;
            /** @brief Number of slip systems from SlipGeom component */
            static constexpr int nslip = SlipGeom::nslip;
            /** @brief Number of EOS parameters obtained from other sources (density, bulk modulus, cvav) */
            static constexpr int nParamsEOSHave = 3;
            /** @brief Number of EOS parameters to provide in parameter vector */
            static constexpr int nParamsEOS = EosModel::nParams - nParamsEOSHave;
            /** @brief Number of slip geometry parameters */
            static constexpr int nParamsSlipGeom = SlipGeom::nParams;
            /** @brief Number of kinetics parameters */
            static constexpr int nParamsKinetics = Kinetics::nParams;
            /** @brief Number of thermoelastic parameters */
            static constexpr int nParamsThermoElastN = ThermoElastN::nParams;
            /** @brief Total number of parameters required */
            static constexpr int nParams =
               2 + 1 + // density0, cvav, tolerance
               nParamsSlipGeom + nParamsKinetics + nParamsThermoElastN + nParamsEOS;

            /** @} */

            /**
             * @name Constructors and Destructor
             * @{
             */

            /**
             * @brief Default constructor initializes stride configuration.
             * 
             * Sets up default array strides for vectorized operations. Model is not
             * functional until initFromParams() and complete() are called.
             */
            __ecmech_host__
            matModel()
               : matModelBase()
            {
               // Should the tangent stiff matrix be included in these stride calculations?
               m_strides[istride_def_rate] = ecmech::nsvp;
               m_strides[istride_spin_v] = ecmech::ndim;
               m_strides[istride_vol_ratio] = ecmech::nvr;
               m_strides[istride_int_eng] = ecmech::ne;
               m_strides[istride_stress] = ecmech::nsvp;
               m_strides[istride_history] = NumHist<SlipGeom, Kinetics, ThermoElastN, EosModel>::numHist;
               m_strides[istride_tkelv] = 1;
               m_strides[istride_sdd ] = ecmech::nsdd;
            }

            /**
             * @brief Constructor with custom strides.
             * 
             * Allows specification of custom memory layout for array indexing.
             * Useful for integration with host codes using specific data layouts.
             * 
             * @param[in] strides Vector of stride values [nstride elements]
             *                    - Must satisfy size == ECMECH_NSTRIDE
             *                    - Each stride >= corresponding minimum size
             * 
             * @see updateStrides() for stride details
             */
            __ecmech_host__
            matModel(std::vector<size_t> strides)
               : matModelBase()
            {
               updateStrides(strides);
            }

            /**
             * @brief Destructor - frees managed component objects.
             * 
             * Releases memory for:
             * - m_slipGeom (slip system geometry)
             * - m_kinetics (rate-dependent strength)
             * - m_eosModel (equation of state)
             * - m_elastN (thermoelastic response)
             * 
             * Uses .free() method which works for both chai::managed_ptr
             * and CPU-only pointers.
             */
            __ecmech_host__
            ~matModel()
            {
#if (defined(ECMECH_SNLS_TEAM) || defined(ECMECH_PERSISTENT_EXPT)) && defined(__ecmech_gpu_active__)
               if (m_persistCounterAlloc) { m_persistCounter.free(); }
#endif
               m_slipGeom.free();
               m_kinetics.free();
               m_eosModel.free();
               m_elastN.free();
            }

            /** @} */

            /**
             * @name Material Model Interface (Overrides from matModelBase)
             * @{
             */

            /**
             * @brief Update array stride configuration.
             * 
             * Overrides base class method. Must be called before complete().
             * Validates stride sizes and stores for use in getResponseECM().
             * 
             * @param[in] strides Vector of stride values
             * 
             * **Validation**:
             * - Size must equal ECMECH_NSTRIDE (8)
             * - Each stride must be >= minimum required size
             * 
             * **Errors**:
             * - ECMECH_FAIL if called after complete()
             * - ECMECH_FAIL if wrong size
             * - ECMECH_FAIL if any stride too small
             * 
             * @note Cannot be called after complete()
             */
            __ecmech_host__
            virtual void
            updateStrides(std::vector<size_t> strides) override final {
               if (m_complete) {
                  ECMECH_FAIL(__func__, "updateStrides should only be called before object is completed");
               }
               if (strides.size() != ecmech::nstride) {
                  // the order here needs to be consistent with ISTRIDE_* macros in ECMECH_const.h
                  std::ostringstream os;
                  os << "Stride vector needs to have a size of " << ecmech::nstride << " with strides of at least: " <<
                     ecmech::nsvp << ", " << ecmech::ndim << ", " << ecmech::nvr << ", " <<
                     ecmech::ne << ", " << ecmech::nsvp << ", " << numHist << ", 1, " << ecmech::nsdd
                  ;
                  ECMECH_FAIL(__func__, os.str().c_str());
               }
               // Need to make sure all of the strides provided at least make sense
               if (strides[istride_def_rate] < ecmech::nsvp) {
                  std::ostringstream os;
                  os << "strides[istride_def_rate] should have at least a length of: " << ecmech::nsvp;
                  ECMECH_FAIL(__func__, os.str().c_str());
               }
               if (strides[istride_spin_v] < ecmech::ndim) {
                  std::ostringstream os;
                  os << "strides[istride_spin_v] should have at least a length of: " << ecmech::ndim;
                  ECMECH_FAIL(__func__, os.str().c_str());
               }
               if (strides[istride_vol_ratio] < ecmech::nvr) {
                  std::ostringstream os;
                  os << "strides[istride_int_eng] should have at least a length of: " << ecmech::nvr;
                  ECMECH_FAIL(__func__, os.str().c_str());
               }
               if (strides[istride_int_eng] < ecmech::ne) {
                  std::ostringstream os;
                  os << "strides[istride_int_eng] should have at least a length of: " << ecmech::ne;
                  ECMECH_FAIL(__func__, os.str().c_str());
               }
               if (strides[istride_stress] < ecmech::nsvp) {
                  std::ostringstream os;
                  os << "strides[istride_stress] should have at least a length of: " << ecmech::nsvp;
                  ECMECH_FAIL(__func__, os.str().c_str());
               }
               if (strides[istride_history] < numHist) {
                  std::ostringstream os;
                  os << "strides[istride_history] should have at least a length of: " << numHist;
                  ECMECH_FAIL(__func__, os.str().c_str());
               }
               if (strides[istride_tkelv] < 1) {
                  std::ostringstream os;
                  os << "strides[istride_tkelv] should have at least a length of: " << 1;
                  ECMECH_FAIL(__func__, os.str().c_str());
               }
               if (strides[istride_sdd] < ecmech::nsdd) {
                  std::ostringstream os;
                  os << "strides[istride_sdd] should have at least a length of: " << ecmech::nsdd;
                  ECMECH_FAIL(__func__, os.str().c_str());
               }
               for (unsigned int i = 0; i < strides.size(); i++) {
                  m_strides[i] = strides[i];
               }
            }

            using matModelBase::initFromParams;
            /**
             * @brief Initialize material model from parameter vectors.
             * 
             * Parses parameter vectors and initializes all component models:
             * 1. Validates parameter counts
             * 2. Frees any existing components
             * 3. Extracts base parameters (density, cvav, tolerance)
             * 4. Creates SlipGeom via factory
             * 5. Creates ThermoElastN via factory
             * 6. Creates Kinetics via factory
             * 7. Creates EosModel via factory (combines base + additional params)
             * 8. Sets up history variable metadata
             * 
             * @param[in] opts Integer options (must be empty for this implementation)
             * @param[in] pars Double parameters [nParams elements]
             *                 Order: density0, cvav, tolerance, slipgeom, elastN, kinetics, eos
             * @param[in] strs String parameters (size ≤ 1, optional model name)
             * @param[in] callBackVoid Unused callback (reserved for future use)
             * 
             * **Parameter order** (critical):
             * ```
             * pars[0] = density0
             * pars[1] = cvav
             * pars[2] = tolerance
             * pars[3:3+nParamsSlipGeom-1] = slip geometry params
             * pars[...] = thermoelastic params
             * pars[...] = kinetics params
             * pars[...] = additional EOS params (gamma, etc.)
             * ```
             * 
             * **Errors**:
             * - ECMECH_FAIL if wrong number of opts (must be 0)
             * - ECMECH_FAIL if wrong number of pars (must equal nParams)
             * - ECMECH_FAIL if strs.size() > 1
             * - ECMECH_FAIL if parameter validation fails in components
             * 
             * @note Stores opts, pars, strs internally for getParams()
             * @note Does NOT call complete() - caller must do this
             */
            __ecmech_host__
            void initFromParams(const std::vector<int>& opts,
                                const std::vector<double>& pars,
                                const std::vector<std::string>& strs,
                                void* /*callBackVoid*/ = nullptr
                                ) override final
            {
               // keep parameters for later
               m_opts = opts;
               m_pars = pars;
               m_strs = strs;

               if (pars.size() != (unsigned int) nParams) {
                  ECMECH_FAIL(__func__, "wrong number of pars");
               }
               if (opts.size() != 0) {
                  ECMECH_FAIL(__func__, "wrong number of opts");
               }
               if (strs.size() > 1) {
                  // strs[0] is optionally a name -- see makeMatModel
                  ECMECH_FAIL(__func__, "wrong number of strs");
               }

               // Want to make sure we free up any old memory before setting parameters just in-case we had a model around already...
               m_slipGeom.free();
               m_kinetics.free();
               m_eosModel.free();
               m_elastN.free();

               std::vector<double>::const_iterator parsIt = pars.begin();

               m_density0 = *parsIt; ++parsIt;
               m_cvav = *parsIt; ++parsIt;

               m_tolerance = *parsIt; ++parsIt;

               {
                  const std::vector<double> paramsThese(parsIt, parsIt + SlipGeom::nParams);
                  m_slipGeom = internal::make_class_factory<SlipGeom>(paramsThese);
                  parsIt += SlipGeom::nParams;
                  // m_slipGeom.setParams(paramsThese); parsIt += SlipGeom::nParams;
               }
               {
                  const std::vector<double> paramsThese(parsIt, parsIt + ThermoElastN::nParams);
                  m_elastN = internal::make_class_factory<ThermoElastN>(paramsThese);
                  parsIt += ThermoElastN::nParams;
                  // m_elastN.setParams(paramsThese); parsIt += ThermoElastN::nParams;
               }
               {
                  const std::vector<double> paramsThese(parsIt, parsIt + Kinetics::nParams);
                  m_kinetics = internal::make_class_factory<Kinetics>(paramsThese, SlipGeom::nslip);
                  parsIt += Kinetics::nParams;
                  // m_kinetics.setParams(paramsThese); parsIt += Kinetics::nParams;
               }
               {
                  double bulk_modulus = (*m_elastN).getBulkMod();
                  std::vector<double> paramsThese(EosModel::nParams);
                  paramsThese[0] = m_density0;
                  paramsThese[1] = bulk_modulus;
                  paramsThese[2] = m_cvav;
                  std::copy(parsIt, parsIt + nParamsEOS, paramsThese.begin() + nParamsEOSHave);

                  m_eosModel = internal::make_class_factory<EosModel>(paramsThese);
                  parsIt += nParamsEOS;
                  // m_eosModel.setParams(paramsThese); parsIt += nParamsEOS;

                  {
                     double rel_vol_min, rel_vol_max;
                     (*m_eosModel).getInfo(rel_vol_min, rel_vol_max, m_energy0, m_rel_vol0);
                  }
               }

               int iParam = parsIt - pars.begin();
               if (iParam != nParams) {
                  ECMECH_FAIL(__func__, "wrong number of params");
               }
               //////////////////////////////

               m_rhvNames.clear();
               m_rhvVals.clear();
               m_rhvPlot.clear();
               m_rhvState.clear();

#if defined(ECMECH_USE_DPEFF)
               m_rhvNames.push_back("dplas_eff"); m_rhvVals.push_back(0.); m_rhvPlot.push_back(true); m_rhvState.push_back(true); // iHistA_shrateEff
               m_rhvNames.push_back("eps"); m_rhvVals.push_back(0.); m_rhvPlot.push_back(true); m_rhvState.push_back(true); // iHistA_shrEff
#else
               m_rhvNames.push_back("shrate_eff"); m_rhvVals.push_back(0.); m_rhvPlot.push_back(true); m_rhvState.push_back(true); // iHistA_shrateEff
               m_rhvNames.push_back("shr_eff"); m_rhvVals.push_back(0.); m_rhvPlot.push_back(true); m_rhvState.push_back(true); // iHistA_shrEff
#endif
               m_rhvNames.push_back("flow_str"); m_rhvVals.push_back(0.); m_rhvPlot.push_back(true); m_rhvState.push_back(false); // iHistA_flowStr
               m_rhvNames.push_back("n_feval"); m_rhvVals.push_back(0.); m_rhvPlot.push_back(true); m_rhvState.push_back(false); // iHistA_nFEval
               // numHistAux
               //
               for (int iTvec = 0; iTvec < ecmech::ntvec; ++iTvec) {
                  std::ostringstream os;
                  os << "t" << iTvec + 1;
                  m_rhvNames.push_back(os.str()); m_rhvVals.push_back(0.); m_rhvPlot.push_back(true); m_rhvState.push_back(true);
               }

               //
               {
                  double qVal = 1.0;
                  for (int iQ = 0; iQ < ecmech::qdim; ++iQ) {
                     std::ostringstream os;
                     os << "quat_" << iQ + 1;
                     m_rhvNames.push_back(os.str()); m_rhvVals.push_back(qVal); m_rhvPlot.push_back(true); m_rhvState.push_back(true);
                     qVal = 0.0;
                  }
               }
               //
               (*m_kinetics).getHistInfo(m_rhvNames, m_rhvVals, m_rhvPlot, m_rhvState);
               //
               for (int iSlip = 0; iSlip < SlipGeom::nslip; ++iSlip) {
                  std::ostringstream os;
                  os << "shrate_" << iSlip + 1;
                  m_rhvNames.push_back(os.str()); m_rhvVals.push_back(0.); m_rhvPlot.push_back(true); m_rhvState.push_back(true);
               }

               //
               if (m_rhvNames.size() != numHist) {
                  ECMECH_FAIL(__func__, "mismatch in numHist");
               }
            }

            using matModelBase::getParams;
            /**
             * @brief Retrieve stored parameter vectors.
             * 
             * Simple accessor returning internally stored parameters from
             * most recent initFromParams() call.
             * 
             * @param[out] opts Integer options (cleared and set to m_opts)
             * @param[out] pars Double parameters (cleared and set to m_pars)
             * @param[out] strs String parameters (cleared and set to m_strs)
             * 
             * @note Does not reconstruct from components, just returns stored values
             */
            __ecmech_host__
            void getParams(std::vector<int>& opts,
                           std::vector<double>& pars,
                           std::vector<std::string>& strs) const override final
            {
               opts = m_opts;
               pars = m_pars;
               strs = m_strs;
            }

            using matModelBase::getResponseECM;
            /**
             * @brief Compute constitutive response for batch of material points.
             * 
             * Main computational kernel - evaluates crystal plasticity model for nPassed
             * independent material points in parallel. Uses SNLS nonlinear solver for
             * implicit time integration of elastic-viscoplastic equations.
             * 
             * **Execution flow**:
             * 1. Validate m_complete flag
             * 2. Extract stride configuration
             * 3. Capture component pointers for GPU lambda
             * 4. Launch RAJA parallel loop over material points
             * 5. For each point:
             *    - Set up problem state (deformation, orientation, history)
             *    - Create SNLS solver and problem instance
             *    - Solve nonlinear system
             *    - Update stress, history, temperature
             *    - Optionally compute tangent stiffness
             * 
             * **Parallelization**:
             * - RAJA execution strategy set by setExecutionStrategy()
             * - snls::forall handles CPU/GPU/OpenMP dispatch
             * - Each material point is independent (embarrassingly parallel)
             * 
             * **Convergence**:
             * - Uses SNLS trust-region Newton solver
             * - Tolerance from m_tolerance
             * - Failed convergence: sets nFEval < 0 in history
             * 
             * Array shapes provided down below are minimal size of each array but they could be greater
             * based on stride sizes
             * 
             * @param[in] dt Time step size
             * @param[in] def_rate_d6vV Deformation rate [nsvp × nPassed]
             * @param[in] spin_vecV Spin vector [ndim × nPassed]
             * @param[in] rel_vol_ratiosV Volume ratios [nvr × nPassed]
             * @param[in,out] internal_energyV Internal energy [ne × nPassed]
             * @param[in,out] cauchy_stress_d6pV Cauchy stress [nsvp × nPassed]
             * @param[in,out] histV History variables [numHist × nPassed]
             * @param[out] tkelvV Temperature [1 × nPassed]
             * @param[out] sddV Auxiliary outputs [nsdd × nPassed]
             * @param[out] mtanSDV Tangent stiffness [nsvec×nsvec × nPassed] or nullptr
             * @param[in] nPassed Number of material points
             * 
             * **Array indexing**: point-major order
             * ```cpp
             * stress_component = cauchy_stress_d6pV[i * stress_stride + component]
             * ```
             * 
             * **Errors**:
             * - ECMECH_FAIL if !m_complete
             * - ECMECH_FAIL if convergence fails (in debug mode)
             * 
             * @note const method for thread safety (no model parameter modification)
             * @note GPU memory: Arrays must be accessible from execution space
             * 
             * @see setExecutionStrategy() for parallel configuration
             * @see setSNLSExecutionStrategy() for SNLS backend configuration
             */
            __ecmech_host__
            void getResponseECM(const double & dt,
                                const double * def_rate_d6vV,
                                const double * spin_vecV,
                                const double * rel_vol_ratiosV,
                                double * internal_energyV,
                                double * cauchy_stress_d6pV,
                                double * histV,
                                double * tkelvV,
                                double * sddV,
                                double * mtanSDV,
                                const int& nPassed) const override final
            {
               if (!m_complete) {
                  ECMECH_FAIL(__func__, "not complete");
               }

#if defined(ECMECH_SIMD_EXPT) || defined(ECMECH_GPU_PACK_EXPT)
               const simd::PackArrays packArrays{dt, m_tolerance,
                  def_rate_d6vV, m_strides[istride_def_rate], spin_vecV, m_strides[istride_spin_v],
                  rel_vol_ratiosV, m_strides[istride_vol_ratio], internal_energyV, m_strides[istride_int_eng],
                  cauchy_stress_d6pV, m_strides[istride_stress], histV, m_strides[istride_history],
                  tkelvV, m_strides[istride_tkelv], sddV, m_strides[istride_sdd], mtanSDV, nPassed};
#endif
#if defined(ECMECH_SIMD_EXPT)
               // Experiment: SIMD across points on the CPU/OpenMP backends (ECMECH_SIMD=1)
               if constexpr (simd::simd_supported<SlipGeom, Kinetics, ThermoElastN>::value) {
                  if (m_accel != ECM_EXEC_STRAT_GPU && simd::simdEnabled()) {
                     simd::getResponseSimd<SlipGeom, Kinetics, ThermoElastN, EosModel>
                        (*m_slipGeom, *m_kinetics, *m_elastN, *m_eosModel, packArrays, m_accel == ECM_EXEC_STRAT_OPENMP);
                     if (this->reduceStatus(histV, nPassed)) {
                        getResponseRetry(dt, def_rate_d6vV, spin_vecV, rel_vol_ratiosV, internal_energyV,
                                         cauchy_stress_d6pV, histV, tkelvV, sddV, mtanSDV, nPassed);
                     }
                     return;
                  }
               }
#endif
#if defined(ECMECH_TEAM_SPLIT_EXPT) && defined(__ecmech_gpu_active__)
               // Experiment: NT threads per point, splitting the slip-system work (ECMECH_TEAM_SPLIT=NT)
               if (m_accel == ECM_EXEC_STRAT_GPU && teamSplitSize() > 1) {
                  bool ran = true;
                  switch (teamSplitSize()) {
                     case 4: getResponseTeamSplit<4>(dt, def_rate_d6vV, spin_vecV, rel_vol_ratiosV, internal_energyV, cauchy_stress_d6pV, histV, tkelvV, sddV, mtanSDV, nPassed); break;
                     case 8: getResponseTeamSplit<8>(dt, def_rate_d6vV, spin_vecV, rel_vol_ratiosV, internal_energyV, cauchy_stress_d6pV, histV, tkelvV, sddV, mtanSDV, nPassed); break;
                     case 16: getResponseTeamSplit<16>(dt, def_rate_d6vV, spin_vecV, rel_vol_ratiosV, internal_energyV, cauchy_stress_d6pV, histV, tkelvV, sddV, mtanSDV, nPassed); break;
                     case 32: getResponseTeamSplit<32>(dt, def_rate_d6vV, spin_vecV, rel_vol_ratiosV, internal_energyV, cauchy_stress_d6pV, histV, tkelvV, sddV, mtanSDV, nPassed); break;
                     default: ran = false; break;
                  }
                  if (ran) {
                     if (this->reduceStatus(histV, nPassed)) {
                        getResponseRetry(dt, def_rate_d6vV, spin_vecV, rel_vol_ratiosV, internal_energyV,
                                         cauchy_stress_d6pV, histV, tkelvV, sddV, mtanSDV, nPassed);
                     }
                     return;
                  }
               }
#endif
#if defined(ECMECH_GPU_PACK_EXPT) && defined(__ecmech_gpu_active__)
               // Experiment: N points per GPU thread (ECMECH_GPU_PACK=2 or 4)
               // or warp-uniform lanes, one point per thread (ECMECH_GPU_WARP=1)
               if (m_accel == ECM_EXEC_STRAT_GPU && (gpuPackSize() > 1 || gpuWarpLanes())) {
                  bool ran = true;
                  if (gpuWarpLanes()) { ran = getResponseGpuPack<ecmech::dpack<1, true>>(packArrays); }
                  else {
                     switch (gpuPackSize()) {
                        case 2: ran = getResponseGpuPack<ecmech::dpack<2>>(packArrays); break;
                        case 4: ran = getResponseGpuPack<ecmech::dpack<4>>(packArrays); break;
                        default: ran = false; break;
                     }
                  }
                  if (ran) {
                     if (this->reduceStatus(histV, nPassed)) {
                        getResponseRetry(dt, def_rate_d6vV, spin_vecV, rel_vol_ratiosV, internal_energyV,
                                         cauchy_stress_d6pV, histV, tkelvV, sddV, mtanSDV, nPassed);
                     }
                     return;
                  }
               }
#endif

#if defined(ECMECH_SNLS_TEAM) && defined(__ecmech_gpu_active__)
               if (m_accel == ECM_EXEC_STRAT_GPU) {
                  bool ran = true;
                  switch (snlsTeamSize()) {
                     case 1: getResponseTeam<1>(dt, def_rate_d6vV, spin_vecV, rel_vol_ratiosV, internal_energyV, cauchy_stress_d6pV, histV, tkelvV, sddV, mtanSDV, nPassed); break;
                     case 2: getResponseTeam<2>(dt, def_rate_d6vV, spin_vecV, rel_vol_ratiosV, internal_energyV, cauchy_stress_d6pV, histV, tkelvV, sddV, mtanSDV, nPassed); break;
                     case 4: getResponseTeam<4>(dt, def_rate_d6vV, spin_vecV, rel_vol_ratiosV, internal_energyV, cauchy_stress_d6pV, histV, tkelvV, sddV, mtanSDV, nPassed); break;
                     case 8: getResponseTeam<8>(dt, def_rate_d6vV, spin_vecV, rel_vol_ratiosV, internal_energyV, cauchy_stress_d6pV, histV, tkelvV, sddV, mtanSDV, nPassed); break;
                     default: ran = false; break;
                  }
                  if (ran) {
                     if (this->reduceStatus(histV, nPassed)) {
                        getResponseRetry(dt, def_rate_d6vV, spin_vecV, rel_vol_ratiosV, internal_energyV,
                                         cauchy_stress_d6pV, histV, tkelvV, sddV, mtanSDV, nPassed);
                     }
                     return;
                  }
               }
#endif
#if (defined(ECMECH_SNLS_TEAM) || defined(ECMECH_PERSISTENT_EXPT)) && defined(__ecmech_gpu_active__)
               // Experiment: persistent, iteration-uniform kernel (ECMECH_PERSISTENT=1)
               if (m_accel == ECM_EXEC_STRAT_GPU && persistentEnabled()) {
                  getResponsePersistent(dt, def_rate_d6vV, spin_vecV, rel_vol_ratiosV, internal_energyV, cauchy_stress_d6pV, histV, tkelvV, sddV, mtanSDV, nPassed);
                  if (this->reduceStatus(histV, nPassed)) {
                     getResponseRetry(dt, def_rate_d6vV, spin_vecV, rel_vol_ratiosV, internal_energyV,
                                      cauchy_stress_d6pV, histV, tkelvV, sddV, mtanSDV, nPassed);
                  }
                  return;
               }
#endif
#if defined(ECMECH_SNLS_TEAM)
               // Experiment: GPU block size for the one-thread-per-point
               // kernel, chosen at run time with ECMECH_GPU_BLOCK.
               switch (m_accel == ECM_EXEC_STRAT_GPU ? gpuBlockSize() : ECMECH_GPU_THREADS) {
                  case 64:  getResponseSolo<64>(dt, def_rate_d6vV, spin_vecV, rel_vol_ratiosV, internal_energyV, cauchy_stress_d6pV, histV, tkelvV, sddV, mtanSDV, nPassed); break;
                  case 128: getResponseSolo<128>(dt, def_rate_d6vV, spin_vecV, rel_vol_ratiosV, internal_energyV, cauchy_stress_d6pV, histV, tkelvV, sddV, mtanSDV, nPassed); break;
                  case 512: getResponseSolo<512>(dt, def_rate_d6vV, spin_vecV, rel_vol_ratiosV, internal_energyV, cauchy_stress_d6pV, histV, tkelvV, sddV, mtanSDV, nPassed); break;
                  default:  getResponseSolo<ECMECH_GPU_THREADS>(dt, def_rate_d6vV, spin_vecV, rel_vol_ratiosV, internal_energyV, cauchy_stress_d6pV, histV, tkelvV, sddV, mtanSDV, nPassed); break;
               }
#else
               getResponseSolo<ECMECH_GPU_THREADS>(dt, def_rate_d6vV, spin_vecV, rel_vol_ratiosV, internal_energyV, cauchy_stress_d6pV, histV, tkelvV, sddV, mtanSDV, nPassed);
#endif

               if (this->reduceStatus(histV, nPassed)) {
                  getResponseRetry(dt, def_rate_d6vV, spin_vecV, rel_vol_ratiosV, internal_energyV,
                                   cauchy_stress_d6pV, histV, tkelvV, sddV, mtanSDV, nPassed);
               }
            }// End of getResponse

#if defined(ECMECH_SNLS_TEAM) && defined(__ecmech_gpu_active__)
            /// Team size for the experimental SNLS team path on the GPU, read
            /// once from the ECMECH_SNLS_TEAM environment variable: 1, 2, 4
            /// or 8 threads per point; anything else (or unset) keeps the
            /// one-thread-per-point path.
            __ecmech_host__
            static int snlsTeamSize()
            {
               static const int nt = [] {
                  const char* env = std::getenv("ECMECH_SNLS_TEAM");
                  return env ? std::atoi(env) : 0;
               }();
               return nt;
            }

            /// getResponse with NT threads per point (snls::forall_team +
            /// SNLSTrDlDenseG::solveTeam). ECMECH_TEAM_PTS_PER_BLOCK points
            /// share a block, so the block has ECMECH_TEAM_PTS_PER_BLOCK*NT
            /// threads and that many TeamSlots of shared memory.
            template<int NT>
            __ecmech_host__
            void getResponseTeam(const double & dt,
                                 const double * def_rate_d6vV,
                                 const double * spin_vecV,
                                 const double * rel_vol_ratiosV,
                                 double * internal_energyV,
                                 double * cauchy_stress_d6pV,
                                 double * histV,
                                 double * tkelvV,
                                 double * sddV,
                                 double * mtanSDV,
                                 const int& nPassed) const
            {
               const unsigned int def_rate_stride = m_strides[istride_def_rate];
               const unsigned int spin_v_stride = m_strides[istride_spin_v];
               const unsigned int vol_ratio_stride = m_strides[istride_vol_ratio];
               const unsigned int int_eng_stride = m_strides[istride_int_eng];
               const unsigned int stress_stride = m_strides[istride_stress];
               const unsigned int history_stride = m_strides[istride_history];
               const unsigned int tkelv_stride = m_strides[istride_tkelv];
               const unsigned int sdd_stride = m_strides[istride_sdd];

               const auto slipGeom = m_slipGeom;
               const auto kinetics = m_kinetics;
               const auto elastN = m_elastN;
               const auto eosModel = m_eosModel;
               const auto tolerance = m_tolerance;
               const auto outputLevel = m_outputLevel;

               using Slots = evptn::TeamSlots<SlipGeom, Kinetics, ThermoElastN, EosModel>;
               constexpr int BLOCK = ECMECH_TEAM_PTS_PER_BLOCK * NT;
               constexpr int IPB = snls::ForallTeamPacking<BLOCK, NT>::itemsPerBlock;

               snls::forall_team<BLOCK, NT>(0, nPassed, [=]
                  __ecmech_hdev__
                  (int i, int tid, int nthreads, int teamBase, snls::TeamConsensus& consensus)
               {
                  RAJA_TEAM_SHARED Slots slots[IPB];
                  double *mtanSDThis = (mtanSDV ? &mtanSDV[ecmech::nsvec2 * i] : nullptr);
                  auto get_response = [&] (const SlipGeom& slip_geom) -> bool {
                     return evptn::getResponseSnglTeam<SlipGeom, Kinetics, ThermoElastN, EosModel>
                     (tid, nthreads, teamBase, consensus, slots[teamBase],
                        slip_geom, *kinetics, *elastN, *eosModel,
                        dt,
                        tolerance,
                        &def_rate_d6vV[def_rate_stride * i],
                        &spin_vecV[spin_v_stride * i],
                        &rel_vol_ratiosV[vol_ratio_stride * i],
                        &internal_energyV[int_eng_stride * i],
                        &cauchy_stress_d6pV[stress_stride * i],
                        &histV[history_stride * i],
                        tkelvV[tkelv_stride * i],
                        &sddV[sdd_stride * i],
                        mtanSDThis,
                        outputLevel);
                  };
                  bool status;
                  auto slipGeom_tmp = slipGeom;
                  if constexpr (SlipGeom::dynamic) {
                     SlipGeom slip_geom = *slipGeom_tmp;
                     status = get_response(slip_geom);
                  } else {
                     status = get_response(*slipGeom_tmp);
                  }
                  if (!status && tid == 0) {
                     histV[history_stride * i + iHistA_nFEval] *= -1;
                  }
               });
            }
#endif

#if (defined(ECMECH_SNLS_TEAM) || defined(ECMECH_PERSISTENT_EXPT)) && defined(__ecmech_gpu_active__)
            __ecmech_host__
            static bool persistentEnabled()
            {
               static const bool on = std::getenv("ECMECH_PERSISTENT") != nullptr;
               return on;
            }

            /// Experiment: a fixed number of worker threads (ECMECH_PERSIST_WORKERS,
            /// default 4096) each take points from an atomic counter. Every round,
            /// each worker advances its current point by one Newton iteration
            /// (the same code for all lanes); a worker whose point has finished
            /// completes it and takes the next one. Each point's computation is
            /// that of getResponseSngl; only the lane computing it changes.
            __ecmech_host__
            void getResponsePersistent(const double & dt,
                                       const double * def_rate_d6vV,
                                       const double * spin_vecV,
                                       const double * rel_vol_ratiosV,
                                       double * internal_energyV,
                                       double * cauchy_stress_d6pV,
                                       double * histV,
                                       double * tkelvV,
                                       double * sddV,
                                       double * mtanSDV,
                                       const int& nPassed) const
            {
               const unsigned int def_rate_stride = m_strides[istride_def_rate];
               const unsigned int spin_v_stride = m_strides[istride_spin_v];
               const unsigned int vol_ratio_stride = m_strides[istride_vol_ratio];
               const unsigned int int_eng_stride = m_strides[istride_int_eng];
               const unsigned int stress_stride = m_strides[istride_stress];
               const unsigned int history_stride = m_strides[istride_history];
               const unsigned int tkelv_stride = m_strides[istride_tkelv];
               const unsigned int sdd_stride = m_strides[istride_sdd];

               const auto slipGeom = m_slipGeom;
               const auto kinetics = m_kinetics;
               const auto elastN = m_elastN;
               const auto eosModel = m_eosModel;
               const auto tolerance = m_tolerance;
               const auto outputLevel = m_outputLevel;

               static const int workers_env = [] {
                  const char* env = std::getenv("ECMECH_PERSIST_WORKERS");
                  return env ? std::atoi(env) : 4096;
               }();
               const int nWorkers = (workers_env < nPassed) ? workers_env : nPassed;

               if (!m_persistCounterAlloc) {
                  auto mm = snls::memoryManager::getInstance();
                  m_persistCounter = mm.allocManagedArray<int>(1);
                  m_persistCounterAlloc = true;
               }
               int* counter = m_persistCounter.data(chai::ExecutionSpace::GPU);
               snls::forall<1>(0, 1, [=] __ecmech_hdev__ (int) { counter[0] = 0; });

               // ECMECH_PERSIST_SIMPLE=1: refill per point instead of per Newton step:
               // each worker takes the next point from the queue and solves it with
               // getResponseSngl (the warp's threads then run the same solver loop on
               // different points at different iterations)
               static const bool simple = std::getenv("ECMECH_PERSIST_SIMPLE") != nullptr;
               if (simple) {
                  snls::forall<ECMECH_GPU_THREADS>(0, nWorkers, [=] __ecmech_hdev__ (int)
                  {
                     auto slipGeom_tmp = slipGeom;
                     while (true) {
                        const int i = RAJA::atomicAdd<RAJA::auto_atomic>(counter, 1);
                        if (i >= nPassed) { break; }
                        double *mtanSDThis = (mtanSDV ? &mtanSDV[ecmech::nsvec2 * i] : nullptr);
                        auto get_response = [&] (const SlipGeom& slip_geom) -> bool {
                           return getResponseSngl<SlipGeom, Kinetics, ThermoElastN, EosModel>
                           (slip_geom, *kinetics, *elastN, *eosModel, dt, tolerance,
                            &def_rate_d6vV[def_rate_stride * i], &spin_vecV[spin_v_stride * i],
                            &rel_vol_ratiosV[vol_ratio_stride * i], &internal_energyV[int_eng_stride * i],
                            &cauchy_stress_d6pV[stress_stride * i], &histV[history_stride * i],
                            tkelvV[tkelv_stride * i], &sddV[sdd_stride * i], mtanSDThis, outputLevel);
                        };
                        bool status;
                        if constexpr (SlipGeom::dynamic) {
                           SlipGeom slip_geom = *slipGeom_tmp;
                           status = get_response(slip_geom);
                        } else {
                           status = get_response(*slipGeom_tmp);
                        }
                        if (!status) { histV[history_stride * i + iHistA_nFEval] *= -1; }
                     }
                  });
                  return;
               }

               using PP = evptn::PersistentPoint<SlipGeom, Kinetics, ThermoElastN, EosModel>;
               snls::forall<ECMECH_GPU_THREADS>(0, nWorkers, [=] __ecmech_hdev__ (int)
               {
                  PP pp;
                  auto slipGeom_tmp = slipGeom;
                  // local: lane-local copy of a dynamic slip geometry, rebuilt
                  // for every point (as getResponseSolo copies it per point);
                  // nullptr for static geometries, which are shared.
                  auto run = [&] (const SlipGeom& sg, SlipGeom* local) {
                     int i = 0;
                     bool active = false;
                     while (true) {
                        if (active) {
                           if (!pp.step()) {
                              const bool ok = pp.finish(*elastN);
                              if (!ok) { histV[history_stride * i + iHistA_nFEval] *= -1; }
                              active = false;
                           }
                        }
                        if (!active) {
                           i = RAJA::atomicAdd<RAJA::auto_atomic>(counter, 1);
                           if (i >= nPassed) { break; }
                           if (local) { local->~SlipGeom(); new (local) SlipGeom(*slipGeom_tmp); }
                           double *mtanSDThis = (mtanSDV ? &mtanSDV[ecmech::nsvec2 * i] : nullptr);
                           active = pp.start(sg, *kinetics, *elastN, *eosModel, dt, tolerance,
                                             &def_rate_d6vV[def_rate_stride * i],
                                             &spin_vecV[spin_v_stride * i],
                                             &rel_vol_ratiosV[vol_ratio_stride * i],
                                             &internal_energyV[int_eng_stride * i],
                                             &cauchy_stress_d6pV[stress_stride * i],
                                             &histV[history_stride * i],
                                             tkelvV[tkelv_stride * i],
                                             &sddV[sdd_stride * i],
                                             mtanSDThis, outputLevel);
                           if (!active) { histV[history_stride * i + iHistA_nFEval] *= -1; }
                        }
                     }
                  };
                  if constexpr (SlipGeom::dynamic) {
                     SlipGeom slip_geom_local = *slipGeom_tmp;
                     run(slip_geom_local, &slip_geom_local);
                  } else {
                     run(*slipGeom_tmp, nullptr);
                  }
               });
            }

            mutable chai::ManagedArray<int> m_persistCounter;
            mutable bool m_persistCounterAlloc = false;
#endif

            /// One thread per point: the original getResponse kernel, with the
            /// GPU block size as a template parameter.
#if defined(ECMECH_GPU_PACK_EXPT) && defined(__ecmech_gpu_active__)
            /// Experiment: points per GPU thread, read once from ECMECH_GPU_PACK
            __ecmech_host__
            static int gpuPackSize()
            {
               static const int n = [] {
                  const char* env = std::getenv("ECMECH_GPU_PACK");
                  return env ? std::atoi(env) : 1;
               }();
               return n;
            }

            /// Experiment: warp-uniform lanes (ECMECH_GPU_WARP=1)
            __ecmech_host__
            static bool gpuWarpLanes()
            {
               static const bool on = std::getenv("ECMECH_GPU_WARP") != nullptr;
               return on;
            }

            /// Experiment: getResponse with value type R per GPU thread (simd::getResponsePack;
            /// R = dpack<N>: N points per thread; dpack<1, true>: one point, warp-uniform
            /// control flow); false if the model is not supported.
            template<class R>
            __ecmech_host__
            bool getResponseGpuPack(const simd::PackArrays& a) const
            {
               constexpr int N = lanes_of<R>();
               if constexpr (simd::pack_supported<SlipGeom, Kinetics, ThermoElastN, R>::value) {
                  if (a.history_stride > (unsigned)simd::maxPackHist) {
                     ECMECH_FAIL(__func__, "GPU pack path: history too long for the padding scratch");
                  }
                  const auto slipGeom = m_slipGeom;
                  const auto kinetics = m_kinetics;
                  const auto elastN = m_elastN;
                  const auto eosModel = m_eosModel;
                  const simd::PackArrays arrays = a; // (nvcc: no first capture of `a` inside if constexpr)
                  const int nPacks = (a.nPassed + N - 1) / N;
                  snls::forall<ECMECH_GPU_THREADS>(0, nPacks, [=] __ecmech_hdev__ (int ip) {
                     simd::getResponsePack<R>(ip, *slipGeom, *kinetics, *elastN, *eosModel, arrays);
                  });
                  return true;
               }
               else {
                  return false;
               }
            }
#endif

#if defined(ECMECH_TEAM_SPLIT_EXPT) && defined(__ecmech_gpu_active__)
            /// Experiment: threads per point for the team split, from ECMECH_TEAM_SPLIT
            __ecmech_host__
            static int teamSplitSize()
            {
               static const int n = [] {
                  const char* env = std::getenv("ECMECH_TEAM_SPLIT");
                  return env ? std::atoi(env) : 1;
               }();
               return n;
            }

            /// Experiment: NT threads per point (one team per GPU block). Every thread
            /// runs the whole point (bit-identically); the slip-system kinetics and the
            /// slip-rate derivative rows are split across the team (evptn::SlipTeam).
            /// Thread 0 writes the results, the others work on copies.
            template<int NT>
            __ecmech_host__
            void getResponseTeamSplit(const double & dt,
                                      const double * def_rate_d6vV,
                                      const double * spin_vecV,
                                      const double * rel_vol_ratiosV,
                                      double * internal_energyV,
                                      double * cauchy_stress_d6pV,
                                      double * histV,
                                      double * tkelvV,
                                      double * sddV,
                                      double * mtanSDV,
                                      const int& nPassed) const
            {
               const unsigned int def_rate_stride = m_strides[istride_def_rate];
               const unsigned int spin_v_stride = m_strides[istride_spin_v];
               const unsigned int vol_ratio_stride = m_strides[istride_vol_ratio];
               const unsigned int int_eng_stride = m_strides[istride_int_eng];
               const unsigned int stress_stride = m_strides[istride_stress];
               const unsigned int history_stride = m_strides[istride_history];
               const unsigned int tkelv_stride = m_strides[istride_tkelv];
               const unsigned int sdd_stride = m_strides[istride_sdd];
               if (history_stride > 128 || stress_stride > 8 || int_eng_stride > 8 || sdd_stride > 8) {
                  ECMECH_FAIL(__func__, "team split: per-thread copies too small");
               }
               const auto slipGeom = m_slipGeom;
               const auto kinetics = m_kinetics;
               const auto elastN = m_elastN;
               const auto eosModel = m_eosModel;
               const auto tolerance = m_tolerance;
               const auto outputLevel = m_outputLevel;
               snls::forall_team<NT, NT>(0, nPassed, [=] __ecmech_hdev__ (int i, int tid, int nthreads, int, snls::TeamConsensus&)
               {
                  constexpr int ns = SlipGeom::nslip;
                  RAJA_TEAM_SHARED double sh[2 * ns + ecmech::ntvec * ecmech::ntvec + ecmech::nwvec * ecmech::ntvec];
                  const evptn::SlipTeam team{tid, nthreads, &sh[0], &sh[ns], &sh[2 * ns]};
                  // thread 0 on the point's arrays, the others on copies
                  double histC[128], stressC[8], energyC[8], sddC[8], tC, mtanC[ecmech::nsvec2];
                  double* hist = &histV[history_stride * i];
                  double* stress = &cauchy_stress_d6pV[stress_stride * i];
                  double* energy = &internal_energyV[int_eng_stride * i];
                  double* sdd = &sddV[sdd_stride * i];
                  double* tk = &tkelvV[tkelv_stride * i];
                  double* mtan = (mtanSDV ? &mtanSDV[ecmech::nsvec2 * i] : nullptr);
                  if (tid != 0) {
                     for (unsigned int k = 0; k < history_stride; ++k) { histC[k] = hist[k]; }
                     for (unsigned int k = 0; k < stress_stride; ++k) { stressC[k] = stress[k]; }
                     for (unsigned int k = 0; k < int_eng_stride; ++k) { energyC[k] = energy[k]; }
                     for (unsigned int k = 0; k < sdd_stride; ++k) { sddC[k] = sdd[k]; }
                     tC = *tk;
                     hist = histC; stress = stressC; energy = energyC; sdd = sddC; tk = &tC;
                     if (mtan) { mtan = mtanC; }
                  }
                  auto get_response = [&] (const SlipGeom& slip_geom) -> bool {
                     return getResponseSngl<SlipGeom, Kinetics, ThermoElastN, EosModel>
                     (slip_geom, *kinetics, *elastN, *eosModel, dt, tolerance,
                      &def_rate_d6vV[def_rate_stride * i], &spin_vecV[spin_v_stride * i],
                      &rel_vol_ratiosV[vol_ratio_stride * i], energy, stress, hist, *tk, sdd, mtan,
                      outputLevel, &team);
                  };
                  bool status;
                  auto slipGeom_tmp = slipGeom;
                  if constexpr (SlipGeom::dynamic) {
                     SlipGeom slip_geom = *slipGeom_tmp;
                     status = get_response(slip_geom);
                  } else {
                     status = get_response(*slipGeom_tmp);
                  }
                  if (tid == 0 && !status) { hist[iHistA_nFEval] *= -1; }
               });
            }
#endif

            template<int BLOCK>
            __ecmech_host__
            void getResponseSolo(const double & dt,
                                 const double * def_rate_d6vV,
                                 const double * spin_vecV,
                                 const double * rel_vol_ratiosV,
                                 double * internal_energyV,
                                 double * cauchy_stress_d6pV,
                                 double * histV,
                                 double * tkelvV,
                                 double * sddV,
                                 double * mtanSDV,
                                 const int& nPassed) const
            {
               // All of the stride lengths are constant within this function
               const unsigned int def_rate_stride = m_strides[istride_def_rate];
               const unsigned int spin_v_stride = m_strides[istride_spin_v];
               const unsigned int vol_ratio_stride = m_strides[istride_vol_ratio];
               const unsigned int int_eng_stride = m_strides[istride_int_eng];
               const unsigned int stress_stride = m_strides[istride_stress];
               const unsigned int history_stride = m_strides[istride_history];
               const unsigned int tkelv_stride = m_strides[istride_tkelv];
               const unsigned int sdd_stride = m_strides[istride_sdd];

               const auto slipGeom = m_slipGeom;
               const auto kinetics = m_kinetics;
               const auto elastN = m_elastN;
               const auto eosModel = m_eosModel;
               const auto tolerance = m_tolerance;
               const auto outputLevel = m_outputLevel;

               snls::forall<BLOCK>(0, nPassed, [=]
                  __ecmech_hdev__
                  (int i)
               {
                  double *mtanSDThis = (mtanSDV ? &mtanSDV[ecmech::nsvec2 * i] : nullptr);
                  auto get_response = [=] (const SlipGeom& slip_geom) -> bool {
                     return getResponseSngl<SlipGeom, Kinetics, ThermoElastN, EosModel>
                     (slip_geom, *kinetics, *elastN, *eosModel,
                        dt,
                        tolerance,
                        &def_rate_d6vV[def_rate_stride * i],
                        &spin_vecV[spin_v_stride * i],
                        &rel_vol_ratiosV[vol_ratio_stride * i],
                        &internal_energyV[int_eng_stride * i],
                        &cauchy_stress_d6pV[stress_stride * i],
                        &histV[history_stride * i],
                        tkelvV[tkelv_stride * i],
                        &sddV[sdd_stride * i],
                        mtanSDThis,
                        outputLevel);
                  };
                  bool status;
                  // Thanks to NVCC being difficult we have to create an unnecessary temp variable just so we can use our
                  // lambda expression in an if constexpr...
                  auto slipGeom_tmp = slipGeom;
                  // If we have dynamic slip systems then we need to create
                  // a thread local slip geometery class or else we might run into
                  // race condition issues...
                  if constexpr (SlipGeom::dynamic) {
                     SlipGeom slip_geom = *slipGeom_tmp;
                     status = get_response(slip_geom);
                  } else {
                     status = get_response(*slipGeom_tmp);
                  }
                  if (!status) {
                     histV[history_stride * i + iHistA_nFEval] *= -1;
                  }
               });
            }

#if defined(ECMECH_SNLS_TEAM)
            /// Experiment: block size for the one-thread-per-point GPU kernel,
            /// read once from ECMECH_GPU_BLOCK (64, 128, 256 or 512).
            __ecmech_host__
            static int gpuBlockSize()
            {
               static const int nb = [] {
                  const char* env = std::getenv("ECMECH_GPU_BLOCK");
                  return env ? std::atoi(env) : ECMECH_GPU_THREADS;
               }();
               return nb;
            }
#endif

            /**
             * @brief Retry constitutive response for failed material points using backup solver.
             * 
             * Called by getResponseECM() when reduceStatus() detects convergence failures
             * (indicated by negative nFEval in history). Attempts to recover failed points
             * using an alternative solver strategy with decoupled implicit updates.
             * 
             * **run-time strategy** (compile-time configured):
             * 
             * **With ECMECH_EXTRA_SOLVERS defined**:
             * - Validates m_complete flag
             * - Extracts stride configuration
             * - Launches parallel loop over all nPassed points
             * - For each point:
             *   - Skips if nFEval >= 0 (already converged successfully)
             *   - For failed points (nFEval < 0):
             *     - Calls getResponseNRSngl() with decoupled implicit update strategy
             *     - Updates stress, history, temperature if successful
             *     - Keeps nFEval negative if retry also fails
             * - After retry loop: calls reduceStatus() again
             *   - If still failures exist: triggers ECMECH_FAIL (backup failed)
             *   - Otherwise: all points recovered successfully
             * 
             * **Without ECMECH_EXTRA_SOLVERS defined**:
             * - Immediately triggers ECMECH_FAIL
             * - Convergence failure is fatal (no backup solver available)
             * 
             * **Performance implications**:
             * - Only processes failed points (early return for successful ones)
             * - Backup solver has simpler decoupled systems
             * - May converge where the normal solver fails
             * - Trade-off: Robustness vs accuracy in coupled state evolution
             * 
             * @param[in] dt Time step size [s]
             * @param[in] def_rate_d6vV Deformation rate array [nsvp × nPassed]
             * @param[in] spin_vecV Spin vector array [ndim × nPassed]
             * @param[in] rel_vol_ratiosV Volume ratio array [nvr × nPassed]
             * @param[in,out] internal_energyV Internal energy array [ne × nPassed]
             * @param[in,out] cauchy_stress_d6pV Stress array [nsvp × nPassed]
             * @param[in,out] histV History array [numHist × nPassed]
             *                - nFEval < 0 marks points needing retry
             *                - Remains negative if retry fails
             *                - Becomes positive if retry succeeds
             * @param[out] tkelvV Temperature array [1 × nPassed]
             * @param[out] sddV Auxiliary output array [nsdd × nPassed]
             * @param[out] mtanSDV Tangent stiffness array [nsvec² × nPassed] or nullptr
             * @param[in] nPassed Number of material points
             * 
             * **Error handling**:
             * - ECMECH_FAIL if !m_complete (when ECMECH_EXTRA_SOLVERS defined)
             * - ECMECH_FAIL if backup solver fails (any point still has nFEval < 0)
             * - ECMECH_FAIL immediately if ECMECH_EXTRA_SOLVERS not defined
             * 
             * @note const method despite modifying arrays (modifies via pointers)
             * @note Parameters marked UNUSED_EXTRA for non-ECMECH_EXTRA_SOLVERS builds
             * @note Dynamic slip systems: creates thread-local copy to avoid race conditions
             * 
             * @warning If backup solver also fails, simulation terminates with ECMECH_FAIL
             */
            __ecmech_host__
            inline
            void getResponseRetry( const double & UNUSED_EXTRA(dt),
                                   const double * UNUSED_EXTRA(def_rate_d6vV),
                                   const double * UNUSED_EXTRA(spin_vecV),
                                   const double * UNUSED_EXTRA(rel_vol_ratiosV),
                                   double * UNUSED_EXTRA(internal_energyV),
                                   double * UNUSED_EXTRA(cauchy_stress_d6pV),
                                   double * UNUSED_EXTRA(histV),
                                   double * UNUSED_EXTRA(tkelvV),
                                   double * UNUSED_EXTRA(sddV),
                                   double * UNUSED_EXTRA(mtanSDV),
                                   const int& UNUSED_EXTRA(nPassed)
                                 ) const
            {
#if defined(ECMECH_EXTRA_SOLVERS)
               if (!m_complete) {
                  ECMECH_FAIL(__func__, "not complete");
               }

               // All of the stride lengths are constant within this function
               const unsigned int def_rate_stride = m_strides[istride_def_rate];
               const unsigned int spin_v_stride = m_strides[istride_spin_v];
               const unsigned int vol_ratio_stride = m_strides[istride_vol_ratio];
               const unsigned int int_eng_stride = m_strides[istride_int_eng];
               const unsigned int stress_stride = m_strides[istride_stress];
               const unsigned int history_stride = m_strides[istride_history];
               const unsigned int tkelv_stride = m_strides[istride_tkelv];
               const unsigned int sdd_stride = m_strides[istride_sdd];

               const auto slipGeom = m_slipGeom;
               const auto kinetics = m_kinetics;
               const auto elastN = m_elastN;
               const auto eosModel = m_eosModel;
               const auto tolerance = m_tolerance;
               const auto outputLevel = m_outputLevel;

               snls::forall(0, nPassed, [=]
                  __ecmech_hdev__
                  (int i)
               {
                  // skip elements that were successful
                  if (histV[history_stride * i + iHistA_nFEval] >= 0) return;

                  double *mtanSDThis = (mtanSDV ? &mtanSDV[ecmech::nsvec2 * i] : nullptr);
                  auto get_response = [=] (const SlipGeom& slip_geom) -> bool {
                     return getResponseNRSngl<SlipGeom, Kinetics, ThermoElastN, EosModel>
                     (slip_geom, *kinetics, *elastN, *eosModel,
                        dt,
                        tolerance,
                        &def_rate_d6vV[def_rate_stride * i],
                        &spin_vecV[spin_v_stride * i],
                        &rel_vol_ratiosV[vol_ratio_stride * i],
                        &internal_energyV[int_eng_stride * i],
                        &cauchy_stress_d6pV[stress_stride * i],
                        &histV[history_stride * i],
                        tkelvV[tkelv_stride * i],
                        &sddV[sdd_stride * i],
                        mtanSDThis,
                        outputLevel);
                  };
                  bool status;
                  // Thanks to NVCC being difficult we have to create an unnecessary temp variable just so we can use our
                  // lambda expression in an if constexpr...
                  auto slipGeom_tmp = slipGeom;
                  // If we have dynamic slip systems then we need to create
                  // a thread local slip geometery class or else we might run into
                  // race condition issues...
                  if constexpr (SlipGeom::dynamic) {
                     SlipGeom slip_geom = *slipGeom_tmp;
                     status = get_response(slip_geom);
                  } else {
                     status = get_response(*slipGeom_tmp);
                  }
                  if (!status) {
                     histV[history_stride * i + iHistA_nFEval] *= -1;
                  }
               });

               if (this->reduceStatus(histV, nPassed)) {
                  ECMECH_FAIL(__func__, "Back-up solvers failed to converge for at least one point!");
               }
#else
               ECMECH_FAIL(__func__, "Solver failed to converge for at least one point!");
#endif
            } // End of getResponseRetry

            /**
             * @brief Check if any material points failed to converge.
             * 
             * Performs parallel reduction over all material points to detect convergence
             * failures. Used by getResponseECM() to determine if getResponseRetry() should
             * be called.
             * 
             * **Failure detection**:
             * - Checks histV[i * history_stride + iHistA_nFEval] for each point i
             * - nFEval < 0 indicates convergence failure (set by getResponseECM)
             * - Returns true if ANY point failed
             * 
             * **Parallel strategy** (execution-dependent):
             * - **CPU**: RAJA::seq_exec with RAJA::ReduceBitOr<seq_reduce>
             * - **OpenMP**: RAJA::omp_parallel_for_exec with RAJA::ReduceSum<omp_reduce_ordered>
             * - **GPU**: RAJA::cuda_exec/hip_exec with RAJA::ReduceBitOr<cuda_reduce/hip_reduce>
             * 
             * @param[in] histV History variable array
             * @param[in] nPassed Number of material points
             * 
             * @return true if any point has nFEval < 0 (convergence failure)
             * @return false if all points converged successfully
             * 
             * @note Const method - does not modify model state
             * @note Performance: O(nPassed) reduction, parallelized across execution strategy
             */
            __ecmech_host__
            bool reduceStatus(const double* const histV,
                              const int& nPassed) const
            {
               RAJA::RangeSegment default_range(0, nPassed);
               const unsigned int history_stride = m_strides[istride_history];
               switch (m_accel) {
#if defined(RAJA_ENABLE_OPENMP)
                  case ECM_EXEC_STRAT_OPENMP:
                  {
                     RAJA::ReduceSum<RAJA::omp_reduce_ordered, int> status_all(0);
                     RAJA::forall<RAJA::omp_parallel_for_exec>(default_range, [ = ] (int i) {
                        status_all += (histV[history_stride * i + iHistA_nFEval] < 0);
                     });
                     return status_all.get() > 0;
                     break;
                  }
#endif
#if defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_HIP)
                  case ECM_EXEC_STRAT_GPU:
                  {
#if defined(RAJA_ENABLE_CUDA)
                     using gpu_reduce = RAJA::cuda_reduce;
                     using gpu_policy = RAJA::cuda_exec<ECMECH_GPU_THREADS>;
#else
                     using gpu_reduce = RAJA::hip_reduce;
                     using gpu_policy = RAJA::hip_exec<ECMECH_GPU_THREADS>;
#endif
                     RAJA::ReduceBitOr<gpu_reduce, bool> status_all(false);
                     RAJA::forall<gpu_policy>(default_range, [=] RAJA_DEVICE(int i) {
                        status_all |= (histV[history_stride * i + iHistA_nFEval] < 0);
                     });
                     return status_all.get();
                     break;
                  }
#endif
                  case ECM_EXEC_STRAT_CPU:
                  default: // fall through to CPU if other options are not available
                  {
                     RAJA::ReduceBitOr<RAJA::seq_reduce, bool> status_all(false);
                     RAJA::forall<RAJA::seq_exec>(default_range, [ = ] (int i) {
                        status_all |= (histV[history_stride * i + iHistA_nFEval] < 0);
                     });
                     return status_all.get();
                     break;
                  }
               } // switch _accel
            }

            using matModelBase::getHistInfo;
            /**
             * @brief Get history variable information.
             * 
             * Retrieves metadata for all history variables including names, initial
             * values, and output flags. Data is stored in m_rhv* member vectors during
             * initFromParams().
             * 
             * @param[out] names Variable names (resized and filled)
             * @param[out] vals Initial values (resized and filled)
             * @param[out] plot Output flags (resized and filled)
             * @param[out] state State variable flags (resized and filled)
             * 
             * **History variable categories** (in order):
             * 1. Effective shear rate / plastic strain rate
             * 2. Effective shear / plastic strain
             * 3. Flow strength
             * 4. Number of function evaluations
             * 5. Elastic lattice strain (5 components)
             * 6. Crystal orientation (quaternion, 4 components)
             * 7. Kinetics hardness state (nH components)
             * 8. Slip system shear rates (nslip components)
             * 
             * **Errors**:
             * - ECMECH_FAIL if m_rhvNames not initialized (before initFromParams)
             * - ECMECH_FAIL if size mismatch with numHist
             * 
             * @note All output vectors resized to numHist and filled
             */
            __ecmech_host__
            void getHistInfo(std::vector<std::string> & names,
                             std::vector<double>      & vals,
                             std::vector<bool>        & plot,
                             std::vector<bool>        & state) const override final {
               if (m_rhvNames.size() != numHist) {
                  ECMECH_FAIL(__func__, "have not yet set up history information");
               }
               names.resize(numHist); std::copy(m_rhvNames.begin(), m_rhvNames.end(), names.begin() );
               vals.resize(numHist); std::copy(m_rhvVals.begin(), m_rhvVals.end(), vals.begin() );
               plot.resize(numHist); std::copy(m_rhvPlot.begin(), m_rhvPlot.end(), plot.begin() );
               state.resize(numHist); std::copy(m_rhvState.begin(), m_rhvState.end(), state.begin() );
            }

            /**
             * @brief Get total number of history variables.
             * 
             * Returns compile-time constant numHist computed from template parameters.
             * 
             * @return Number of history variables per material point
             * 
             * **Components**:
             * - 4: Auxiliary (shrate_eff, shr_eff, flow_str, n_feval)
             * - 5: Elastic strain deviatoric
             * - 4: Quaternion orientation
             * - nH: Kinetics hardness state
             * - nslip: Slip system shear rates
             * 
             * Total: 4 + 5 + 4 + nH + nslip = numHist
             */
            __ecmech_host__
            int getNumHist( ) const override final {
               return numHist;
            }

            /**
             * @brief Finalize model setup.
             * 
             * Called after initFromParams() to finalize model configuration:
             * 1. Extract reference bulk modulus from EOS model
             * 2. Set m_complete = true
             * 
             * After this call:
             * - getResponseECM() can be called
             * - initFromParams() can be called again (reinitializes)
             * - updateStrides() cannot be called
             * 
             * @note Must be called before getResponseECM()
             */
            __ecmech_host__
            void complete( ) override final
            {
               m_bulkRef = (*m_eosModel).getBulkRef();
               m_complete = true;
            }

            using ExecStrat = ecmech::ExecutionStrategy;

            /**
             * @brief Set execution strategy for parallel operations.
             * 
             * Configures both ECMech and SNLS execution backends to use consistent
             * parallelization strategy.
             * 
             * @param[in] accel Desired execution strategy
             * 
             * **Available strategies** (compile-time dependent):
             * - ECM_EXEC_STRAT_CPU: Always available (serial)
             * - ECM_EXEC_STRAT_GPU: If __ecmech_gpu_active__ defined
             * - ECM_EXEC_STRAT_OPENMP: If RAJA_ENABLE_OPENMP and OPENMP_ENABLE defined
             * 
             * **Behavior**:
             * - Sets m_accel for ECMech
             * - Calls setSNLSExecutionStrategy() for SNLS backend
             * - Unsupported strategies fall back to CPU
             * 
             * @note Can be called multiple times to switch strategies
             * @note Caller responsible for ensuring data accessibility (e.g., GPU memory)
             * 
             * @see setSNLSExecutionStrategy()
             */
            __ecmech_host__
            void setExecutionStrategy(ExecStrat accel) override final  {
               switch (accel) {
#ifdef __ecmech_gpu_active__
                  case (ECM_EXEC_STRAT_GPU): {
                     m_accel = ECM_EXEC_STRAT_GPU;
                     break;
                  }
#endif
#if defined(RAJA_ENABLE_OPENMP) && defined(OPENMP_ENABLE)
                  case (ECM_EXEC_STRAT_OPENMP): {
                     m_accel = ECM_EXEC_STRAT_OPENMP;
                     break;
                  }
#endif
                  case (ECM_EXEC_STRAT_CPU):
                  default: {
                     m_accel = ECM_EXEC_STRAT_CPU;
                     break;
                  }
               }
               this->setSNLSExecutionStrategy(m_accel);
            }

            /**
             * @brief Configure SNLS solver execution backend.
             * 
             * Sets the SNLS (nonlinear solver) library's execution strategy to match
             * ECMech's strategy. Called automatically by setExecutionStrategy().
             * 
             * @param[in] accel Execution strategy
             * 
             * **Backend mapping**:
             * - ECM_EXEC_STRAT_CPU → snls::ExecutionStrategy::CPU
             * - ECM_EXEC_STRAT_GPU → snls::ExecutionStrategy::GPU
             * - ECM_EXEC_STRAT_OPENMP → snls::ExecutionStrategy::OPENMP
             * 
             * **SNLS Device singleton**:
             * - Uses snls::Device::GetInstance()
             * - Configures global backend for all SNLS operations
             * - Settings persist across matModel instances
             * 
             * @note Typically called indirectly via setExecutionStrategy()
             * @note Affects all SNLS operations, not just this model
             */
            __ecmech_host__
            void setSNLSExecutionStrategy(ExecStrat accel) const
            {
               snls::Device &device = snls::Device::GetInstance();
               switch (accel) {
#ifdef __ecmech_gpu_active__
                  case (ECM_EXEC_STRAT_GPU): {
                     device.SetBackend(snls::ExecutionStrategy::GPU);
                     break;
                  }
#endif
#if defined(RAJA_ENABLE_OPENMP) && defined(OPENMP_ENABLE)
                  case (ECM_EXEC_STRAT_OPENMP): {
                     device.SetBackend(snls::ExecutionStrategy::OPENMP);
                     break;
                  }
#endif
                  case (ECM_EXEC_STRAT_CPU):
                  default: {
                     device.SetBackend(snls::ExecutionStrategy::CPU);
                     break;
                  }
               }
            }

            /** @} */


            /**
             * @name Component Accessors
             * @brief Access to underlying templated component objects.
             * 
             * These accessors provide const references to the internal component models.
             * Useful for:
             * - Computing derived quantities (e.g., sample frame plastic deformation rate)
             * - Accessing slip system geometry (Schmid tensors)
             * - Extracting material properties (elastic constants)
             * 
             * **Stability warning**: The API of the returned classes may change between
             * point releases. These accessors are provided for advanced usage but are
             * not guaranteed to be stable.
             * 
             * @note All accessors dereference managed pointers to return references
             * @note Only available after initFromParams() has been called
             * @{
             */

            /**
             * @brief Get slip system geometry component.
             * @return Const reference to SlipGeom instance
             */
            const SlipGeom & getSlipGeom() const { return *m_slipGeom; }

            /**
             * @brief Get kinetics (strength evolution) component.
             * @return Const reference to Kinetics instance
             */
            const Kinetics & getKinetics() const { return *m_kinetics; }

            /**
             * @brief Get thermoelastic component.
             * @return Const reference to ThermoElastN instance
             */
            const ThermoElastN & getThermoElastN() const { return *m_elastN; }

            /**
             * @brief Get equation of state component.
             * @return Const reference to EosModel instance
             */
            const EosModel & getEosModel() const { return *m_eosModel; }
            /** @} */

         private:

            /**
             * @name Private Member Variables
             * @brief Internal state and component storage.
             * @{
             */


#if defined(SNLS_RAJA_PORT_SUITE)
            /**
             * @brief Managed pointer to slip geometry component.
             */
            chai::managed_ptr<SlipGeom> m_slipGeom;
            /**
             * @brief Managed pointer to slip kinetics / slip hardening component.
             */
            chai::managed_ptr<Kinetics> m_kinetics;
            /**
             * @brief Managed pointer to thermoelastic component .
             */
            chai::managed_ptr<ThermoElastN> m_elastN;
            /**
             * @brief Managed pointer to EOS component .
             */
            chai::managed_ptr<EosModel> m_eosModel;
#else
            /**
             * @brief Managed pointer to slip geometry component.
             */
            internal::pcmptr<SlipGeom> m_slipGeom;
            /**
             * @brief Managed pointer to slip kinetics / slip hardening component.
             */
            internal::pcmptr<Kinetics> m_kinetics;
            /**
             * @brief Managed pointer to thermoelastic component .
             */
            internal::pcmptr<ThermoElastN> m_elastN;
            /**
             * @brief Managed pointer to EOS component .
             */
            internal::pcmptr<EosModel> m_eosModel;
#endif
            /**
             * @brief Nonlinear solver tolerance.
             * 
             * Convergence criterion for SNLS Newton-Raphson iterations.
             * Typical values: 1e-8 to 1e-12
             */
            double m_tolerance;
            /**
             * @brief Array stride configuration [ECMECH_NSTRIDE elements].
             * 
             * Defines memory layout for getResponseECM arrays:
             * - [0]: def_rate stride (≥ nsvp)
             * - [1]: spin_v stride (≥ ndim)
             * - [2]: vol_ratio stride (≥ nvr)
             * - [3]: int_eng stride (≥ ne)
             * - [4]: stress stride (≥ nsvp)
             * - [5]: history stride (≥ numHist)
             * - [6]: tkelv stride (≥ 1)
             * - [7]: sdd stride (≥ nsdd)
             */
            unsigned int m_strides[ecmech::nstride];

            /**
             * @brief History variable names.
             * Size: numHist
             * Populated during initFromParams()
             */
            std::vector<std::string> m_rhvNames;
            /**
             * @brief History variable initial values.
             * Size: numHist
             * Populated during initFromParams()
             */
            std::vector<double> m_rhvVals;
            /**
             * @brief History variable plot flags.
             * Size: numHist
             * true = include in standard output
             */
            std::vector<bool> m_rhvPlot;
            /**
             * @brief History variable state flags.
             * Size: numHist
             * true = essential state variable (not derived)
             */
            std::vector<bool> m_rhvState;

            /**
             * @brief Stored integer options from initFromParams.
             * Used by getParams() to return original values.
             */
            std::vector<int> m_opts;
            /**
             * @brief Stored double parameters from initFromParams.
             * Used by getParams() to return original values.
             */
            std::vector<double> m_pars;
            /**
             * @brief Stored string parameters from initFromParams.
             * Used by getParams() to return original values.
             */
            std::vector<std::string> m_strs;
            /** @} */

      }; // class matModel
   } // namespace evptn
} // namespace ecmech

#endif // ecmech_evptnWrap_include
