# Developer Guide
This is a working document and will likely need to be updated as further refactors occur in the future. Alternatively, if we realize things need better documentation then this will be updated.

# Name convention:
We currently don't have a consistent naming convention defined as we didn't create one beforehand. However, here is one we'll try and strive towards in future refactors.

Class and Struct types generally should follow a PascalCase convention aka `SuperCoolClassName`.

Variable names should generally be lower case with `_` between words. Although, we're still working on improving consistency in-regards to that convention across the library.

Class member variables, we generally want those to start `m_` and then `var_name` so they would be `m_var_name`. The `m_` here denotes a member variable. However, you will notice that for the `static constexpr int/size_t` variables we drop the `m_` and just have the name. The decision behind this was that these are static and constant properties of the class, and don't represent parameters that the user is able to define at runtime or that might be derived from runtime defined parameters.

Struct member variables differ from class member variables as they are generally just data containers without any functions. So, we don't require the names to look like `m_var_name` but we instead expect them to be `var_name`.

Function names should also be lowercase with `_` between names. However as earlier noted, we still have not exactly enforced this at this point.

# Variable Names:

We have a wide range of different variables used through out the code. This is a short guide on at least some of them and is by no means a complete guide to everything in the library.

Next, we try to add the following substrings to names if the variables refer to quantities that either are some form of a deviatoric, deviatoric + pressure, or some variation of those vector quantities in Voigt notation. `d5` refers to a deviatoric vector that makes use of the actual 5D space representation. `d6` refers to a deviatoric vector that has values in the expanded 6D space. If the `d5` or `d6` has a character following that it refers to either the pressure (`p`) or volume (`v`), and these vectors are concatenation of the deviatoric and volumetric/pressure terms. So, a variable with a `d6p` substring would be of length 7 where the first 6 terms are the deviatoric terms and the final is the pressure term.

First, `elast` suffix names refer to the elastic strain and generally this involves quantities in the crystal/lattice frame.

`def_rate` is the deforamtion rate which is the symmetric portion of the velocity gradient.

`spin_vec` is `vect(W)` where W is the skew portion of the velocity gradient.

`rel_vol_ratios` is maybe not the best named array but it's made up of what can be thought of terms related to the relative volume of the point your dealing with where the 2 first indices (0 & 1) correspond to the old and new relative volumes respectively. Here, a relative volume > 1 signals material under volumetric expansion and a relative volume < 1 signals material under volumetric compression. Traditionally, one might set this to begin as a value of 1 but that is not a hard requirement.

Terms with `quat` in the name generally refer to the unit quaternion that maps the rotation of the starting crystal lattice reference frame to the current sample frame.

Terms that look like `h_state` refer to the hardening state.

Terms that have the name `gdot` refer to the macroscopic shearing rate per slip system of the material commonly seen written as $\dot{\gamma}$ in the literature.

`tkelv` is the temperature which is most commonly provided in Kelvin.

Terms with the suffix `_n` commonly refer to the begining of time step information.

Terms with the suffix `_u` or `_f` commonly refer to the end of time step information.

# The value type and the CPU SIMD path:

ExaCMech's point-wise numerics (the slip geometry's projections, the kinetics, the elasticity, the `EvptnUpdstProblem` residual and Jacobian, and the update driver) are templates on a value type `R`:

- **`R = double`: one material point.** This is the GPU path (one point per thread) and the CPU path for models without pack support.
- **`R = ecmech::simd::PackDouble`: a pack of material points solved in lockstep.** This is `snls::simd::pack<ECMECH_SIMD_LANES>`, one double per point. It is the path for every supported model on the CPU and OpenMP backends, and it is always on: the compiler vectorizes the operations on a pack, which is where the CPU speed-up comes from.

The value-type layer (masks, `select`, `set_where`, `any`, `all`, the `_where` math functions) and the pack type come from SNLS (`SNLS_value_type.h`, `SNLS_pack.h`); `ECMech_value_type.h` brings them into namespace `ecmech`. [SIMDGuide.md](SIMDGuide.md) shows, with before/after examples, how to write and convert code for it.

## Code path

`matModel::getResponseECM` (`ECMech_evptnWrap.h`) chooses the path for each call:

1. **CPU and OpenMP backends, model supported by the pack path** (`simd::simd_supported`): `simd::getResponseSimd` (`simd/ECMech_simd_evptn.h`) splits the points into packs of `ECMECH_SIMD_LANES` points. It runs the packs with `RAJA::forall` (OpenMP when SNLS and RAJA were built with OpenMP, else sequentially) and calls `evptn::getResponseLanes<PackDouble>` for each pack. If the last pack is not full, its extra lanes repeat the last point on scratch copies, so they write nothing.
2. **The GPU, and models the pack path does not support:** `getResponsePerPoint` runs `evptn::getResponseSngl` for each point with `snls::forall` (blocks of `ECMECH_GPU_THREADS` threads on the GPU). `getResponseSngl` is `getResponseLanes<double>` on the point's arrays.
3. **On all backends,** points that failed to converge are marked (`hist[iHistA_nFEval] *= -1`), and `getResponseRetry` solves them again with the backup solvers (`ENABLE_EXTRA_SOLVERS`).

`evptn::getResponseLanes<R>` (`evptn/ECMech_evptnLanes.h`) is the one update driver for both value types. It runs these steps:

1. **Per point:** the `ProblemState` and `preprocess` (equation of state, hardness update).
   - For a pack, kinetics that provide `updateHLanes` update the hardness on the whole pack instead (the Orowan kinetics' reduced solve).
2. **The problem state the solve works on.**
   - For one point this is the point's own `ProblemState`.
   - For a pack, the per-point states are gathered into a `ProblemStatePack` (and a stress-dependent slip geometry into a `SlipGeomPack`).
3. **The Newton solve** of `EvptnUpdstProblem<..., R>` with SNLS's trust-region solver on `R` (`snls::SNLSTrDlDenseG<Prob, n, R>`). Each lane runs exactly the one-point iteration.
4. **The tangent, the state update and the slip quantities** for postprocessing, on `R`.
5. **Per point:** the results are scattered back, then `postprocess` runs.

For `R = double` the gather and scatter steps compile away (`if constexpr`), and the driver performs the same operations as the original one-point code.

## Build options

| CMake option | default | what it does |
|---|---|---|
| `ECMECH_SIMD_LANES` | 4 | material points per pack on the CPU (the pack width). Wider packs can help models with expensive kinetics, narrower ones the others; measure before changing it |
| `ECMECH_GPU_THREADS` | 256 | threads per block of ExaCMech's GPU kernels |
| `ENABLE_PACK_VECTOR_MATH` (SNLS) | OFF | lets the pack's `exp`/`log`/`pow` use the compiler's vector math library. Set when building SNLS. It currently applies to CPU builds with Clang on Linux; the flags are passed on to ExaCMech through the SNLS target |

Both ExaCMech options reach the code through the generated `ECMech_config.h`.

## Supporting a model on the pack path

A model runs on packs when its kinetics class provides `evalGdots` templated on the value type (`evptn::pack_supported`, checked by `simd::simd_supported`). Models without it run one point at a time on the CPU, with no code change needed. To add pack support to a model, its classes need the following (SIMDGuide.md shows how to convert the code):

- **Kinetics.** `template<class R> void evalGdots(R* gdot, R* dgdot_dtau, const R* tau, const R* vals) const`, and whatever it calls, templated on `R`.
  - `getVals` stays scalar; the driver calls it point by point (`kinetic_values`).
  - The hardness update stays scalar (`updateH` in `preprocess`), unless the kinetics also provides `template<class R> mask_t<R> updateHLanes(R* hs_u, const R* hs_o, double dt, const R* gdot) const`, which updates a whole pack and returns the points that converged.
- **Slip geometry.** A fixed geometry (P and Q the same for every point) works as is. A stress-dependent geometry (`dynamic = true`) is gathered per lane into a `SlipGeomPack`. If its resolved shear stress depends on the stress only, it can declare `static constexpr bool rss_from_stress = true` and provide a templated `evalRSS`, as `SlipGeomBCCNonSchmid` does.
- **Elasticity.** The `ThermoElastN` methods used by `EvptnUpdstProblem` are templated on `R` (`ThermoElastNCubic`, `ThermoElastNHexag`).
- **Equation of state.** It stays per point (called in `preprocess`).

## GPU code

Device code always uses `double`; packs are host-only. In code that runs on the GPU:

- **No `std::` math, no exceptions, no placement new.** Call math functions unqualified (`exp`, not `std::exp`).
- **`__ecmech_hdev__`** marks functions for both host and device, as elsewhere in ExaCMech.
- **`__ecmech_device_noinline__`** (`ECMech_gpu_portability.h`) keeps a function out of line in device code only; it is still optimized, and on the host it is inlined as usual. `SlipGeomBCCNonSchmid::NSprojection` uses it: inlined into the very large, fully inlined getResponse kernel of the non-Schmid models, it led a GPU compiler to wrong results at `-O2` and above.

## Testing

**Pack and one-point results must agree bit for bit.** For `R = double` the templated code performs the original operations, so with floating-point contraction off a pack gives each point exactly its one-point result. Check that after any change to templated code:

1. Build twice with `-ffp-contract=off` in `CMAKE_CXX_FLAGS`: once with `ECMECH_SIMD_LANES=1` (one point per pack) and once with the default width.
2. Run the miniapp on the CPU backend with both builds.
3. Compare the `CHECKSUM` lines the miniapp prints at the end: they must be identical.

With contraction on (the default), the compiler may fuse multiply-adds differently in the two versions, so the last digits can differ.

**GPU and CPU agree to rounding.** The GPU backend uses the one-point path. Compare its `CHECKSUM` lines with the CPU backend's; they agree to rounding (about 1e-12 relative or better), not bitwise, because the device math library differs from the host's.

# Further Notes:
If you find something unclear in either the documentation or code please open an issue asking for clarification. Alternatively, if you find issues with those same topics feel free to open a PR with a fix as generally our developer team's bandwidth is limited so some of these items will be lower on the priority list for us.

