# Writing code for the value type and the SIMD path

ExaCMech's point-wise numerics are written once and run in two ways:

- **One point at a time** with `R = double`: on the GPU (one point per thread), and on the CPU for models without pack support.
- **A pack of points in lockstep** with `R = ecmech::simd::PackDouble`, on the CPU and OpenMP backends. That is `snls::simd::pack<n_lanes>`, `n_lanes` doubles, one per material point; the compiler turns the operations on them into SIMD instructions.

This guide shows how to write, or convert, code so that it works for both. [DeveloperGuide.md](DeveloperGuide.md) describes the code path, the build options and how a model gets pack support. The value-type layer itself lives in SNLS: `SNLS_value_type.h` and `SNLS_pack.h` (see the SNLS README).

The examples are pseudo-code close to the real code; the names are those of `ECMech_value_type.h`, which brings the SNLS functions into namespace `ecmech`.

## The rules in short

1. **Template on the value type.** Write `template<class R>` and use `R` for every per-point quantity. Parameters, constants and other values that are the same for all points stay `double`.
2. **Never branch on a per-point value directly.** A comparison of `R` values gives a mask, `mask_t<R>` (a `bool` for `double`). Use masks with `select`, `set_where`, `any` and `all`.
3. **Call math functions unqualified:** `exp(x)`, `sqrt(x)`, `fmax(a, b)`, never `std::exp`. The pack overloads are found by argument-dependent lookup, and `std::` math is not available in GPU device code.
4. **For `R = double` the code must do exactly what the original scalar code did.** Then the GPU path and the one-point path are unchanged, and a pack gives each point the one-point result.
5. **Packs are for host code only.** GPU code always uses `double`. Do not put packs in device memory or in objects copied to the GPU.

## The tools

| need | `double` | pack | use |
|---|---|---|---|
| per-point condition | `bool` | one flag per lane | `const auto m = (x > y);` |
| pick a value per point | `m ? a : b` | lane-wise blend | `select(m, a, b)` |
| assign only where a condition holds | `if (m) x = v;` | lane-wise | `set_where(m, x, v)` |
| does any / every point need it | `m` | reduction | `any(m)`, `all(m)` |
| expensive function for some points only | `if (m) y = exp(x);` | only the needed lanes | `y = exp_where(m, x)` (also `log_where`, `pow_where`) |
| all or no lanes | `true` / `false` | | `all_lanes<R>()`, `no_lanes<R>()` |
| one point's value | the value | lane `l` | `lane(x, l)`, `set_lane(x, l, v)`, `mask_lane(m, l)`, `set_mask_lane(m, l, b)` |
| number of points in `R` | 1 | `n_lanes` | `lanes_of<R>()` |
| code only for one point | | | `if constexpr (snls::is_scalar_value<R>()) { ... }` |

Masks combine with `&&`, `||` and `!`. These operate lane by lane and **do not short-circuit**: both sides are always evaluated.

## Example 1: a function of one point

Before:

```cpp
double flow_stress(double g, double tau, double rate_sens) {
   return g * pow(fabs(tau) / g, rate_sens);
}
```

After:

```cpp
template<class R>
__ecmech_hdev__ inline
R flow_stress(const R& g, const R& tau, double rate_sens) {   // rate_sens: same for all points
   return g * pow(fabs(tau) / g, rate_sens);
}
```

Nothing else changes: operators and math functions exist for packs, and a `double` argument mixes with `R` (`pow(R, double)`, `fmax(R, double)`, `2.0 * x`).

## Example 2: an if/else on a per-point value

Before:

```cpp
double gdot, dgdot_dtau;
if (fabs(tau) < t_min * g) {        // inactive slip system
   gdot = 0.0;
   dgdot_dtau = 0.0;
}
else {
   gdot = gam_w * pow(fabs(tau) / g, 1.0 / xm) * sign(tau);
   dgdot_dtau = gdot / (xm * tau);
}
```

After:

```cpp
R gdot = 0.0, dgdot_dtau = 0.0;
const mask_t<R> active = !(fabs(tau) < t_min * g);
if (any(active)) {                   // skip the work when no point needs it
   // a safe argument for the inactive lanes, so they cannot produce inf/NaN
   const R tau_safe = select(active, tau, R(1.0));
   const R ratio = fabs(tau_safe) / g;
   const R gdot_a = gam_w * pow_where(active, ratio, 1.0 / xm) * copysign(R(1.0), tau_safe);
   set_where(active, gdot, gdot_a);
   set_where(active, dgdot_dtau, gdot_a / (xm * tau_safe));
}
```

How this works:

- **For `double`** it does exactly what the original does. `any(active)` is the `if`, `set_where` is the assignment, and `select(active, tau, 1.0)` returns `tau` whenever the branch runs.
- **For a pack**, the branch runs if any lane needs it. All lanes compute, but only the active lanes take the results.
- **Inactive lanes need a safe argument.** They still execute the arithmetic, so give them one before dividing, taking a log, or raising to a power. Otherwise they produce inf or NaN: harmless in the result, but slow, and it can raise floating-point exceptions.
- **Use the `_where` math functions** (`exp_where`, `log_where`, `pow_where`) for expensive calls that only some lanes need. For a pack they evaluate only the lanes in the mask.

## Example 3: an early return or a per-point flag

Before:

```cpp
bool update(double& h, double dt) {
   if (dt * rate(h) > limit) { return false; }   // this point fails
   h += dt * rate(h);
   return true;
}
```

After:

```cpp
template<class R>
mask_t<R> update(R& h, double dt) {
   const R r = rate(h);
   const mask_t<R> ok = !(dt * r > limit);
   set_where(ok, h, h + dt * r);       // failed points keep h
   return ok;                          // the caller handles the points that failed
}
```

A pack cannot return early for some of its points. Turn the early return into a mask that the function returns, and let the points that failed keep their values. The callers then work with masks too. In `getResponseLanes`, for example, a point whose preprocess failed rides along in the pack on another point's data and is marked as failed at the end.

## Example 4: a loop that runs until each point converges

Before:

```cpp
for (int it = 0; it < max_iter; ++it) {
   double f, J;
   eval(f, J, x);
   if (fabs(f) < tol) { break; }
   x -= f / J;
}
```

After:

```cpp
mask_t<R> active = all_lanes<R>();
for (int it = 0; it < max_iter && any(active); ++it) {
   R f, J;
   eval(f, J, x);
   active = active && !(fabs(f) < tol);   // converged points drop out
   set_where(active, x, x - f / J);       // ... and keep their x
}
```

The loop continues while any lane is still iterating. A converged lane keeps its value, so it gets exactly the iterates of the one-point loop.

Do not write such a loop yourself for a nonlinear solve. SNLS's solvers do it, and they take a value type:

- **Trust region:** `snls::SNLSTrDlDenseG<Problem, n, R>`. The problem's `computeRJ(R* r, R* J, const R* x)` returns `mask_t<R>`. Read the results with `solver.getConverged()` and, per lane, `solver.getNFEvals(l)` and `solver.getStatus(l)`. Pass a mask to `solver.solve(active)` to solve only some lanes.
- **One equation:** `snls::NewtonBB<FJ, R, Step>`, whose `computeFJ(R& f, R& J, const R& x)` may also take `const mask_t<R>& active`. The `solve_orowan_mobile` function in the Orowan kinetics is an example.

## Example 5: code that stays per point

Some code is cheap, rarely called, or awkward to vectorize: parameter lookups, the equation of state, the kinetics' `getVals`. It can stay scalar. Wrap it in a loop over the lanes:

```cpp
template<class K, class R>
R kinetic_values(const K& k, R* vals, const R& pressure, const R& tkelv, const R* h_state) {
   if constexpr (snls::is_scalar_value<R>()) {
      return k.getVals(vals, pressure, tkelv, h_state);     // the scalar function itself
   }
   else {
      R scale;
      for (int l = 0; l < lanes_of<R>(); ++l) {             // one point at a time
         double v[K::nVals], h[K::nH];
         for (int j = 0; j < K::nH; ++j) { h[j] = lane(h_state[j], l); }
         set_lane(scale, l, k.getVals(v, lane(pressure, l), lane(tkelv, l), h));
         for (int j = 0; j < K::nVals; ++j) { set_lane(vals[j], l, v[j]); }
      }
      return scale;
   }
}
```

`kinetic_values` in `evptn/ECMech_evptn.h` is this function. The same pattern gathers point data into packs, and scatters results back, in `getResponseLanes`.

## Example 6: side effects that only make sense for one point

Some code writes state that belongs to one point, such as a slip geometry that stores its stress-dependent P and Q. Keep that part scalar with `if constexpr`:

```cpp
template<class R>
void projection(R* taua, double* P_out, const R* kirchoff, bool fill_PQ) const {
   for (int i_slip = 0; i_slip < nslip; ++i_slip) {
      // ... resolved shear stress of the two senses, tau_s[0] and tau_s[1], of type R
      const auto second = (tau_s[1] > tau_s[0]);
      taua[i_slip] = select(second, tau_s[1], tau_s[0]);
      if constexpr (snls::is_scalar_value<R>()) {   // P and Q belong to one point
         if (fill_PQ) { /* store the chosen sense's P and Q, using: second ? 1 : 0 */ }
      }
   }
}
```

`SlipGeomBCCNonSchmid::NSprojection` does this. For a pack, `getResponseLanes` keeps a copy of the geometry per lane, and `SlipGeomPack` gathers the per-lane P and Q.

## Example 7: per-point integers and indices

Pivot rows in an LU factorization, or a selected slip system, differ per point. Use `index_t<R>`: `int` for `double`, one integer per lane for a pack. Read through it with `value_traits<R>::gather(array, index)`. SNLS's dense LU (`SNLS_LUP_SolveXV`) shows the pattern.

## Pitfalls

- **No `?:` on masks.** `m ? a : b` compiles for `double` but not for a pack. Use `select(m, a, b)`.
- **Write constants of type `R` as `R(0.0)`** where a function needs two values of the same type, as in `select(m, x, R(0.0))`.
- **`any(m)` branches run every lane.** Inside them, assign through `set_where` or `select`, never with plain `=` to a per-point value that other lanes must keep.
- **Count per point.** For a per-point iteration count or failure flag, use a mask or an `int` array with one entry per lane. A single `int` or `bool` is shared by the whole pack.
- **The order of floating-point operations is part of the result.** For `R = double` keep the original operations and their order. Then with `-ffp-contract=off` a pack gives exactly the one-point results, which is how the code is tested (DeveloperGuide.md, "Testing").
