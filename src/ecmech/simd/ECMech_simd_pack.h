#pragma once
/**
 * @file ECMech_simd_pack.h
 * @brief Experiment (ECMECH_SIMD_EXPT, CPU only): the pack value type, W
 * points in one std::experimental::native_simd<double> (W = 4 with AVX2),
 * plugged into the value-type layer (SNLS_value_type.h), and vector math.
 */

#include <experimental/simd>
#include <cmath>
#include "SNLS_value_type.h"

namespace ecmech {
namespace simd {
namespace stdx = std::experimental;
using vdouble = stdx::native_simd<double>;
using vmask = vdouble::mask_type;
constexpr int W = static_cast<int>(vdouble::size());
} // namespace simd
} // namespace ecmech

namespace snls {
template<>
struct value_traits<ecmech::simd::vdouble> {
   using R = ecmech::simd::vdouble;
   using mask = ecmech::simd::vmask;
   using index = R; // pivot rows, held as doubles
   static constexpr int lanes = ecmech::simd::W;
   [[gnu::always_inline]] static inline R select(const mask& m, const R& a, const R& b) {
      R r = b; ecmech::simd::stdx::where(m, r) = a; return r;
   }
   static inline double get(const R& v, int l) { return v[l]; }
   static inline void set(R& v, int l, double x) { v[l] = x; }
   static inline R gather(const R* const b, const index& i) {
      R r;
      for (int l = 0; l < lanes; ++l) { r[l] = b[static_cast<int>(i[l])][l]; }
      return r;
   }
};
template<>
struct mask_traits<ecmech::simd::vmask> {
   using M = ecmech::simd::vmask;
   static inline bool any(const M& m) { return ecmech::simd::stdx::any_of(m); }
   static inline bool all(const M& m) { return ecmech::simd::stdx::all_of(m); }
   static inline bool get(const M& m, int l) { return m[l]; }
   static inline void set(M& m, int l, bool x) { m[l] = x; }
};
} // namespace snls

namespace ecmech {
// Unqualified exp/log/pow/sin/cos in namespace ecmech find these first, so
// the double overloads are brought in alongside (same functions as ::exp).
using std::exp; using std::log; using std::pow; using std::sin; using std::cos;
// Vector versions from glibc's libmvec (link -lmvec); libstdc++'s
// <experimental/simd> evaluates these one lane at a time. Non-template, so
// they win over the std::experimental templates found by ADL. libmvec is
// accurate to a few ulp, so pack results differ from double at rounding.
#if defined(__AVX2__) && !defined(__AVX512F__) && !defined(ECMECH_SIMD_NO_MVEC)
} // namespace ecmech
#include <immintrin.h>
extern "C" __m256d _ZGVdN4v_exp(__m256d);
extern "C" __m256d _ZGVdN4v_log(__m256d);
extern "C" __m256d _ZGVdN4v_sin(__m256d);
extern "C" __m256d _ZGVdN4v_cos(__m256d);
extern "C" __m256d _ZGVdN4vv_pow(__m256d, __m256d);
namespace ecmech {
using simd::vdouble;
inline vdouble exp(const vdouble& x) { return vdouble(_ZGVdN4v_exp(static_cast<__m256d>(x))); }
inline vdouble log(const vdouble& x) { return vdouble(_ZGVdN4v_log(static_cast<__m256d>(x))); }
inline vdouble sin(const vdouble& x) { return vdouble(_ZGVdN4v_sin(static_cast<__m256d>(x))); }
inline vdouble cos(const vdouble& x) { return vdouble(_ZGVdN4v_cos(static_cast<__m256d>(x))); }
inline vdouble pow(const vdouble& x, const vdouble& y) { return vdouble(_ZGVdN4vv_pow(static_cast<__m256d>(x), static_cast<__m256d>(y))); }
inline vdouble pow(const vdouble& x, double y) { return pow(x, vdouble(y)); }
#endif
} // namespace ecmech
