#pragma once
/**
 * @file ECMech_dpack.h
 * @brief Experiment: dpack<N, Wp>, N points in one value, usable on the GPU
 * (each GPU thread then works on N points) and on the CPU. A plain
 * array-of-doubles type with element-wise arithmetic, comparisons that give
 * an mpack<N, Wp>, element-wise math, and the value-type traits
 * (SNLS_value_type.h).
 */

#include <cmath>
#include <type_traits>
#include "SNLS_value_type.h"
#include "ECMech_gpu_portability.h"

namespace ecmech {

template<int N, bool Warp = false>
struct mpack {
   bool v[N];
   __ecmech_hdev__ mpack() = default;
   __ecmech_hdev__ explicit mpack(bool b) { for (int i = 0; i < N; ++i) { v[i] = b; } }
};
template<int N, bool Wp> __ecmech_hdev__ inline mpack<N, Wp> operator&&(const mpack<N, Wp>& a, const mpack<N, Wp>& b) { mpack<N, Wp> r; for (int i = 0; i < N; ++i) { r.v[i] = a.v[i] && b.v[i]; } return r; }
template<int N, bool Wp> __ecmech_hdev__ inline mpack<N, Wp> operator||(const mpack<N, Wp>& a, const mpack<N, Wp>& b) { mpack<N, Wp> r; for (int i = 0; i < N; ++i) { r.v[i] = a.v[i] || b.v[i]; } return r; }
template<int N, bool Wp> __ecmech_hdev__ inline mpack<N, Wp> operator!(const mpack<N, Wp>& a) { mpack<N, Wp> r; for (int i = 0; i < N; ++i) { r.v[i] = !a.v[i]; } return r; }

template<int N, bool Warp = false>
struct dpack {
   double v[N];
   __ecmech_hdev__ dpack() = default;
   __ecmech_hdev__ dpack(double x) { for (int i = 0; i < N; ++i) { v[i] = x; } }
   __ecmech_hdev__ dpack& operator+=(const dpack& b) { for (int i = 0; i < N; ++i) { v[i] += b.v[i]; } return *this; }
   __ecmech_hdev__ dpack& operator-=(const dpack& b) { for (int i = 0; i < N; ++i) { v[i] -= b.v[i]; } return *this; }
   __ecmech_hdev__ dpack& operator*=(const dpack& b) { for (int i = 0; i < N; ++i) { v[i] *= b.v[i]; } return *this; }
   __ecmech_hdev__ dpack& operator/=(const dpack& b) { for (int i = 0; i < N; ++i) { v[i] /= b.v[i]; } return *this; }
};

#define ECMECH_DPACK_BINOP(OP) \
   template<int N, bool Wp> __ecmech_hdev__ inline dpack<N, Wp> operator OP(const dpack<N, Wp>& a, const dpack<N, Wp>& b) { dpack<N, Wp> r; for (int i = 0; i < N; ++i) { r.v[i] = a.v[i] OP b.v[i]; } return r; } \
   template<int N, bool Wp> __ecmech_hdev__ inline dpack<N, Wp> operator OP(const dpack<N, Wp>& a, double b) { dpack<N, Wp> r; for (int i = 0; i < N; ++i) { r.v[i] = a.v[i] OP b; } return r; } \
   template<int N, bool Wp> __ecmech_hdev__ inline dpack<N, Wp> operator OP(double a, const dpack<N, Wp>& b) { dpack<N, Wp> r; for (int i = 0; i < N; ++i) { r.v[i] = a OP b.v[i]; } return r; }
ECMECH_DPACK_BINOP(+)
ECMECH_DPACK_BINOP(-)
ECMECH_DPACK_BINOP(*)
ECMECH_DPACK_BINOP(/)
#undef ECMECH_DPACK_BINOP
template<int N, bool Wp> __ecmech_hdev__ inline dpack<N, Wp> operator-(const dpack<N, Wp>& a) { dpack<N, Wp> r; for (int i = 0; i < N; ++i) { r.v[i] = -a.v[i]; } return r; }

#define ECMECH_DPACK_CMP(OP) \
   template<int N, bool Wp> __ecmech_hdev__ inline mpack<N, Wp> operator OP(const dpack<N, Wp>& a, const dpack<N, Wp>& b) { mpack<N, Wp> r; for (int i = 0; i < N; ++i) { r.v[i] = a.v[i] OP b.v[i]; } return r; } \
   template<int N, bool Wp> __ecmech_hdev__ inline mpack<N, Wp> operator OP(const dpack<N, Wp>& a, double b) { mpack<N, Wp> r; for (int i = 0; i < N; ++i) { r.v[i] = a.v[i] OP b; } return r; } \
   template<int N, bool Wp> __ecmech_hdev__ inline mpack<N, Wp> operator OP(double a, const dpack<N, Wp>& b) { mpack<N, Wp> r; for (int i = 0; i < N; ++i) { r.v[i] = a OP b.v[i]; } return r; }
ECMECH_DPACK_CMP(<)
ECMECH_DPACK_CMP(<=)
ECMECH_DPACK_CMP(>)
ECMECH_DPACK_CMP(>=)
ECMECH_DPACK_CMP(==)
ECMECH_DPACK_CMP(!=)
#undef ECMECH_DPACK_CMP

#define ECMECH_DPACK_FN1(F) \
   template<int N, bool Wp> __ecmech_hdev__ inline dpack<N, Wp> F(const dpack<N, Wp>& a) { dpack<N, Wp> r; for (int i = 0; i < N; ++i) { r.v[i] = ::F(a.v[i]); } return r; }
ECMECH_DPACK_FN1(exp)
ECMECH_DPACK_FN1(log)
ECMECH_DPACK_FN1(sqrt)
ECMECH_DPACK_FN1(fabs)
ECMECH_DPACK_FN1(sin)
ECMECH_DPACK_FN1(cos)
#undef ECMECH_DPACK_FN1
#define ECMECH_DPACK_FN2(F) \
   template<int N, bool Wp> __ecmech_hdev__ inline dpack<N, Wp> F(const dpack<N, Wp>& a, const dpack<N, Wp>& b) { dpack<N, Wp> r; for (int i = 0; i < N; ++i) { r.v[i] = ::F(a.v[i], b.v[i]); } return r; } \
   template<int N, bool Wp> __ecmech_hdev__ inline dpack<N, Wp> F(const dpack<N, Wp>& a, double b) { dpack<N, Wp> r; for (int i = 0; i < N; ++i) { r.v[i] = ::F(a.v[i], b); } return r; }
ECMECH_DPACK_FN2(pow)
ECMECH_DPACK_FN2(copysign)
ECMECH_DPACK_FN2(fmax)
ECMECH_DPACK_FN2(fmin)
#undef ECMECH_DPACK_FN2

} // namespace ecmech

namespace snls {
template<int N, bool Wp>
struct value_traits<ecmech::dpack<N, Wp>> {
   using R = ecmech::dpack<N, Wp>;
   using mask = ecmech::mpack<N, Wp>;
   /// pivot rows: held as doubles for several lanes, an int for one
   using index = std::conditional_t<(N > 1), R, int>;
   static constexpr int lanes = N;
   __snls_hdev__ static inline R select(const mask& m, const R& a, const R& b) {
      R r; for (int i = 0; i < N; ++i) { r.v[i] = m.v[i] ? a.v[i] : b.v[i]; } return r;
   }
   __snls_hdev__ static inline double get(const R& x, int l) { return x.v[l]; }
   __snls_hdev__ static inline void set(R& x, int l, double y) { x.v[l] = y; }
   __snls_hdev__ static inline R gather(const R* const b, const index& i) {
      if constexpr (N > 1) {
         R r; for (int l = 0; l < N; ++l) { r.v[l] = b[static_cast<int>(i.v[l])].v[l]; } return r;
      }
      else { return b[i]; }
   }
};
template<int N, bool Wp>
struct mask_traits<ecmech::mpack<N, Wp>> {
   using M = ecmech::mpack<N, Wp>;
   /// for a warp mask (Wp) the reduction is over the lanes of the GPU warp
   /// as well: every thread of the warp then takes the same branch
   __snls_hdev__ static inline bool any(const M& m) {
      bool r = false; for (int i = 0; i < N; ++i) { r = r || m.v[i]; }
#if defined(__CUDA_ARCH__)
      if constexpr (Wp) { r = __any_sync(__activemask(), r); }
#elif defined(__HIP_DEVICE_COMPILE__)
      if constexpr (Wp) { r = __any(r); }
#endif
      return r;
   }
   __snls_hdev__ static inline bool all(const M& m) {
      bool r = true; for (int i = 0; i < N; ++i) { r = r && m.v[i]; }
#if defined(__CUDA_ARCH__)
      if constexpr (Wp) { r = __all_sync(__activemask(), r); }
#elif defined(__HIP_DEVICE_COMPILE__)
      if constexpr (Wp) { r = __all(r); }
#endif
      return r;
   }
   __snls_hdev__ static inline bool get(const M& m, int l) { return m.v[l]; }
   __snls_hdev__ static inline void set(M& m, int l, bool x) { m.v[l] = x; }
};
} // namespace snls
