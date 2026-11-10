#pragma once
/**
 * @file ECMech_simd_core.h
 * @brief Experiment (ECMECH_SIMD_EXPT, CPU only): SIMD across material
 * points. A "pack" holds W points (W = 4 doubles with AVX2, 8 with
 * AVX-512) and every per-point quantity becomes one pack: the code below is
 * the original ExaCMech arithmetic templated on the value type R, which is
 * either double (one point) or vdouble (W points). Branches on per-point
 * values become select(mask, a, b); everything else reads like the
 * original.
 */

#include <experimental/simd>
#include <cmath>

#include "ECMech_core.h"
#include "ECMech_util.h"

namespace ecmech {
namespace simd {

namespace stdx = std::experimental;
using vdouble = stdx::native_simd<double>;
using vmask = vdouble::mask_type;
constexpr int W = static_cast<int>(vdouble::size());

// math: std:: for double, std::experimental:: (found by ADL) for vdouble
using std::exp; using std::log; using std::sqrt; using std::fabs; using std::copysign;
using std::fmax; using std::fmin; using std::sin; using std::cos; using std::pow;

// Vector exp/log/sin/cos from glibc's libmvec (link -lmvec); libstdc++'s
// <experimental/simd> evaluates these one lane at a time. Non-template
// overloads, so they win over the std::experimental templates. libmvec is
// accurate to a few ulp, so results differ from the scalar path at rounding.
#if defined(__AVX2__) && !defined(__AVX512F__) && !defined(ECMECH_SIMD_NO_MVEC)
#include <immintrin.h>
extern "C" __m256d _ZGVdN4v_exp(__m256d);
extern "C" __m256d _ZGVdN4v_log(__m256d);
extern "C" __m256d _ZGVdN4v_sin(__m256d);
extern "C" __m256d _ZGVdN4v_cos(__m256d);
inline vdouble exp(const vdouble& x) { return vdouble(_ZGVdN4v_exp(static_cast<__m256d>(x))); }
inline vdouble log(const vdouble& x) { return vdouble(_ZGVdN4v_log(static_cast<__m256d>(x))); }
inline vdouble sin(const vdouble& x) { return vdouble(_ZGVdN4v_sin(static_cast<__m256d>(x))); }
inline vdouble cos(const vdouble& x) { return vdouble(_ZGVdN4v_cos(static_cast<__m256d>(x))); }
extern "C" __m256d _ZGVdN4vv_pow(__m256d, __m256d);
inline vdouble pow(const vdouble& x, const vdouble& y) { return vdouble(_ZGVdN4vv_pow(static_cast<__m256d>(x), static_cast<__m256d>(y))); }
inline vdouble pow(const vdouble& x, double y) { return pow(x, vdouble(y)); }
#endif

template<class R> struct real_traits { using mask = bool; };
template<> struct real_traits<vdouble> { using mask = vmask; };
/// mask type of R: bool for double, vmask for vdouble
template<class R> using mask_t = typename real_traits<R>::mask;

inline double  select(bool m, double a, double b) { return m ? a : b; }
[[gnu::always_inline]] inline vdouble select(const vmask& m, const vdouble& a, const vdouble& b) { vdouble r = b; stdx::where(m, r) = a; return r; }
inline bool any(bool m) { return m; }
inline bool any(const vmask& m) { return stdx::any_of(m); }
inline bool all(bool m) { return m; }
inline bool all(const vmask& m) { return stdx::all_of(m); }
/// dst = src where m
template<class R> inline void set_where(const mask_t<R>& m, R& dst, const R& src) { dst = select(m, src, dst); }

// ---------------------------------------------------------------------
// vector helpers (ECMech_util.h), each pointer argument with its own type
// so per-point packs (R) and shared model constants (double) mix freely
// ---------------------------------------------------------------------
template<int n, class R, class S, class T> inline void vecsVxa(R* const p, const S a, const T* const v)
{ for (int i = 0; i < n; ++i) { p[i] = a * v[i]; } }
template<int n, class R, class T, class U> inline void vecsVapb(R* const p, const T* const a, const U* const b)
{ for (int i = 0; i < n; ++i) { p[i] = a[i] + b[i]; } }
template<int n, class R, class S> inline void vecsVsa(R* const p, const S a)
{ for (int i = 0; i < n; ++i) { p[i] = p[i] * a; } }
template<int n, class R> inline R vecNorm(const R* const v)
{ R s = v[0] * v[0]; for (int i = 1; i < n; ++i) { s += v[i] * v[i]; } return sqrt(s); }
template<int n, class R, class T, class U> inline void vecsVAdiagB(R* const v, const T* const a, const U* const b)
{ for (int i = 0; i < n; ++i) { v[i] = a[i] * b[i]; } }
template<int n, int q, class R, class T, class U> inline void vecsVMa(R* const p, const T* const M, const U* const a)
{
   for (int iN = 0; iN < n; ++iN) {
      p[iN] = 0.0;
      for (int iQ = 0; iQ < q; ++iQ) { p[iN] += M[ECMECH_NM_INDX(iN, iQ, n, q)] * a[iQ]; }
   }
}
template<int n, int q, class R, class T, class U> inline void vecsVaTM(R* const p, const T* const a, const U* const M)
{
   for (int iQ = 0; iQ < q; ++iQ) {
      p[iQ] = 0.0;
      for (int iN = 0; iN < n; ++iN) { p[iQ] += a[iN] * M[ECMECH_NM_INDX(iN, iQ, n, q)]; }
   }
}
template<int n, class R, class T, class U> inline void vecsVMTa(R* const p, const T* const M, const U* const v)
{
   for (int i = 0; i < n; ++i) {
      p[i] = 0.0;
      for (int j = 0; j < n; ++j) { p[i] += M[ECMECH_NN_INDX(j, i, n)] * v[j]; }
   }
}
template<int n, int m, int q, class R, class T, class U> inline void vecsMAB(R* const P, const T* const A, const U* const B)
{
   for (int ij = 0; ij < n * m; ++ij) { P[ij] = 0.0; }
   for (int iN = 0; iN < n; ++iN) {
      for (int iQ = 0; iQ < q; ++iQ) {
         for (int jM = 0; jM < m; ++jM) {
            P[ECMECH_NM_INDX(iN, jM, n, m)] += A[ECMECH_NM_INDX(iN, iQ, n, q)] * B[ECMECH_NM_INDX(iQ, jM, q, m)];
         }
      }
   }
}
template<int n, int m, int q, class R, class T, class U> inline void vecsMABT(R* const P, const T* const A, const U* const B)
{
   for (int ij = 0; ij < n * m; ++ij) { P[ij] = 0.0; }
   for (int iN = 0; iN < n; ++iN) {
      for (int jM = 0; jM < m; ++jM) {
         for (int iQ = 0; iQ < q; ++iQ) {
            P[ECMECH_NM_INDX(iN, jM, n, m)] += A[ECMECH_NM_INDX(iN, iQ, n, q)] * B[ECMECH_NM_INDX(jM, iQ, m, q)];
         }
      }
   }
}

// ---------------------------------------------------------------------
// rotation utilities: generated from ECMech_util.h (double -> R)
// ---------------------------------------------------------------------
template<class R>
inline void quat_prod(R* const q, const R* const a, const R* const b)
   {
      q[0] = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3];

      q[1] = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2];
      q[2] = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1];
      q[3] = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0];
   }

template<class R>
inline void quat_to_tensor(R* const c, const R* const quat )
   {
      R x0sq = quat[0] * quat[0];
      R x1sq = quat[1] * quat[1];
      R x2sq = quat[2] * quat[2];
      R x3sq = quat[3] * quat[3];

      R x0x1 = quat[0] * quat[1];
      R x0x2 = quat[0] * quat[2];
      R x0x3 = quat[0] * quat[3];

      R x1x2 = quat[1] * quat[2];
      R x1x3 = quat[1] * quat[3];

      R x2x3 = quat[2] * quat[3];

      c[ECMECH_NN_INDX(0, 0, ndim)] = x0sq + x1sq - x2sq - x3sq;
      c[ECMECH_NN_INDX(0, 1, ndim)] = two * (x1x2 - x0x3);
      c[ECMECH_NN_INDX(0, 2, ndim)] = two * (x1x3 + x0x2);
      c[ECMECH_NN_INDX(1, 0, ndim)] = two * (x1x2 + x0x3);
      c[ECMECH_NN_INDX(1, 1, ndim)] = x0sq - x1sq + x2sq - x3sq;
      c[ECMECH_NN_INDX(1, 2, ndim)] = two * (x2x3 - x0x1);
      c[ECMECH_NN_INDX(2, 0, ndim)] = two * (x1x3 - x0x2);
      c[ECMECH_NN_INDX(2, 1, ndim)] = two * (x2x3 + x0x1);
      c[ECMECH_NN_INDX(2, 2, ndim)] = x0sq - x1sq - x2sq + x3sq;
   }

template<class R>
inline void get_rot_mat_vecd(R* const qr5x5_raw, const R* const c )
   {
      
      
      const R  c11 = c[ECMECH_NN_INDX(0, 0, ndim)];
      const R  c21 = c[ECMECH_NN_INDX(1, 0, ndim)];
      const R  c31 = c[ECMECH_NN_INDX(2, 0, ndim)];
      const R  c12 = c[ECMECH_NN_INDX(0, 1, ndim)];
      const R  c22 = c[ECMECH_NN_INDX(1, 1, ndim)];
      const R  c32 = c[ECMECH_NN_INDX(2, 1, ndim)];
      const R  c13 = c[ECMECH_NN_INDX(0, 2, ndim)];
      const R  c23 = c[ECMECH_NN_INDX(1, 2, ndim)];
      const R  c33 = c[ECMECH_NN_INDX(2, 2, ndim)];


      
      
      
      

      RAJA::View<R, RAJA::Layout<2> > qr5x5(qr5x5_raw, ecmech::ntvec, ecmech::ntvec);

      
      

      qr5x5(0, 0) = onehalf * (c11 * c11 - c12 * c12 - c21 * c21 + c22 * c22);
      qr5x5(0, 1) = sqr3 * onehalf * (c13 * c13 - c23 * c23);
      qr5x5(0, 2) = c11 * c12 - c21 * c22;
      qr5x5(0, 3) = c11 * c13 - c21 * c23;
      qr5x5(0, 4) = c12 * c13 - c22 * c23;
      qr5x5(1, 0) = sqr3 * onehalf * (c31 * c31 - c32 * c32);
      qr5x5(1, 1) = thrhalf * c33 * c33 - onehalf;
      qr5x5(1, 2) = sqr3 * c31 * c32;
      qr5x5(1, 3) = sqr3 * c31 * c33;
      qr5x5(1, 4) = sqr3 * c32 * c33;
      qr5x5(2, 0) = c11 * c21 - c12 * c22;
      qr5x5(2, 1) = sqr3 * c13 * c23;
      qr5x5(2, 2) = c11 * c22 + c12 * c21;
      qr5x5(2, 3) = c11 * c23 + c13 * c21;
      qr5x5(2, 4) = c12 * c23 + c13 * c22;
      qr5x5(3, 0) = c11 * c31 - c12 * c32;
      qr5x5(3, 1) = sqr3 * c13 * c33;
      qr5x5(3, 2) = c11 * c32 + c12 * c31;
      qr5x5(3, 3) = c11 * c33 + c13 * c31;
      qr5x5(3, 4) = c12 * c33 + c13 * c32;
      qr5x5(4, 0) = c21 * c31 - c22 * c32;
      qr5x5(4, 1) = sqr3 * c23 * c33;
      qr5x5(4, 2) = c21 * c32 + c22 * c31;
      qr5x5(4, 3) = c21 * c33 + c23 * c31;
      qr5x5(4, 4) = c22 * c33 + c23 * c32;
   }

template<class R>
inline void M35_d_AAoB_dA(R* const M35, const R* const cmv6b )
   {
      const R vb1 = cmv6b[0];
      const R vb2 = cmv6b[1];
      const R vb3 = cmv6b[2];
      const R vb4 = cmv6b[3];
      const R vb5 = cmv6b[4];

      M35[ECMECH_NM_INDX(0, 0, nwvec, ntvec)] = vb5 * onehalf;
      M35[ECMECH_NM_INDX(1, 0, nwvec, ntvec)] = vb4 * onehalf;
      M35[ECMECH_NM_INDX(2, 0, nwvec, ntvec)] = -vb3;
      M35[ECMECH_NM_INDX(0, 1, nwvec, ntvec)] = vb5 * halfsqr3;
      M35[ECMECH_NM_INDX(1, 1, nwvec, ntvec)] = -vb4 * halfsqr3;
      M35[ECMECH_NM_INDX(2, 1, nwvec, ntvec)] = zero;
      M35[ECMECH_NM_INDX(0, 2, nwvec, ntvec)] = -vb4 * onehalf;
      M35[ECMECH_NM_INDX(1, 2, nwvec, ntvec)] = vb5 * onehalf;
      M35[ECMECH_NM_INDX(2, 2, nwvec, ntvec)] = vb1;
      M35[ECMECH_NM_INDX(0, 3, nwvec, ntvec)] = vb3 * onehalf;
      M35[ECMECH_NM_INDX(1, 3, nwvec, ntvec)] = halfsqr3 * vb2 - vb1 * onehalf;
      M35[ECMECH_NM_INDX(2, 3, nwvec, ntvec)] = -vb5 * onehalf;
      M35[ECMECH_NM_INDX(0, 4, nwvec, ntvec)] = -vb1 * onehalf - halfsqr3 * vb2;
      M35[ECMECH_NM_INDX(1, 4, nwvec, ntvec)] = -vb3 * onehalf;
      M35[ECMECH_NM_INDX(2, 4, nwvec, ntvec)] = vb4 * onehalf;
      
      
      
   }

template<class R>
inline void d_rot_mat_vecd_latop(R* const dvdc_raw, const R* const c, const R* const vec_sm )
   {
      const R  c11 = c[ECMECH_NN_INDX(0, 0, ndim)];
      const R  c21 = c[ECMECH_NN_INDX(1, 0, ndim)];
      const R  c31 = c[ECMECH_NN_INDX(2, 0, ndim)];
      const R  c12 = c[ECMECH_NN_INDX(0, 1, ndim)];
      const R  c22 = c[ECMECH_NN_INDX(1, 1, ndim)];
      const R  c32 = c[ECMECH_NN_INDX(2, 1, ndim)];
      const R  c13 = c[ECMECH_NN_INDX(0, 2, ndim)];
      const R  c23 = c[ECMECH_NN_INDX(1, 2, ndim)];
      const R  c33 = c[ECMECH_NN_INDX(2, 2, ndim)];
      const R va1 = vec_sm[0];
      const R va2 = vec_sm[1];
      const R va3 = vec_sm[2];
      const R va4 = vec_sm[3];
      const R va5 = vec_sm[4];

      
      RAJA::View<R, RAJA::Layout<3> > dvdc(dvdc_raw, ecmech::ntvec, ecmech::ndim, ecmech::ndim);
      dvdc(0, 0, 0) = c11 * va1 - c11 * sqr3 * va2 * onethird + c21 * va3 + c31 * va4;
      dvdc(1, 0, 0) = -sqr3 * (3 * c11 * va1 - c11 * sqr3 * va2 + 3 * c21 * va3 + 3 * c31 * va4) * oneninth;
      dvdc(2, 0, 0) = c12 * va1 - c12 * sqr3 * va2 * onethird + c22 * va3 + c32 * va4;
      dvdc(3, 0, 0) = c13 * va1 - c13 * sqr3 * va2 * onethird + c23 * va3 + c33 * va4;
      dvdc(4, 0, 0) = zero;
      dvdc(0, 1, 0) = c11 * va3 - c21 * va1 - c21 * sqr3 * va2 * onethird + c31 * va5;
      dvdc(1, 1, 0) = sqr3 * (-3 * c11 * va3 + 3 * c21 * va1 + c21 * sqr3 * va2 - 3 * c31 * va5) * oneninth;
      dvdc(2, 1, 0) = c12 * va3 - c22 * va1 - c22 * sqr3 * va2 * onethird + c32 * va5;
      dvdc(3, 1, 0) = c13 * va3 - c23 * va1 - c23 * sqr3 * va2 * onethird + c33 * va5;
      dvdc(4, 1, 0) = zero;
      dvdc(0, 2, 0) = c11 * va4 + c21 * va5 + twothird * c31 * sqr3 * va2;
      dvdc(1, 2, 0) = -sqr3 * c11 * va4 * onethird - sqr3 * c21 * va5 * onethird - twothird * c31 * va2;
      dvdc(2, 2, 0) = c12 * va4 + c22 * va5 + twothird * c32 * sqr3 * va2;
      dvdc(3, 2, 0) = c13 * va4 + c23 * va5 + twothird * c33 * sqr3 * va2;
      dvdc(4, 2, 0) = zero;
      dvdc(0, 0, 1) = -c12 * va1 + c12 * sqr3 * va2 * onethird - c22 * va3 - c32 * va4;
      dvdc(1, 0, 1) = sqr3 * (-3 * c12 * va1 + c12 * sqr3 * va2 - 3 * c22 * va3 - 3 * c32 * va4) * oneninth;
      dvdc(2, 0, 1) = c11 * va1 - c11 * sqr3 * va2 * onethird + c21 * va3 + c31 * va4;
      dvdc(3, 0, 1) = zero;
      dvdc(4, 0, 1) = c13 * va1 - c13 * sqr3 * va2 * onethird + c23 * va3 + c33 * va4;
      dvdc(0, 1, 1) = -c12 * va3 + c22 * va1 + c22 * sqr3 * va2 * onethird - c32 * va5;
      dvdc(1, 1, 1) = sqr3 * (-3 * c12 * va3 + 3 * c22 * va1 + c22 * sqr3 * va2 - 3 * c32 * va5) * oneninth;
      dvdc(2, 1, 1) = c11 * va3 - c21 * va1 - c21 * sqr3 * va2 * onethird + c31 * va5;
      dvdc(3, 1, 1) = zero;
      dvdc(4, 1, 1) = c13 * va3 - c23 * va1 - c23 * sqr3 * va2 * onethird + c33 * va5;
      dvdc(0, 2, 1) = -c12 * va4 - c22 * va5 - twothird * c32 * sqr3 * va2;
      dvdc(1, 2, 1) = -sqr3 * c12 * va4 * onethird - sqr3 * c22 * va5 * onethird - twothird * c32 * va2;
      dvdc(2, 2, 1) = c11 * va4 + c21 * va5 + twothird * c31 * sqr3 * va2;
      dvdc(3, 2, 1) = zero;
      dvdc(4, 2, 1) = c13 * va4 + c23 * va5 + twothird * c33 * sqr3 * va2;
      dvdc(0, 0, 2) = zero;
      dvdc(1, 0, 2) = -twothird * sqr3i * (-3 * c13 * va1 + sqr3 * c13 * va2 - 3 * c23 * va3 - 3 * c33 * va4);
      dvdc(2, 0, 2) = zero;
      dvdc(3, 0, 2) = c11 * va1 - c11 * sqr3 * va2 * onethird + c21 * va3 + c31 * va4;
      dvdc(4, 0, 2) = c12 * va1 - c12 * sqr3 * va2 * onethird + c22 * va3 + c32 * va4;
      dvdc(0, 1, 2) = zero;
      dvdc(1, 1, 2) = -twothird * sqr3i * (-3 * va3 * c13 + 3 * va1 * c23 + sqr3 * c23 * va2 - 3 * va5 * c33);
      dvdc(2, 1, 2) = zero;
      dvdc(3, 1, 2) = c11 * va3 - c21 * va1 - c21 * sqr3 * va2 * onethird + c31 * va5;
      dvdc(4, 1, 2) = c12 * va3 - c22 * va1 - c22 * sqr3 * va2 * onethird + c32 * va5;
      dvdc(0, 2, 2) = zero;
      dvdc(1, 2, 2) = twothird * sqr3 * c13 * va4 + twothird * sqr3 * c23 * va5 + fourthirds * c33 * va2;
      dvdc(2, 2, 2) = zero;
      dvdc(3, 2, 2) = c11 * va4 + c21 * va5 + twothird * c31 * sqr3 * va2;
      dvdc(4, 2, 2) = c12 * va4 + c22 * va5 + twothird * c32 * sqr3 * va2;
   }

template<class R>
inline void d_rot_mat_wveccp_latop(R* const dvdc_raw, const R* const cmv3w )
   {
      const R vw1 = cmv3w[0];
      const R vw2 = cmv3w[1];
      const R vw3 = cmv3w[2];

      
      RAJA::View<R, RAJA::Layout<3> > dvdc(dvdc_raw, ecmech::nwvec, ecmech::ndim, ecmech::ndim);

      dvdc(0, 0, 0) = vw1;
      dvdc(1, 0, 0) = zero;
      dvdc(2, 0, 0) = zero;
      dvdc(0, 1, 0) = vw2;
      dvdc(1, 1, 0) = zero;
      dvdc(2, 1, 0) = zero;
      dvdc(0, 2, 0) = vw3;
      dvdc(1, 2, 0) = zero;
      dvdc(2, 2, 0) = zero;
      dvdc(0, 0, 1) = zero;
      dvdc(1, 0, 1) = vw1;
      dvdc(2, 0, 1) = zero;
      dvdc(0, 1, 1) = zero;
      dvdc(1, 1, 1) = vw2;
      dvdc(2, 1, 1) = zero;
      dvdc(0, 2, 1) = zero;
      dvdc(1, 2, 1) = vw3;
      dvdc(2, 2, 1) = zero;
      dvdc(0, 0, 2) = zero;
      dvdc(1, 0, 2) = zero;
      dvdc(2, 0, 2) = vw1;
      dvdc(0, 1, 2) = zero;
      dvdc(1, 1, 2) = zero;
      dvdc(2, 1, 2) = vw2;
      dvdc(0, 2, 2) = zero;
      dvdc(1, 2, 2) = zero;
      dvdc(2, 2, 2) = vw3;
   }

template<class R>
inline void d_quat_to_tensor(R* const dcdq_raw, const R* const quat )
   {
      R tqa = two * quat[0];
      R tqb = two * quat[1];
      R tqc = two * quat[2];
      R tqd = two * quat[3];

      RAJA::View<R, RAJA::Layout<3> > dcdq(dcdq_raw, ecmech::ndim, ecmech::ndim, ecmech::qdim);

      
      dcdq(0, 0, 0) = tqa;
      dcdq(0, 0, 1) = tqb;
      dcdq(0, 0, 2) = -tqc;
      dcdq(0, 0, 3) = -tqd;

      
      dcdq(0, 1, 0) = -tqd;
      dcdq(0, 1, 1) = tqc;
      dcdq(0, 1, 2) = tqb;
      dcdq(0, 1, 3) = -tqa;

      
      dcdq(0, 2, 0) = tqc;
      dcdq(0, 2, 1) = tqd;
      dcdq(0, 2, 2) = tqa;
      dcdq(0, 2, 3) = tqb;

      
      dcdq(1, 0, 0) = tqd;
      dcdq(1, 0, 1) = tqc;
      dcdq(1, 0, 2) = tqb;
      dcdq(1, 0, 3) = tqa;

      
      dcdq(1, 1, 0) = tqa;
      dcdq(1, 1, 1) = -tqb;
      dcdq(1, 1, 2) = tqc;
      dcdq(1, 1, 3) = -tqd;

      
      dcdq(1, 2, 0) = -tqb;
      dcdq(1, 2, 1) = -tqa;
      dcdq(1, 2, 2) = tqd;
      dcdq(1, 2, 3) = tqc;

      
      dcdq(2, 0, 0) = -tqc;
      dcdq(2, 0, 1) = tqd;
      dcdq(2, 0, 2) = -tqa;
      dcdq(2, 0, 3) = tqb;

      
      dcdq(2, 1, 0) = tqb;
      dcdq(2, 1, 1) = tqa;
      dcdq(2, 1, 2) = tqd;
      dcdq(2, 1, 3) = tqc;

      
      dcdq(2, 2, 0) = tqa;
      dcdq(2, 2, 1) = -tqb;
      dcdq(2, 2, 2) = -tqc;
      dcdq(2, 2, 3) = tqd;
   }

template<class R>
inline void inv_to_quat(R* const quat, const R* const inv)
   {
      R a = inv[0] * 0.5;
      quat[0] = cos(a);
      a = sin(a);
      vecsVxa<nwvec>(&(quat[1]), a, &(inv[1]));
   }

template<class R>
inline void d_rot_mat_vecd_smop(R* const dvdc_raw, const R* const c, const R* const vec_lat )
   {
      
      
      
      
      
      

      const R  c11 = c[ECMECH_NN_INDX(0, 0, ndim)];
      const R  c21 = c[ECMECH_NN_INDX(1, 0, ndim)];
      const R  c31 = c[ECMECH_NN_INDX(2, 0, ndim)];
      const R  c12 = c[ECMECH_NN_INDX(0, 1, ndim)];
      const R  c22 = c[ECMECH_NN_INDX(1, 1, ndim)];
      const R  c32 = c[ECMECH_NN_INDX(2, 1, ndim)];
      const R  c13 = c[ECMECH_NN_INDX(0, 2, ndim)];
      const R  c23 = c[ECMECH_NN_INDX(1, 2, ndim)];
      const R  c33 = c[ECMECH_NN_INDX(2, 2, ndim)];
      const R va1 = vec_lat[0];
      const R va2 = vec_lat[1];
      const R va3 = vec_lat[2];
      const R va4 = vec_lat[3];
      const R va5 = vec_lat[4];

      
      RAJA::View<R, RAJA::Layout<3> > dvdc(dvdc_raw, ecmech::ntvec, ecmech::ndim, ecmech::ndim);
      dvdc(0, 0, 0) = c11 * va1 - c11 * sqr3 * va2 * onethird + c12 * va3 + c13 * va4;
      dvdc(1, 0, 0) = sqr3 * (-3 * c11 * va1 + sqr3 * c11 * va2 - 3 * c12 * va3 - 3 * c13 * va4) * oneninth;
      dvdc(2, 0, 0) = c21 * va1 - c21 * sqr3 * va2 * onethird + c22 * va3 + c23 * va4;
      dvdc(3, 0, 0) = c31 * va1 - c31 * sqr3 * va2 * onethird + c32 * va3 + c33 * va4;
      dvdc(4, 0, 0) = zero;
      dvdc(0, 1, 0) = -c21 * va1 + c21 * sqr3 * va2 * onethird - c22 * va3 - c23 * va4;
      dvdc(1, 1, 0) = -sqr3 * (3 * c21 * va1 - c21 * sqr3 * va2 + 3 * c22 * va3 + 3 * c23 * va4) * oneninth;
      dvdc(2, 1, 0) = c11 * va1 - c11 * sqr3 * va2 * onethird + c12 * va3 + c13 * va4;
      dvdc(3, 1, 0) = zero;
      dvdc(4, 1, 0) = c31 * va1 - c31 * sqr3 * va2 * onethird + c32 * va3 + c33 * va4;
      dvdc(0, 2, 0) = zero;
      dvdc(1, 2, 0) = -twothird * sqr3i * (-3 * c31 * va1 + c31 * sqr3 * va2 - 3 * c32 * va3 - 3 * c33 * va4);
      dvdc(2, 2, 0) = zero;
      dvdc(3, 2, 0) = c11 * va1 - c11 * sqr3 * va2 * onethird + c12 * va3 + c13 * va4;
      dvdc(4, 2, 0) = c21 * va1 - c21 * sqr3 * va2 * onethird + c22 * va3 + c23 * va4;
      dvdc(0, 0, 1) = c11 * va3 - c12 * va1 - c12 * sqr3 * va2 * onethird + c13 * va5;
      dvdc(1, 0, 1) = sqr3 * (-3 * c11 * va3 + 3 * c12 * va1 + c12 * sqr3 * va2 - 3 * c13 * va5) * oneninth;
      dvdc(2, 0, 1) = c21 * va3 - c22 * va1 - c22 * sqr3 * va2 * onethird + c23 * va5;
      dvdc(3, 0, 1) = c31 * va3 - c32 * va1 - c32 * sqr3 * va2 * onethird + c33 * va5;
      dvdc(4, 0, 1) = zero;
      dvdc(0, 1, 1) = -c21 * va3 + c22 * va1 + c22 * sqr3 * va2 * onethird - c23 * va5;
      dvdc(1, 1, 1) = sqr3 * (-3 * c21 * va3 + 3 * c22 * va1 + sqr3 * va2 * c22 - 3 * c23 * va5) * oneninth;
      dvdc(2, 1, 1) = c11 * va3 - c12 * va1 - c12 * sqr3 * va2 * onethird + c13 * va5;
      dvdc(3, 1, 1) = zero;
      dvdc(4, 1, 1) = c31 * va3 - c32 * va1 - c32 * sqr3 * va2 * onethird + c33 * va5;
      dvdc(0, 2, 1) = zero;
      dvdc(1, 2, 1) = -twothird * sqr3i * (-3 * c31 * va3 + 3 * c32 * va1 + sqr3 * va2 * c32 - 3 * c33 * va5);
      dvdc(2, 2, 1) = zero;
      dvdc(3, 2, 1) = c11 * va3 - c12 * va1 - c12 * sqr3 * va2 * onethird + c13 * va5;
      dvdc(4, 2, 1) = c21 * va3 - c22 * va1 - c22 * sqr3 * va2 * onethird + c23 * va5;
      dvdc(0, 0, 2) = c11 * va4 + c12 * va5 + twothird * c13 * sqr3 * va2;
      dvdc(1, 0, 2) = -sqr3 * c11 * va4 * onethird - sqr3 * c12 * va5 * onethird - twothird * c13 * va2;
      dvdc(2, 0, 2) = c21 * va4 + c22 * va5 + twothird * c23 * sqr3 * va2;
      dvdc(3, 0, 2) = c31 * va4 + c32 * va5 + twothird * c33 * sqr3 * va2;
      dvdc(4, 0, 2) = zero;
      dvdc(0, 1, 2) = -c21 * va4 - c22 * va5 - twothird * c23 * sqr3 * va2;
      dvdc(1, 1, 2) = -sqr3 * c21 * va4 * onethird - sqr3 * c22 * va5 * onethird - twothird * c23 * va2;
      dvdc(2, 1, 2) = c11 * va4 + c12 * va5 + twothird * c13 * sqr3 * va2;
      dvdc(3, 1, 2) = zero;
      dvdc(4, 1, 2) = c31 * va4 + c32 * va5 + twothird * c33 * sqr3 * va2;
      dvdc(0, 2, 2) = zero;
      dvdc(1, 2, 2) = twothird * sqr3 * c31 * va4 + twothird * sqr3 * c32 * va5 + fourthirds * c33 * va2;
      dvdc(2, 2, 2) = zero;
      dvdc(3, 2, 2) = c11 * va4 + c12 * va5 + twothird * c13 * sqr3 * va2;
      dvdc(4, 2, 2) = c21 * va4 + c22 * va5 + twothird * c23 * sqr3 * va2;
   }

template<int n, bool l_T, class R>
inline void qr6x6_pre_mul(R* const M_out, const R* const M_in, const R* const qr5x5 )
   {
      for (int iM = 0; iM<ecmech::ntvec; ++iM) { 
         for (int jM = 0; jM<n; ++jM) {
            int ijM = ECMECH_NM_INDX(iM, jM, ecmech::nsvec, n);
            M_out[ijM] = 0.0;
            for (int pTvec = 0; pTvec<ecmech::ntvec; ++pTvec) {
               if (l_T) {
                  M_out[ijM] += qr5x5[ECMECH_NN_INDX(pTvec, iM, ecmech::ntvec)] * M_in[ECMECH_NM_INDX(pTvec, jM, ecmech::nsvec, n)];
               }
               else {
                  M_out[ijM] += qr5x5[ECMECH_NN_INDX(iM, pTvec, ecmech::ntvec)] * M_in[ECMECH_NM_INDX(pTvec, jM, ecmech::nsvec, n)];
               }
            }
         }
      }

      for (int jM = 0; jM<n; ++jM) {
         int ijM = ECMECH_NM_INDX(iSvecS, jM, ecmech::nsvec, n);
         M_out[ijM] = M_in[ijM];
      }
   }


// ---------------------------------------------------------------------
// rotation utilities with per-point branches (ECMech_util.h), by hand
// ---------------------------------------------------------------------
template<class R>
inline void emap_to_quat(R* const quat, const R* const emap)
{
   R inv[invdim] = { 0.0, 1.0, 0.0, 0.0 };
   inv[0] = vecNorm<emapdim>(emap);
   const auto big = inv[0] > idp_tiny_sqrt;
   const R invInv = 1.0 / select(big, inv[0], R(1.0));
   for (int i = 0; i < emapdim; ++i) { inv[1 + i] = select(big, invInv * emap[i], inv[1 + i]); }
   inv_to_quat(quat, inv);
}

template<class R>
inline void dquat_demap_T(R* const dqdeT_raw, const R* const emap)
{
   const double theta_sm_a = 1e-9;
   const double oo48 = 1.0 / 48.0;
   const R theta = vecNorm<emapdim>(emap);
   const auto small = fabs(theta) < theta_sm_a;
   const auto tiny = fabs(theta) < idp_tiny_sqrt;
   const R theta_s = select(tiny, R(1.0), theta);
   const R shalf = sin(theta * onehalf);
   const R sthhbyth = select(small, onehalf - theta * theta * oo48, shalf / theta_s);
   const R halfsthh = select(small, theta * oneqrtr, shalf * onehalf);
   const R theta_inv = one / theta_s;
   const R na = select(tiny, R(one), emap[0] * theta_inv);
   const R nb = select(tiny, R(zero), emap[1] * theta_inv);
   const R nc = select(tiny, R(zero), emap[2] * theta_inv);
   const R halfcthh = cos(theta * onehalf) * onehalf;
   RAJA::View<R, RAJA::Layout<2> > dqdeT(dqdeT_raw, ecmech::emapdim, ecmech::qdim);
   dqdeT(0, 0) = -halfsthh * na;
   dqdeT(1, 0) = -halfsthh * nb;
   dqdeT(2, 0) = -halfsthh * nc;
   R temp = na * na;
   dqdeT(0, 1) = halfcthh * temp + sthhbyth * (one - temp);
   temp = nb * nb;
   dqdeT(1, 2) = halfcthh * temp + sthhbyth * (one - temp);
   temp = nc * nc;
   dqdeT(2, 3) = halfcthh * temp + sthhbyth * (one - temp);
   temp = halfcthh - sthhbyth;
   R tempb = temp * na * nb;
   dqdeT(1, 1) = tempb;
   dqdeT(0, 2) = tempb;
   tempb = temp * na * nc;
   dqdeT(2, 1) = tempb;
   dqdeT(0, 3) = tempb;
   tempb = temp * nb * nc;
   dqdeT(2, 2) = tempb;
   dqdeT(1, 3) = tempb;
}

template<class R>
inline void get_c_quat(R* const c_quat, const R* const dr_quat, const R* const cn_quat)
{
   quat_prod(c_quat, cn_quat, dr_quat);
}

template<class R>
inline void eval_d_dxi_impl_quat(R* const dxtal_ori_quat_dxi_T,
                                 R* const dDapp_dxi,
                                 R* const dWapp_dxi,
                                 const R* const def_rate_d5_sample,
                                 const R* const w_vec_sm,
                                 const R* const xi,
                                 const R* const xtal_ori_quat_n,
                                 const R* const xtal_rmat,
                                 const R* const xtal_ori_quat)
{
   {
      R dA_quat_dxi_T[ ecmech::ndim * ecmech::qdim ];
      dquat_demap_T(dA_quat_dxi_T, xi);
      quat_prod(&(dxtal_ori_quat_dxi_T[ecmech::qdim * 0]), xtal_ori_quat_n, &(dA_quat_dxi_T[ecmech::qdim * 0]) );
      quat_prod(&(dxtal_ori_quat_dxi_T[ecmech::qdim * 1]), xtal_ori_quat_n, &(dA_quat_dxi_T[ecmech::qdim * 1]) );
      quat_prod(&(dxtal_ori_quat_dxi_T[ecmech::qdim * 2]), xtal_ori_quat_n, &(dA_quat_dxi_T[ecmech::qdim * 2]) );
   }
   R dxtal_rmat_dxi[ (ecmech::ndim * ecmech::ndim) * ecmech::nwvec ];
   {
      R dCmatx_dq[ (ecmech::ndim * ecmech::ndim) * ecmech::qdim ];
      d_quat_to_tensor(dCmatx_dq, xtal_ori_quat);
      vecsMABT<ndim*ndim, nwvec, qdim>(dxtal_rmat_dxi, dCmatx_dq, dxtal_ori_quat_dxi_T);
   }
   {
      R dD_dxtal_rmat[ ecmech::ntvec * (ecmech::ndim * ecmech::ndim) ];
      d_rot_mat_vecd_latop(dD_dxtal_rmat, xtal_rmat, def_rate_d5_sample);
      vecsMAB<ntvec, nwvec, ndim*ndim>(dDapp_dxi, dD_dxtal_rmat, dxtal_rmat_dxi);
   }
   {
      R dW_dxtal_rmat[ ecmech::nwvec * (ecmech::ndim * ecmech::ndim) ];
      d_rot_mat_wveccp_latop(dW_dxtal_rmat, w_vec_sm);
      vecsMAB<nwvec, nwvec, ndim*ndim>(dWapp_dxi, dW_dxtal_rmat, dxtal_rmat_dxi);
   }
}

} // namespace simd
} // namespace ecmech
