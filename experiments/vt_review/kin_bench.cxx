// Design-review microbenchmark (not part of the library): one slip-rate law
// (KMBalD, MTS + drag, the most branchy kinetics) evaluated for N points x
// nslip slip systems in four ways:
//   orig   : the original scalar evalGdot (kept double overload), point by point
//   vt1    : the value-type template with R = double, point by point
//   pack   : the value-type template with R = native_simd<double> (CPU SIMD)
//   ompsimd: `#pragma omp simd` over points calling the original scalar
//            evalGdot (compiler auto-vectorization of the unchanged code)
// Build: see build_kin_bench.sh next to this file.
#include <chrono>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include "ECMech_core.h"
#include "ECMech_cases.h"
#include "kinetics/ECMech_kinetics_KMBalD.h"

#if defined(KB_DECLARE_SIMD_MATH)
// make libmvec visible to the vectorizer without -ffast-math
extern "C" {
#pragma omp declare simd notinbranch
double exp(double);
#pragma omp declare simd notinbranch
double log(double);
#pragma omp declare simd notinbranch
double pow(double, double);
}
#endif

using Kin = ecmech::KineticsKMBalD<false, false, false, false, 1>;
constexpr int NS = 12;

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// SoA: tau[s*N + p], vals[v*N + p]; outputs gdot/dg[s*N + p]
__attribute__((noinline)) void run_orig(const Kin& k, int N, const double* tau, const double* vals, double* gd, double* dg, double mu)
{
   for (int p = 0; p < N; ++p) {
      double v[Kin::nVals];
      for (int j = 0; j < Kin::nVals; ++j) { v[j] = vals[j * N + p]; }
      for (int s = 0; s < NS; ++s) {
         bool a;
         k.evalGdot(gd[s * N + p], a, dg[s * N + p], v, s, tau[s * N + p], mu);
      }
   }
}

__attribute__((noinline)) void run_vt1(const Kin& k, int N, const double* tau, const double* vals, double* gd, double* dg, double mu)
{
   for (int p = 0; p < N; ++p) {
      double v[Kin::nVals];
      for (int j = 0; j < Kin::nVals; ++j) { v[j] = vals[j * N + p]; }
      for (int s = 0; s < NS; ++s) {
         bool a;
         k.evalGdot<double>(gd[s * N + p], a, dg[s * N + p], v, s, tau[s * N + p], mu);
      }
   }
}

#if defined(ECMECH_SIMD_EXPT)
using V = ecmech::simd::vdouble;
constexpr int W = ecmech::simd::W;
__attribute__((noinline)) void run_pack(const Kin& k, int N, const double* tau, const double* vals, double* gd, double* dg, double mu)
{
   auto load = [](V& x, const double* src) { std::memcpy(&x, src, sizeof(V)); };
   auto store = [](const V& x, double* dst) { std::memcpy(dst, &x, sizeof(V)); };
   for (int p = 0; p < N; p += W) {
      V v[Kin::nVals];
      for (int j = 0; j < Kin::nVals; ++j) { load(v[j], &vals[j * N + p]); }
      for (int s = 0; s < NS; ++s) {
         V t; load(t, &tau[s * N + p]);
         V g, d; ecmech::mask_t<V> a;
         k.evalGdot<V>(g, a, d, v, s, t, mu);
         store(g, &gd[s * N + p]);
         store(d, &dg[s * N + p]);
      }
   }
}
#endif

__attribute__((noinline)) void run_ompsimd(const Kin& k, int N, const double* tau, const double* vals, double* gd, double* dg, double mu)
{
   for (int s = 0; s < NS; ++s) {
      #pragma omp simd
      for (int p = 0; p < N; ++p) {
         double v[Kin::nVals];
         for (int j = 0; j < Kin::nVals; ++j) { v[j] = vals[j * N + p]; }
         bool a;
         k.evalGdot(gd[s * N + p], a, dg[s * N + p], v, s, tau[s * N + p], mu);
      }
   }
}

int main(int argc, char** argv)
{
   const int N = (argc > 1) ? std::atoi(argv[1]) : 4096;
   const int reps = (argc > 2) ? std::atoi(argv[2]) : 200;
   // fcc_mts parameters (cases/props_cp_mts.txt): p = q = 1 at run time
   std::vector<double> par = {47.26, 300.0, 1944.106926, 4e-4, 1.0, 1.0, 1.0, 1.0, 0.03, 0.008, 0.1,
                              3e-4, 5e-5, 0.1, 0.01, 9e-4};
   if (argc > 3) { par[4] = std::atof(argv[3]); par[5] = std::atof(argv[3]); } // p = q (exercise pow)
   Kin k(NS);
   k.setParams(par.data());
   std::mt19937_64 rng(42);
   std::uniform_real_distribution<double> u(-1.0, 1.0), uh(0.5, 2.0);
   std::vector<double> tau(NS * N), vals(Kin::nVals * N);
   for (int p = 0; p < N; ++p) {
      double v[Kin::nVals], h = 9e-4 * uh(rng);
      k.getVals(v, 0.0, 300.0, &h);
      for (int j = 0; j < Kin::nVals; ++j) { vals[j * N + p] = v[j]; }
      // resolved shear stresses spread over the inactive, thermally activated and drag regimes
      for (int s = 0; s < NS; ++s) { tau[s * N + p] = 0.02 * u(rng) * std::fabs(u(rng)); }
   }
   std::vector<double> g0(NS * N), d0(NS * N), g1(NS * N), d1(NS * N);
   auto bench = [&](const char* name, auto fn) {
      double best = 1e30;
      for (int r = 0; r < 5; ++r) {
         const double t0 = now();
         for (int i = 0; i < reps; ++i) { fn(k, N, tau.data(), vals.data(), g1.data(), d1.data(), 47.26); }
         best = std::fmin(best, now() - t0);
      }
      double maxrel = 0.0; long nbit = 0;
      for (int i = 0; i < NS * N; ++i) {
         const double a = g0[i], b = g1[i];
         if (a == b && d0[i] == d1[i]) { ++nbit; }
         const double den = std::fmax(std::fabs(a), 1e-300);
         if (a != b) { maxrel = std::fmax(maxrel, std::fabs(a - b) / den); }
      }
      std::printf("%-8s %8.3f ns/slip-system  bitwise %6.2f%%  max rel diff %.2e\n", name,
                  1e9 * best / (double(reps) * N * NS), 100.0 * nbit / (NS * N), maxrel);
   };
   run_orig(k, N, tau.data(), vals.data(), g0.data(), d0.data(), 47.26);
   long act = 0; for (double g : g0) { act += (g != 0.0); }
   std::printf("N=%d p=q=%g active fraction %.2f\n", N, par[4], double(act) / (NS * N));
   bench("orig", run_orig);
   bench("vt1", run_vt1);
#if defined(ECMECH_SIMD_EXPT)
   bench("pack", run_pack);
#endif
   bench("ompsimd", run_ompsimd);
   return 0;
}
