//#include "mlir/Analysis/PresburgerSet.h"
//#include "mlir/Analysis/Presburger/Simplex.h"

#include "mlir/Analysis/Presburger/Matrix.h"


#include <iostream>
#include "benchmark/benchmark.h"
#include <thread>
#include <chrono>
#include <omp.h>

using namespace mlir;
/*
//typedef float float4 __attribute__((__ext_vector_type__(4)));
//typedef float float2 __attribute__((__ext_vector_type__(2)));
typedef float float128 __attribute__((__ext_vector_type__(128)));
//typedef __attribute__((vector_size(128))) float float128;

 
// Helper funcs

/// Construct a FlatAffineConstraints from a set of inequality and
/// equality constraints.
static FlatAffineConstraints
makeFACFromConstraints(unsigned dims, ArrayRef<SmallVector<int64_t, 4>> ineqs,
                       ArrayRef<SmallVector<int64_t, 4>> eqs) {
  FlatAffineConstraints fac(ineqs.size(), eqs.size(), dims + 1, dims);
  for (const SmallVector<int64_t, 4> &eq : eqs)
    fac.addEquality(eq);
  for (const SmallVector<int64_t, 4> &ineq : ineqs)
    fac.addInequality(ineq);
  return fac;
}


static FlatAffineConstraints
makeFACFromIneqs(unsigned dims, ArrayRef<SmallVector<int64_t, 4>> ineqs) {
  return makeFACFromConstraints(dims, ineqs, {});
}

static PresburgerSet makeSetFromFACs(unsigned dims,
                                     ArrayRef<FlatAffineConstraints> facs) {
  PresburgerSet set = PresburgerSet::getEmptySet(dims);
  for (const FlatAffineConstraints &fac : facs)
    set.unionFACInPlace(fac);
  return set;
}

// Main
PresburgerSet set1 = PresburgerSet::getUniverse();
PresburgerSet set2 = PresburgerSet::getUniverse();

void set_up() {
   set1 = makeSetFromFACs(1, {
      makeFACFromIneqs(1, {{1, -2},     // x >= 2.
                          {-1, 8}}),    // x <= 8.
      makeFACFromIneqs(1, {{1, -10},    // x >= 10.
                          {-1, 20}}),   // x <= 20.
    });

    //PresburgerSet 
   set2 = makeSetFromFACs(1, {
      makeFACFromIneqs(1, {{1, -2},         // x >= 2.
                          {-1, 8}}),        // x <= 8.
      makeFACFromIneqs(1, {{1, -10},        // x >= 10.
                          {-1, 600000}}),   // x <= 20.
    });

}

static void BM_PresburgerSet_subtract(benchmark::State& state) {
  for (auto _ : state) {
      state.PauseTiming();
      set_up();
      state.ResumeTiming();
    
      set1.subtract(set2);
  }
}
BENCHMARK(BM_PresburgerSet_subtract);

static void BM_Simplex_emptyRollback(benchmark::State& state) {
  for (auto _ : state) {
    Simplex simplex(2);
    // (u - v) >= 0
    simplex.addInequality({1, -1, 0});

    unsigned snapshot = simplex.getSnapshot();
    // (u - v) <= -1
    simplex.addInequality({-1, 1, -1});
    simplex.rollback(snapshot);
  }
}
BENCHMARK(BM_Simplex_emptyRollback);
*/

/*
#if __AVX__
static void BM_print_AVX(benchmark::State& state) {
  printf("AVX enabled\n");
  for (auto _ : state) {
    int x = 0;
    int y = 1;
    x += y;
  }
}
BENCHMARK(BM_print_AVX);
#endif

#if __AVX2__
static void BM_print_AVX2(benchmark::State& state) {
  printf("AVX2 enabled\n");
  for (auto _ : state) {
    int x = 0;
    int y = 1;
    x += y;
  }
}
BENCHMARK(BM_print_AVX2);
#endif
*/
/*
#if __AVX2__
static void BM_print_AVX2(benchmark::State& state) {
  printf("AVX512 enabled\n");
  for (auto _ : state) {
    int x = 0;
    int y = 1;
    x += y;
  }
}
BENCHMARK(BM_print_AVX2);
#endif
*/

Matrix createMatrix(unsigned rows, unsigned columns) {
  Matrix mat(rows, columns);
  for (unsigned row = 0; row < rows; ++row)
    for (unsigned col = 0; col < columns; ++col)
      mat(row, col) = col == 3 ? 1 : 0;

  return mat;
}

static void BM_Matrix_addToRow(benchmark::State& state) {
  Matrix benchmat = createMatrix(512,512);
  for (auto _ : state) {
    benchmat.addToRow(0, 1, 5);
    benchmat.addToRow(0, 1, 6);
    benchmat.addToRow(0, 1, 7);
    benchmat.addToRow(0, 1, 8);
    benchmat.addToRow(0, 1, 9);
    benchmat.addToRow(0, 1, 10);
  }
}
BENCHMARK(BM_Matrix_addToRow);

static void BM_Matrix_addToColumn(benchmark::State& state) {
  Matrix benchmat = createMatrix(512,512);
  for (auto _ : state) {
    benchmat.addToColumn(0, 1, 5);
    benchmat.addToColumn(0, 1, 6);
    benchmat.addToColumn(0, 1, 7);
    benchmat.addToColumn(0, 1, 8);
    benchmat.addToColumn(0, 1, 9);
    benchmat.addToColumn(0, 1, 10);
  }
}
BENCHMARK(BM_Matrix_addToColumn);

// Matrix multiply scalar

void mul_rows_by_const_scalar(Matrix& mat, int scaling_const) {
  for (unsigned row = 0; row < mat.getNumRows(); row++) {
    mat.ScaleRowScalar(row, scaling_const);
  }
}

static void BM_Matrix_mul_by_const_scalar(benchmark::State& state) {
  Matrix benchmat = createMatrix(2048,2048);
  for (auto _ : state) {
    mul_rows_by_const_scalar(benchmat, 5);
    mul_rows_by_const_scalar(benchmat, -2);
    mul_rows_by_const_scalar(benchmat, 3);
  }
}
BENCHMARK(BM_Matrix_mul_by_const_scalar);

// Matrix Multiply simd
void mul_rows_by_const_simd(Matrix& mat, int scaling_const) {
  for (unsigned row = 0; row < mat.getNumRows(); row++) {
    mat.ScaleRowVector(row, scaling_const);
  }
}

static void BM_Matrix_mul_by_const_simd(benchmark::State& state) {
  Matrix benchmat = createMatrix(2048,2048);
  for (auto _ : state) {
    mul_rows_by_const_simd(benchmat, 5);
    mul_rows_by_const_simd(benchmat, -2);
    mul_rows_by_const_simd(benchmat, 3);

  }
}
BENCHMARK(BM_Matrix_mul_by_const_simd);


#define N 128
#define M 256

void print_A(float A[][M]) {
  for (long i = 0; i < N; i++)
    for (long j = 0; j < M; j++)
      if (A[i][j] == 1.1102)
        printf("%f", A[i][j]);
}

static void BM_scaleRows_s(benchmark::State& state) {
  float A[N][M];
  float scale = 20.0;
  for (auto _ : state) {
    for (int i = 0; i < 1000; i++)
      #pragma clang loop vectorize(disable)
      for (long i = 0; i < N; i++)
        for (long j = 0; j < M; j++)
          A[i][j] *= scale;
  }
  print_A(A);
}
BENCHMARK(BM_scaleRows_s);

static void BM_scaleRows_v(benchmark::State& state) {
  float A[N][M];
  float scale = 20.0;
  for (auto _ : state) {
    for (int i = 0; i < 1000; i++)
      #pragma clang loop vectorize(enable)
      for (long i = 0; i < N; i++)
        for (long j = 0; j < M; j++)
          A[i][j] *= scale;
  }
  print_A(A);
}
BENCHMARK(BM_scaleRows_v);



BENCHMARK_MAIN();