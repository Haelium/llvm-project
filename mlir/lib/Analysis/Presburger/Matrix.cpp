//===- Matrix.cpp - MLIR Matrix Class -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Presburger/Matrix.h"

// Delete these - Bourke's tests
#include <omp.h>
#include <immintrin.h>

namespace mlir {

Matrix::Matrix(unsigned rows, unsigned columns)
    : nRows(rows), nColumns(columns), data(nRows * nColumns) {}

Matrix Matrix::identity(unsigned dimension) {
  Matrix matrix(dimension, dimension);
  for (unsigned i = 0; i < dimension; ++i)
    matrix(i, i) = 1;
  return matrix;
}

int64_t &Matrix::at(unsigned row, unsigned column) {
  assert(row < getNumRows() && "Row outside of range");
  assert(column < getNumColumns() && "Column outside of range");
  return data[row * nColumns + column];
}

int64_t Matrix::at(unsigned row, unsigned column) const {
  assert(row < getNumRows() && "Row outside of range");
  assert(column < getNumColumns() && "Column outside of range");
  return data[row * nColumns + column];
}

int64_t &Matrix::operator()(unsigned row, unsigned column) {
  return at(row, column);
}

int64_t Matrix::operator()(unsigned row, unsigned column) const {
  return at(row, column);
}

unsigned Matrix::getNumRows() const { return nRows; }

unsigned Matrix::getNumColumns() const { return nColumns; }

void Matrix::resizeVertically(unsigned newNRows) {
  nRows = newNRows;
  data.resize(nRows * nColumns);
}

void Matrix::swapRows(unsigned row, unsigned otherRow) {
  assert((row < getNumRows() && otherRow < getNumRows()) &&
         "Given row out of bounds");
  if (row == otherRow)
    return;
  for (unsigned col = 0; col < nColumns; col++)
    std::swap(at(row, col), at(otherRow, col));
}

void Matrix::swapColumns(unsigned column, unsigned otherColumn) {
  assert((column < getNumColumns() && otherColumn < getNumColumns()) &&
         "Given column out of bounds");
  if (column == otherColumn)
    return;
  for (unsigned row = 0; row < nRows; row++)
    std::swap(at(row, column), at(row, otherColumn));
}

ArrayRef<int64_t> Matrix::getRow(unsigned row) const {
  return {&data[row * nColumns], nColumns};
}

void Matrix::addToRow(unsigned sourceRow, unsigned targetRow, int64_t scale) {
  if (scale == 0)
    return;
  for (unsigned col = 0; col < nColumns; ++col)
    at(targetRow, col) += scale * at(sourceRow, col);
  return;
}

void Matrix::addToColumn(unsigned sourceColumn, unsigned targetColumn,
                         int64_t scale) {
  if (scale == 0)
    return;
  for (unsigned row = 0, e = getNumRows(); row < e; ++row)
    at(row, targetColumn) += scale * at(row, sourceColumn);
}

void Matrix::negateColumn(unsigned column) {
  for (unsigned row = 0, e = getNumRows(); row < e; ++row)
    at(row, column) = -at(row, column);
}

void Matrix::print(raw_ostream &os) const {
  for (unsigned row = 0; row < nRows; ++row) {
    for (unsigned column = 0; column < nColumns; ++column)
      os << at(row, column) << ' ';
    os << '\n';
  }
}

void Matrix::dump() const { print(llvm::errs()); }

// Remove these - Bourke's tests
void Matrix::ScaleRowScalar(unsigned targetRow, int64_t scale) {
  #pragma clang loop vectorize(disable)
  for (unsigned i = 0; i < getNumColumns(); i++) {
    data[targetRow * nColumns + i] = data[targetRow * nColumns + i] + scale;
  }
}

void Matrix::ScaleRowVector(unsigned targetRow, int64_t scale)  {
  #pragma clang loop vectorize(enable)
  for (unsigned i = 0; i < getNumColumns(); i++) {
    data[targetRow * nColumns + i] = data[targetRow * nColumns + i] + scale;
  }
  /*
  __m256i scale_m256i = __m256i_set1_i64(scale);

  for (unsigned i = 0; i < getNumColumns()-4; i+=4) {
    __m256d input = _m256i_loadu_i64(data[targetRow * nColumns + i]);
    __mm256_storeu_pd(_m256i_mul_i64(), data[targetRow * nColumns + i]);
  }*/
}




} // namespace mlir

