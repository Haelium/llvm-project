add_mlir_unittest(MLIRPresburgerMatrixTests
  MatrixTest.cpp
)

target_link_libraries(MLIRPresburgerMatrixTests
  PRIVATE MLIRPresburger)

