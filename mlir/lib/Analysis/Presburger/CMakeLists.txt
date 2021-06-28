add_mlir_library(MLIRPresburger
  Matrix.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  )

target_compile_options(MLIRPresburger PUBLIC -fno-vectorize)