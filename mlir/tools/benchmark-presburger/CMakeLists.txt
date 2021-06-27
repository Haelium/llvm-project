set (LIBS
        MLIRLoopAnalysis
        )

add_llvm_tool(benchmark-presburger
        benchmark-presburger.cpp

        DEPENDS
        ${LIBS}
        )

target_link_libraries(benchmark-presburger PRIVATE ${LIBS} benchmark)
target_compile_options(benchmark-presburger PRIVATE -march=native -fopenmp)

llvm_update_compile_flags(benchmark-presburger)
mlir_check_all_link_libraries(benchmark-presburger)