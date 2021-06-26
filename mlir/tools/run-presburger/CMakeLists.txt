set (LIBS
        MLIRLoopAnalysis
        )

add_llvm_tool(run-presburger
        run-presburger-set-ops.cpp

        DEPENDS
        ${LIBS}
        )

target_link_libraries(run-presburger PRIVATE ${LIBS} benchmark)
target_compile_options(run-presburger PRIVATE -march=native -fopenmp)

#include(CheckCXXCompilerFlag)
#CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
#if(COMPILER_SUPPORTS_MARCH_NATIVE)
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
#endif()

llvm_update_compile_flags(run-presburger)
mlir_check_all_link_libraries(run-presburger)