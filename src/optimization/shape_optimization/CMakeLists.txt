set(SHAPE_OPTIMIZATION_SOURCE
    flow_constraints.cpp 
    rol_objective.cpp
    )


foreach(dim RANGE 1 3)
    # Output library
    string(CONCAT ShapeOptimizationLib ShapeOptimization_${dim}D)
    add_library(${ShapeOptimizationLib} STATIC ${SHAPE_OPTIMIZATION_SOURCE})
    # Replace occurences of PHILIP_DIM with 1, 2, or 3 in the code
    target_compile_definitions(${ShapeOptimizationLib} PRIVATE PHILIP_DIM=${dim})

    # Link with OptimizationLib 
    string(CONCAT OptimizationLib Optimization_${dim}D)
    target_link_libraries(${ShapeOptimizationLib} ${OptimizationLib})

    # Setup target with deal.II
    if(NOT DOC_ONLY)
        DEAL_II_SETUP_TARGET(${ShapeOptimizationLib})
    endif()

    unset(ShapeOptimizationLib)
    unset(OptimizationLib)

endforeach()
