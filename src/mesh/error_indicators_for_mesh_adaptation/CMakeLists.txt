set(MESH_ERROR_INDICATORS_SRC
    mesh_error_factory.cpp
    mesh_error_estimate_base.cpp
    residual_based_error_indicator.cpp
    dual_weighted_residual.cpp
    direct_goal_oriented_error.cpp
    )

foreach(dim RANGE 1 3)
    # Output library
    string(CONCAT MeshErrorIndicatorsLib MeshErrorIndicators_${dim}D)
    add_library(${MeshErrorIndicatorsLib} STATIC ${MESH_ERROR_INDICATORS_SRC})

    # Replace occurences of PHILIP_DIM with dim in the code
    target_compile_definitions(${MeshErrorIndicatorsLib} PRIVATE PHILIP_DIM=${dim})
    
    # Library dependency
    string(CONCAT FunctionalLib Functional_${dim}D)
    target_link_libraries(${MeshErrorIndicatorsLib} ${FunctionalLib})

    # Setup target with deal.II
    if(NOT DOC_ONLY)
        DEAL_II_SETUP_TARGET(${MeshErrorIndicatorsLib})
    endif()

    unset(MeshErrorIndicatorsLib)
    unset(FunctionalLib)

endforeach()
