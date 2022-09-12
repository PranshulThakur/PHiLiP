set(GRID_SOURCE
    high_order_grid.cpp
    gmsh_reader.cpp
    meshmover_linear_elasticity.cpp
    free_form_deformation.cpp
    mesh_adaptation.cpp) 

add_subdirectory(grids)

add_subdirectory(error_indicators_for_mesh_adaptation)

foreach(dim RANGE 1 3)
    # Output library
    string(CONCAT HighOrderGridLib HighOrderGrid_${dim}D)
    add_library(${HighOrderGridLib} STATIC ${GRID_SOURCE})
    # Replace occurences of PHILIP_DIM with 1, 2, or 3 in the code
    target_compile_definitions(${HighOrderGridLib} PRIVATE PHILIP_DIM=${dim})

    # Library dependency
    string(CONCAT FunctionalLib Functional_${dim}D)
    target_link_libraries(${HighOrderGridLib} ${FunctionalLib})

    string(CONCAT MeshErrorIndicatorsLib MeshErrorIndicators_${dim}D)
    target_link_libraries(${HighOrderGridLib} ${MeshErrorIndicatorsLib})

    # Setup target with deal.II
    if(NOT DOC_ONLY)
        DEAL_II_SETUP_TARGET(${HighOrderGridLib})
    endif()

    unset(HighOrderGridLib)
    unset(FunctionalLib)
    unset(MeshErrorIndicatorsLib)

endforeach()
