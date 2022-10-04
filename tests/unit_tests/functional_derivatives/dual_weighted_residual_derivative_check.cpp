#include <deal.II/grid/grid_generator.h>
#include "dg/dg_factory.hpp"
#include "optimization/design_parameterization/inner_vol_parameterization.hpp"
#include "physics/physics_factory.h"
#include <deal.II/numerics/vector_tools.h>
#include "functional/dual_weighted_residual_obj_func.h"

const int nstate = 1;
int main (int argc, char * argv[])
{
    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);    
    int mpi_rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    dealii::ConditionalOStream pcout(std::cout, mpi_rank==0);

    const int dim = PHILIP_DIM;

    using namespace PHiLiP;   
    
    using VectorType = typename dealii::LinearAlgebra::distributed::Vector<double>;
    using MatrixType = dealii::TrilinosWrappers::SparseMatrix;

#if PHILIP_DIM == 1
    using MeshType = typename dealii::Triangulation<dim>;
#else
    using MeshType = typename dealii::parallel::distributed::Triangulation<dim>;
#endif
    
    // Create grid and dg. 
    std::shared_ptr<MeshType> grid = std::make_shared<MeshType>(MPI_COMM_WORLD);
    unsigned int grid_refinement_val = 5;
    dealii::GridGenerator::hyper_cube(*grid);
    grid->refine_global(grid_refinement_val);

    dealii::ParameterHandler parameter_handler; // Using default parameters. 
    Parameters::AllParameters::declare_parameters (parameter_handler);
    Parameters::AllParameters all_parameters;
    all_parameters.parse_parameters (parameter_handler);
    const unsigned int poly_degree = 3;
    const unsigned int grid_degree = 1;

    std::shared_ptr < DGBase<dim, double> > dg = DGFactory<dim, double>::create_discontinuous_galerkin(&all_parameters, poly_degree,poly_degree + 1, grid_degree, grid);
    dg->allocate_system();

    std::shared_ptr <Physics::PhysicsBase<dim,nstate,double>> physics_double = Physics::PhysicsFactory<dim, nstate, double>::create_Physics(&all_parameters);
    VectorType solution_with_ghost;
    solution_with_ghost.reinit(dg->locally_owned_dofs,dg->ghost_dofs, MPI_COMM_WORLD);
    dealii::VectorTools::interpolate(dg->dof_handler, *(physics_double->manufactured_solution_function), solution_with_ghost);
    dg->solution = solution_with_ghost;
    for (auto it = dg->solution.begin(); it != dg->solution.end(); ++it) {
        // Interpolating the exact manufactured solution caused some problems at the boundary conditions.
        // The manufactured solution is exactly equal to the manufactured_solution_function at the boundary,
        // therefore, the finite difference will change whether the flow is incoming or outgoing.
        // As a result, we would be differentiating at a non-differentiable point.
        // Hence, we fix this issue by taking the second derivative at a non-exact solution.
        (*it) += 1.0;
    }
    dg->solution.update_ghost_values();
    
    VectorType solution_coarse = dg->solution;
    std::unique_ptr<DualWeightedResidualObjFunc<dim, nstate, double>> dwr_objfunc = std::make_unique<DualWeightedResidualObjFunc<dim, nstate, double>> (dg);
    MatrixType interpolation_matrix;
    interpolation_matrix.copy_from(dwr_objfunc->interpolation_matrix);
    
    dg->change_cells_fe_degree_by_deltadegree_and_interpolate_solution(1);
    VectorType solution_fine_from_solution_transfer = dg->solution;
    dg->change_cells_fe_degree_by_deltadegree_and_interpolate_solution(-1);


    VectorType solution_fine_interpolated = solution_fine_from_solution_transfer;

    solution_fine_interpolated *= 0.0;

    interpolation_matrix.vmult(solution_fine_interpolated, solution_coarse);

    VectorType diff = solution_fine_interpolated;

    diff -= solution_fine_from_solution_transfer;

    if(diff.l2_norm() > 1.0e-15) 
    {
        pcout<<"Test failed. Interpolation isn't done properly."<<std::endl;
        pcout<<" Diff l2 norm = "<<diff.l2_norm()<<std::endl;
        return 1;
    }


    pcout<<"Interpolation matrix is good"<<std::endl;

    return 0; // Test passed
}
