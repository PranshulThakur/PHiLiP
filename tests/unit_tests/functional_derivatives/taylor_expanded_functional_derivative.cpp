#include <deal.II/grid/grid_generator.h>
#include "dg/dg_factory.hpp"
#include "physics/physics_factory.h"
#include <deal.II/numerics/vector_tools.h>
#include "functional/functional.h"

using namespace PHiLiP;   
const int nstate = 1;

int main (int argc, char * argv[])
{
    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);    
    int mpi_rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    dealii::ConditionalOStream pcout(std::cout, mpi_rank==0);

    const int dim = PHILIP_DIM;

    using VectorType = typename dealii::LinearAlgebra::distributed::Vector<double>;
    using MatrixType = dealii::TrilinosWrappers::SparseMatrix;

#if PHILIP_DIM == 1
    using MeshType = typename dealii::Triangulation<dim>;
#else
    using MeshType = typename dealii::parallel::distributed::Triangulation<dim>;
#endif
    
    // Create grid and dg. 
    std::shared_ptr<MeshType> grid = std::make_shared<MeshType>(
            #if PHILIP_DIM != 1
            MPI_COMM_WORLD
            #endif
            );
    unsigned int grid_refinement_val = 3;
    dealii::GridGenerator::hyper_cube(*grid);
    grid->refine_global(grid_refinement_val);

    dealii::ParameterHandler parameter_handler; // Using default parameters. 
    Parameters::AllParameters::declare_parameters (parameter_handler);
    Parameters::AllParameters all_parameters;
    all_parameters.parse_parameters(parameter_handler);
    all_parameters.linear_solver_param.linear_residual = 1.0e-14;
    all_parameters.manufactured_convergence_study_param.manufactured_solution_param.use_manufactured_source_term = true;
    all_parameters.manufactured_convergence_study_param.manufactured_solution_param.manufactured_solution_type = Parameters::ManufacturedSolutionParam::ManufacturedSolutionType::exp_solution;

    const unsigned int poly_degree = 1;
    const unsigned int grid_degree = 1;

    std::shared_ptr < DGBase<dim, double> > dg = DGFactory<dim, double>::create_discontinuous_galerkin(&all_parameters, poly_degree,poly_degree + 1, grid_degree, grid);
    dg->allocate_system();
    pcout<<"Created and allocated DG."<<std::endl;

    std::shared_ptr <Physics::PhysicsBase<dim,nstate,double>> physics_double = Physics::PhysicsFactory<dim, nstate, double>::create_Physics(&all_parameters);
    pcout<<"Created physics double."<<std::endl;
    VectorType solution_no_ghost;
    solution_no_ghost.reinit(dg->locally_owned_dofs, MPI_COMM_WORLD);
    dealii::VectorTools::interpolate(dg->dof_handler, *(physics_double->manufactured_solution_function), solution_no_ghost);
    pcout<<"Interpolated solution."<<std::endl;
    dg->solution = solution_no_ghost;
    for (auto it = dg->solution.begin(); it != dg->solution.end(); ++it) {
        // Interpolating the exact manufactured solution caused some problems at the boundary conditions.
        // The manufactured solution is exactly equal to the manufactured_solution_function at the boundary,
        // therefore, the finite difference will change whether the flow is incoming or outgoing.
        // As a result, we would be differentiating at a non-differentiable point.
        // Hence, we fix this issue by taking the second derivative at a non-exact solution.
        (*it) += 1.0;
    }
    dg->solution.update_ghost_values();
    return 0;
}
