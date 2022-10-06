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
    unsigned int grid_refinement_val = 3;
    dealii::GridGenerator::hyper_cube(*grid);
    grid->refine_global(grid_refinement_val);

    dealii::ParameterHandler parameter_handler; // Using default parameters. 
    Parameters::AllParameters::declare_parameters (parameter_handler);
    Parameters::AllParameters all_parameters;
    all_parameters.parse_parameters (parameter_handler);
    const unsigned int poly_degree = 1;
    const unsigned int grid_degree = 1;

    std::shared_ptr < DGBase<dim, double> > dg = DGFactory<dim, double>::create_discontinuous_galerkin(&all_parameters, poly_degree,poly_degree + 1, grid_degree, grid);
    dg->allocate_system();
    unsigned int n_dofs_coarse = dg->n_dofs();

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

    if(diff.l2_norm() > 1.0e-12) 
    {
        pcout<<"Test failed. Interpolation isn't done properly."<<std::endl;
        pcout<<" Diff l2 norm = "<<diff.l2_norm()<<std::endl;
        return 1;
    }


    pcout<<"Interpolation matrix is good"<<std::endl;
// ====== Dof indices check ==========================================================================================
/*
    pcout<<"Now checking if stored dof_indices are good..."<<std::endl;
    
    dg->change_cells_fe_degree_by_deltadegree_and_interpolate_solution(1);

    std::vector<dealii::types::global_dof_index> dof_indices;
    
    for(const auto &cell : dg->dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned()) continue;

        const dealii::types::global_dof_index cell_index = cell->active_cell_index();
        const unsigned int i_fele = cell->active_fe_index();
        const dealii::FESystem<dim,dim> &fe_ref = dg->fe_collection[i_fele];
        const unsigned int n_dofs_cell = fe_ref.n_dofs_per_cell();

        dof_indices.resize(n_dofs_cell);
        cell->get_dof_indices (dof_indices);
        const std::vector<dealii::types::global_dof_index> &dof_indices_fine = dwr_objfunc->cellwise_dofs_fine[cell_index];

        for(unsigned int i_dof=0; i_dof < n_dofs_cell; ++i_dof)
        {
            if( dof_indices_fine[i_dof] != dof_indices[i_dof])
            {
                pcout<<"Dof indices are different"<<std::endl;
                return 1;
            }
        }

    } // cell loop ends

    pcout<<"Dof indices are the same."<<std::endl;
*/
// ====== Check dIdw finite difference ==========================================================================================
    pcout<<"Now checking dIdw analytical vs finite difference."<<std::endl; 
    dwr_objfunc->evaluate_functional(true, true, false);
    pcout<<"Evaluated analytical dIdw."<<std::endl; 

    VectorType dIdw_fd;
    dIdw_fd.reinit(dg->solution);

    double value_original = dwr_objfunc->current_functional_value;
    double value_perturbed = value_original;

    const dealii::IndexSet &dof_range = dg->solution.get_partitioner()->locally_owned_range();
    // const dealii::IndexSet &vol_nodes_range = dg->high_order_grid->volume_nodes.get_partitioner()->locally_owned_range();

    unsigned int n_dofs = dg->solution.size();
    AssertDimension(n_dofs, n_dofs_coarse);
    AssertDimension(n_dofs, dwr_objfunc->dIdw.size());
    AssertDimension(n_dofs, dIdw_fd.size());
    pcout<<"All dimensions are good."<<std::endl; 
   // unsigned int n_vol_nodes = dg->high_order_grid->volume_nodes.size();

    double step_size = 1.0e-5;
    pcout<<"Evaluating dIdw using finite difference."<<std::endl; 

    for(unsigned int i_dof = 0; i_dof < n_dofs; ++i_dof)
    {
        if(! dof_range.is_element(i_dof)) {continue;}

        dg->solution(i_dof) += step_size; // perturb solution

        value_perturbed = dwr_objfunc->evaluate_functional();

        dIdw_fd(i_dof) = (value_perturbed - value_original)/step_size;

        dg->solution(i_dof) -= step_size; // reset solution
    }
    pcout<<"Done evaluating dIdw using finite difference."<<std::endl; 

    VectorType diff_dIdw = dwr_objfunc->dIdw;
    diff_dIdw -= dIdw_fd;
    diff_dIdw.update_ghost_values();

    if(diff_dIdw.l2_norm() > 1.0e-15)
    {
        pcout<<"Difference between finite difference and analytical dIdw is high. L2 norm of diff_dIdw = "<<diff_dIdw.l2_norm()<<std::endl;
        return 1;
    }


    return 0; // Test passed
}
