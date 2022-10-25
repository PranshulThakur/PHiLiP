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
    all_parameters.linear_solver_param.linear_residual = 1.0e-14;
    all_parameters.optimization_param.mesh_weight_factor = 1.0e-2;
    all_parameters.optimization_param.mesh_volume_power = -2;
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
    
    const bool uses_solution_values = true;
    const bool uses_solution_gradient = false;
    const bool use_coarse_residual = false;
    std::unique_ptr<DualWeightedResidualObjFunc<dim, nstate, double>> dwr_func = std::make_unique<DualWeightedResidualObjFunc<dim, nstate, double>> (dg,
                                                                                                                                                     uses_solution_values, 
                                                                                                                                                     uses_solution_gradient, 
                                                                                                                                                     use_coarse_residual);
    MatrixType interpolation_matrix;
    interpolation_matrix.copy_from(dwr_func->interpolation_matrix);
    
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
        const std::vector<dealii::types::global_dof_index> &dof_indices_fine = dwr_func->cellwise_dofs_fine[cell_index];

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
    dg->change_cells_fe_degree_by_deltadegree_and_interpolate_solution(-1);

// ======= Check if volume nodes and the solution remain the same after evaluating the functional ============================================================================
    // This check ensures that volume_node/solution configuration stays the same. If this same configuration is used again, already computed values aren't re-evaluated.
    std::unique_ptr<Functional<dim, nstate, double>> dwr_objfunc = std::make_unique<DualWeightedResidualObjFunc<dim, nstate, double>> ( dg, 
                                                                                                                                        uses_solution_values, 
                                                                                                                                        uses_solution_gradient, 
                                                                                                                                        use_coarse_residual);
    VectorType diff_vol_nodes = dg->high_order_grid->volume_nodes;
    VectorType diff_sol = dg->solution;
 
    dwr_objfunc->evaluate_functional(true, true, false);
    const double value_original = dwr_objfunc->current_functional_value;
    
    diff_vol_nodes -= dg->high_order_grid->volume_nodes;
    diff_sol -= dg->solution;
    diff_vol_nodes.update_ghost_values();
    diff_sol.update_ghost_values();
    if(diff_vol_nodes.l2_norm() != 0.0) 
    {
        pcout<<"Volume nodes have been changed when not expected to change"<<std::endl;
        return 1;
    }
    if(diff_sol.l2_norm() != 0.0) 
    {
        pcout<<"Solution has changed. It wasn't expected to change after evaluating objective function."<<std::endl;
        return 1;
    }
    
//================ Check if Hessian-vector products work ====================================================
    pcout<<"Checking if Hessian vector products work without segfaults.\n"
          <<"If it gives an error, consider re-running the test in debug mode to check which assert statement has been triggered."<<std::endl; 
    VectorType vector_u_size1 (dg->solution); 
    VectorType vector_x_size1 (dg->high_order_grid->volume_nodes); 
    VectorType vector_u_size2 (dg->solution); 
    VectorType vector_x_size2 (dg->high_order_grid->volume_nodes); 
    
    dwr_objfunc->d2IdWdW_vmult(vector_u_size1, dg->solution);
    if(vector_u_size1.size() != dg->solution.size()) {return 1;}

    dwr_objfunc->d2IdXdX_vmult(vector_x_size1, dg->high_order_grid->volume_nodes);
    if(vector_x_size1.size() != dg->high_order_grid->volume_nodes.size()) {return 1;}
    
    
    dwr_objfunc->d2IdWdX_vmult(vector_u_size2, dg->high_order_grid->volume_nodes);
    if(vector_u_size2.size() != dg->solution.size()) {return 1;}
    
    
    dwr_objfunc->d2IdWdX_Tvmult(vector_x_size2, dg->solution);
    if(vector_x_size2.size() != dg->high_order_grid->volume_nodes.size()) {return 1;}
   
    pcout<<"d2IdWdW*dg->solution = "<<vector_u_size1.l2_norm()<<std::endl;
    pcout<<"d2IdXdX*dg->high_order_grid->volume_nodes = "<<vector_x_size1.l2_norm()<<std::endl;
    pcout<<"d2IdWdX*dg->high_order_grid->volume_nodes = "<<vector_u_size2.l2_norm()<<std::endl;
    pcout<<"d2IdWdX^T*dg->solution = "<<vector_x_size2.l2_norm()<<std::endl;

    
    pcout<<"Hessian vector products seem to work."<<std::endl;
    
// ====== Check dIdX finite difference ==========================================================================================
    pcout<<"Checking dIdX analytical vs finite difference."<<std::endl; 
    dwr_objfunc->evaluate_functional(true, true, false); // Shouldn't re-evaluate derivatives as it's already computed above.
    if(value_original != dwr_objfunc->current_functional_value) 
    {
        pcout<<"Value of the objective function has changed. Something's wrong.."<<std::endl;
        pcout<<"Difference = "<<value_original - dwr_objfunc->current_functional_value<<std::endl;
        return 1;
    }

    pcout<<"Evaluated analytical dIdX."<<std::endl; 

    VectorType dIdX_fd;
    dIdX_fd.reinit(dg->high_order_grid->volume_nodes); 
    
    double value_perturbed = value_original;

    const dealii::IndexSet &vol_range = dg->high_order_grid->volume_nodes.get_partitioner()->locally_owned_range();

    unsigned int n_vol_nodes = dg->high_order_grid->volume_nodes.size();
    AssertDimension(n_vol_nodes, dwr_objfunc->dIdX.size());
    AssertDimension(n_vol_nodes, dIdX_fd.size()); 
    pcout<<"All dimensions are good."<<std::endl; 

    double step_size_delx = 1.0e-8;
    pcout<<"Evaluating dIdX using finite difference."<<std::endl;

    for(unsigned int i_node = 0; i_node < n_vol_nodes; ++i_node)
    {
        if(vol_range.is_element(i_node)) {
            dg->high_order_grid->volume_nodes(i_node) += step_size_delx; // perturb node
        }
        dg->high_order_grid->volume_nodes.update_ghost_values();
        
        value_perturbed = dwr_objfunc->evaluate_functional();

        if(vol_range.is_element(i_node)) {
            dIdX_fd(i_node) = (value_perturbed - value_original)/step_size_delx; 
            dg->high_order_grid->volume_nodes(i_node) -= step_size_delx; // reset 
        }
        dg->high_order_grid->volume_nodes.update_ghost_values();
    }

    dIdX_fd.update_ghost_values();
    pcout<<"Done evaluating dIdX using finite difference."<<std::endl;

    VectorType diff_dIdX = dwr_objfunc->dIdX; 
    diff_dIdX -= dIdX_fd;
    diff_dIdX.update_ghost_values();

    pcout<<"dIdX analytical = "<<std::endl;
    dwr_objfunc->dIdX.print(std::cout, 3, true, false);
    
    pcout<<"dIdX finite difference = "<<std::endl;
    dIdX_fd.print(std::cout, 3, true, false);
// ====== Check dIdw finite difference ==========================================================================================
    pcout<<"Now checking dIdw analytical vs finite difference."<<std::endl; 
    dwr_objfunc->evaluate_functional(true, true, false); 
    pcout<<"Evaluated analytical dIdw."<<std::endl; 

    VectorType dIdw_fd;
    dIdw_fd.reinit(dg->solution); 

    const dealii::IndexSet &dof_range = dg->solution.get_partitioner()->locally_owned_range();  

    unsigned int n_dofs = dg->solution.size(); 
    AssertDimension(n_dofs, n_dofs_coarse);
    AssertDimension(n_dofs, dwr_objfunc->dIdw.size());
    AssertDimension(n_dofs, dIdw_fd.size()); 
    pcout<<"All dimensions are good."<<std::endl; 

    double step_size_delu = 1.0e-6;
    pcout<<"Evaluating dIdw using finite difference."<<std::endl; 

    for(unsigned int i_dof = 0; i_dof < n_dofs; ++i_dof)
    {
        if(dof_range.is_element(i_dof)){
            dg->solution(i_dof) += step_size_delu; // perturb solution
        }
        dg->solution.update_ghost_values();
        
        value_perturbed = dwr_objfunc->evaluate_functional();

        if(dof_range.is_element(i_dof)){
            dIdw_fd(i_dof) = (value_perturbed - value_original)/step_size_delu; 
            dg->solution(i_dof) -= step_size_delu; // reset solution 
        }
        dg->solution.update_ghost_values();
    }

    dIdw_fd.update_ghost_values();
    pcout<<"Done evaluating dIdw using finite difference."<<std::endl;

    VectorType diff_dIdw = dwr_objfunc->dIdw; 
    diff_dIdw -= dIdw_fd;
    diff_dIdw.update_ghost_values();

    pcout<<"dIdw analytical = "<<std::endl;
    dwr_objfunc->dIdw.print(std::cout, 3, true, false);
    
    pcout<<"dIdw finite difference = "<<std::endl;
    dIdw_fd.print(std::cout, 3, true, false);
    
    double tol_dIdw = 1.0e-4;
    double tol_dIdX = 1.0e-2;
  
    pcout<<"Analytical - finite difference dIdw = "<<diff_dIdw.l2_norm()<<std::endl;
    pcout<<"Analytical - finite difference dIdX = "<<diff_dIdX.l2_norm()<<std::endl;
    
    if(diff_dIdw.l2_norm() > tol_dIdw || diff_dIdX.l2_norm() > tol_dIdX)
    {
        pcout<<"Difference between finite difference and analytical dIdw or dIdX is high."<<std::endl;
        return 1;
    }

    pcout<<"Analytical derivatives match well with finite difference."<<std::endl;
    return 0; // Test passed
}
