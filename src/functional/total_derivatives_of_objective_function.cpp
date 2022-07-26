#include "total_derivatives_of_objective_function.h"


namespace PHiLiP {

template <int dim, int nstate, typename real, typename MeshType>
TotalDerivativeObjfunc<dim, nstate, real, MeshType>::TotalDerivativeObjfunc(std::shared_ptr<DGBase<dim, real, MeshType>> _dg)
    :dg(_dg)
    {}

template <int dim, int nstate, typename real, typename MeshType>
void TotalDerivativeObjfunc<dim, nstate, real, MeshType>::refine_or_coarsen_dg(unsigned int degree)
{
    dealii::LinearAlgebra::distributed::Vector<double> old_solution(dg->solution);
    old_solution.update_ghost_values();
    // NOTE: IMPLEMENTED FOR 1D ONLY FOR NOW. Change SolutionTransfer while using MPI.
    dealii::SolutionTransfer<dim, dealii::LinearAlgebra::distributed::Vector<double>, dealii::DoFHandler<dim>> solution_transfer(dg->dof_handler);
    solution_transfer.prepare_for_coarsening_and_refinement(old_solution);
    dg->set_all_cells_fe_degree(degree);
    dg->allocate_system();
    dg->solution.zero_out_ghosts();
    solution_transfer.interpolate(old_solution, dg->solution);
    dg->solution.update_ghost_values();
}

template <int dim, int nstate, typename real, typename MeshType>
void TotalDerivativeObjfunc<dim, nstate, real, MeshType>::compute_solution_tilde_and_solution_fine()
{
    // Compute solution coarse tilde
    bool compute_dRdW = true, compute_dRdX=false;
    dg->assemble_residual(compute_dRdW, compute_dRdX);
    dg->system_matrix *= -1.0;
    solution_coarse_taylor_expanded.reinit(dg->solution.size());
    solve_linear(dg->system_matrix, dg->right_hand_side, solution_coarse_taylor_expanded, dg->all_parameters->linear_solver_param);
    solution_coarse_taylor_expanded += dg->solution;
    // Interpolate solution on finer grid
    // NOTE : Yet to be implemented

    // Store r_u and r_x 
    dealii::LinearAlgebra::distributed::Vector<real> solution_coarse_old = dg->solution;
    dg->solution = solution_coarse_taylor_expanded;
    compute_dRdW = true; compute_dRdX=false;
    dg->assemble_residual(compute_dRdW, compute_dRdX);
    r_u.copy_from(dg->system_matrix);
    
    compute_dRdW = false; compute_dRdX=true;
    dg->assemble_residual(compute_dRdW, compute_dRdX);
    r_x.copy_from(dg->dRdXv);

    // Get solution back
    dg->solution = solution_coarse_old;

//===============================================================================================================================================================================
    // Refine and interpolate solution on finer p.
    refine_or_coarsen_dg(dg->initial_degree+1);

    // Compute solution_fine taylor expanded
    compute_dRdW = true, compute_dRdX=false;
    dg->assemble_residual(compute_dRdW, compute_dRdX);
    dg->system_matrix *= -1.0;
    solution_coarse_taylor_expanded.reinit(dg->solution.size());
    solve_linear(dg->system_matrix, dg->right_hand_side, solution_fine, dg->all_parameters->linear_solver_param);
    solution_fine += dg->solution;

    // Store R_u and R_x 
    dealii::LinearAlgebra::distributed::Vector<real> solution_fine_old = dg->solution;
    dg->solution = solution_fine;
    compute_dRdW = true; compute_dRdX=false;
    dg->assemble_residual(compute_dRdW, compute_dRdX);
    R_u.copy_from(dg->system_matrix);
    
    compute_dRdW = false; compute_dRdX=true;
    dg->assemble_residual(compute_dRdW, compute_dRdX);
    R_x.copy_from(dg->dRdXv);
    
    // Get solution back
    dg->solution = solution_fine_old;

    // Coarsen and interpolate back
    refine_or_coarsen_dg(dg->initial_degree);







}





template class TotalDerivativeObjfunc<PHILIP_DIM, 1, double, dealii::Triangulation<PHILIP_DIM>>;
template class TotalDerivativeObjfunc<PHILIP_DIM, 2, double, dealii::Triangulation<PHILIP_DIM>>;
template class TotalDerivativeObjfunc<PHILIP_DIM, 3, double, dealii::Triangulation<PHILIP_DIM>>;
template class TotalDerivativeObjfunc<PHILIP_DIM, 4, double, dealii::Triangulation<PHILIP_DIM>>;
template class TotalDerivativeObjfunc<PHILIP_DIM, 5, double, dealii::Triangulation<PHILIP_DIM>>;
#if PHILIP_DIM != 1
template class TotalDerivativeObjfunc<PHILIP_DIM, 1, double,  dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class TotalDerivativeObjfunc<PHILIP_DIM, 2, double,  dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class TotalDerivativeObjfunc<PHILIP_DIM, 3, double,  dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class TotalDerivativeObjfunc<PHILIP_DIM, 4, double,  dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class TotalDerivativeObjfunc<PHILIP_DIM, 5, double,  dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif
} // namespace PHiLiP
