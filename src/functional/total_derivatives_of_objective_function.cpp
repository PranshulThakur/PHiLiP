#include "total_derivatives_of_objective_function.h"


namespace PHiLiP {

template <int dim, int nstate, typename real, typename MeshType>
TotalDerivativeObjfunc<dim, nstate, real, MeshType>::TotalDerivativeObjfunc(std::shared_ptr<DGBase<dim, real, MeshType>> _dg)
    :dg(_dg)
{
    form_interpolation_matrix(); // Also resizes solution_tilde_fine
    compute_solution_tilde_and_solution_fine();
    
    objfunc = std::make_unique<ObjectiveFunctionMeshAdaptation<dim, nstate, real, MeshType>>(dg, solution_fine, solution_tilde_fine);
    objective_function_val = objfunc->evaluate_objective_function_and_derivatives();
    refine_or_coarsen_dg(dg->initial_degree); //coarsen dg back.

    compute_adjoints();
    compute_total_derivative();
}


template <int dim, int nstate, typename real, typename MeshType>
void TotalDerivativeObjfunc<dim, nstate, real, MeshType>::form_interpolation_matrix()
{
    
    const unsigned int coarse_degree = dg->initial_degree;
    const unsigned int fine_degree = coarse_degree + 1;
    const dealii::FE_DGQ<dim> fe_dg_coarse(coarse_degree);
    const dealii::FE_DGQ<dim> fe_dg_fine(fine_degree);

    dealii::FullMatrix<real> local_interpolation_matrix(fe_dg_fine.n_dofs_per_cell(), fe_dg_coarse.n_dofs_per_cell());
    dealii::FETools::get_interpolation_matrix(fe_dg_coarse, fe_dg_fine, local_interpolation_matrix);
    
    const unsigned int n_rows_local = fe_dg_fine.n_dofs_per_cell();
    const unsigned int n_cols_local = fe_dg_coarse.n_dofs_per_cell();
    const unsigned int n_rows_global = n_rows_local*dg->triangulation->n_active_cells();
    const unsigned int n_cols_global = n_cols_local*dg->triangulation->n_active_cells();


    dealii::DynamicSparsityPattern dsp(n_rows_global, n_cols_global);
    
    for(unsigned int cell_no = 0; cell_no < dg->triangulation->n_active_cells(); cell_no++)
    {
        unsigned int i_global = cell_no*n_rows_local;
        unsigned int j_global = cell_no*n_cols_local;
        for(unsigned int i=0; i<n_rows_local; i++)
        {
            for(unsigned int j=0; j<n_cols_local; j++)
            {
                dsp.add(i_global + i, j_global + j);
            }
        }
    }
    dealii::SparsityPattern      sparsity_pattern;
    sparsity_pattern.copy_from(dsp);
    interpolation_matrix.reinit(sparsity_pattern); 


    for(unsigned int cell_no = 0; cell_no < dg->triangulation->n_active_cells(); cell_no++)
    {
        unsigned int i_global = cell_no*n_rows_local;
        unsigned int j_global = cell_no*n_cols_local;
        for(unsigned int i=0; i<n_rows_local; i++)
        {
            for(unsigned int j=0; j<n_cols_local; j++)
            {
                interpolation_matrix.set(i_global + i, j_global + j, local_interpolation_matrix(i,j));
            }
        }
    }

    solution_tilde_fine.reinit(n_rows_global);
}

template <int dim, int nstate, typename real, typename MeshType>
void TotalDerivativeObjfunc<dim, nstate, real, MeshType>::refine_or_coarsen_dg(unsigned int degree)
{
    dealii::LinearAlgebra::distributed::Vector<double> old_solution(dg->solution);
    old_solution.update_ghost_values();
    using VectorType       = typename dealii::LinearAlgebra::distributed::Vector<double>;
    using DoFHandlerType   = typename dealii::DoFHandler<dim>;
    using SolutionTransfer = typename MeshTypeHelper<MeshType>::template SolutionTransfer<dim,VectorType,DoFHandlerType>;

    SolutionTransfer solution_transfer(dg->dof_handler);
    solution_transfer.prepare_for_coarsening_and_refinement(old_solution);
    dg->set_all_cells_fe_degree(degree);
    dg->allocate_system();
    dg->solution.zero_out_ghosts();
    
    if constexpr (std::is_same_v<typename dealii::SolutionTransfer<dim,VectorType,DoFHandlerType>, decltype(solution_transfer)>) 
    {
         solution_transfer.interpolate(old_solution, dg->solution);
    } 
    else 
    {
        solution_transfer.interpolate(dg->solution);
    }

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
    interpolation_matrix.vmult(solution_tilde_fine, solution_coarse_taylor_expanded);

    // Store r_u and r_x 
    dealii::LinearAlgebra::distributed::Vector<real> solution_coarse_old = dg->solution;
    dg->solution = solution_coarse_taylor_expanded;
    compute_dRdW = true; compute_dRdX=false;
    dg->assemble_residual(compute_dRdW, compute_dRdX);
    r_u.copy_from(dg->system_matrix);
    r_u_transpose.copy_from(dg->system_matrix_transpose);
    
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
    compute_dRdW = true; compute_dRdX = false;
    dg->assemble_residual(compute_dRdW, compute_dRdX);
    R_u.copy_from(dg->system_matrix);
    R_u_transpose.copy_from(dg->system_matrix_transpose);
    
    compute_dRdW = false; compute_dRdX=true;
    dg->assemble_residual(compute_dRdW, compute_dRdX);
    R_x.copy_from(dg->dRdXv);
    
    // Get solution back
    dg->solution = solution_fine_old;
}


template <int dim, int nstate, typename real, typename MeshType>
void TotalDerivativeObjfunc<dim, nstate, real, MeshType>::compute_adjoints()
{
    Assert(objfunc->is_derivative_computed, dealii::ExcMessage( "Derivative of objective function is not computed."));
    R_u_transpose *= -1.0; 
    r_u_transpose *= -1.0;

    dealii::LinearAlgebra::distributed::Vector<real> dF_dUH (solution_coarse_taylor_expanded.size()); // U_H_tilde
    interpolation_matrix.Tvmult(dF_dUH, objfunc->derivative_objfunc_wrt_solution_tilde);

    solve_linear(R_u_transpose, objfunc->derivative_objfunc_wrt_solution_fine, adjoint_fine, dg->all_parameters->linear_solver_param);
    solve_linear(r_u_transpose, dF_dUH, adjoint_tilde, dg->all_parameters->linear_solver_param);
}


template <int dim, int nstate, typename real, typename MeshType>
void TotalDerivativeObjfunc<dim, nstate, real, MeshType>::compute_total_derivative()
{
    Assert(objfunc->is_derivative_computed, dealii::ExcMessage( "Derivative of objective function is not computed."));
    dF_dX_total = objfunc->derivative_objfunc_wrt_metric_nodes;
    R_x.Tvmult_add(dF_dX_total, adjoint_fine);
    r_x.Tvmult_add(dF_dX_total, adjoint_tilde);
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
