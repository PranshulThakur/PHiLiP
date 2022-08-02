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
    compute_total_hessian();
}


template <int dim, int nstate, typename real, typename MeshType>
void TotalDerivativeObjfunc<dim, nstate, real, MeshType>::form_interpolation_matrix()
{
    std::cout<<"Forming interpolation matrix..."<<std::endl;
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
    std::cout<<std::endl<<"Refining or coarsening dg to degree "<<degree<<"..."<<std::endl;
    dealii::LinearAlgebra::distributed::Vector<real> old_solution(dg->solution);
    old_solution.update_ghost_values();
    using VectorType       = typename dealii::LinearAlgebra::distributed::Vector<real>;
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
         std::cout<<"Using normal solution transfer."<<std::endl;
    } 
    else 
    {
        solution_transfer.interpolate(dg->solution);
         std::cout<<"Using distributed solution transfer."<<std::endl;
    }

    dg->solution.update_ghost_values();
}

template <int dim, int nstate, typename real, typename MeshType>
void TotalDerivativeObjfunc<dim, nstate, real, MeshType>::compute_solution_tilde_and_solution_fine()
{
    std::cout<<"Computing solution fine and solution tilde..."<<std::endl;
    // Compute solution coarse tilde
    bool compute_dRdW = true, compute_dRdX=false;
    dg->assemble_residual(compute_dRdW, compute_dRdX);
    dg->system_matrix *= -1.0;
    solution_coarse_taylor_expanded.reinit(dg->solution.size());
    solve_linear(dg->system_matrix, dg->right_hand_side, solution_coarse_taylor_expanded, dg->all_parameters->linear_solver_param);
    solution_coarse_taylor_expanded += dg->solution;
    std::cout<<"Computed solution coarse taylor expanded."<<std::endl;
    // Interpolate solution on finer grid
    interpolation_matrix.vmult(solution_tilde_fine, solution_coarse_taylor_expanded);
    
    // Store r_u and r_x 
    solution_coarse_old = dg->solution;
    dg->solution = solution_coarse_taylor_expanded;
    compute_dRdW = true; compute_dRdX=false;
    dg->assemble_residual(compute_dRdW, compute_dRdX);
    r_u.copy_from(dg->system_matrix);
    r_u_transpose.copy_from(dg->system_matrix_transpose);
    std::cout<<"Stored r_u."<<std::endl;
    
    compute_dRdW = false; compute_dRdX=true;
    dg->assemble_residual(compute_dRdW, compute_dRdX);
    r_x.copy_from(dg->dRdXv);
    std::cout<<"Stored r_x."<<std::endl;

    // Get solution back
    dg->solution = solution_coarse_old;

//===============================================================================================================================================================================
    // Refine and interpolate solution on finer p.
    refine_or_coarsen_dg(dg->initial_degree+1);

    // Compute solution_fine taylor expanded
    compute_dRdW = true, compute_dRdX=false;
    dg->assemble_residual(compute_dRdW, compute_dRdX);
    dg->system_matrix *= -1.0;
    solution_fine.reinit(dg->solution.size());
    solve_linear(dg->system_matrix, dg->right_hand_side, solution_fine, dg->all_parameters->linear_solver_param);
    solution_fine += dg->solution;
    std::cout<<"Computed solution fine."<<std::endl;

    // Store R_u and R_x 
    dealii::LinearAlgebra::distributed::Vector<real> solution_fine_old = dg->solution;
    dg->solution = solution_fine;
    compute_dRdW = true; compute_dRdX = false;
    dg->assemble_residual(compute_dRdW, compute_dRdX);
    R_u.copy_from(dg->system_matrix);
    R_u_transpose.copy_from(dg->system_matrix_transpose);
    std::cout<<"Stored R_u."<<std::endl;
    
    compute_dRdW = false; compute_dRdX=true;
    dg->assemble_residual(compute_dRdW, compute_dRdX);
    R_x.copy_from(dg->dRdXv);
    std::cout<<"Stored R_x."<<std::endl;
    
    // Get solution back
    dg->solution = solution_fine_old;
}


template <int dim, int nstate, typename real, typename MeshType>
void TotalDerivativeObjfunc<dim, nstate, real, MeshType>::compute_adjoints()
{
    std::cout<<"Computing adjoints..."<<std::endl;
    Assert(objfunc->is_derivative_computed, dealii::ExcMessage("Derivative of objective function is not computed."));
    R_u_transpose *= -1.0; 
    r_u_transpose *= -1.0;

    dealii::LinearAlgebra::distributed::Vector<real> dF_dUH (solution_coarse_taylor_expanded.size()); // U_H_tilde
    adjoint_fine.reinit(solution_fine.size());
    adjoint_tilde.reinit(solution_coarse_taylor_expanded.size());
    interpolation_matrix.Tvmult(dF_dUH, objfunc->derivative_objfunc_wrt_solution_tilde);

    solve_linear(R_u_transpose, objfunc->derivative_objfunc_wrt_solution_fine, adjoint_fine, dg->all_parameters->linear_solver_param);
    solve_linear(r_u_transpose, dF_dUH, adjoint_tilde, dg->all_parameters->linear_solver_param);
}


template <int dim, int nstate, typename real, typename MeshType>
void TotalDerivativeObjfunc<dim, nstate, real, MeshType>::compute_total_derivative()
{
    std::cout<<"Computing total first order derivative..."<<std::endl;
    Assert(objfunc->is_derivative_computed, dealii::ExcMessage( "Derivative of objective function is not computed."));
    dF_dX_total = objfunc->derivative_objfunc_wrt_metric_nodes;
    R_x.Tvmult_add(dF_dX_total, adjoint_fine);
    r_x.Tvmult_add(dF_dX_total, adjoint_tilde);
    
    std::cout<<"solution_coarse_taylor_expanded = "<<std::endl;
    solution_coarse_taylor_expanded.print(std::cout, 3, true, false);
    std::cout<<"solution_fine = "<<std::endl;
    solution_fine.print(std::cout, 3, true, false);
    std::cout<<"solution_tilde_fine = "<<std::endl;
    solution_tilde_fine.print(std::cout, 3, true, false);
    
    std::cout<<"dF_dX_total = "<<std::endl;
    dF_dX_total.print(std::cout, 3, true, false);
}

template <int dim, int nstate, typename real, typename MeshType>
void TotalDerivativeObjfunc<dim, nstate, real, MeshType>::compute_total_hessian()
{
    dealii::FullMatrix<real> r_u_full; r_u_full.copy_from(r_u);
    dealii::FullMatrix<real> R_u_full; R_u_full.copy_from(R_u);

    dealii::FullMatrix<real> r_u_inverse = r_u_full; r_u_inverse.invert(r_u_full);
    dealii::FullMatrix<real> R_u_inverse = R_u_full; R_u_inverse.invert(R_u_full);
    r_u_inverse *= -1.0;
    R_u_inverse *= -1.0;
    
    dealii::FullMatrix<real> r_x_full; r_x_full.copy_from(r_x);
    dealii::FullMatrix<real> R_x_full; R_x_full.copy_from(R_x);

    // Store adjoint times d2R, d2r
    dg->solution = solution_coarse_taylor_expanded; 
    dg->set_dual(adjoint_tilde);
    dg->assemble_residual(false,false,true);
    dealii::FullMatrix<real> adjoint_times_d2rdudu; adjoint_times_d2rdudu.copy_from(dg->d2RdWdW);
    dealii::FullMatrix<real> adjoint_times_d2rdxdx; adjoint_times_d2rdxdx.copy_from(dg->d2RdXdX);
    dealii::FullMatrix<real> adjoint_times_d2rdudx; adjoint_times_d2rdudx.copy_from(dg->d2RdWdX);

    refine_or_coarsen_dg(dg->initial_degree + 1);
    dg->solution = solution_fine;
    dg->set_dual(adjoint_fine);
    dg->assemble_residual(false,false,true);
    dealii::FullMatrix<real> adjoint_times_d2RdUdU; adjoint_times_d2RdUdU.copy_from(dg->d2RdWdW);
    dealii::FullMatrix<real> adjoint_times_d2Rdxdx; adjoint_times_d2Rdxdx.copy_from(dg->d2RdXdX);
    dealii::FullMatrix<real> adjoint_times_d2RdUdx; adjoint_times_d2RdUdx.copy_from(dg->d2RdWdX);

    refine_or_coarsen_dg(dg->initial_degree);
    dg->solution = solution_coarse_old;

    // Form lagrangian with Uh 
    dealii::FullMatrix<real> dUh_dx = R_x_full;
    R_u_inverse.mmult(dUh_dx, R_x_full); // get dUh_dx
    
    dealii::FullMatrix<real> Lxx; Lxx.copy_from(objfunc->d2F_dX_dX);// get Lxx, Lxu and Luu
    dealii::FullMatrix<real> Lux; Lux.copy_from(objfunc->d2F_dWfine_dX);
    dealii::FullMatrix<real> Luu; Luu.copy_from(objfunc->d2F_dWfine_dWfine);

    Lxx.add(1.0, adjoint_times_d2Rdxdx);
    Lux.add(1.0, adjoint_times_d2RdUdx);
    Luu.add(1.0, adjoint_times_d2RdUdU);

    dealii::FullMatrix<real> term1 = Lxx;
    dUh_dx.Tmmult(term1, Lux, true);
    Lux.Tmmult(term1, dUh_dx, true);
    term1.triple_product(Luu, dUh_dx, dUh_dx, true);


    // Form lagrangian with U_h^H
    dealii::FullMatrix<real> dUH_dx = r_x_full;
    r_u_inverse.mmult(dUH_dx, r_x_full);
    dealii::FullMatrix<real> interpolation_matrix_full; interpolation_matrix_full.copy_from(interpolation_matrix);

    Lxx.copy_from(objfunc->d2F_dX_dX);
    Lxx.add(1.0, adjoint_times_d2rdxdx); // Lxx done
    Lux.copy_from(adjoint_times_d2rdudx);
    Luu.copy_from(adjoint_times_d2rdudu);
    dealii::FullMatrix<real> Fux; Fux.copy_from(objfunc->d2F_dWtilde_dX);
    dealii::FullMatrix<real> Fuu; Fuu.copy_from(objfunc->d2F_dWtilde_dWtilde);
    dealii::FullMatrix<real> Fxx; Fxx.copy_from(objfunc->d2F_dX_dX);

    interpolation_matrix_full.Tmmult(Lux ,Fux, true); // Lux done

    Luu.triple_product(Fuu, interpolation_matrix_full, interpolation_matrix_full, true);

    dealii::FullMatrix<real> term2 = Lxx;
    dUH_dx.Tmmult(term2,Lux,true);
    Lux.Tmmult(term2, dUH_dx, true);
    term2.triple_product(Luu,dUH_dx, dUH_dx, true);

    dealii::FullMatrix<real> F_Uh_uhH; F_Uh_uhH.copy_from(objfunc->d2F_dWfine_dWtilde);
    dealii::FullMatrix<real> term3 = Lxx;
    term3 *= 0.0;
    term3.triple_product(F_Uh_uhH, dUh_dx, dUH_dx, true);
    dealii::FullMatrix<real> Hessian_total = term1;
    Hessian_total.add(1.0, term2);
    Hessian_total.add(-1.0,Fxx);
    Hessian_total.add(1.0, term3);
    Hessian_total.Tadd(1.0, term3);
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
