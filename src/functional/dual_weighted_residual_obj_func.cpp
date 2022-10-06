#include "dual_weighted_residual_obj_func.h"
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>
#include "linear_solver/linear_solver.h"

namespace PHiLiP {

template <int dim, int nstate, typename real>
DualWeightedResidualObjFunc<dim, nstate, real> :: DualWeightedResidualObjFunc( 
    std::shared_ptr<DGBase<dim,real>> dg_input,
    const bool uses_solution_values,
    const bool uses_solution_gradient)
    : Functional<dim, nstate, real> (dg_input, uses_solution_values, uses_solution_gradient)
    , R_u(std::make_unique<MatrixType>())
    , R_u_transpose(std::make_unique<MatrixType>())
    , matrix_ux(std::make_unique<MatrixType>())
    , matrix_uu(std::make_unique<MatrixType>())
    , interpolation_matrix(std::make_unique<MatrixType>())
{
    compute_interpolation_matrix(); // also stores cellwise_dofs_fine, vector coarse and vector fine.
    functional = FunctionalFactory<dim,nstate,real>::create_Functional(this->dg->all_parameters->functional_param, this->dg);
}

//===================================================================================================================================================
//                          Functions used only once in constructor
//===================================================================================================================================================
template<int dim, int nstate, typename real>
void DualWeightedResidualObjFunc<dim, nstate, real> :: compute_interpolation_matrix()
{ 
    vector_coarse = this->dg->solution; // copies values and parallel layout.
    unsigned int n_dofs_coarse = this->dg->n_dofs();
    this->dg->change_cells_fe_degree_by_deltadegree_and_interpolate_solution(1);
    vector_fine = this->dg->solution;
    unsigned int n_dofs_fine = this->dg->n_dofs();
    const dealii::IndexSet dofs_fine_locally_relevant_range = this->dg->locally_relevant_dofs;
    cellwise_dofs_fine = get_cellwise_dof_indices();
    this->dg->change_cells_fe_degree_by_deltadegree_and_interpolate_solution(-1);
    AssertDimension(vector_coarse.size(), this->dg->solution.size());     

    // Get all possible interpolation matrices for available poly order combinations.
    dealii::Table<2,dealii::FullMatrix<real>> interpolation_hp;
    extract_interpolation_matrices(interpolation_hp);

    // Get locally owned dofs
    const dealii::IndexSet &dofs_range_coarse = vector_coarse.get_partitioner()->locally_owned_range();
    const dealii::IndexSet &dofs_range_fine = vector_fine.get_partitioner()->locally_owned_range();

    dealii::DynamicSparsityPattern dsp(n_dofs_fine, n_dofs_coarse, dofs_range_fine);
    std::vector<dealii::types::global_dof_index> dof_indices;
    
    for(const auto &cell : this->dg->dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned()) continue;

        const dealii::types::global_dof_index cell_index = cell->active_cell_index();
        const unsigned int i_fele = cell->active_fe_index();
        const dealii::FESystem<dim,dim> &fe_ref = this->dg->fe_collection[i_fele];
        const unsigned int n_dofs_cell = fe_ref.n_dofs_per_cell();

        dof_indices.resize(n_dofs_cell);
        cell->get_dof_indices (dof_indices);
        const std::vector<dealii::types::global_dof_index> &dof_indices_fine = cellwise_dofs_fine[cell_index];

        for(unsigned int i=0; i < dof_indices_fine.size(); ++i)
        {
            for(unsigned int j=0; j < n_dofs_cell; ++j)
            {
                dsp.add(dof_indices_fine[i], dof_indices[j]);
            }
        }

    } // cell loop ends

    dealii::SparsityTools::distribute_sparsity_pattern(dsp, dofs_range_fine, MPI_COMM_WORLD, dofs_fine_locally_relevant_range);
    interpolation_matrix->reinit(dofs_range_fine, dofs_range_coarse, dsp, MPI_COMM_WORLD);

    for(const auto &cell : this->dg->dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned()) continue;

        const dealii::types::global_dof_index cell_index = cell->active_cell_index();
        const unsigned int i_fele = cell->active_fe_index();
        const dealii::FESystem<dim,dim> &fe_ref = this->dg->fe_collection[i_fele];
        const unsigned int n_dofs_cell = fe_ref.n_dofs_per_cell();

        dof_indices.resize(n_dofs_cell);
        cell->get_dof_indices (dof_indices);
        const std::vector<dealii::types::global_dof_index> &dof_indices_fine = cellwise_dofs_fine[cell_index];
        
        assert(i_fele + 1 <= this->dg->max_degree);
        const dealii::FullMatrix<real> &interpolation_matrix_local = interpolation_hp(i_fele + 1, i_fele);
        AssertDimension(interpolation_matrix_local.m(), dof_indices_fine.size());
        AssertDimension(interpolation_matrix_local.n(), n_dofs_cell);

        for(unsigned int i=0; i < dof_indices_fine.size(); ++i)
        {
            for(unsigned int j=0; j < n_dofs_cell; ++j)
            {
                interpolation_matrix->set(dof_indices_fine[i], dof_indices[j], interpolation_matrix_local(i,j));
            }
        }

    } // cell loop ends

    interpolation_matrix->compress(dealii::VectorOperation::insert);
}

template<int dim, int nstate, typename real>
void DualWeightedResidualObjFunc<dim, nstate, real> :: extract_interpolation_matrices(
    dealii::Table<2, dealii::FullMatrix<real>> &interpolation_hp)
{
    const dealii::hp::FECollection<dim> &fe = this->dg->dof_handler.get_fe_collection();
    interpolation_hp.reinit(fe.size(), fe.size());

    for(unsigned int i=0; i<fe.size(); ++i)
    {
        for(unsigned int j=0; j<fe.size(); ++j)
        {
            if(i != j)
            {
                interpolation_hp(i, j).reinit(fe[i].n_dofs_per_cell(), fe[j].n_dofs_per_cell());
                try
                {
                    fe[i].get_interpolation_matrix(fe[j], interpolation_hp(i,j));
                } 
                // If interpolation matrix cannot be generated, reset matrix size to 0.
                catch (const typename dealii::FiniteElement<dim>::ExcInterpolationNotImplemented &)
                {
                    interpolation_hp(i,j).reinit(0,0);
                }

            }
        }
    }
}

template<int dim, int nstate, typename real>
std::vector<std::vector<dealii::types::global_dof_index>> DualWeightedResidualObjFunc<dim, nstate, real> :: get_cellwise_dof_indices()
{
    unsigned int n_cells_global = this->dg->triangulation->n_global_active_cells();
    std::vector<std::vector<dealii::types::global_dof_index>> cellwise_dof_indices(n_cells_global);
    std::vector<dealii::types::global_dof_index> dof_indices;
    for(const auto &cell : this->dg->dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned()) continue;

        const unsigned int i_fele = cell->active_fe_index();
        const dealii::FESystem<dim,dim> &fe_ref = this->dg->fe_collection[i_fele];
        const unsigned int n_dofs_cell = fe_ref.n_dofs_per_cell();

        dof_indices.resize(n_dofs_cell);
        cell->get_dof_indices (dof_indices);

        const dealii::types::global_dof_index cell_index = cell->active_cell_index();
        cellwise_dof_indices[cell_index] = dof_indices;
    }

    return cellwise_dof_indices;
}

//===================================================================================================================================================
//                          Functions used in evaluate_functional
//===================================================================================================================================================

template<int dim, int nstate, typename real>
real DualWeightedResidualObjFunc<dim, nstate, real> :: evaluate_functional(
    const bool compute_dIdW,
    const bool compute_dIdX,
    const bool compute_d2I)
{
    bool actually_compute_value = true;
    bool actually_compute_dIdW = compute_dIdW;
    bool actually_compute_dIdX = compute_dIdX;
    bool actually_compute_d2I  = compute_d2I;


    if(compute_dIdW || compute_dIdX || compute_d2I)
    {
        actually_compute_dIdW = true;
        actually_compute_dIdX = true;
        actually_compute_d2I  = true; 
    }

    this->need_compute(actually_compute_value, actually_compute_dIdW, actually_compute_dIdX, actually_compute_d2I);
    
    bool compute_derivatives = false;
    if(actually_compute_dIdW || actually_compute_dIdX || actually_compute_d2I) {compute_derivatives = true;}

    if(actually_compute_value)
    {
        this->current_functional_value = evaluate_objective_function(); // also stores adjoint, residual_fine and J_u.
    }

    if(compute_derivatives)
    {
        compute_common_vectors_and_matrices();
        store_dIdX();
        store_dIdW();
    }

    return this->current_functional_value;
}


template<int dim, int nstate, typename real>
real DualWeightedResidualObjFunc<dim, nstate, real> :: evaluate_objective_function()
{
    eta.reinit(this->dg->triangulation->n_active_cells());

    // Evaluate adjoint and residual fine
    this->dg->change_cells_fe_degree_by_deltadegree_and_interpolate_solution(1);
    const bool compute_dRdW = true;
    this->dg->assemble_residual(compute_dRdW);
    
    residual_fine = this->dg->right_hand_side;
    residual_fine.update_ghost_values();
    adjoint.reinit(residual_fine);
    const bool compute_dIdW = true;
    functional->evaluate_functional(compute_dIdW);
    
    solve_linear(this->dg->system_matrix_transpose, functional->dIdw, adjoint, this->dg->all_parameters->linear_solver_param);
    adjoint *= -1.0;
    adjoint.update_ghost_values();
    
    this->dg->change_cells_fe_degree_by_deltadegree_and_interpolate_solution(-1);
    
    for(const auto &cell : this->dg->dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned()) continue;

        const dealii::types::global_dof_index cell_index = cell->active_cell_index();
        
        const std::vector<dealii::types::global_dof_index> &dof_indices_fine = cellwise_dofs_fine[cell_index];
        eta[cell_index] = 0.0;

        for(unsigned int i_dof=0; i_dof < dof_indices_fine.size(); ++i_dof)
        {
            eta[cell_index] += adjoint(dof_indices_fine[i_dof])*residual_fine(dof_indices_fine[i_dof]);
        }

    } // cell loop ends

    real obj_func_local = eta*eta;
    real obj_func_global = dealii::Utilities::MPI::sum(obj_func_local, MPI_COMM_WORLD);
    return obj_func_global;
}

template<int dim, int nstate, typename real>
void DualWeightedResidualObjFunc<dim, nstate, real> :: compute_common_vectors_and_matrices()
{
    this->dg->change_cells_fe_degree_by_deltadegree_and_interpolate_solution(1);
    
    // Store derivatives related to the residual
    bool compute_dRdW = true, compute_dRdX=false, compute_d2R=false;
    this->dg->assemble_residual(compute_dRdW, compute_dRdX, compute_d2R);
    R_u->copy_from(this->dg->system_matrix);
    R_u_transpose->copy_from(this->dg->system_matrix_transpose);
    
    compute_dRdW = false, compute_dRdX = true, compute_d2R = false;
    this->dg->assemble_residual(compute_dRdW, compute_dRdX, compute_d2R);
    R_x->copy_from(this->dg->dRdXv);
    
    this->dg->set_dual(adjoint);
    compute_dRdW = false, compute_dRdX = false, compute_d2R = true;
    this->dg->assemble_residual(compute_dRdW, compute_dRdX, compute_d2R);
    matrix_ux->copy_from(this->dg->d2RdWdX);
    matrix_uu->copy_from(this->dg->d2RdWdW);

    // Store derivatives relate to functional J.
    const bool compute_dIdW = false,  compute_dIdX = false, compute_d2I = true;
    functional->evaluate_functional(compute_dIdW, compute_dIdX, compute_d2I);
    matrix_ux->add(1.0, *functional->d2IdWdX);
    matrix_uu->add(1.0, *functional->d2IdWdW);

    (*matrix_ux) *= -1.0;
    (*matrix_uu) *= -1.0;

    this->dg->change_cells_fe_degree_by_deltadegree_and_interpolate_solution(-1);
}

template<int dim, int nstate, typename real>
void DualWeightedResidualObjFunc<dim, nstate, real> :: store_dIdX()
{ 
    eta_x_Tvmult(this->dIdX, eta);
}

template<int dim, int nstate, typename real>
void DualWeightedResidualObjFunc<dim, nstate, real> :: store_dIdW()
{
    eta_u_Tvmult(this->dIdw, eta);
}

template<int dim, int nstate, typename real>
void DualWeightedResidualObjFunc<dim, nstate, real> :: d2IdWdW_vmult(
    VectorType &out_vector, 
    const VectorType &in_vector) const
{ 
    NormalVector v_interm;
    eta_u_vmult(v_interm, in_vector);
    eta_u_Tvmult(out_vector, v_interm);
}

template<int dim, int nstate, typename real>
void DualWeightedResidualObjFunc<dim, nstate, real> :: d2IdWdX_vmult(
    VectorType &out_vector, 
    const VectorType &in_vector) const
{ 
    NormalVector v_interm;
    eta_x_vmult(v_interm, in_vector);
    eta_u_Tvmult(out_vector, v_interm);
}

template<int dim, int nstate, typename real>
void DualWeightedResidualObjFunc<dim, nstate, real> :: d2IdWdX_Tvmult(
    VectorType &out_vector, 
    const VectorType &in_vector) const
{ 
    NormalVector v_interm;
    eta_u_vmult(v_interm, in_vector);
    eta_x_Tvmult(out_vector, v_interm);
}

template<int dim, int nstate, typename real>
void DualWeightedResidualObjFunc<dim, nstate, real> :: d2IdXdX_vmult(
    VectorType &out_vector, 
    const VectorType &in_vector) const
{ 
    NormalVector v_interm;
    eta_x_vmult(v_interm, in_vector);
    eta_x_Tvmult(out_vector, v_interm);
}

//===================================================================================================================================================
//                          Functions used to evaluate vmults and Tvmults
//===================================================================================================================================================

template<int dim, int nstate, typename real>
void DualWeightedResidualObjFunc<dim, nstate, real> :: eta_psi_vmult(
    NormalVector &out_vector, 
    const VectorType &in_vector) const
{
    out_vector.reinit(this->dg->triangulation->n_active_cells());
    AssertDimension(in_vector.size(), vector_fine.size());
    
    for(const auto &cell : this->dg->dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned()) continue;

        const dealii::types::global_dof_index cell_index = cell->active_cell_index();
        
        const std::vector<dealii::types::global_dof_index> &dof_indices_fine = cellwise_dofs_fine[cell_index];
        out_vector[cell_index] = 0.0;

        for(unsigned int i_dof=0; i_dof < dof_indices_fine.size(); ++i_dof)
        {
            out_vector[cell_index] += residual_fine(dof_indices_fine[i_dof])*in_vector(dof_indices_fine[i_dof]);
        }

    } // cell loop ends

}

template<int dim, int nstate, typename real>
void DualWeightedResidualObjFunc<dim, nstate, real> :: eta_R_vmult(
    NormalVector &out_vector, 
    const VectorType &in_vector) const
{
    AssertDimension(in_vector.size(), vector_fine.size());
    out_vector.reinit(this->dg->triangulation->n_active_cells());
    
    for(const auto &cell : this->dg->dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned()) continue;

        const dealii::types::global_dof_index cell_index = cell->active_cell_index();
        
        const std::vector<dealii::types::global_dof_index> &dof_indices_fine = cellwise_dofs_fine[cell_index];
        out_vector[cell_index] = 0.0;

        for(unsigned int i_dof=0; i_dof < dof_indices_fine.size(); ++i_dof)
        {
            out_vector[cell_index] += adjoint(dof_indices_fine[i_dof])*in_vector(dof_indices_fine[i_dof]);
        }

    } // cell loop ends

}

template<int dim, int nstate, typename real>
void DualWeightedResidualObjFunc<dim, nstate, real> :: eta_psi_Tvmult(
    VectorType &out_vector, 
    const NormalVector &in_vector) const
{
    AssertDimension(in_vector.size(), this->dg->triangulation->n_active_cells());
    out_vector.reinit(vector_fine);
    
    for(const auto &cell : this->dg->dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned()) continue;

        const dealii::types::global_dof_index cell_index = cell->active_cell_index();
        
        const std::vector<dealii::types::global_dof_index> &dof_indices_fine = cellwise_dofs_fine[cell_index];

        for(unsigned int i_dof = 0; i_dof < dof_indices_fine.size(); ++i_dof)
        {
            out_vector(dof_indices_fine[i_dof]) = in_vector[cell_index] * residual_fine(dof_indices_fine[i_dof]);
        }
    } // cell loop ends

    out_vector.update_ghost_values();
}

template<int dim, int nstate, typename real>
void DualWeightedResidualObjFunc<dim, nstate, real> :: eta_R_Tvmult(
    VectorType &out_vector, 
    const NormalVector &in_vector) const
{
    AssertDimension(in_vector.size(), this->dg->triangulation->n_active_cells());
    out_vector.reinit(vector_fine);
    
    for(const auto &cell : this->dg->dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned()) continue;

        const dealii::types::global_dof_index cell_index = cell->active_cell_index();
        
        const std::vector<dealii::types::global_dof_index> &dof_indices_fine = cellwise_dofs_fine[cell_index];

        for(unsigned int i_dof = 0; i_dof < dof_indices_fine.size(); ++i_dof)
        {
            out_vector(dof_indices_fine[i_dof]) = in_vector[cell_index] * adjoint(dof_indices_fine[i_dof]);
        }
    } // cell loop ends

    out_vector.update_ghost_values();
}

template<int dim, int nstate, typename real>
void DualWeightedResidualObjFunc<dim, nstate, real> :: eta_x_vmult(
    NormalVector &out_vector, 
    const VectorType &in_vector) const
{
    AssertDimension(in_vector.size(), this->dg->high_order_grid->volume_nodes.size());
    out_vector.reinit(this->dg->triangulation->n_active_cells());
//========================================================================================
    // Compute v1 = Rx*in_vector.
    VectorType v1;
    v1.reinit(vector_fine);
    R_x->vmult(v1, in_vector);
    
    // Compute v2 = eta_R*v1 = eta_R*Rx*in_vector.
    NormalVector v2;
    eta_R_vmult(v2, v1);
//========================================================================================
//========================================================================================
    VectorType v3;
    v3.reinit(vector_fine);

    matrix_ux->vmult(v3, in_vector);

    VectorType v4;
    v4.reinit(vector_fine);

    solve_linear(*R_u_transpose, v3, v4, this->dg->all_parameters->linear_solver_param);
    v4.update_ghost_values();
    
    //v5 = eta_psi*R_u^{-T} * matrix_ux * in_vector 
    NormalVector v5;
    eta_psi_vmult(v5, v4);
//========================================================================================
    out_vector = v2;
    out_vector += v5;
}


template<int dim, int nstate, typename real>
void DualWeightedResidualObjFunc<dim, nstate, real> :: eta_u_vmult(
    NormalVector &out_vector, 
    const VectorType &in_vector) const
{
    AssertDimension(in_vector.size(), vector_coarse.size());
    out_vector.reinit(this->dg->triangulation->n_active_cells());

    VectorType in_vector_fine;
    in_vector_fine.reinit(vector_fine);
    interpolation_matrix->vmult(in_vector_fine, in_vector);
//========================================================================================
    // Compute v1 = Ru*in_vector.
    VectorType v1;
    v1.reinit(vector_fine);
    R_u->vmult(v1, in_vector_fine);
    
    // Compute v2 = eta_R*v1 = eta_R*Ru*in_vector.
    NormalVector v2;
    eta_R_vmult(v2, v1);
//========================================================================================
//========================================================================================
    VectorType v3;
    v3.reinit(vector_fine);
    // v3 = Muu*I_h*in_vector
    matrix_uu->vmult(v3, in_vector_fine);

    VectorType v4;
    v4.reinit(vector_fine);

    solve_linear(*R_u_transpose, v3, v4, this->dg->all_parameters->linear_solver_param);
    v4.update_ghost_values();
    
    //v5 = eta_psi*R_u^{-T} * matrix_uu * I_h*in_vector 
    NormalVector v5;
    eta_psi_vmult(v5, v4);
//========================================================================================
    out_vector = v2;
    out_vector += v5;
}

template<int dim, int nstate, typename real>
void DualWeightedResidualObjFunc<dim, nstate, real> :: eta_x_Tvmult(
    VectorType &out_vector, 
    const NormalVector &in_vector) const
{
    AssertDimension(in_vector.size(), this->dg->triangulation->n_active_cells());
    out_vector.reinit(this->dg->high_order_grid->volume_nodes);
//========================================================================================
    VectorType v1;
    eta_psi_Tvmult(v1, in_vector);

    VectorType v2;
    v2.reinit(vector_fine);

    solve_linear(*R_u, v1, v2, this->dg->all_parameters->linear_solver_param);
    v2.update_ghost_values();
    
    // v3 = in_vector^T*eta_psi*R_u^{-T}*matrix_ux
    VectorType v3 (this->dg->high_order_grid->volume_nodes);
    matrix_ux->Tvmult(v3, v2);
    v3.update_ghost_values();
//========================================================================================
//========================================================================================

    VectorType v4;
    eta_R_Tvmult(v4, in_vector);
    
    // v5 = in_vector^T * eta_R * R_x
    VectorType v5 (this->dg->high_order_grid->volume_nodes);
    R_x->Tvmult(v5, v4);
    v5.update_ghost_values();
//========================================================================================
    out_vector = v3;
    out_vector += v5;
    out_vector.update_ghost_values();
}

template<int dim, int nstate, typename real>
void DualWeightedResidualObjFunc<dim, nstate, real> :: eta_u_Tvmult(
    VectorType &out_vector, 
    const NormalVector &in_vector) const
{
    AssertDimension(in_vector.size(), this->dg->triangulation->n_active_cells());
    out_vector.reinit(vector_coarse);
//========================================================================================
    VectorType v1;
    eta_psi_Tvmult(v1, in_vector);

    VectorType v2;
    v2.reinit(vector_fine);

    solve_linear(*R_u, v1, v2, this->dg->all_parameters->linear_solver_param);
    v2.update_ghost_values();
    
    // v3 = in_vector^T*eta_psi*R_u^{-T}*matrix_uu
    VectorType v3 (vector_fine);
    matrix_uu->Tvmult(v3, v2);
    v3.update_ghost_values();
//========================================================================================
//========================================================================================
    VectorType v4;
    eta_R_Tvmult(v4, in_vector);
    
    // v5 = in_vector^T * eta_R * R_u
    VectorType v5 (vector_fine);
    R_u->Tvmult(v5, v4);
    v5.update_ghost_values();
//========================================================================================
    VectorType v6 = v5;
    v6 += v3;
    interpolation_matrix->Tvmult(out_vector, v6);
    out_vector.update_ghost_values();
}

template class DualWeightedResidualObjFunc <PHILIP_DIM, 1, double>;
template class DualWeightedResidualObjFunc <PHILIP_DIM, 2, double>;
template class DualWeightedResidualObjFunc <PHILIP_DIM, 3, double>;
template class DualWeightedResidualObjFunc <PHILIP_DIM, 4, double>;
template class DualWeightedResidualObjFunc <PHILIP_DIM, 5, double>;
} // namespace PHiLiP

