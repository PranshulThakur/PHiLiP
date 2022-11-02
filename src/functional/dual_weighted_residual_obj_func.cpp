#include "dual_weighted_residual_obj_func.h"
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>
#include "linear_solver/linear_solver.h"
#include "cell_volume_objective_function.h"

namespace PHiLiP {

template <int dim, int nstate, typename real>
DualWeightedResidualObjFunc<dim, nstate, real> :: DualWeightedResidualObjFunc( 
    std::shared_ptr<DGBase<dim,real>> dg_input,
    const real _functional_exact,
    const bool uses_solution_values,
    const bool uses_solution_gradient,
    const bool _use_coarse_residual)
    : Functional<dim, nstate, real> (dg_input, uses_solution_values, uses_solution_gradient)
    , use_coarse_residual(_use_coarse_residual)
    , functional_exact(_functional_exact)
{
    AssertDimension(this->dg->high_order_grid->max_degree, 1);
    compute_interpolation_matrix(); // also stores cellwise_dofs_fine, vector coarse and vector fine.
    functional = FunctionalFactory<dim,nstate,real>::create_Functional(this->dg->all_parameters->functional_param, this->dg);
    //cell_weight_functional = std::make_unique<CellVolumeObjFunc<dim, nstate, real>> (this->dg);
}

//===================================================================================================================================================
//                          Functions used only once in constructor
//===================================================================================================================================================
template<int dim, int nstate, typename real>
void DualWeightedResidualObjFunc<dim, nstate, real> :: compute_interpolation_matrix()
{ 
    vector_coarse = this->dg->solution; // copies values and parallel layout
    vector_vol_nodes = this->dg->high_order_grid->volume_nodes;
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
    interpolation_matrix.reinit(dofs_range_fine, dofs_range_coarse, dsp, MPI_COMM_WORLD);

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
                interpolation_matrix.set(dof_indices_fine[i], dof_indices[j], interpolation_matrix_local(i,j));
            }
        }

    } // cell loop ends

    interpolation_matrix.compress(dealii::VectorOperation::insert);
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
    unsigned int n_cells_global = this->dg->triangulation->n_global_active_cells(); // CHANHE TO n_ctive_cells();
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
        actually_compute_d2I  = false; // Hessian-vector products are evaluated directly without storing d2Is. 
    }

    this->need_compute(actually_compute_value, actually_compute_dIdW, actually_compute_dIdX, actually_compute_d2I);
    
    bool compute_derivatives = false;
    if(actually_compute_dIdW || actually_compute_dIdX) {compute_derivatives = true;}

    if(actually_compute_value)
    {
        this->current_functional_value = evaluate_objective_function(); // also stores adjoint and residual_used.
        this->pcout<<"Evaluated objective function."<<std::endl;
        AssertDimension(this->dg->solution.size(), vector_coarse.size());
    }

    if(compute_derivatives)
    {
        this->pcout<<"Computing common vectors and matrices."<<std::endl;
        compute_common_vectors_and_matrices();
        AssertDimension(this->dg->solution.size(), vector_coarse.size());
        this->pcout<<"Computed common vectors and matrices."<<std::endl;
        store_dIdX();
        this->pcout<<"Stored dIdX."<<std::endl;
        store_dIdW();
        this->pcout<<"Stored dIdw."<<std::endl;
    }

    return this->current_functional_value;
}


template<int dim, int nstate, typename real>
real DualWeightedResidualObjFunc<dim, nstate, real> :: evaluate_objective_function()
{
    const real functional_val = functional->evaluate_functional();
    functional_error = functional_exact - functional_val;
    real obj_func_global = 0.5 * pow(functional_error, 2);
    //obj_func_global += cell_weight_functional->evaluate_functional();
    return obj_func_global;
}

template<int dim, int nstate, typename real>
void DualWeightedResidualObjFunc<dim, nstate, real> :: compute_common_vectors_and_matrices()
{
    functional->evaluate_functional(true, false, false);
    functional_u = functional->dIdw;
    functional_u.update_ghost_values();
    
    functional->evaluate_functional(false, true, false);
    functional_x = functional->dIdX;
    functional_x.update_ghost_values();

    functional->evaluate_functional(false, false, true);
    functional_uu.copy_from(*functional->d2IdWdW);
    functional_ux.copy_from(*functional->d2IdWdX);
    functional_xx.copy_from(*functional->d2IdXdX);

    // Compress all matrices
    functional_uu.compress(dealii::VectorOperation::add);
    functional_ux.compress(dealii::VectorOperation::add);
    functional_xx.compress(dealii::VectorOperation::add);
}

template<int dim, int nstate, typename real>
void DualWeightedResidualObjFunc<dim, nstate, real> :: store_dIdX()
{ 
    this->dIdX = functional_x;
    this->dIdX *= functional_error;
    this->dIdX *= -1.0;
    this->dIdX.update_ghost_values();
/*
    //Add derivative of mesh weight.
    const bool compute_dIdW = false, compute_dIdX = true, compute_d2I = false;
    cell_weight_functional->evaluate_functional(compute_dIdW, compute_dIdX, compute_d2I);
    this->dIdX += cell_weight_functional->dIdX;
    this->dIdX.update_ghost_values();
*/
}

template<int dim, int nstate, typename real>
void DualWeightedResidualObjFunc<dim, nstate, real> :: store_dIdW()
{
    this->dIdw = functional_u;
    this->dIdw *= functional_error;
    this->dIdw *= -1.0;
    this->dIdw.update_ghost_values();
}

template<int dim, int nstate, typename real>
void DualWeightedResidualObjFunc<dim, nstate, real> :: d2IdWdW_vmult(
    VectorType &out_vector, 
    const VectorType &in_vector) const
{
    AssertDimension(in_vector.size(), vector_coarse.size());
    AssertDimension(out_vector.size(), vector_coarse.size());

//========== Get term 1 ==================================
    VectorType term1(vector_coarse);
    functional_uu.vmult(term1, in_vector);
    term1 *= functional_error;
    term1 *= -1.0;
    term1.update_ghost_values();

//========== Get term 2 ==================================
    const real dot_product = functional_u * in_vector;
    VectorType term2 = functional_u;
    term2 *= dot_product;
    term2.update_ghost_values();

//===========================================
    out_vector = term1;
    out_vector += term2;
    out_vector.update_ghost_values();
}

template<int dim, int nstate, typename real>
void DualWeightedResidualObjFunc<dim, nstate, real> :: d2IdWdX_vmult(
    VectorType &out_vector, 
    const VectorType &in_vector) const
{ 
    AssertDimension(in_vector.size(), this->dg->high_order_grid->volume_nodes.size());
    AssertDimension(out_vector.size(), vector_coarse.size());

//========== Get term 1 ==================================
    VectorType term1(vector_coarse);
    functional_ux.vmult(term1, in_vector);
    term1 *= functional_error;
    term1 *= -1.0;
    term1.update_ghost_values();

//========== Get term 2 ==================================
    const real dot_product = functional_x * in_vector;
    VectorType term2 = functional_u;
    term2 *= dot_product;
    term2.update_ghost_values();

//===========================================
    out_vector = term1;
    out_vector += term2;
    out_vector.update_ghost_values();
}

template<int dim, int nstate, typename real>
void DualWeightedResidualObjFunc<dim, nstate, real> :: d2IdWdX_Tvmult(
    VectorType &out_vector, 
    const VectorType &in_vector) const
{ 
    AssertDimension(in_vector.size(), vector_coarse.size());
    AssertDimension(out_vector.size(), this->dg->high_order_grid->volume_nodes.size());

//========== Get the first term  ===================================
    const real dot_product = in_vector * functional_u;
    VectorType term1 = functional_x;
    term1 *= dot_product;
    term1.update_ghost_values();
//========== Get the second term  ===================================
    VectorType term2(vector_vol_nodes);
    functional_ux.Tvmult(term2, in_vector);
    term2.update_ghost_values();
    term2 *= functional_error;
    term2 *= -1.0;
    term2.update_ghost_values();
//====================================================================
    out_vector = term1;
    out_vector += term2;
    out_vector.update_ghost_values();
}

template<int dim, int nstate, typename real>
void DualWeightedResidualObjFunc<dim, nstate, real> :: d2IdXdX_vmult(
    VectorType &out_vector, 
    const VectorType &in_vector) const
{
    AssertDimension(in_vector.size(), this->dg->high_order_grid->volume_nodes.size());
    AssertDimension(out_vector.size(), this->dg->high_order_grid->volume_nodes.size());

//========== Get term 1 ==================================
    VectorType term1(vector_vol_nodes);
    functional_xx.vmult(term1, in_vector);
    term1 *= functional_error;
    term1 *= -1.0;
    term1.update_ghost_values();

//========== Get term 2 ==================================
    const real dot_product = functional_x * in_vector;
    VectorType term2 = functional_x;
    term2 *= dot_product;
    term2.update_ghost_values();

//===========================================
    out_vector = term1;
    out_vector += term2;
    out_vector.update_ghost_values();

/* 
   // Add (cell_weight)_xx * in_vector
    const bool compute_dIdW = false, compute_dIdX = false, compute_d2I = true;
    cell_weight_functional->evaluate_functional(compute_dIdW, compute_dIdX, compute_d2I);
    VectorType out_vector2(out_vector);
    cell_weight_functional->d2IdXdX_vmult(out_vector2, in_vector);

    out_vector += out_vector2;
    out_vector.update_ghost_values();
*/    
}

template class DualWeightedResidualObjFunc <PHILIP_DIM, 1, double>;
template class DualWeightedResidualObjFunc <PHILIP_DIM, 2, double>;
template class DualWeightedResidualObjFunc <PHILIP_DIM, 3, double>;
template class DualWeightedResidualObjFunc <PHILIP_DIM, 4, double>;
template class DualWeightedResidualObjFunc <PHILIP_DIM, 5, double>;
} // namespace PHiLiP

