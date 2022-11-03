#include "dual_weighted_residual_obj_func.h"
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>
#include "linear_solver/linear_solver.h"
#include "cell_volume_objective_function.h"

namespace PHiLiP {

template <int dim, int nstate, typename real>
DualWeightedResidualObjFunc<dim, nstate, real> :: DualWeightedResidualObjFunc( 
    std::shared_ptr<DGBase<dim,real>> dg_input,
    const bool uses_solution_values,
    const bool uses_solution_gradient,
    const bool _use_coarse_residual)
    : Functional<dim, nstate, real> (dg_input, uses_solution_values, uses_solution_gradient)
    , use_coarse_residual(_use_coarse_residual)
{
    AssertDimension(this->dg->high_order_grid->max_degree, 1);
    compute_interpolation_matrix(); // also stores cellwise_dofs_fine, vector coarse and vector fine.
    functional = FunctionalFactory<dim,nstate,real>::create_Functional(this->dg->all_parameters->functional_param, this->dg);
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
    dwr_error.reinit(this->dg->triangulation->n_active_cells());
    dwr_error = 0;
    // Evaluate adjoint and residual fine
    const VectorType solution_coarse_stored = this->dg->solution;
    this->dg->change_cells_fe_degree_by_deltadegree_and_interpolate_solution(1);
    const bool compute_dRdW = true;
    this->dg->assemble_residual(compute_dRdW);
    
    VectorType residual_fine = this->dg->right_hand_side;
    residual_fine.update_ghost_values();
    adjoint.reinit(vector_fine);
    const bool compute_dIdW = true;
    functional->evaluate_functional(compute_dIdW);

    solve_linear(this->dg->system_matrix_transpose, functional->dIdw, adjoint, this->dg->all_parameters->linear_solver_param);
    adjoint *= -1.0;
    adjoint.update_ghost_values();

    this->dg->change_cells_fe_degree_by_deltadegree_and_interpolate_solution(-1);
    /* Interpolating one poly order up and then down changes solution by ~1.0e-12, which causes functional to be re-evaluated when the solution-node configuration is the same. 
    Resetting of solution to stored coarse solution prevents this issue.     */
    this->dg->solution = solution_coarse_stored; 
    this->dg->solution.update_ghost_values();
    
    residual_used = residual_fine;
    if(use_coarse_residual)
    {
        this->dg->assemble_residual();
        VectorType coarse_residual_interpolated;
        coarse_residual_interpolated.reinit(residual_fine);
        interpolation_matrix.vmult(coarse_residual_interpolated, this->dg->right_hand_side);
        residual_used -= coarse_residual_interpolated;
    }
    residual_used.update_ghost_values();

    for(const auto &cell : this->dg->dof_handler.active_cell_iterators())
    {
        if(! cell->is_locally_owned()) {continue;}

        const dealii::types::global_dof_index cell_index = cell->active_cell_index();

        const std::vector<dealii::types::global_dof_index> &dof_indices_fine = cellwise_dofs_fine[cell_index];

        dwr_error[cell_index] = 0.0;

        for(unsigned int i_dof = 0; i_dof < dof_indices_fine.size(); ++i_dof)
        {
            dwr_error[cell_index] += adjoint(dof_indices_fine[i_dof]) * residual_used(dof_indices_fine[i_dof]);
        }
    } // cell loop ends

    real obj_func_local = dwr_error * dwr_error;
    obj_func_local *= 1.0/2.0;

    real obj_func_global = dealii::Utilities::MPI::sum(obj_func_local, MPI_COMM_WORLD);
    return obj_func_global;
}

template<int dim, int nstate, typename real>
void DualWeightedResidualObjFunc<dim, nstate, real> :: compute_common_vectors_and_matrices()
{
    const VectorType solution_coarse_stored = this->dg->solution;
    this->dg->change_cells_fe_degree_by_deltadegree_and_interpolate_solution(1);
    
    // Store derivatives related to the residual
    bool compute_dRdW = true, compute_dRdX=false, compute_d2R=false;
    this->dg->assemble_residual(compute_dRdW, compute_dRdX, compute_d2R);
    R_u.reinit(this->dg->system_matrix);
    R_u.copy_from(this->dg->system_matrix);
    R_u_transpose.reinit(this->dg->system_matrix_transpose);
    R_u_transpose.copy_from(this->dg->system_matrix_transpose);
    
    compute_dRdW = false, compute_dRdX = true, compute_d2R = false;
    this->dg->assemble_residual(compute_dRdW, compute_dRdX, compute_d2R);
    R_x.reinit(this->dg->dRdXv);
    R_x.copy_from(this->dg->dRdXv);
 
    AssertDimension(adjoint.size(), vector_fine.size());
    AssertDimension(adjoint.size(), this->dg->solution.size());
    this->dg->set_dual(adjoint);
    compute_dRdW = false, compute_dRdX = false, compute_d2R = true;
    this->dg->assemble_residual(compute_dRdW, compute_dRdX, compute_d2R);
    matrix_ux.reinit(this->dg->d2RdWdX);
    matrix_ux.copy_from(this->dg->d2RdWdX);
    matrix_uu.reinit(this->dg->d2RdWdW);
    matrix_uu.copy_from(this->dg->d2RdWdW);

    // Store derivatives relate to functional J.
    const bool compute_dIdW = false,  compute_dIdX = false, compute_d2I = true;
    functional->evaluate_functional(compute_dIdW, compute_dIdX, compute_d2I);
    matrix_ux.add(1.0, *functional->d2IdWdX);
    matrix_uu.add(1.0, *functional->d2IdWdW);

    matrix_ux *= -1.0;
    matrix_uu *= -1.0;

    // Store dual_dwr_R times d2R
    VectorType dwr_dwr_R(vector_fine);
    dwr_residual_Tvmult(dwr_dwr_R, dwr_error);
    this->dg->set_dual(dwr_dwr_R);
    compute_dRdW = false, compute_dRdX = false, compute_d2R = true;
    this->dg->assemble_residual(compute_dRdW, compute_dRdX, compute_d2R);
    dwr_dwr_R_times_Rxx.copy_from(this->dg->d2RdXdX);
    dwr_dwr_R_times_Rux.copy_from(this->dg->d2RdWdX);
    dwr_dwr_R_times_Ruu.copy_from(this->dg->d2RdWdW);


    this->dg->change_cells_fe_degree_by_deltadegree_and_interpolate_solution(-1);
    
    /* Interpolating one poly order up and then down changes solution by ~1.0e-12, which causes functional to be re-evaluated when the solution-node configuration is the same. 
    Resetting of solution to stored coarse solution prevents this issue.     */
    this->dg->solution = solution_coarse_stored; 
    this->dg->solution.update_ghost_values();
    // Compute r_u and r_x
    if(use_coarse_residual)
    {
        compute_dRdW = true; compute_dRdX = false;
        this->dg->assemble_residual(compute_dRdW, compute_dRdX);
        r_u.copy_from(this->dg->system_matrix);
        
        compute_dRdW = false; compute_dRdX = true;
        this->dg->assemble_residual(compute_dRdW, compute_dRdX);
        r_x.copy_from(this->dg->dRdXv);
    }

    // Compress all matrices
    R_u.compress(dealii::VectorOperation::add);
    R_u_transpose.compress(dealii::VectorOperation::add);
    R_x.compress(dealii::VectorOperation::add);
    matrix_ux.compress(dealii::VectorOperation::add);
    matrix_uu.compress(dealii::VectorOperation::add);
    dwr_dwr_R_times_Rxx.compress(dealii::VectorOperation::add);
    dwr_dwr_R_times_Rux.compress(dealii::VectorOperation::add);
    dwr_dwr_R_times_Ruu.compress(dealii::VectorOperation::add);
    if(use_coarse_residual)
    {
        r_u.compress(dealii::VectorOperation::add);
        r_x.compress(dealii::VectorOperation::add);
    }
}

template<int dim, int nstate, typename real>
void DualWeightedResidualObjFunc<dim, nstate, real> :: store_dIdX()
{ 
    this->dIdX.reinit(vector_vol_nodes);
    dwr_x_Tvmult(this->dIdX, dwr_error);
}

template<int dim, int nstate, typename real>
void DualWeightedResidualObjFunc<dim, nstate, real> :: store_dIdW()
{
    this->dIdw.reinit(vector_coarse);
    dwr_u_Tvmult(this->dIdw, dwr_error);
}

template<int dim, int nstate, typename real>
void DualWeightedResidualObjFunc<dim, nstate, real> :: d2IdWdW_vmult(
    VectorType &out_vector, 
    const VectorType &in_vector) const
{
    AssertDimension(in_vector.size(), vector_coarse.size());
    AssertDimension(out_vector.size(), vector_coarse.size());

    //===== Compute term 1 ==================
    VectorType term1(vector_coarse);
    dwr_times_dwr_uu_vmult(term1, in_vector);

    //===== Compute term 2 ==================
    NormalVector v1(this->dg->triangulation->n_active_cells());
    dwr_u_vmult(v1, in_vector);
    VectorType term2(vector_coarse);
    dwr_u_Tvmult(term2, v1);
    //===================================

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
    
    //===== Compute term 1 ==================
    VectorType term1(vector_coarse);
    dwr_times_dwr_ux_vmult(term1, in_vector);

    //===== Compute term 2 ==================
    NormalVector v1(this->dg->triangulation->n_active_cells());
    dwr_x_vmult(v1, in_vector);
    VectorType term2(vector_coarse);
    dwr_u_Tvmult(term2, v1);
    //===================================
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

    //======== Compute term 1 =================================================
    NormalVector v1(this->dg->triangulation->n_active_cells());
    dwr_u_vmult(v1, in_vector);
    VectorType term1(vector_vol_nodes);
    dwr_x_Tvmult(term1, v1);

    //======== Compute term 2 =================================================
    VectorType term2(vector_vol_nodes);
    dwr_times_dwr_ux_Tvmult(term2, in_vector);
    //=======================================================================

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
    
    //===== Compute term 1 ==================
    VectorType term1(vector_vol_nodes);
    dwr_times_dwr_xx_vmult(term1, in_vector);

    //===== Compute term 2 ==================
    NormalVector v1(this->dg->triangulation->n_active_cells());
    dwr_x_vmult(v1, in_vector);
    VectorType term2(vector_vol_nodes);
    dwr_x_Tvmult(term2, v1);
    //===================================
    out_vector = term1;
    out_vector += term2;
    out_vector.update_ghost_values();
}

//===================================================================================================================================================
//                          Functions used to evaluate vmults and Tvmults
//===================================================================================================================================================

//===================================================================================================================================================
//                          vmults and Tvmults of \eta_{\psi} and \eta_{R}
//===================================================================================================================================================

template<int dim, int nstate, typename real>
void DualWeightedResidualObjFunc<dim, nstate, real> :: dwr_adjoint_vmult(
    NormalVector &out_vector, 
    const VectorType &in_vector) const
{
    AssertDimension(out_vector.size(), this->dg->triangulation->n_active_cells());
    AssertDimension(in_vector.size(), vector_fine.size());

    for(const auto &cell : this->dg->dof_handler.active_cell_iterators())
    {
        if(! cell->is_locally_owned()) {continue;}

        const dealii::types::global_dof_index cell_index = cell->active_cell_index();
        const std::vector<dealii::types::global_dof_index> &dof_indices_fine = cellwise_dofs_fine[cell_index];

        out_vector[cell_index] = 0.0;

        for(unsigned int i_dof=0; i_dof < dof_indices_fine.size(); ++i_dof)
        {
            out_vector[cell_index] += residual_used(dof_indices_fine[i_dof])*in_vector(dof_indices_fine[i_dof]);
        }
    } // cell loop ends
}

template<int dim, int nstate, typename real>
void DualWeightedResidualObjFunc<dim, nstate, real> :: dwr_residual_vmult(
    NormalVector &out_vector, 
    const VectorType &in_vector) const
{
    AssertDimension(out_vector.size(), this->dg->triangulation->n_active_cells());
    AssertDimension(in_vector.size(), vector_fine.size());
    
    for(const auto &cell : this->dg->dof_handler.active_cell_iterators())
    {
        if(! cell->is_locally_owned()) {continue;}

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
void DualWeightedResidualObjFunc<dim, nstate, real> :: dwr_adjoint_Tvmult(
    VectorType &out_vector, 
    const NormalVector &in_vector) const
{
    AssertDimension(out_vector.size(), vector_fine.size());
    AssertDimension(in_vector.size(), this->dg->triangulation->n_active_cells());
    
    for(const auto &cell : this->dg->dof_handler.active_cell_iterators())
    {
        if(! cell->is_locally_owned()) {continue;}

        const dealii::types::global_dof_index cell_index = cell->active_cell_index();
        const std::vector<dealii::types::global_dof_index> &dof_indices_fine = cellwise_dofs_fine[cell_index];

        for(unsigned int i_dof=0; i_dof < dof_indices_fine.size(); ++i_dof)
        {
            out_vector(dof_indices_fine[i_dof]) = in_vector[cell_index] * residual_used(dof_indices_fine[i_dof]);
        }
    } // cell loop ends

    out_vector.update_ghost_values();
}

template<int dim, int nstate, typename real>
void DualWeightedResidualObjFunc<dim, nstate, real> :: dwr_residual_Tvmult(
    VectorType &out_vector, 
    const NormalVector &in_vector) const
{
    AssertDimension(out_vector.size(), vector_fine.size());
    AssertDimension(in_vector.size(), this->dg->triangulation->n_active_cells());
    
    for(const auto &cell : this->dg->dof_handler.active_cell_iterators())
    {
        if(! cell->is_locally_owned()) {continue;}

        const dealii::types::global_dof_index cell_index = cell->active_cell_index();
        const std::vector<dealii::types::global_dof_index> &dof_indices_fine = cellwise_dofs_fine[cell_index];

        for(unsigned int i_dof=0; i_dof < dof_indices_fine.size(); ++i_dof)
        {
            out_vector(dof_indices_fine[i_dof]) = in_vector[cell_index] * adjoint(dof_indices_fine[i_dof]);
        }
    } // cell loop ends

    out_vector.update_ghost_values();
}

//===================================================================================================================================================
//                          vmults and Tvmults of \eta_x and \eta_u
//===================================================================================================================================================
template<int dim, int nstate, typename real>
void DualWeightedResidualObjFunc<dim, nstate, real> :: dwr_x_vmult(
    NormalVector &out_vector,
    const VectorType &in_vector) const
{
    AssertDimension(out_vector.size(), this->dg->triangulation->n_active_cells());
    AssertDimension(in_vector.size(), this->dg->high_order_grid->volume_nodes.size());

    //======= Get the first term ===================================
    VectorType v1(vector_fine);
    R_x.vmult(v1, in_vector);
    v1.update_ghost_values();
    NormalVector term1 (this->dg->triangulation->n_active_cells());
    dwr_residual_vmult(term1, v1);

    //======= Get the second term =================================
    VectorType v2(vector_fine);
    adjoint_x_vmult(v2, in_vector);
    NormalVector term2 (this->dg->triangulation->n_active_cells());
    dwr_adjoint_vmult(term2, v2);
    //==================================================================

    out_vector = term1;
    out_vector += term2;
}

template<int dim, int nstate, typename real>
void DualWeightedResidualObjFunc<dim, nstate, real> :: dwr_u_vmult(
    NormalVector &out_vector,
    const VectorType &in_vector) const
{
    AssertDimension(out_vector.size(), this->dg->triangulation->n_active_cells());
    AssertDimension(in_vector.size(), vector_coarse.size());

    VectorType in_vector_fine(vector_fine);
    interpolation_matrix.vmult(in_vector_fine, in_vector);
    in_vector_fine.update_ghost_values();
    
    //======= Get the first term ===================================
    VectorType v1(vector_fine);
    R_u.vmult(v1, in_vector_fine);
    v1.update_ghost_values();
    NormalVector term1 (this->dg->triangulation->n_active_cells());
    dwr_residual_vmult(term1, v1);

    //======= Get the second term =================================
    VectorType v2(vector_fine);
    adjoint_u_vmult(v2, in_vector_fine);
    NormalVector term2(this->dg->triangulation->n_active_cells());
    dwr_adjoint_vmult(term2, v2);
    //==================================================================

    out_vector = term1;
    out_vector += term2;
}

template<int dim, int nstate, typename real>
void DualWeightedResidualObjFunc<dim, nstate, real> :: dwr_x_Tvmult(
    VectorType &out_vector,
    const NormalVector &in_vector) const
{
    AssertDimension(out_vector.size(), this->dg->high_order_grid->volume_nodes.size());
    AssertDimension(in_vector.size(), this->dg->triangulation->n_active_cells());

    //======= Get the first term =====================================
    VectorType v1(vector_fine);
    dwr_adjoint_Tvmult(v1, in_vector);
    VectorType term1(vector_vol_nodes);
    adjoint_x_Tvmult(term1, v1);

    //====== Get the second term =====================================
    VectorType v2(vector_fine);
    dwr_residual_Tvmult(v2, in_vector);
    VectorType term2(vector_vol_nodes);
    R_x.Tvmult(term2, v2);
    term2.update_ghost_values();
    //===============================================================

    out_vector = term1;
    out_vector += term2;
}

template<int dim, int nstate, typename real>
void DualWeightedResidualObjFunc<dim, nstate, real> :: dwr_u_Tvmult(
    VectorType &out_vector,
    const NormalVector &in_vector) const
{
    AssertDimension(out_vector.size(), vector_coarse.size());
    AssertDimension(in_vector.size(), this->dg->triangulation->n_active_cells());

    //======= Get the first term =====================================
    VectorType v1(vector_fine);
    dwr_adjoint_Tvmult(v1, in_vector);
    VectorType term1(vector_fine);
    adjoint_u_Tvmult(term1, v1);

    //====== Get the second term =====================================
    VectorType v2(vector_fine);
    dwr_residual_Tvmult(v2, in_vector);
    VectorType term2(vector_fine);
    R_u.Tvmult(term2, v2);
    term2.update_ghost_values();
    //===============================================================

    VectorType out_vector_fine = term1;
    out_vector_fine += term2;
    out_vector_fine.update_ghost_values();

    interpolation_matrix.Tvmult(out_vector, out_vector_fine);
    out_vector.update_ghost_values();
}

//===================================================================================================================================================
//                          vmults and Tvmults required to compute second derivatives
//===================================================================================================================================================
template<int dim, int nstate, typename real>
void DualWeightedResidualObjFunc<dim, nstate, real> :: dwr_diagonal_vmult(
    VectorType &out_vector, 
    const VectorType &in_vector) const
{
    AssertDimension(in_vector.size(), vector_fine.size());
    AssertDimension(out_vector.size(), vector_fine.size());

    for(const auto &cell : this->dg->dof_handler.active_cell_iterators())
    {
        if(! cell->is_locally_owned()) {continue;}

        const dealii::types::global_dof_index cell_index = cell->active_cell_index();
        const std::vector<dealii::types::global_dof_index> &dof_indices_fine = cellwise_dofs_fine[cell_index];

        for(unsigned int i_dof=0; i_dof < dof_indices_fine.size(); ++i_dof)
        {
            out_vector(dof_indices_fine[i_dof]) = dwr_error[cell_index] * in_vector(dof_indices_fine[i_dof]);
        }
    } // cell loop ends

    out_vector.update_ghost_values();
}

template<int dim, int nstate, typename real>
void DualWeightedResidualObjFunc<dim, nstate, real> :: dwr_times_dwr_adjoint_x_vmult(
    VectorType &out_vector, 
    const VectorType &in_vector) const
{
    AssertDimension(in_vector.size(), vector_vol_nodes.size());
    AssertDimension(out_vector.size(), vector_fine.size());

    VectorType v1(vector_fine);
    R_x.vmult(v1, in_vector);
    v1.update_ghost_values();

    dwr_diagonal_vmult(out_vector, v1);
}

template<int dim, int nstate, typename real>
void DualWeightedResidualObjFunc<dim, nstate, real> :: dwr_times_dwr_residual_x_vmult(
    VectorType &out_vector, 
    const VectorType &in_vector) const
{
    AssertDimension(in_vector.size(), vector_vol_nodes.size());
    AssertDimension(out_vector.size(), vector_fine.size());

    VectorType v1(vector_fine);
    adjoint_x_vmult(v1, in_vector);

    dwr_diagonal_vmult(out_vector, v1);
}

template<int dim, int nstate, typename real>
void DualWeightedResidualObjFunc<dim, nstate, real> :: dwr_times_dwr_adjoint_u_vmult(
    VectorType &out_vector, 
    const VectorType &in_vector) const
{
    AssertDimension(in_vector.size(), vector_fine.size());
    AssertDimension(out_vector.size(), vector_fine.size());

    VectorType v1(vector_fine);
    R_u.vmult(v1, in_vector);
    v1.update_ghost_values();

    dwr_diagonal_vmult(out_vector, v1);
}

template<int dim, int nstate, typename real>
void DualWeightedResidualObjFunc<dim, nstate, real> :: dwr_times_dwr_residual_u_vmult(
    VectorType &out_vector, 
    const VectorType &in_vector) const
{
    AssertDimension(in_vector.size(), vector_fine.size());
    AssertDimension(out_vector.size(), vector_fine.size());

    VectorType v1(vector_fine);
    adjoint_u_vmult(v1, in_vector);

    dwr_diagonal_vmult(out_vector, v1);
}

template<int dim, int nstate, typename real>
void DualWeightedResidualObjFunc<dim, nstate, real> :: dwr_times_dwr_adjoint_x_Tvmult(
    VectorType &out_vector, 
    const VectorType &in_vector) const
{
    AssertDimension(in_vector.size(), vector_fine.size());
    AssertDimension(out_vector.size(), vector_vol_nodes.size());

    VectorType v1(vector_fine);
    dwr_diagonal_vmult(v1, in_vector);

    R_x.Tvmult(out_vector, v1);
    out_vector.update_ghost_values();
}

template<int dim, int nstate, typename real>
void DualWeightedResidualObjFunc<dim, nstate, real> :: dwr_times_dwr_residual_x_Tvmult(
    VectorType &out_vector, 
    const VectorType &in_vector) const
{
    AssertDimension(in_vector.size(), vector_fine.size());
    AssertDimension(out_vector.size(), vector_vol_nodes.size());

    VectorType v1(vector_fine);
    dwr_diagonal_vmult(v1, in_vector);

    adjoint_x_Tvmult(out_vector, v1);
    out_vector.update_ghost_values();
}

template<int dim, int nstate, typename real>
void DualWeightedResidualObjFunc<dim, nstate, real> :: dwr_times_dwr_adjoint_u_Tvmult(
    VectorType &out_vector, 
    const VectorType &in_vector) const
{
    AssertDimension(in_vector.size(), vector_fine.size());
    AssertDimension(out_vector.size(), vector_fine.size());

    VectorType v1(vector_fine);
    dwr_diagonal_vmult(v1, in_vector);

    R_u.Tvmult(out_vector, v1);
    out_vector.update_ghost_values();
}

template<int dim, int nstate, typename real>
void DualWeightedResidualObjFunc<dim, nstate, real> :: dwr_times_dwr_residual_u_Tvmult(
    VectorType &out_vector, 
    const VectorType &in_vector) const
{
    AssertDimension(in_vector.size(), vector_fine.size());
    AssertDimension(out_vector.size(), vector_fine.size());

    VectorType v1(vector_fine);
    dwr_diagonal_vmult(v1, in_vector);

    adjoint_u_Tvmult(out_vector, v1);
    out_vector.update_ghost_values();
}
//===================================================================================================================================================
//                          vmults and Tvmults of \eta^T \eta_{xx, ux, uu}
//===================================================================================================================================================
template<int dim, int nstate, typename real>
void DualWeightedResidualObjFunc<dim, nstate, real> :: dwr_times_dwr_xx_vmult(
    VectorType &out_vector, 
    const VectorType &in_vector) const
{
    AssertDimension(in_vector.size(), vector_vol_nodes.size());
    AssertDimension(out_vector.size(), vector_vol_nodes.size());

    //=== Compute term 1 =======================
    VectorType term1(vector_vol_nodes);
    dwr_dwr_R_times_Rxx.vmult(term1, in_vector);
    term1.update_ghost_values();

    //=== Compute term 2 =======================
    VectorType v1(vector_fine);
    dwr_times_dwr_residual_x_vmult(v1, in_vector);
    VectorType term2(vector_vol_nodes);
    R_x.Tvmult(term2, v1);
    term2.update_ghost_values();

    //=== Compute term 3 =======================
    VectorType v2(vector_fine);
    dwr_times_dwr_adjoint_x_vmult(v2, in_vector);
    VectorType term3(vector_vol_nodes);
    adjoint_x_Tvmult(term3, v2);
    //=============================================================
    out_vector = term1;
    out_vector += term2;
    out_vector += term3;
    out_vector.update_ghost_values();
}

template<int dim, int nstate, typename real>
void DualWeightedResidualObjFunc<dim, nstate, real> :: dwr_times_dwr_ux_vmult(
    VectorType &out_vector, 
    const VectorType &in_vector) const
{
    AssertDimension(in_vector.size(), vector_vol_nodes.size());
    AssertDimension(out_vector.size(), vector_coarse.size());
    
    //=== Compute term 1 =======================
    VectorType term1(vector_fine);
    dwr_dwr_R_times_Rux.vmult(term1, in_vector);
    term1.update_ghost_values();

    //=== Compute term 2 =======================
    VectorType v1(vector_fine);
    dwr_times_dwr_residual_x_vmult(v1, in_vector);
    VectorType term2(vector_fine);
    R_u.Tvmult(term2, v1);
    term2.update_ghost_values();

    //=== Compute term 3 =======================
    VectorType v2(vector_fine);
    dwr_times_dwr_adjoint_x_vmult(v2, in_vector);
    VectorType term3(vector_fine);
    adjoint_u_Tvmult(term3, v2);
    //=============================================================
    VectorType out_vector_fine = term1;
    out_vector_fine += term2;
    out_vector_fine += term3;
    out_vector_fine.update_ghost_values();

    interpolation_matrix.Tvmult(out_vector, out_vector_fine);
    out_vector.update_ghost_values();
}

template<int dim, int nstate, typename real>
void DualWeightedResidualObjFunc<dim, nstate, real> :: dwr_times_dwr_ux_Tvmult(
    VectorType &out_vector, 
    const VectorType &in_vector) const
{
    AssertDimension(in_vector.size(), vector_coarse.size());
    AssertDimension(out_vector.size(), vector_vol_nodes.size());

    VectorType in_vector_fine(vector_fine);
    interpolation_matrix.vmult(in_vector_fine, in_vector);
    in_vector_fine.update_ghost_values();
    
    //===== Compute term 1 =======================================
    VectorType v1(vector_fine);
    adjoint_u_vmult(v1, in_vector_fine);
    VectorType term1(vector_vol_nodes);
    dwr_times_dwr_adjoint_x_Tvmult(term1, v1);

    //===== Compute term 2 =======================================
    VectorType v2(vector_fine);
    R_u.vmult(v2, in_vector_fine);
    v2.update_ghost_values();
    VectorType term2(vector_vol_nodes);
    dwr_times_dwr_residual_x_Tvmult(term2, v2);

    //===== Compute term 3 =======================================
    VectorType term3(vector_vol_nodes);
    dwr_dwr_R_times_Rux.Tvmult(term3, in_vector_fine);
    term3.update_ghost_values();
    //============================================================

    out_vector = term1;
    out_vector += term2;
    out_vector += term3;
    out_vector.update_ghost_values();
}

template<int dim, int nstate, typename real>
void DualWeightedResidualObjFunc<dim, nstate, real> :: dwr_times_dwr_uu_vmult(
    VectorType &out_vector, 
    const VectorType &in_vector) const
{
    AssertDimension(in_vector.size(), vector_coarse.size());
    AssertDimension(out_vector.size(), vector_coarse.size());

    VectorType in_vector_fine(vector_fine);
    interpolation_matrix.vmult(in_vector_fine, in_vector);
    in_vector_fine.update_ghost_values();
    
    //=== Compute term 1 =======================
    VectorType term1(vector_fine);
    dwr_dwr_R_times_Ruu.vmult(term1, in_vector_fine);
    term1.update_ghost_values();

    //=== Compute term 2 =======================
    VectorType v1(vector_fine);
    dwr_times_dwr_residual_u_vmult(v1, in_vector_fine);
    VectorType term2(vector_fine);
    R_u.Tvmult(term2, v1);
    term2.update_ghost_values();

    //=== Compute term 3 =======================
    VectorType v2(vector_fine);
    dwr_times_dwr_adjoint_u_vmult(v2, in_vector_fine);
    VectorType term3(vector_fine);
    adjoint_u_Tvmult(term3, v2);
    //=============================================================
    VectorType out_vector_fine = term1;
    out_vector_fine += term2;
    out_vector_fine += term3;
    out_vector_fine.update_ghost_values();

    interpolation_matrix.Tvmult(out_vector, out_vector_fine);
    out_vector.update_ghost_values();
}


//===================================================================================================
//                          Vmults and Tvmults of adjoint
//===================================================================================================
template<int dim, int nstate, typename real>
void DualWeightedResidualObjFunc<dim, nstate, real> :: adjoint_x_vmult(
    VectorType &out_vector, 
    const VectorType &in_vector) const
{
    AssertDimension(in_vector.size(), this->dg->high_order_grid->volume_nodes.size());
    AssertDimension(out_vector.size(), vector_fine.size());
//========================================================================================
    VectorType v1;
    v1.reinit(vector_fine);
    matrix_ux.vmult(v1, in_vector);
    v1.update_ghost_values();

    solve_linear(R_u_transpose, v1, out_vector, this->dg->all_parameters->linear_solver_param);
    out_vector.update_ghost_values();
}


template<int dim, int nstate, typename real>
void DualWeightedResidualObjFunc<dim, nstate, real> :: adjoint_u_vmult(
    VectorType &out_vector, 
    const VectorType &in_vector) const
{
    AssertDimension(in_vector.size(), vector_fine.size());
    AssertDimension(out_vector.size(), vector_fine.size());
    //===============================================================================
    VectorType v1;
    v1.reinit(vector_fine);
    matrix_uu.vmult(v1, in_vector);
    v1.update_ghost_values();

    solve_linear(R_u_transpose, v1, out_vector, this->dg->all_parameters->linear_solver_param);
    out_vector.update_ghost_values();
}

template<int dim, int nstate, typename real>
void DualWeightedResidualObjFunc<dim, nstate, real> :: adjoint_x_Tvmult(
    VectorType &out_vector, 
    const VectorType &in_vector) const
{
    AssertDimension(in_vector.size(), vector_fine.size());
    AssertDimension(out_vector.size(), this->dg->high_order_grid->volume_nodes.size());
    //===========================================================================================
    VectorType v1;
    v1.reinit(vector_fine);
    VectorType in_vector_copy = in_vector; // because solve_linear() does not take it as a const.
    solve_linear(R_u, in_vector_copy, v1, this->dg->all_parameters->linear_solver_param);
    v1.update_ghost_values();

    matrix_ux.Tvmult(out_vector, v1);
    out_vector.update_ghost_values();
}

template<int dim, int nstate, typename real>
void DualWeightedResidualObjFunc<dim, nstate, real> :: adjoint_u_Tvmult(
    VectorType &out_vector, 
    const VectorType &in_vector) const
{
    AssertDimension(in_vector.size(), vector_fine.size());
    AssertDimension(out_vector.size(), vector_fine.size());
    //===========================================================================================
    VectorType v1;
    v1.reinit(vector_fine);
    VectorType in_vector_copy = in_vector;
    solve_linear(R_u, in_vector_copy, v1, this->dg->all_parameters->linear_solver_param);
    v1.update_ghost_values();

    matrix_uu.Tvmult(out_vector, v1);
    out_vector.update_ghost_values();
}

template class DualWeightedResidualObjFunc <PHILIP_DIM, 1, double>;
template class DualWeightedResidualObjFunc <PHILIP_DIM, 2, double>;
template class DualWeightedResidualObjFunc <PHILIP_DIM, 3, double>;
template class DualWeightedResidualObjFunc <PHILIP_DIM, 4, double>;
template class DualWeightedResidualObjFunc <PHILIP_DIM, 5, double>;
} // namespace PHiLiP

