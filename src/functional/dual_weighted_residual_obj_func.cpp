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
   // cell_weight_functional = std::make_unique<CellVolumeObjFunc<dim, nstate, real>> (this->dg);
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
    // Evaluate adjoint and residual fine
    VectorType solution_coarse_stored = this->dg->solution;
    this->dg->change_cells_fe_degree_by_deltadegree_and_interpolate_solution(1);
    const bool compute_dRdW = true;
    this->dg->assemble_residual(compute_dRdW);
    
    VectorType residual_fine = this->dg->right_hand_side;
    residual_fine.update_ghost_values();
    adjoint.reinit(residual_fine);
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
    
    dwr_error = adjoint*residual_used; // dealii takes care of summing over all processors.

    real obj_func_global = 1.0/2.0 * dwr_error * dwr_error;
   // obj_func_global += cell_weight_functional->evaluate_functional();
    return obj_func_global;
}

template<int dim, int nstate, typename real>
void DualWeightedResidualObjFunc<dim, nstate, real> :: compute_common_vectors_and_matrices()
{
    VectorType solution_coarse_stored = this->dg->solution;
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
    adjoint_times_R_uu.reinit(this->dg->d2RdWdW);
    adjoint_times_R_uu.copy_from(this->dg->d2RdWdW);
    adjoint_times_R_ux.reinit(this->dg->d2RdWdX);
    adjoint_times_R_ux.copy_from(this->dg->d2RdWdX);
    adjoint_times_R_xx.reinit(this->dg->d2RdXdX);
    adjoint_times_R_xx.copy_from(this->dg->d2RdXdX);
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
    adjoint_times_R_uu.compress(dealii::VectorOperation::add);
    adjoint_times_R_ux.compress(dealii::VectorOperation::add);
    adjoint_times_R_xx.compress(dealii::VectorOperation::add);
    if(use_coarse_residual)
    {
        r_u.compress(dealii::VectorOperation::add);
        r_x.compress(dealii::VectorOperation::add);
    }

    common_vector.reinit(vector_fine);
    solve_linear(R_u, residual_used, common_vector, this->dg->all_parameters->linear_solver_param);
    common_vector.update_ghost_values();
}

template<int dim, int nstate, typename real>
void DualWeightedResidualObjFunc<dim, nstate, real> :: store_dIdX()
{ 
    dwr_error_x.reinit(this->dg->high_order_grid->volume_nodes); 
    matrix_ux.Tvmult(dwr_error_x, common_vector);
    dwr_error_x.update_ghost_values();

    R_x.Tvmult_add(dwr_error_x, adjoint);
    dwr_error_x.update_ghost_values();
  
    this->dIdX = dwr_error_x;
    this->dIdX *= dwr_error;
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
    VectorType dwr_error_u_fine(vector_fine);
    matrix_uu.Tvmult(dwr_error_u_fine, common_vector);
    dwr_error_u_fine.update_ghost_values();

    R_u.Tvmult_add(dwr_error_u_fine, adjoint);
    dwr_error_u_fine.update_ghost_values();

    dwr_error_u.reinit(vector_coarse);
    interpolation_matrix.Tvmult(dwr_error_u, dwr_error_u_fine);
    dwr_error_u.update_ghost_values();

    this->dIdw = dwr_error_u;
    this->dIdw *= dwr_error;
    this->dIdw.update_ghost_values();
}

template<int dim, int nstate, typename real>
void DualWeightedResidualObjFunc<dim, nstate, real> :: d2IdWdW_vmult(
    VectorType &out_vector, 
    const VectorType &in_vector) const
{
    AssertDimension(in_vector.size(), vector_coarse.size());
    AssertDimension(out_vector.size(), vector_coarse.size());
    //========= Get the first term=========================================
    real dot_product = dwr_error_u*in_vector;
    VectorType term1 = dwr_error_u;
    term1 *= dot_product;
    term1.update_ghost_values();
    //========= Get the second term=========================================
    VectorType v1 = in_vector;
    v1 *= dwr_error;
    v1.update_ghost_values();
    VectorType term2(vector_coarse);
    dwr_uu_vmult(term2, v1);

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

    // ==== Get the first term ==========================================================
    real dot_product = dwr_error_x * in_vector;
    VectorType term1 = dwr_error_u;
    term1 *= dot_product;
    term1.update_ghost_values();
    // ==== Get the second term ==========================================================
    VectorType v1 = in_vector;
    v1 *= dwr_error;
    v1.update_ghost_values();
    VectorType term2(vector_coarse);
    dwr_ux_vmult(term2, v1);

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

    //===== Get the first term =============================================================
    real dot_product = in_vector * dwr_error_u;
    VectorType term1 = dwr_error_x;
    term1 *= dot_product;
    term1.update_ghost_values();
    //===== Get the second term =============================================================
    VectorType v1 = in_vector;
    v1 *= dwr_error;
    v1.update_ghost_values();
    VectorType term2(this->dg->high_order_grid->volume_nodes);
    dwr_ux_Tvmult(term2, v1);

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
    // ========== Get the first term ===================================================================
    real dot_product = dwr_error_x * in_vector;
    VectorType term1 = dwr_error_x;
    term1 *= dot_product;
    term1.update_ghost_values();
    // ========== Get the second term ===================================================================
    VectorType v1 = in_vector;
    v1 *= dwr_error;
    v1.update_ghost_values();
    VectorType term2(this->dg->high_order_grid->volume_nodes);
    dwr_xx_vmult(term2, v1);

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

//===================================================================================================================================================
//                          Functions used to evaluate vmults and Tvmults
//===================================================================================================================================================
template<int dim, int nstate, typename real>
void DualWeightedResidualObjFunc<dim, nstate, real> :: dwr_uu_vmult(
    VectorType &out_vector, 
    const VectorType &in_vector) const
{
    AssertDimension(in_vector.size(), vector_coarse.size());
    AssertDimension(out_vector.size(), vector_coarse.size());

    VectorType in_vector_fine(vector_fine);
    interpolation_matrix.vmult(in_vector_fine, in_vector);
    in_vector_fine.update_ghost_values();

    VectorType term1(vector_fine);
    adjoint_times_R_uu.vmult(term1, in_vector_fine);
    term1.update_ghost_values();

    VectorType v1(vector_fine);
    adjoint_u_vmult(v1, in_vector_fine);
    VectorType term2(vector_fine);
    R_u.Tvmult(term2, v1);
    term2.update_ghost_values();

    VectorType v2(vector_fine);
    R_u.vmult(v2, in_vector_fine);
    v2.update_ghost_values();
    VectorType term3(vector_fine);
    adjoint_u_Tvmult(term3, v2);

    VectorType term_sum = term1;
    term_sum += term2;
    term_sum += term3;
    term_sum.update_ghost_values();

    interpolation_matrix.Tvmult(out_vector, term_sum);
    out_vector.update_ghost_values();
}

template<int dim, int nstate, typename real>
void DualWeightedResidualObjFunc<dim, nstate, real> :: dwr_xx_vmult(
    VectorType &out_vector, 
    const VectorType &in_vector) const
{
    VectorType vol_nodes_vector = this->dg->high_order_grid->volume_nodes;
    AssertDimension(in_vector.size(), vol_nodes_vector.size());
    AssertDimension(out_vector.size(), vol_nodes_vector.size());

    VectorType term1(vol_nodes_vector);
    adjoint_times_R_xx.vmult(term1, in_vector);
    term1.update_ghost_values();

    VectorType v1(vector_fine);
    adjoint_x_vmult(v1, in_vector);
    VectorType term2(vol_nodes_vector);
    R_x.Tvmult(term2, v1);
    term2.update_ghost_values();

    VectorType v2(vector_fine);
    R_x.vmult(v2, in_vector);
    v2.update_ghost_values();
    VectorType term3(vol_nodes_vector);
    adjoint_x_Tvmult(term3, v2);

    out_vector = term1;
    out_vector += term2;
    out_vector += term3;
    out_vector.update_ghost_values();
}

template<int dim, int nstate, typename real>
void DualWeightedResidualObjFunc<dim, nstate, real> :: dwr_ux_vmult(
    VectorType &out_vector, 
    const VectorType &in_vector) const
{
    VectorType vol_nodes_vector = this->dg->high_order_grid->volume_nodes;
    AssertDimension(in_vector.size(), vol_nodes_vector.size());
    AssertDimension(out_vector.size(), vector_coarse.size());

    VectorType term1(vector_fine);
    adjoint_times_R_ux.vmult(term1, in_vector);
    term1.update_ghost_values();

    VectorType v1(vector_fine);
    adjoint_x_vmult(v1, in_vector);
    VectorType term2(vector_fine);
    R_u.Tvmult(term2, v1);
    term2.update_ghost_values();

    VectorType v2(vector_fine);
    R_x.vmult(v2, in_vector);
    v2.update_ghost_values();
    VectorType term3(vector_fine);
    adjoint_u_Tvmult(term3, v2);

    VectorType term_sum = term1;
    term_sum += term2;
    term_sum += term3;
    term_sum.update_ghost_values();

    interpolation_matrix.Tvmult(out_vector, term_sum);
    out_vector.update_ghost_values();
}

template<int dim, int nstate, typename real>
void DualWeightedResidualObjFunc<dim, nstate, real> :: dwr_ux_Tvmult(
    VectorType &out_vector, 
    const VectorType &in_vector) const
{
    VectorType vol_nodes_vector = this->dg->high_order_grid->volume_nodes;
    AssertDimension(in_vector.size(), vector_coarse.size());
    AssertDimension(out_vector.size(), vol_nodes_vector.size());
    
    VectorType in_vector_fine(vector_fine);
    interpolation_matrix.vmult(in_vector_fine, in_vector);
    in_vector_fine.update_ghost_values();

    VectorType v1(vector_fine);
    adjoint_u_vmult(v1, in_vector_fine);
    VectorType term1(vol_nodes_vector);
    R_x.Tvmult(term1, v1);
    term1.update_ghost_values();

    VectorType v2(vector_fine);
    R_u.vmult(v2, in_vector_fine);
    v2.update_ghost_values();
    VectorType term2(vol_nodes_vector);
    adjoint_x_Tvmult(term2, v2);

    VectorType term3(vol_nodes_vector);
    adjoint_times_R_ux.Tvmult(term3, in_vector_fine);
    term3.update_ghost_values();

    out_vector = term1;
    out_vector += term2;
    out_vector += term3;
    out_vector.update_ghost_values();
}

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

