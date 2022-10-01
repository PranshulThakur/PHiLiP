#include "dual_weighted_residual_obj_func.h"

namespace PHiLiP {

template <int dim, int nstate, typename real>
DualWeightedResidualObjFunc<dim, nstate, real> :: DualWeightedResidualObjFunc( 
    std::shared_ptr<DGBase<dim,real>> dg_input,
    const bool uses_solution_values,
    const bool uses_solution_gradient)
    : Functional<dim, nstate, real> (dg_input, uses_solution_values, uses_solution_gradient)
{
    compute_cell_index_range();
}

template<int dim, int nstate, typename real>
void DualWeightedResidualObjFunc<dim, nstate, real> :: compute_cell_index_range()
{
    const unsigned int n_global_cells = this->dg->triangulation->n_global_active_cells();
    cell_index_range.set_size(n_global_cells);
    for(const auto &cell : this->dg->dof_handler.active_cell_iterators())
    {
        if(cell->is_locally_owned())
        {
            cell_index_range.add_index(cell->active_cell_index());
        }
    }
}

template<int dim, int nstate, typename real>
void DualWeightedResidualObjFunc<dim, nstate, real> :: compute_interpolation_matrix()
{
/*
    const unsigned int coarse_degree = this->dg->initial_degree;
    const unsigned int fine_degree = coarse_degree + 1;
    const dealii::FE_DGQ<dim> fe_dg_coarse(coarse_degree);
    const dealii::FE_DGQ<dim> fe_dg_fine(fine_degree);
    
    vector_coarse = this->dg->solution; // copies values and parallel layout.
    unsigned int n_dofs_coarse = this->dg->n_dofs();
    this->dg->change_cells_fe_degree_by_deltadegree_and_interpolate_solution(1);
    vector_fine = this->dg->solution;
    unsigned int n_dofs_fine = this->dg->n_dofs();
    dealii::IndexSet dofs_fine_locally_relevant_range = this->dg->locally_relevant_dofs;
    this->dg->change_cells_fe_degree_by_deltadegree_and_interpolate_solution(-1);
    AssertDimension(vector_coarse.size(), this->dg->solution.size());
    
    // Get local interpolation matrix. Needs to be changed. Check out dealii solution_transfer.cc, interpolation_hp and extract_interpolation_matrices().
    dealii::FullMatrix<real> local_interpolation_matrix(fe_dg_fine.n_dofs_per_cell(), fe_dg_coarse.n_dofs_per_cell());
    dealii::FETools::get_interpolation_matrix(fe_dg_coarse, fe_dg_fine, local_interpolation_matrix);

    // Get locally owned and locally relevant dofs
    dealii::IndexSet &dofs_range_coarse = vector_coarse.get_partitioner()->locally_owned_range();
    dealii::IndexSet &dofs_range_fine = vector_fine.get_partitioner()->locally_owned_range();

    dealii::DynamicSparsityPattern dsp(n_dofs_fine, n_dofs_coarse, dofs_range_fine);
    for(const auto &cell : this->dg->dof_handler.active_cell_iterators())
    {
        if(! cell->is_locally_owned()) {continue;}
        
    }
*/


    
}
template class DualWeightedResidualObjFunc <PHILIP_DIM, 1, double>;
template class DualWeightedResidualObjFunc <PHILIP_DIM, 2, double>;
template class DualWeightedResidualObjFunc <PHILIP_DIM, 3, double>;
template class DualWeightedResidualObjFunc <PHILIP_DIM, 4, double>;
template class DualWeightedResidualObjFunc <PHILIP_DIM, 5, double>;
} // namespace PHiLiP

