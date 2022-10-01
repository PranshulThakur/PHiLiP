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

template class DualWeightedResidualObjFunc <PHILIP_DIM, 1, double>;
template class DualWeightedResidualObjFunc <PHILIP_DIM, 2, double>;
template class DualWeightedResidualObjFunc <PHILIP_DIM, 3, double>;
template class DualWeightedResidualObjFunc <PHILIP_DIM, 4, double>;
template class DualWeightedResidualObjFunc <PHILIP_DIM, 5, double>;
} // namespace PHiLiP

