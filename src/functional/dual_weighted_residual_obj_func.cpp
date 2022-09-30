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

template class DualWeightedResidualObjFunc <PHILIP_DIM, 1, double>;
template class DualWeightedResidualObjFunc <PHILIP_DIM, 2, double>;
template class DualWeightedResidualObjFunc <PHILIP_DIM, 3, double>;
template class DualWeightedResidualObjFunc <PHILIP_DIM, 4, double>;
template class DualWeightedResidualObjFunc <PHILIP_DIM, 5, double>;
} // namespace PHiLiP

