#include "residual_based_error_indicator.h"

namespace PHiLiP {

template <int dim, typename real, typename MeshType>
ResidualErrorEstimate<dim, real, MeshType> :: ResidualErrorEstimate(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input)
    : MeshErrorEstimateBase<dim, real, MeshType> (dg_input)
{}

template <int dim, typename real, typename MeshType>
dealii::Vector<real> ResidualErrorEstimate<dim, real, MeshType> :: compute_cellwise_errors()
{
    std::vector<dealii::types::global_dof_index> dofs_indices;
    dealii::Vector<real> cellwise_errors (this->dg->high_order_grid->triangulation->n_active_cells());

    for (const auto &cell : this->dg->dof_handler.active_cell_iterators()) 
    {
         if (!cell->is_locally_owned()) 
         continue;

         const int i_fele = cell->active_fe_index();
         const dealii::FESystem<dim,dim> &fe_ref = this->dg->fe_collection[i_fele];
         const unsigned int n_dofs_cell = fe_ref.n_dofs_per_cell();
         dofs_indices.resize(n_dofs_cell);
         cell->get_dof_indices (dofs_indices);
         real max_residual = 0;
         for (unsigned int idof = 0; idof < n_dofs_cell; ++idof) 
         {
            const unsigned int index = dofs_indices[idof];
            const real res = std::abs(this->dg->right_hand_side[index]);
            if (res > max_residual) 
                max_residual = res;
         }
         cellwise_errors[cell->active_cell_index()] = max_residual;
     }

     return cellwise_errors;
}

template class ResidualErrorEstimate<PHILIP_DIM, double, dealii::Triangulation<PHILIP_DIM>>;
template class ResidualErrorEstimate<PHILIP_DIM, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
#if PHILIP_DIM != 1
template class ResidualErrorEstimate<PHILIP_DIM, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif
} // namespace PHiLiP

