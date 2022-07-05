#include "objective_function_for_mesh_adaptation.h"
#include <deal.II/dofs/dof_tools.h>

namespace PHiLiP {

template <int dim, int nstate, typename real, typename MeshType>
ObjectiveFunctionMeshAdaptation<dim,nstate,real,MeshType>::ObjectiveFunctionMeshAdaptation(std::shared_ptr<DGBase<dim,real,MeshType>> _dg)
    : dg(_dg)
    , pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
{}



template <int dim, int nstate, typename real, typename MeshType>
template <typename real2>
real2 ObjectiveFunctionMeshAdaptation<dim,nstate,real,MeshType>::evaluate_objective_function()
{

    

}


template <int dim, int nstate, typename real, typename MeshType>
void ObjectiveFunctionMeshAdaptation<dim,nstate,real,MeshType>::evaluate_objective_function_hessian()
{
    dealii::IndexSet locally_owned_dofs = dg->high_order_grid->dof_handler_grid.locally_owned_dofs();
    dealii::IndexSet locally_relevant_dofs, ghost_dofs;
    dealii::DoFTools::extract_locally_relevant_dofs(dg->high_order_grid->dof_handler_grid, locally_relevant_dofs);
    ghost_dofs = locally_relevant_dofs;
    ghost_dofs.subtract_set(locally_owned_dofs);
    dFdX.reinit(locally_owned_dofs, ghost_dofs, MPI_COMM_WORLD);
    std::cout<<"Size of dFdX = "<<dFdX.size()<<std::endl;
}

template class ObjectiveFunctionMeshAdaptation<PHILIP_DIM, PHILIP_DIM, double, dealii::Triangulation<PHILIP_DIM>>;
#if PHILIP_DIM != 1
template class ObjectiveFunctionMeshAdaptation<PHILIP_DIM, PHILIP_DIM, double,  dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif

} // namespace PHiLiP
