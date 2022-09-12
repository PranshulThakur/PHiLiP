#include "mesh_error_estimate_base.h"

namespace PHiLiP {

template <int dim, typename real, typename MeshType>
MeshErrorEstimateBase<dim, real, MeshType> :: ~MeshErrorEstimateBase(){}

template <int dim, typename real, typename MeshType>
MeshErrorEstimateBase<dim, real, MeshType> :: MeshErrorEstimateBase(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input)
    : dg(dg_input)
{}


template class MeshErrorEstimateBase<PHILIP_DIM, double, dealii::Triangulation<PHILIP_DIM>>;
template class MeshErrorEstimateBase<PHILIP_DIM, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
#if PHILIP_DIM != 1
template class MeshErrorEstimateBase<PHILIP_DIM, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif

} // namespace PHiLiP
