#include "total_derivatives_of_objective_function.h"


namespace PHiLiP {

template <int dim, int nstate, typename real, typename MeshType>
TotalDerivativeObjfunc<dim, nstate, real, MeshType>::TotalDerivativeObjfunc(std::shared_ptr<DGBase<dim, real, MeshType>> _dg)
    :dg(_dg)
    {}

template class TotalDerivativeObjfunc<PHILIP_DIM, 1, double, dealii::Triangulation<PHILIP_DIM>>;
template class TotalDerivativeObjfunc<PHILIP_DIM, 2, double, dealii::Triangulation<PHILIP_DIM>>;
template class TotalDerivativeObjfunc<PHILIP_DIM, 3, double, dealii::Triangulation<PHILIP_DIM>>;
template class TotalDerivativeObjfunc<PHILIP_DIM, 4, double, dealii::Triangulation<PHILIP_DIM>>;
template class TotalDerivativeObjfunc<PHILIP_DIM, 5, double, dealii::Triangulation<PHILIP_DIM>>;
#if PHILIP_DIM != 1
template class TotalDerivativeObjfunc<PHILIP_DIM, 1, double,  dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class TotalDerivativeObjfunc<PHILIP_DIM, 2, double,  dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class TotalDerivativeObjfunc<PHILIP_DIM, 3, double,  dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class TotalDerivativeObjfunc<PHILIP_DIM, 4, double,  dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class TotalDerivativeObjfunc<PHILIP_DIM, 5, double,  dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif
} // namespace PHiLiP
