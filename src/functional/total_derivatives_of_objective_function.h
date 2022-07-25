#ifndef __TOTAL_DERIVATIVES_OBJFUNC__
#define __TOTAL_DERIVATIVES_OBJFUNC__

#include "objective_function_for_mesh_adaptation.h"

namespace PHiLiP {

#if PHILIP_DIM==1
template <int dim, int nstate, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, int nstate, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif

class TotalDerivativeObjfunc 
{
public:
    TotalDerivativeObjfunc(std::shared_ptr<DGBase<dim,real,MeshType>> _dg);

    std::shared_ptr<DGBase<dim,real,MeshType>> dg;

    dealii::Vector<real> dF_dX;
};

} // namespace PHiLiP

#endif
