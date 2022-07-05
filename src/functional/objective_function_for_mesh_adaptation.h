#ifndef __MESH_OBJ_FUNCTION_H__
#define __MESH_OBJ_FUNCTION_H__

#include "functional.h"

namespace PHiLiP {

#if PHILIP_DIM==1
template <int dim, int nstate, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, int nstate, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif

class ObjectiveFunctionMeshAdaptation 
{

public:

    std::shared_ptr<DGBase<dim,real,MeshType>> dg;
    dealii::LinearAlgebra::distributed::Vector<real> dFdX;
    dealii::LinearAlgebra::distributed::Vector<real> dFdUh;
    dealii::LinearAlgebra::distributed::Vector<real> dFdUhH_tilde;
    dealii::TrilinosWrappers::SparseMatrix d2FdXdX;
    dealii::ConditionalOStream pcout;

    ObjectiveFunctionMeshAdaptation(std::shared_ptr<DGBase<dim,real,MeshType>> _dg);


    template <typename real2>
    real2 evaluate_objective_function();

    template <typename real2>
    void evaluate_objective_function_derivative();

    void evaluate_objective_function_hessian();
};

} // namespace PHiLiP

#endif

