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
    const dealii::UpdateFlags volume_update_flags = dealii::update_values | dealii::update_gradients | dealii::update_quadrature_points | dealii::update_JxW_values;
    const dealii::UpdateFlags face_update_flags = dealii::update_values | dealii::update_gradients | dealii::update_quadrature_points | dealii::update_JxW_values | dealii::update_normal_vectors;
    
    ObjectiveFunctionMeshAdaptation(std::shared_ptr<DGBase<dim,real,MeshType>> _dg);


    real2 evaluate_objective_function_and_derivatives();

    template <typename real2>
    void evaluate_objective_function_derivative();

    void evaluate_objective_function_hessian();
};

} // namespace PHiLiP

#endif

