#ifndef __MESH_OBJ_FUNCTION_H__
#define __MESH_OBJ_FUNCTION_H__

#include "functional.h"
#include "physics/physics.h"
#include "physics/physics_factory.h"
#include "physics/model.h"
#include "physics/model_factory.h"
#include "dg/dg.h"
#include "functional.h"

namespace PHiLiP {

#if PHILIP_DIM==1
template <int dim, int nstate, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, int nstate, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif

class ObjectiveFunctionMeshAdaptation 
{
    using FadType = Sacado::Fad::DFad<real>; 
    using FadFadType = Sacado::Fad::DFad<FadType>;

public:

    std::shared_ptr<DGBase<dim,real,MeshType>> dg; ///< fine dg 
    std::shared_ptr<Functional<dim, nstate, real, MeshType> > functional;
    std::shared_ptr<Physics::PhysicsBase<dim,nstate,FadFadType>> physics_fad_fad;

    const dealii::UpdateFlags volume_update_flags = dealii::update_values | dealii::update_gradients | dealii::update_quadrature_points | dealii::update_JxW_values;
    const dealii::UpdateFlags face_update_flags = dealii::update_values | dealii::update_gradients | dealii::update_quadrature_points | dealii::update_JxW_values | dealii::update_normal_vectors;
    
    dealii::LinearAlgebra::distributed::Vector<real> derivative_objfunc_wrt_solution_fine;
    dealii::LinearAlgebra::distributed::Vector<real> derivative_objfunc_wrt_solution_tilde;
    dealii::LinearAlgebra::distributed::Vector<real> derivative_objfunc_wrt_metric_nodes;
    
    dealii::TrilinosWrappers::SparseMatrix d2F_dWfine_dWfine;
    dealii::TrilinosWrappers::SparseMatrix d2F_dWfine_dWtilde;
    dealii::TrilinosWrappers::SparseMatrix d2F_dWfine_dX;
    
    dealii::TrilinosWrappers::SparseMatrix d2F_dWtilde_dWtilde;
    dealii::TrilinosWrappers::SparseMatrix d2F_dWtilde_dX;
    
    dealii::TrilinosWrappers::SparseMatrix d2F_dX_dX;
    
    dealii::LinearAlgebra::distributed::Vector<real> solution_fine;
    dealii::LinearAlgebra::distributed::Vector<real> solution_tilde;    
    
    dealii::ConditionalOStream pcout;

    ObjectiveFunctionMeshAdaptation(std::shared_ptr<DGBase<dim,real,MeshType>> _dg, 
                                    dealii::LinearAlgebra::distributed::Vector<real> & _solution_fine,  
                                    dealii::LinearAlgebra::distributed::Vector<real> & _solution_tilde);

    void allocate_derivatives();


    real evaluate_objective_function_and_derivatives();

    void evaluate_objective_function_hessian();
    
    template <typename real2>
    real2 evaluate_volume_cell_objective_function(
        const Physics::PhysicsBase<dim,nstate,real2> &physics,
        const std::vector< real2 > &soln_coeff_fine,
        const std::vector< real2 > &soln_coeff_tilde,
        const dealii::FESystem<dim> &fe_solution,
        const std::vector< real2 > &coords_coeff,
        const dealii::FESystem<dim> &fe_metric,
        const dealii::Quadrature<dim> &volume_quadrature) const;

    template <typename real2>
    real2 evaluate_boundary_cell_objective_function(
        const Physics::PhysicsBase<dim,nstate,real2> &physics,
        const unsigned int boundary_id,
        const std::vector< real2 > &soln_coeff_fine,
        const std::vector< real2 > &soln_coeff_tilde,
        const dealii::FESystem<dim> &fe_solution,
        const std::vector< real2 > &coords_coeff,
        const dealii::FESystem<dim> &fe_metric,
        const unsigned int face_number,
        const dealii::Quadrature<dim-1> &face_quadrature) const;

    void truncate_first_derivative(dealii::LinearAlgebra::distributed::Vector<real> &vector_in);

    void truncate_second_derivative(dealii::TrilinosWrappers::SparseMatrix &d2F, bool is_dX_dX);
};


} // namespace PHiLiP

#endif

