#ifndef __CELL_VOLUME_OBJECTIVE_FUNCTION_H__ 
#define __CELL_VOLUME_OBJECTIVE_FUNCTION_H__ 

#include "functional.h"

namespace PHiLiP {

/// Class to compute the weight of mesh to prevent distortion of mesh during optimization.
/**
 *  Comutes \f[\mathcal{J}(\mathbf{u},\mathbf{x}) = \mu \sum_k \frac{1}{\Omega_k^2} \f]
 */
template <int dim, int nstate, typename real>
class CellVolumeObjFunc : public Functional<dim, nstate, real> // using default MeshType
{
    using FadType = Sacado::Fad::DFad<real>; ///< Sacado AD type for first derivatives.
    using FadFadType = Sacado::Fad::DFad<FadType>; ///< Sacado AD type that allows 2nd derivatives.

public: 
    
    /// Constructor
    CellVolumeObjFunc( 
        std::shared_ptr<DGBase<dim,real>> dg_input,
        const bool uses_solution_values = false,
        const bool uses_solution_gradient = false);

    /// Destructor
    ~CellVolumeObjFunc(){}

    /// Templated function to evaluate a cell's volume weight.
    template <typename real2>
    real2 evaluate_volume_cell_functional(
        const Physics::PhysicsBase<dim,nstate,real2> &physics,
        const std::vector< real2 > &soln_coeff,
        const dealii::FESystem<dim> &fe_solution,
        const std::vector< real2 > &coords_coeff,
        const dealii::FESystem<dim> &fe_metric,
        const dealii::Quadrature<dim> &volume_quadrature) const;
    
    /// Corresponding real function to evaluate a cell's volume functional. Overrides function in Functional.
    real evaluate_volume_cell_functional(
        const Physics::PhysicsBase<dim,nstate,real> &physics,
        const std::vector< real > &soln_coeff,
        const dealii::FESystem<dim> &fe_solution,
        const std::vector< real > &coords_coeff,
        const dealii::FESystem<dim> &fe_metric,
        const dealii::Quadrature<dim> &volume_quadrature) const override
    {
        return evaluate_volume_cell_functional<real>(physics, soln_coeff, fe_solution, coords_coeff, fe_metric, volume_quadrature);
    }
    
    /// Corresponding FadFadType function to evaluate a cell's volume functional. Overrides function in Functional.
    FadFadType evaluate_volume_cell_functional(
        const Physics::PhysicsBase<dim,nstate,FadFadType> &physics_fad_fad,
        const std::vector< FadFadType > &soln_coeff,
        const dealii::FESystem<dim> &fe_solution,
        const std::vector< FadFadType > &coords_coeff,
        const dealii::FESystem<dim> &fe_metric,
        const dealii::Quadrature<dim> &volume_quadrature) const override
        {
            return evaluate_volume_cell_functional<FadFadType>(physics_fad_fad, soln_coeff, fe_solution, coords_coeff, fe_metric, volume_quadrature);
        }

private:
    /// Stores the weight of mesh to be used to evaluate this function. 
    /** It is parameter \f[\mu \f] in \f[\mathcal{J}(\mathbf{u},\mathbf{x}) = \mu \sum_k \frac{1}{\Omega_k^2} \f] 
     */
    const real mesh_weight_factor;

    /// Stores power of mesh cell volume
    /** It is parameter \f[\gamma \f] in \f[\mathcal{J}(\mathbf{u},\mathbf{x}) = \mu \sum_k \Omega_k^\gamma \f] 
     */
    const int mesh_volume_power;
}; // class ends

} // namespace PHiLiP
#endif
