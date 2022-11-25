#include "cell_volume_objective_function.h"

namespace PHiLiP {

template<int dim, int nstate, typename real>  
CellVolumeObjFunc<dim, nstate, real> :: CellVolumeObjFunc( 
    std::shared_ptr<DGBase<dim,real>> dg_input,
    const bool uses_solution_values,
    const bool uses_solution_gradient)
    : Functional<dim, nstate, real>(dg_input, uses_solution_values, uses_solution_gradient)
    , mesh_weight_factor(dg_input->all_parameters->optimization_param.mesh_weight_factor)
    , mesh_volume_power(dg_input->all_parameters->optimization_param.mesh_volume_power)
 {}
/*
template<int dim, int nstate, typename real>
template<typename real2>
real2 CellVolumeObjFunc<dim, nstate, real> :: evaluate_volume_cell_functional(
    const Physics::PhysicsBase<dim,nstate,real2> &/physics/,
    const std::vector< real2 > &/soln_coeff/,
    const dealii::FESystem<dim> &/fe_solution/,
    const std::vector< real2 > &coords_coeff,
    const dealii::FESystem<dim> &fe_metric,
    const dealii::Quadrature<dim> &volume_quadrature) const
{
    const unsigned int n_vol_quad_pts = volume_quadrature.size();
    const unsigned int n_metric_dofs_cell = coords_coeff.size();

    real2 cell_distortion_measure = 0.0;
    real2 cell_volume = 0.0;
    for (unsigned int iquad=0; iquad<n_vol_quad_pts; ++iquad) {

        const dealii::Point<dim,double> &ref_point = volume_quadrature.point(iquad);
        const double quad_weight = volume_quadrature.weight(iquad);

        std::array< dealii::Tensor<1,dim,real2>, dim > coord_grad; // Tensor initialize with zeros
        dealii::Tensor<2,dim,real2> metric_jacobian;
        
        for (unsigned int idof=0; idof<n_metric_dofs_cell; ++idof) {
            const unsigned int axis = fe_metric.system_to_component_index(idof).first;
            coord_grad[axis] += coords_coeff[idof] * fe_metric.shape_grad (idof, ref_point);
        }
        real2 jacobian_frobenius_norm_squared = 0.0;
        for (int row=0;row<dim;++row) {
            for (int col=0;col<dim;++col) {
                metric_jacobian[row][col] = coord_grad[row][col];
                jacobian_frobenius_norm_squared += pow(coord_grad[row][col], 2);
            }
        }
        const real2 jacobian_determinant = dealii::determinant(metric_jacobian);

        cell_volume += 1.0 * jacobian_determinant * quad_weight;

        real2 integrand_distortion = jacobian_frobenius_norm_squared/pow(jacobian_determinant, 2/dim);
        integrand_distortion = pow(integrand_distortion, mesh_volume_power);
        cell_distortion_measure += integrand_distortion * jacobian_determinant * quad_weight;
    } // quad loop ends

    real2 cell_volume_obj_func = mesh_weight_factor * cell_distortion_measure/cell_volume;
    
    return cell_volume_obj_func;
}

*/
template class CellVolumeObjFunc<PHILIP_DIM, 1, double>;
template class CellVolumeObjFunc<PHILIP_DIM, 2, double>;
template class CellVolumeObjFunc<PHILIP_DIM, 3, double>;
template class CellVolumeObjFunc<PHILIP_DIM, 4, double>;
template class CellVolumeObjFunc<PHILIP_DIM, 5, double>;
} // namespace PHiLiP
