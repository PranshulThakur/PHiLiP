#include <deal.II/base/tensor.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/dofs/dof_accessor.h>

#include <deal.II/lac/vector.h>

#include "ADTypes.hpp"

#include <deal.II/fe/fe_dgq.h> // Used for flux interpolation

#include "strong_dg.hpp"

/// Returns the value from a CoDiPack variable.
/** The recursive calling allows to retrieve nested CoDiPack types.
 */
template <typename real>
double getValue(const real &x) {
    if constexpr (std::is_same<real, double>::value) {
        return x;
    } else {
        return getValue(x.value());
    }
}

namespace PHiLiP {

template <int dim, int nstate, typename real, typename MeshType>
DGStrong<dim,nstate,real,MeshType>::DGStrong(
    const Parameters::AllParameters *const parameters_input,
    const unsigned int degree,
    const unsigned int max_degree_input,
    const unsigned int grid_degree_input,
    const std::shared_ptr<Triangulation> triangulation_input)
    : DGBaseState<dim,nstate,real,MeshType>::DGBaseState(parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input)
{ }

template <int dim, int nstate, typename real, typename MeshType>
template <typename adtype>
void DGStrong<dim,nstate,real,MeshType>::build_volume_metric_operators(
    const unsigned int poly_degree,
    const unsigned int grid_degree,
    const std::vector<adtype>                    &metric_coeffs,
    OPERATOR::metric_operators<adtype,dim,2*dim> &metric_oper,
    OPERATOR::mapping_shape_functions<dim,2*dim> &mapping_basis,
    std::array<std::vector<adtype>,dim>          &mapping_support_points)
{
    const dealii::FESystem<dim> &fe_metric = this->high_order_grid->fe_system;
    const unsigned int n_metric_dofs = fe_metric.dofs_per_cell;
    const unsigned int n_grid_nodes  = n_metric_dofs / dim;
    //Rewrite the high_order_grid->volume_nodes in a way we can use sum-factorization on.
    //That is, splitting up the vector by the dimension.
    for(int idim=0; idim<dim; idim++){
        mapping_support_points[idim].resize(n_grid_nodes);
    }
    const std::vector<unsigned int > &index_renumbering = dealii::FETools::hierarchic_to_lexicographic_numbering<dim>(grid_degree);
    for (unsigned int idof = 0; idof< n_metric_dofs; ++idof) {
        const adtype val = metric_coeffs[idof];
        const unsigned int istate = fe_metric.system_to_component_index(idof).first; 
        const unsigned int ishape = fe_metric.system_to_component_index(idof).second; 
        const unsigned int igrid_node = index_renumbering[ishape];
        mapping_support_points[istate][igrid_node] = val; 
    }
    metric_oper.build_volume_metric_operators(
        this->volume_quadrature_collection[poly_degree].size(), n_grid_nodes,
        mapping_support_points,
        mapping_basis,
        this->all_parameters->use_invariant_curl_form);
}

/***********************************************************
*
*       Build operators and solve for RHS
*
***********************************************************/

template <int dim, int nstate, typename real, typename MeshType>
template <typename adtype>
void DGStrong<dim,nstate,real,MeshType>::assemble_volume_term_and_build_operators_ad_templated(
    typename dealii::DoFHandler<dim>::active_cell_iterator cell,
    const dealii::types::global_dof_index                  current_cell_index,
    const std::vector<adtype>                              &soln_coeffs,
    const dealii::Tensor<1,dim,std::vector<adtype>>        &/*aux_soln_coeffs*/,
    const std::vector<adtype>                              &/*metric_coeffs*/,
    const std::vector<real>                                &local_dual,
    const std::vector<dealii::types::global_dof_index>     &/*soln_dofs_indices*/,
    const std::vector<dealii::types::global_dof_index>     &/*metric_dofs_indices*/,
    const unsigned int                                     poly_degree,
    const unsigned int                                     grid_degree,
    const Physics::PhysicsBase<dim, nstate, adtype>        &physics,
    OPERATOR::basis_functions<dim,2*dim>                   &soln_basis,
    OPERATOR::basis_functions<dim,2*dim>                   &flux_basis,
    OPERATOR::local_basis_stiffness<dim,2*dim>             &flux_basis_stiffness,
    OPERATOR::vol_projection_operator<dim,2*dim>           &soln_basis_projection_oper_int,
    OPERATOR::vol_projection_operator<dim,2*dim>           &soln_basis_projection_oper_ext,
    OPERATOR::metric_operators<adtype,dim,2*dim>           &metric_oper,
    OPERATOR::mapping_shape_functions<dim,2*dim>           &mapping_basis,
    std::array<std::vector<adtype>,dim>                    &/*mapping_support_points*/,
    dealii::hp::FEValues<dim,dim>                          &/*fe_values_collection_volume*/,
    dealii::hp::FEValues<dim,dim>                          &/*fe_values_collection_volume_lagrange*/,
    const dealii::FESystem<dim,dim>                        &/*fe_soln*/,
    std::vector<adtype>                                    &rhs, 
    dealii::Tensor<1,dim,std::vector<adtype>>              &local_auxiliary_RHS,
    const bool                                             compute_auxiliary_right_hand_side,
    adtype                                                 &dual_dot_residual)
{
    // Check if the current cell's poly degree etc is different then previous cell's.
    // If the current cell's poly degree is different, then we recompute the 1D 
    // polynomial basis functions. Otherwise, we use the previous values in reference space.
    if(poly_degree != soln_basis.current_degree){
        soln_basis.current_degree = poly_degree; 
        flux_basis.current_degree = poly_degree; 
        mapping_basis.current_degree  = poly_degree; 
        this->reinit_operators_for_cell_residual_loop(poly_degree, poly_degree, grid_degree, 
                                                      soln_basis, soln_basis, 
                                                      flux_basis, flux_basis, 
                                                      flux_basis_stiffness, 
                                                      soln_basis_projection_oper_int, soln_basis_projection_oper_ext,
                                                      mapping_basis);
    }

    //Fetch the modal soln coefficients and the modal auxiliary soln coefficients
    //We immediately separate them by state as to be able to use sum-factorization
    //in the interpolation operator. If we left it by n_dofs_cell, then the matrix-vector
    //mult would sum the states at the quadrature point.
    const unsigned int n_dofs_cell = this->fe_collection[poly_degree].dofs_per_cell;
    const unsigned int n_shape_fns = n_dofs_cell / nstate;
    std::array<std::vector<adtype>,nstate> soln_coeff;
    std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate> aux_soln_coeff;
    for (unsigned int idof = 0; idof < n_dofs_cell; ++idof) {
        const unsigned int istate = this->fe_collection[poly_degree].system_to_component_index(idof).first;
        const unsigned int ishape = this->fe_collection[poly_degree].system_to_component_index(idof).second;
        if(ishape == 0)
            soln_coeff[istate].resize(n_shape_fns);
        soln_coeff[istate][ishape] = soln_coeffs[idof];
        /*
        for(int idim=0; idim<dim; idim++){
            if(ishape == 0)
                aux_soln_coeff[istate][idim].resize(n_shape_fns);
            if(this->use_auxiliary_eq){
                aux_soln_coeff[istate][idim][ishape] = aux_soln_coeffs[idim][idof];
            }
            else{
                aux_soln_coeff[istate][idim][ishape] = 0.0;
            }
        }
        */
    }


    if(compute_auxiliary_right_hand_side){
        assemble_volume_term_auxiliary_equation<adtype>(
            soln_coeff,
            poly_degree,
            soln_basis,
            flux_basis,
            metric_oper,
            local_auxiliary_RHS);
    }
    else{
        assemble_volume_term_strong<adtype>(
            cell,
            current_cell_index,
            soln_coeff,
            aux_soln_coeff,
            poly_degree,
            soln_basis,
            flux_basis,
            flux_basis_stiffness,
            soln_basis_projection_oper_int,
            metric_oper,
            physics,
            rhs);
        for(unsigned int idof=0; idof<n_dofs_cell; idof++){
            dual_dot_residual += rhs[idof] * local_dual[idof];
        }
    }
}

template <int dim, int nstate, typename real, typename MeshType>
template<typename adtype>
void DGStrong<dim,nstate,real,MeshType>::assemble_boundary_term_and_build_operators_ad_templated(
    typename dealii::DoFHandler<dim>::active_cell_iterator             /*cell*/,
    const dealii::types::global_dof_index                              current_cell_index,
    const std::vector<adtype>                                          &soln_coeffs,
    const dealii::Tensor<1,dim,std::vector<adtype>>                    &/*aux_soln_coeffs*/,
    const std::vector<adtype>                                          &/*metric_coeffs*/,
    const std::vector<real>                                            &local_dual,
    const unsigned int                                                 face_number,
    const unsigned int                                                 boundary_id,
    const Physics::PhysicsBase<dim, nstate, adtype>                    &physics,
    const NumericalFlux::NumericalFluxConvective<dim, nstate, adtype>  &conv_num_flux,
    const NumericalFlux::NumericalFluxDissipative<dim, nstate, adtype> &diss_num_flux,
    const unsigned int                                                 poly_degree,
    const unsigned int                                                 /*grid_degree*/,
    OPERATOR::basis_functions<dim,2*dim>                               &soln_basis,
    OPERATOR::basis_functions<dim,2*dim>                               &flux_basis,
    OPERATOR::vol_projection_operator<dim,2*dim>                       &soln_basis_projection_oper_int,
    OPERATOR::metric_operators<adtype,dim,2*dim>                       &metric_oper,
    OPERATOR::mapping_shape_functions<dim,2*dim>                       &mapping_basis,
    std::array<std::vector<adtype>,dim>                                &mapping_support_points,
    dealii::hp::FEFaceValues<dim,dim>                                  &/*fe_values_collection_face_int*/,
    const dealii::FESystem<dim,dim>                                    &/*fe_soln*/,
    const real                                                         penalty,
    std::vector<adtype>                                                &rhs,
    dealii::Tensor<1,dim,std::vector<adtype>>                          &local_auxiliary_RHS,
    const bool                                                         compute_auxiliary_right_hand_side,
    adtype                                                             &dual_dot_residual)
{

    const dealii::FESystem<dim> &fe_metric = this->high_order_grid->fe_system;
    const unsigned int n_metric_dofs = fe_metric.dofs_per_cell;
    const unsigned int n_grid_nodes  = n_metric_dofs / dim;
    //build the surface metric operators for interior
    metric_oper.build_facet_metric_operators(
        face_number,
        this->face_quadrature_collection[poly_degree].size(),
        n_grid_nodes,
        mapping_support_points,
        mapping_basis,
        this->all_parameters->use_invariant_curl_form);
    //Fetch the modal soln coefficients and the modal auxiliary soln coefficients
    //We immediately separate them by state as to be able to use sum-factorization
    //in the interpolation operator. If we left it by n_dofs_cell, then the matrix-vector
    //mult would sum the states at the quadrature point.
    const unsigned int n_dofs_cell = this->fe_collection[poly_degree].dofs_per_cell;
    const unsigned int n_shape_fns = n_dofs_cell / nstate;
    std::array<std::vector<adtype>,nstate> soln_coeff;
    std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate> aux_soln_coeff;
    for (unsigned int idof = 0; idof < n_dofs_cell; ++idof) {
        const unsigned int istate = this->fe_collection[poly_degree].system_to_component_index(idof).first;
        const unsigned int ishape = this->fe_collection[poly_degree].system_to_component_index(idof).second;
        if(ishape == 0)
            soln_coeff[istate].resize(n_shape_fns);
        soln_coeff[istate][ishape] = soln_coeffs[idof];
        /*
        for(int idim=0; idim<dim; idim++){
            if(ishape == 0)
                aux_soln_coeff[istate][idim].resize(n_shape_fns);
            if(this->use_auxiliary_eq){
                aux_soln_coeff[istate][idim][ishape] = aux_soln_coeffs[idim][idof];
            }
            else{
                aux_soln_coeff[istate][idim][ishape] = 0.0;
            }
        }
        */
    }

    if(compute_auxiliary_right_hand_side){
        assemble_boundary_term_auxiliary_equation<adtype> (
            face_number, current_cell_index, 
            soln_coeff,
            poly_degree,
            boundary_id,
            soln_basis, metric_oper,
            physics,
            diss_num_flux,
            local_auxiliary_RHS);
    }
    else{
        assemble_boundary_term_strong<adtype> (
            face_number,
            current_cell_index,
            soln_coeff, aux_soln_coeff,
            boundary_id, poly_degree, penalty, 
            soln_basis,
            flux_basis,
            soln_basis_projection_oper_int,
            metric_oper,
            physics, conv_num_flux, diss_num_flux,
            rhs);
        for(unsigned int idof=0; idof<n_dofs_cell; idof++){
            dual_dot_residual += rhs[idof] * local_dual[idof];
        }
    }

}

template <int dim, int nstate, typename real, typename MeshType>
template <typename adtype>
void DGStrong<dim,nstate,real,MeshType>::assemble_face_term_and_build_operators_ad_templated(
    typename dealii::DoFHandler<dim>::active_cell_iterator             /*cell*/,
    typename dealii::DoFHandler<dim>::active_cell_iterator             /*neighbor_cell*/,
    const dealii::types::global_dof_index                              current_cell_index,
    const dealii::types::global_dof_index                              neighbor_cell_index,
    const unsigned int                                                 iface,
    const unsigned int                                                 neighbor_iface,
    const std::vector<adtype>                                          &soln_coeffs_int,
    const std::vector<adtype>                                          &soln_coeffs_ext,
    const dealii::Tensor<1,dim,std::vector<adtype>>                    &/*aux_soln_coeffs_int*/,
    const dealii::Tensor<1,dim,std::vector<adtype>>                    &/*aux_soln_coeffs_ext*/,
    const std::vector<adtype>                                          &/*metric_coeff_int*/,
    const std::vector<adtype>                                          &metric_coeff_ext,
    const std::vector< double >                                        &dual_int,
    const std::vector< double >                                        &dual_ext,
    const unsigned int                                                 poly_degree_int,
    const unsigned int                                                 poly_degree_ext,
    const unsigned int                                                 /*grid_degree_int*/,
    const unsigned int                                                 grid_degree_ext,
    OPERATOR::basis_functions<dim,2*dim>                               &soln_basis_int,
    OPERATOR::basis_functions<dim,2*dim>                               &soln_basis_ext,
    OPERATOR::basis_functions<dim,2*dim>                               &flux_basis_int,
    OPERATOR::basis_functions<dim,2*dim>                               &flux_basis_ext,
    OPERATOR::local_basis_stiffness<dim,2*dim>                         &flux_basis_stiffness,
    OPERATOR::vol_projection_operator<dim,2*dim>                       &soln_basis_projection_oper_int,
    OPERATOR::vol_projection_operator<dim,2*dim>                       &soln_basis_projection_oper_ext,
    OPERATOR::metric_operators<adtype,dim,2*dim>                       &metric_oper_int,
    OPERATOR::metric_operators<adtype,dim,2*dim>                       &metric_oper_ext,
    OPERATOR::mapping_shape_functions<dim,2*dim>                       &mapping_basis,
    std::array<std::vector<adtype>,dim>                                &mapping_support_points,
    const Physics::PhysicsBase<dim, nstate, adtype>                    &physics,
    const NumericalFlux::NumericalFluxConvective<dim, nstate, adtype>  &conv_num_flux,
    const NumericalFlux::NumericalFluxDissipative<dim, nstate, adtype> &diss_num_flux,
    dealii::hp::FEFaceValues<dim,dim>                                  &/*fe_values_collection_face_int*/,
    dealii::hp::FEFaceValues<dim,dim>                                  &/*fe_values_collection_face_ext*/,
    dealii::hp::FESubfaceValues<dim,dim>                               &/*fe_values_collection_subface*/,
    const dealii::FESystem<dim,dim>                                    &/*fe_int*/,
    const dealii::FESystem<dim,dim>                                    &/*fe_ext*/,
    const real                                                         penalty,
    std::vector<adtype>                                                &rhs_int,
    std::vector<adtype>                                                &rhs_ext,
    dealii::Tensor<1,dim,std::vector<adtype>>                          &aux_rhs_int,
    dealii::Tensor<1,dim,std::vector<adtype>>                          &aux_rhs_ext,
    const bool                                                         compute_auxiliary_right_hand_side,
    adtype                                                             &dual_dot_residual,
    const bool                                                         /*is_a_subface*/,
    const unsigned int                                                 /*neighbor_i_subface*/)
{

    const dealii::FESystem<dim> &fe_metric = this->high_order_grid->fe_system;
    const unsigned int n_metric_dofs = fe_metric.dofs_per_cell;
    const unsigned int n_grid_nodes  = n_metric_dofs / dim;
    //build the surface metric operators for interior
    metric_oper_int.build_facet_metric_operators(
        iface,
        this->face_quadrature_collection[poly_degree_int].size(),
        n_grid_nodes,
        mapping_support_points,
        mapping_basis,
        this->all_parameters->use_invariant_curl_form);

    if(poly_degree_ext != soln_basis_ext.current_degree){
        soln_basis_ext.current_degree    = poly_degree_ext; 
        flux_basis_ext.current_degree    = poly_degree_ext; 
        mapping_basis.current_degree     = poly_degree_ext; 
        this->reinit_operators_for_cell_residual_loop(poly_degree_int, poly_degree_ext, grid_degree_ext, 
                                                      soln_basis_int, soln_basis_ext, 
                                                      flux_basis_int, flux_basis_ext, 
                                                      flux_basis_stiffness, 
                                                      soln_basis_projection_oper_int, soln_basis_projection_oper_ext,
                                                      mapping_basis);
    }

    if(!compute_auxiliary_right_hand_side){//only for primary equations
        //get neighbor metric operator
        //rewrite the high_order_grid->volume_nodes in a way we can use sum-factorization on.
        //that is, splitting up the vector by the dimension.
        std::array<std::vector<adtype>,dim> mapping_support_points_neigh;
        for(int idim=0; idim<dim; idim++){
            mapping_support_points_neigh[idim].resize(n_grid_nodes);
        }
        const std::vector<unsigned int > &index_renumbering = dealii::FETools::hierarchic_to_lexicographic_numbering<dim>(grid_degree_ext);
        for (unsigned int idof = 0; idof< n_metric_dofs; ++idof) {
            const adtype val = metric_coeff_ext[idof];
            const unsigned int istate = fe_metric.system_to_component_index(idof).first; 
            const unsigned int ishape = fe_metric.system_to_component_index(idof).second; 
            const unsigned int igrid_node = index_renumbering[ishape];
            mapping_support_points_neigh[istate][igrid_node] = val; 
        }
        //build the metric operators for strong form
        metric_oper_ext.build_volume_metric_operators(
            this->volume_quadrature_collection[poly_degree_ext].size(), n_grid_nodes,
            mapping_support_points_neigh,
            mapping_basis,
            this->all_parameters->use_invariant_curl_form);

        if(this->check_same_coords_strongdg)
        {
            check_same_coords_face_strong(mapping_support_points, mapping_support_points_neigh, mapping_basis, iface, neighbor_iface, poly_degree_int);
        }
    }

    const unsigned int n_dofs_int = this->fe_collection[poly_degree_int].dofs_per_cell;
    const unsigned int n_dofs_ext = this->fe_collection[poly_degree_ext].dofs_per_cell;
    const unsigned int n_shape_fns_int = n_dofs_int / nstate;
    const unsigned int n_shape_fns_ext = n_dofs_ext / nstate;
    // Extract interior modal coefficients of solution
    std::array<std::vector<adtype>,nstate> soln_coeff_int;
    std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate> aux_soln_coeff_int;
    for (unsigned int idof = 0; idof < n_dofs_int; ++idof) {
        const unsigned int istate = this->fe_collection[poly_degree_int].system_to_component_index(idof).first;
        const unsigned int ishape = this->fe_collection[poly_degree_int].system_to_component_index(idof).second;
        if(ishape == 0)
            soln_coeff_int[istate].resize(n_shape_fns_int);

        soln_coeff_int[istate][ishape] = soln_coeffs_int[idof];
        /*
        for(int idim=0; idim<dim; idim++){
            if(ishape == 0){
                aux_soln_coeff_int[istate][idim].resize(n_shape_fns_int);
            }
            if(this->use_auxiliary_eq){
                aux_soln_coeff_int[istate][idim][ishape] = aux_soln_coeffs_int[idim][idof];
            }
            else{
                aux_soln_coeff_int[istate][idim][ishape] = 0.0;
            }
        }
        */
    }

    // Extract exterior modal coefficients of solution
    std::array<std::vector<adtype>,nstate> soln_coeff_ext;
    std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate> aux_soln_coeff_ext;
    for (unsigned int idof = 0; idof < n_dofs_ext; ++idof) {
        const unsigned int istate = this->fe_collection[poly_degree_ext].system_to_component_index(idof).first;
        const unsigned int ishape = this->fe_collection[poly_degree_ext].system_to_component_index(idof).second;
        if(ishape == 0){
            soln_coeff_ext[istate].resize(n_shape_fns_ext);
        }
        soln_coeff_ext[istate][ishape] = soln_coeffs_ext[idof];
        /*
        for(int idim=0; idim<dim; idim++){
            if(ishape == 0){
                aux_soln_coeff_ext[istate][idim].resize(n_shape_fns_ext);
            }
            if(this->use_auxiliary_eq){
                aux_soln_coeff_ext[istate][idim][ishape] = aux_soln_coeffs_ext[idim][idof];
            }
            else{
                aux_soln_coeff_ext[istate][idim][ishape] = 0.0;
            }
        }
        */
    }

    if(compute_auxiliary_right_hand_side){
        assemble_face_term_auxiliary_equation<adtype> (
            iface, neighbor_iface, 
            current_cell_index, neighbor_cell_index,
            soln_coeff_int, soln_coeff_ext,
            poly_degree_int, poly_degree_ext,
            soln_basis_int, soln_basis_ext,
            metric_oper_int,
            physics,
            diss_num_flux,
            aux_rhs_int, aux_rhs_ext);
    }
    else{
        assemble_face_term_strong<adtype> (
            iface, neighbor_iface, 
            current_cell_index,
            neighbor_cell_index,
            soln_coeff_int, soln_coeff_ext,
            aux_soln_coeff_int, aux_soln_coeff_ext,
            poly_degree_int, poly_degree_ext,
            penalty,
            soln_basis_int, soln_basis_ext,
            flux_basis_int, flux_basis_ext,
            soln_basis_projection_oper_int, soln_basis_projection_oper_ext,
            metric_oper_int, metric_oper_ext,
            physics, conv_num_flux, diss_num_flux,
            rhs_int, rhs_ext);
        for(unsigned int idof=0; idof<n_dofs_int; idof++){
            dual_dot_residual += rhs_int[idof] * dual_int[idof];
        }
        for(unsigned int idof=0; idof<n_dofs_ext; idof++){
            dual_dot_residual += rhs_ext[idof] * dual_ext[idof];
        }
    }

}

/*******************************************************************
 *
 *
 *                      AUXILIARY EQUATIONS
 *
 *
 *******************************************************************/

template <int dim, int nstate, typename real, typename MeshType>
void DGStrong<dim,nstate,real,MeshType>::assemble_auxiliary_residual(const bool compute_dRdW, const bool compute_dRdX, const bool compute_d2R)
{
    using PDE_enum = Parameters::AllParameters::PartialDifferentialEquation;
    using ODE_enum = Parameters::ODESolverParam::ODESolverEnum;
    const PDE_enum pde_type = this->all_parameters->pde_type;

    if(pde_type == PDE_enum::burgers_viscous){
        pcout << "DG Strong not yet verified for Burgers' viscous. Aborting..." << std::endl;
        std::abort();
    }
    
    // NOTE: auxiliary currently only works explicit time advancement - not implicit
    if (this->use_auxiliary_eq && !(this->all_parameters->ode_solver_param.ode_solver_type == ODE_enum::implicit_solver)) {
        
        if(compute_dRdW || compute_dRdX || compute_d2R)
        {
            pcout << "DG Strong's viscous terms cannot yet be automatically differentiated. Aborting..."<<std::endl;
            std::abort();
        }
        //set auxiliary rhs to 0
        for(int idim=0; idim<dim; idim++){
            this->auxiliary_right_hand_side[idim] = 0;
        }
        //initialize this to use DG cell residual loop. Note, FEValues to be deprecated in future.
        const auto mapping = (*(this->high_order_grid->mapping_fe_field));

        dealii::hp::MappingCollection<dim> mapping_collection(mapping);

        dealii::hp::FEValues<dim,dim>        fe_values_collection_volume (mapping_collection, this->fe_collection, this->volume_quadrature_collection, this->volume_update_flags); ///< FEValues of volume.
        dealii::hp::FEFaceValues<dim,dim>    fe_values_collection_face_int (mapping_collection, this->fe_collection, this->face_quadrature_collection, this->face_update_flags); ///< FEValues of interior face.
        dealii::hp::FEFaceValues<dim,dim>    fe_values_collection_face_ext (mapping_collection, this->fe_collection, this->face_quadrature_collection, this->neighbor_face_update_flags); ///< FEValues of exterior face.
        dealii::hp::FESubfaceValues<dim,dim> fe_values_collection_subface (mapping_collection, this->fe_collection, this->face_quadrature_collection, this->face_update_flags); ///< FEValues of subface.
         
        dealii::hp::FEValues<dim,dim>        fe_values_collection_volume_lagrange (mapping_collection, this->fe_collection_lagrange, this->volume_quadrature_collection, this->volume_update_flags);

        OPERATOR::basis_functions<dim,2*dim> soln_basis_int(1, this->max_degree, this->max_grid_degree); 
        OPERATOR::basis_functions<dim,2*dim> soln_basis_ext(1, this->max_degree, this->max_grid_degree); 
        OPERATOR::basis_functions<dim,2*dim> flux_basis_int(1, this->max_degree, this->max_grid_degree); 
        OPERATOR::basis_functions<dim,2*dim> flux_basis_ext(1, this->max_degree, this->max_grid_degree); 
        OPERATOR::local_basis_stiffness<dim,2*dim> flux_basis_stiffness(1, this->max_degree, this->max_grid_degree); 
        OPERATOR::mapping_shape_functions<dim,2*dim> mapping_basis(1, this->max_grid_degree, this->max_grid_degree);
        OPERATOR::vol_projection_operator<dim,2*dim> soln_basis_projection_oper_int(1, this->max_degree, this->max_grid_degree); 
        OPERATOR::vol_projection_operator<dim,2*dim> soln_basis_projection_oper_ext(1, this->max_degree, this->max_grid_degree); 
         
        this->reinit_operators_for_cell_residual_loop(
            this->max_degree, this->max_degree, this->max_grid_degree, 
            soln_basis_int, soln_basis_ext, 
            flux_basis_int, flux_basis_ext, 
            flux_basis_stiffness, 
            soln_basis_projection_oper_int, soln_basis_projection_oper_ext,
            mapping_basis);

        auto metric_cell = this->high_order_grid->dof_handler_grid.begin_active();

        // Add right-hand side contributions this cell can compute
        if(compute_d2R)
        {
            //loop over cells solving for auxiliary rhs
            for (auto soln_cell = this->dof_handler.begin_active(); soln_cell != this->dof_handler.end(); ++soln_cell, ++metric_cell) {
                if (!soln_cell->is_locally_owned()) continue;
                this->template assemble_cell_residual_and_ad_derivatives<codi_HessianComputationType>(
                    soln_cell,
                    metric_cell,
                    compute_dRdW, compute_dRdX, compute_d2R,
                    fe_values_collection_volume,
                    fe_values_collection_face_int,
                    fe_values_collection_face_ext,
                    fe_values_collection_subface,
                    fe_values_collection_volume_lagrange,
                    soln_basis_int,
                    soln_basis_ext,
                    flux_basis_int,
                    flux_basis_ext,
                    flux_basis_stiffness,
                    soln_basis_projection_oper_int,
                    soln_basis_projection_oper_ext,
                    mapping_basis,
                    true,
                    this->right_hand_side,
                    this->auxiliary_right_hand_side);
            } // end of cell loop
        }
        else if(compute_dRdW || compute_dRdX)
        {
            //loop over cells solving for auxiliary rhs
            for (auto soln_cell = this->dof_handler.begin_active(); soln_cell != this->dof_handler.end(); ++soln_cell, ++metric_cell) {
                if (!soln_cell->is_locally_owned()) continue;
                this->template assemble_cell_residual_and_ad_derivatives<codi_JacobianComputationType>(
                    soln_cell,
                    metric_cell,
                    compute_dRdW, compute_dRdX, compute_d2R,
                    fe_values_collection_volume,
                    fe_values_collection_face_int,
                    fe_values_collection_face_ext,
                    fe_values_collection_subface,
                    fe_values_collection_volume_lagrange,
                    soln_basis_int,
                    soln_basis_ext,
                    flux_basis_int,
                    flux_basis_ext,
                    flux_basis_stiffness,
                    soln_basis_projection_oper_int,
                    soln_basis_projection_oper_ext,
                    mapping_basis,
                    true,
                    this->right_hand_side,
                    this->auxiliary_right_hand_side);
            } // end of cell loop
        }
        else
        {
            //loop over cells solving for auxiliary rhs
            for (auto soln_cell = this->dof_handler.begin_active(); soln_cell != this->dof_handler.end(); ++soln_cell, ++metric_cell) {
                if (!soln_cell->is_locally_owned()) continue;
                this->template assemble_cell_residual_and_ad_derivatives<double>(
                    soln_cell,
                    metric_cell,
                    compute_dRdW, compute_dRdX, compute_d2R,
                    fe_values_collection_volume,
                    fe_values_collection_face_int,
                    fe_values_collection_face_ext,
                    fe_values_collection_subface,
                    fe_values_collection_volume_lagrange,
                    soln_basis_int,
                    soln_basis_ext,
                    flux_basis_int,
                    flux_basis_ext,
                    flux_basis_stiffness,
                    soln_basis_projection_oper_int,
                    soln_basis_projection_oper_ext,
                    mapping_basis,
                    true,
                    this->right_hand_side,
                    this->auxiliary_right_hand_side);
            } // end of cell loop
        }

        for(int idim=0; idim<dim; idim++){
            //compress auxiliary rhs for solution transfer across mpi ranks
            this->auxiliary_right_hand_side[idim].compress(dealii::VectorOperation::add);
            //update ghost values
            this->auxiliary_right_hand_side[idim].update_ghost_values();

            //solve for auxiliary solution for each dimension
            if(this->all_parameters->use_inverse_mass_on_the_fly)
                this->apply_inverse_global_mass_matrix(this->auxiliary_right_hand_side[idim], this->auxiliary_solution[idim], true);
            else
                this->global_inverse_mass_matrix_auxiliary.vmult(this->auxiliary_solution[idim], this->auxiliary_right_hand_side[idim]);

            //update ghost values of auxiliary solution
            this->auxiliary_solution[idim].update_ghost_values();
        }
    }//end of if statement for diffusive
    else if (this->use_auxiliary_eq && (this->all_parameters->ode_solver_param.ode_solver_type == ODE_enum::implicit_solver)) {
        pcout << "ERROR: " << "auxiliary currently only works for explicit time advancement. Aborting..." << std::endl;
        std::abort();
    } else {
        // Do nothing
    }
}

/**************************************************
 *
 *         AUXILIARY RESIDUAL FUNCTIONS
 *
 **************************************************/

template <int dim, int nstate, typename real, typename MeshType>
template <typename adtype>
void DGStrong<dim,nstate,real,MeshType>::assemble_volume_term_auxiliary_equation(
    const std::array<std::vector<adtype>,nstate> &soln_coeff,
    const unsigned int poly_degree,
    OPERATOR::basis_functions<dim,2*dim> &soln_basis,
    OPERATOR::basis_functions<dim,2*dim> &flux_basis,
    OPERATOR::metric_operators<adtype,dim,2*dim> &metric_oper,
    dealii::Tensor<1,dim,std::vector<adtype>> &local_auxiliary_RHS)
{
    //Please see header file for exact formula we are solving.
    const unsigned int n_quad_pts  = this->volume_quadrature_collection[poly_degree].size();
    const unsigned int n_dofs_cell = this->fe_collection[poly_degree].dofs_per_cell;
    const unsigned int n_shape_fns = n_dofs_cell / nstate;
    const std::vector<double> &quad_weights = this->volume_quadrature_collection[poly_degree].get_weights();

    //Interpolate each state to the quadrature points using sum-factorization
    //with the basis functions in each reference direction.
    for(int istate=0; istate<nstate; istate++){
        std::vector<adtype> soln_at_q(n_quad_pts);
        //interpolate soln coeff to volume cubature nodes
        soln_basis.matrix_vector_mult_1D(soln_coeff[istate], soln_at_q,
                                         soln_basis.oneD_vol_operator);
        //the volume integral for the auxiliary equation is the physical integral of the physical gradient of the solution.
        //That is, we need to physically integrate (we have determinant of Jacobian cancel) the Eq. (12) (with u for chi) in
        //Cicchino, Alexander, et al. "Provably stable flux reconstruction high-order methods on curvilinear elements." Journal of Computational Physics 463 (2022): 111259.

        //apply gradient of reference basis functions on the solution at volume cubature nodes
        dealii::Tensor<1,dim,std::vector<adtype>> ref_gradient_basis_fns_times_soln;
        for(int idim=0; idim<dim; idim++){
            ref_gradient_basis_fns_times_soln[idim].resize(n_quad_pts);
        }
        flux_basis.gradient_matrix_vector_mult_1D(soln_at_q, ref_gradient_basis_fns_times_soln,
                                                  flux_basis.oneD_vol_operator,
                                                  flux_basis.oneD_grad_operator);
        //transform the gradient into a physical gradient operator scaled by determinant of metric Jacobian
        //then apply the inner product in each direction
        for(int idim=0; idim<dim; idim++){
            std::vector<adtype> phys_gradient_u(n_quad_pts);
            for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
                for(int jdim=0; jdim<dim; jdim++){
                    //transform into the physical gradient
                    phys_gradient_u[iquad] += metric_oper.metric_cofactor_vol[idim][jdim][iquad]
                                                 * ref_gradient_basis_fns_times_soln[jdim][iquad];
                }
            }
            //Note that we let the determiant of the metric Jacobian cancel off between the integral and physical gradient
            std::vector<adtype> rhs(n_shape_fns);
            soln_basis.inner_product_1D(phys_gradient_u, quad_weights,
                                        rhs,
                                        soln_basis.oneD_vol_operator,
                                        false, 1.0);//it's added since auxiliary is EQUAL to the gradient of the soln

            //write the the auxiliary rhs for the test function.
            for(unsigned int ishape=0; ishape<n_shape_fns; ishape++){
                local_auxiliary_RHS[idim][istate*n_shape_fns + ishape] += rhs[ishape];
            }
        }
    }
}

template <int dim, int nstate, typename real, typename MeshType>
template <typename adtype>
void DGStrong<dim,nstate,real,MeshType>::assemble_boundary_term_auxiliary_equation(
    const unsigned int                                 iface,
    const dealii::types::global_dof_index              current_cell_index,
    const std::array<std::vector<adtype>,nstate>       &soln_coeff,
    const unsigned int                                 poly_degree,
    const unsigned int                                 boundary_id,
    OPERATOR::basis_functions<dim,2*dim>               &soln_basis,
    OPERATOR::metric_operators<adtype,dim,2*dim>       &metric_oper,
    const Physics::PhysicsBase<dim, nstate, adtype>                    &pde_physics,
    const NumericalFlux::NumericalFluxDissipative<dim, nstate, adtype> &diss_num_flux,
    dealii::Tensor<1,dim,std::vector<adtype>>          &local_auxiliary_RHS)
{
    (void) current_cell_index;

    const unsigned int n_face_quad_pts = this->face_quadrature_collection[poly_degree].size();
    const unsigned int n_quad_pts_vol  = this->volume_quadrature_collection[poly_degree].size();
    const unsigned int n_dofs          = this->fe_collection[poly_degree].dofs_per_cell;
    const unsigned int n_shape_fns     = n_dofs / nstate;

    //Interpolate soln to facet, and gradient to facet.
    std::array<std::vector<adtype>,nstate> soln_at_surf_q;
    std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate> ref_grad_soln_at_vol_q;
    for(int istate=0; istate<nstate; ++istate){
        //allocate
        soln_at_surf_q[istate].resize(n_face_quad_pts);
        //solve soln at facet cubature nodes
        soln_basis.matrix_vector_mult_surface_1D(iface, soln_coeff[istate], soln_at_surf_q[istate],
                                                 soln_basis.oneD_surf_operator,
                                                 soln_basis.oneD_vol_operator);
        //solve reference gradient of soln at facet cubature nodes
        for(int idim=0; idim<dim; idim++){
            ref_grad_soln_at_vol_q[istate][idim].resize(n_quad_pts_vol);
        }
        soln_basis.gradient_matrix_vector_mult_1D(soln_coeff[istate], ref_grad_soln_at_vol_q[istate],
                                                  soln_basis.oneD_vol_operator,
                                                  soln_basis.oneD_grad_operator);
    }

    // Get physical gradient of solution on the surface
    std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate> phys_grad_soln_at_surf_q;
    for(int istate=0; istate<nstate; istate++){
        //transform the gradient into a physical gradient operator
        for(int idim=0; idim<dim; idim++){
            std::vector<adtype> phys_gradient_u(n_quad_pts_vol);
            for(unsigned int iquad=0; iquad<n_quad_pts_vol; iquad++){
                for(int jdim=0; jdim<dim; jdim++){
                    //transform into the physical gradient
                    phys_gradient_u[iquad] += metric_oper.metric_cofactor_vol[idim][jdim][iquad]
                                                 * ref_grad_soln_at_vol_q[istate][jdim][iquad];
                }
                phys_gradient_u[iquad] /= metric_oper.det_Jac_vol[iquad];
            }
            phys_grad_soln_at_surf_q[istate][idim].resize(n_face_quad_pts);
            //interpolate physical volume gradient of the solution to the surface
            soln_basis.matrix_vector_mult_surface_1D(iface, phys_gradient_u, phys_grad_soln_at_surf_q[istate][idim],
                                                     soln_basis.oneD_surf_operator,
                                                     soln_basis.oneD_vol_operator);
        }
    }

    //evaluate physical facet fluxes dot product with physical unit normal scaled by determinant of metric facet Jacobian
    //the outward reference normal dircetion.
    const dealii::Tensor<1,dim,double> unit_ref_normal_int = dealii::GeometryInfo<dim>::unit_normal_vector[iface];
    std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate> surf_num_flux_minus_surf_soln_dot_normal;
    for(unsigned int iquad=0; iquad<n_face_quad_pts; iquad++){
        //Copy Metric Cofactor on the facet in a way can use for transforming Tensor Blocks to reference space
        //The way it is stored in metric_operators is to use sum-factorization in each direction,
        //but here it is cleaner to apply a reference transformation in each Tensor block returned by physics.
        //Note that for a conforming mesh, the facet metric cofactor matrix is the same from either interioir or exterior metric terms. 
        //This is verified for the metric computations in: unit_tests/operator_tests/surface_conforming_test.cpp
        dealii::Tensor<2,dim,adtype> metric_cofactor_surf;
        for(int idim=0; idim<dim; idim++){
            for(int jdim=0; jdim<dim; jdim++){
                metric_cofactor_surf[idim][jdim] = metric_oper.metric_cofactor_surf[idim][jdim][iquad];
            }
        }
        std::array<adtype,nstate> soln_state;
        std::array<dealii::Tensor<1,dim,adtype>,nstate> phys_grad_soln_state;
        for(int istate=0; istate<nstate; istate++){
            soln_state[istate] = soln_at_surf_q[istate][iquad];
            for(int idim=0; idim<dim; idim++){
                phys_grad_soln_state[istate][idim] = phys_grad_soln_at_surf_q[istate][idim][iquad];
            }
        }
        //numerical fluxes
        dealii::Tensor<1,dim,adtype> unit_phys_normal_int;
        metric_oper.transform_reference_to_physical(unit_ref_normal_int,
                                                    metric_cofactor_surf,
                                                    unit_phys_normal_int);
        adtype face_Jac_norm_scaled = 0.0;
        for(int idim=0; idim<dim; idim++){
            face_Jac_norm_scaled += unit_phys_normal_int[idim] * unit_phys_normal_int[idim];
        }
        face_Jac_norm_scaled = sqrt(face_Jac_norm_scaled);
        unit_phys_normal_int /= face_Jac_norm_scaled;//normalize it. 

        std::array<adtype,nstate> soln_boundary;
        std::array<dealii::Tensor<1,dim,adtype>,nstate> grad_soln_boundary;
        dealii::Point<dim,adtype> surf_flux_node;
        for(int idim=0; idim<dim; idim++){
            surf_flux_node[idim] = metric_oper.flux_nodes_surf[iface][idim][iquad];
        }
        pde_physics.boundary_face_values (boundary_id, surf_flux_node, unit_phys_normal_int, soln_state, phys_grad_soln_state, soln_boundary, grad_soln_boundary);

        std::array<adtype,nstate> diss_soln_num_flux;
        diss_soln_num_flux = diss_num_flux.evaluate_solution_flux(soln_state, soln_boundary, unit_phys_normal_int);

        for(int istate=0; istate<nstate; istate++){
            for(int idim=0; idim<dim; idim++){
                //allocate
                if(iquad == 0){
                    surf_num_flux_minus_surf_soln_dot_normal[istate][idim].resize(n_face_quad_pts);
                }
                //solve
                surf_num_flux_minus_surf_soln_dot_normal[istate][idim][iquad]
                    = (diss_soln_num_flux[istate] - soln_at_surf_q[istate][iquad]) * unit_phys_normal_int[idim] * face_Jac_norm_scaled;
            }
        }
    }
    //solve residual and set
    const std::vector<double> &surf_quad_weights = this->face_quadrature_collection[poly_degree].get_weights();
    for(int istate=0; istate<nstate; istate++){
        for(int idim=0; idim<dim; idim++){
            std::vector<adtype> rhs(n_shape_fns);

            soln_basis.inner_product_surface_1D(iface, 
                                                surf_num_flux_minus_surf_soln_dot_normal[istate][idim],
                                                surf_quad_weights, rhs,
                                                soln_basis.oneD_surf_operator,
                                                soln_basis.oneD_vol_operator,
                                                false, 1.0);//it's added since auxiliary is EQUAL to the gradient of the soln
            for(unsigned int ishape=0; ishape<n_shape_fns; ishape++){
                local_auxiliary_RHS[idim][istate*n_shape_fns + ishape] += rhs[ishape]; 
            }
        }
    }
}
/*********************************************************************************/
template <int dim, int nstate, typename real, typename MeshType>
template <typename adtype>
void DGStrong<dim,nstate,real,MeshType>::assemble_face_term_auxiliary_equation(
    const unsigned int                                 iface, 
    const unsigned int                                 neighbor_iface,
    const dealii::types::global_dof_index              current_cell_index,
    const dealii::types::global_dof_index              neighbor_cell_index,
    const std::array<std::vector<adtype>,nstate>       &soln_coeff_int,
    const std::array<std::vector<adtype>,nstate>       &soln_coeff_ext,
    const unsigned int                                 poly_degree_int, 
    const unsigned int                                 poly_degree_ext,
    OPERATOR::basis_functions<dim,2*dim>               &soln_basis_int,
    OPERATOR::basis_functions<dim,2*dim>               &soln_basis_ext,
    OPERATOR::metric_operators<adtype,dim,2*dim>       &metric_oper_int,
    const Physics::PhysicsBase<dim, nstate, adtype>                    &/*pde_physics*/,
    const NumericalFlux::NumericalFluxDissipative<dim, nstate, adtype> &diss_num_flux,
    dealii::Tensor<1,dim,std::vector<adtype>>          &local_auxiliary_RHS_int,
    dealii::Tensor<1,dim,std::vector<adtype>>          &local_auxiliary_RHS_ext)
{
    (void) current_cell_index;
    (void) neighbor_cell_index;

    const unsigned int n_face_quad_pts = this->face_quadrature_collection[poly_degree_int].size();//assume interior cell does the work

    const unsigned int n_dofs_int = this->fe_collection[poly_degree_int].dofs_per_cell;
    const unsigned int n_dofs_ext = this->fe_collection[poly_degree_ext].dofs_per_cell;

    const unsigned int n_shape_fns_int = n_dofs_int / nstate;
    const unsigned int n_shape_fns_ext = n_dofs_ext / nstate;

    //Interpolate soln modal coefficients to the facet
    std::array<std::vector<adtype>,nstate> soln_at_surf_q_int;
    std::array<std::vector<adtype>,nstate> soln_at_surf_q_ext;
    for(int istate=0; istate<nstate; ++istate){
        //allocate
        soln_at_surf_q_int[istate].resize(n_face_quad_pts);
        soln_at_surf_q_ext[istate].resize(n_face_quad_pts);
        //solve soln at facet cubature nodes
        soln_basis_int.matrix_vector_mult_surface_1D(iface,
                                                     soln_coeff_int[istate], soln_at_surf_q_int[istate],
                                                     soln_basis_int.oneD_surf_operator,
                                                     soln_basis_int.oneD_vol_operator);
        soln_basis_ext.matrix_vector_mult_surface_1D(neighbor_iface,
                                                     soln_coeff_ext[istate], soln_at_surf_q_ext[istate],
                                                     soln_basis_ext.oneD_surf_operator,
                                                     soln_basis_ext.oneD_vol_operator);
    }

    //evaluate physical facet fluxes dot product with physical unit normal scaled by determinant of metric facet Jacobian
    //the outward reference normal dircetion.
    const dealii::Tensor<1,dim,double> unit_ref_normal_int = dealii::GeometryInfo<dim>::unit_normal_vector[iface];
    std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate> surf_num_flux_minus_surf_soln_int_dot_normal;
    std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate> surf_num_flux_minus_surf_soln_ext_dot_normal;
    for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {
        //Copy Metric Cofactor on the facet in a way can use for transforming Tensor Blocks to reference space
        //The way it is stored in metric_operators is to use sum-factorization in each direction,
        //but here it is cleaner to apply a reference transformation in each Tensor block returned by physics.
        //Note that for a conforming mesh, the facet metric cofactor matrix is the same from either interioir or exterior metric terms. 
        //This is verified for the metric computations in: unit_tests/operator_tests/surface_conforming_test.cpp
        dealii::Tensor<2,dim,adtype> metric_cofactor_surf;
        for(int idim=0; idim<dim; idim++){
            for(int jdim=0; jdim<dim; jdim++){
                metric_cofactor_surf[idim][jdim] = metric_oper_int.metric_cofactor_surf[idim][jdim][iquad];
            }
        }
        //numerical fluxes
        dealii::Tensor<1,dim,adtype> unit_phys_normal_int;
        metric_oper_int.transform_reference_to_physical(unit_ref_normal_int,
                                                        metric_cofactor_surf,
                                                        unit_phys_normal_int);
        adtype face_Jac_norm_scaled = 0.0;
        for(int idim=0; idim<dim; idim++){
            face_Jac_norm_scaled += unit_phys_normal_int[idim] * unit_phys_normal_int[idim];
        }
        face_Jac_norm_scaled = sqrt(face_Jac_norm_scaled);
        unit_phys_normal_int /= face_Jac_norm_scaled;//normalize it. 

        std::array<adtype,nstate> diss_soln_num_flux;
        std::array<adtype,nstate> soln_state_int;
        std::array<adtype,nstate> soln_state_ext;
        for(int istate=0; istate<nstate; istate++){
            soln_state_int[istate] = soln_at_surf_q_int[istate][iquad];
            soln_state_ext[istate] = soln_at_surf_q_ext[istate][iquad];
        }
        diss_soln_num_flux = diss_num_flux.evaluate_solution_flux(soln_state_int, soln_state_ext, unit_phys_normal_int);

        for(int istate=0; istate<nstate; istate++){
            for(int idim=0; idim<dim; idim++){
                //allocate
                if(iquad == 0){
                    surf_num_flux_minus_surf_soln_int_dot_normal[istate][idim].resize(n_face_quad_pts);
                    surf_num_flux_minus_surf_soln_ext_dot_normal[istate][idim].resize(n_face_quad_pts);
                }
                //solve
                surf_num_flux_minus_surf_soln_int_dot_normal[istate][idim][iquad]
                    = (diss_soln_num_flux[istate] - soln_at_surf_q_int[istate][iquad]) * unit_phys_normal_int[idim] * face_Jac_norm_scaled;

                surf_num_flux_minus_surf_soln_ext_dot_normal[istate][idim][iquad]
                    = (diss_soln_num_flux[istate] - soln_at_surf_q_ext[istate][iquad]) * (- unit_phys_normal_int[idim]) * face_Jac_norm_scaled;
            }
        }
    }
    //solve residual and set
    const std::vector<double> &surf_quad_weights = this->face_quadrature_collection[poly_degree_int].get_weights();
    for(int istate=0; istate<nstate; istate++){
        for(int idim=0; idim<dim; idim++){
            std::vector<adtype> rhs_int(n_shape_fns_int);

            soln_basis_int.inner_product_surface_1D(iface, 
                                                    surf_num_flux_minus_surf_soln_int_dot_normal[istate][idim],
                                                    surf_quad_weights, rhs_int,
                                                    soln_basis_int.oneD_surf_operator,
                                                    soln_basis_int.oneD_vol_operator,
                                                    false, 1.0);//it's added since auxiliary is EQUAL to the gradient of the soln

            for(unsigned int ishape=0; ishape<n_shape_fns_int; ishape++){
                local_auxiliary_RHS_int[idim][istate*n_shape_fns_int + ishape] += rhs_int[ishape]; 
            }
            std::vector<adtype> rhs_ext(n_shape_fns_ext);

            soln_basis_ext.inner_product_surface_1D(neighbor_iface, 
                                                    surf_num_flux_minus_surf_soln_ext_dot_normal[istate][idim],
                                                    surf_quad_weights, rhs_ext,
                                                    soln_basis_ext.oneD_surf_operator,
                                                    soln_basis_ext.oneD_vol_operator,
                                                    false, 1.0);//it's added since auxiliary is EQUAL to the gradient of the soln

            for(unsigned int ishape=0; ishape<n_shape_fns_ext; ishape++){
                local_auxiliary_RHS_ext[idim][istate*n_shape_fns_ext + ishape] += rhs_ext[ishape]; 
            }
        }
    }
}

/****************************************************
*
* PRIMARY EQUATIONS STRONG FORM
*
****************************************************/
template <int dim, int nstate, typename real, typename MeshType>
template <typename adtype>
void DGStrong<dim,nstate,real,MeshType>::assemble_volume_term_strong(
    typename dealii::DoFHandler<dim>::active_cell_iterator             /*cell*/,
    const dealii::types::global_dof_index                              /*current_cell_index*/,
    const std::array<std::vector<adtype>,nstate>                       &soln_coeff,
    const std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate> &/*aux_soln_coeff*/,
    const unsigned int                                                 poly_degree,
    OPERATOR::basis_functions<dim,2*dim>                               &soln_basis,
    OPERATOR::basis_functions<dim,2*dim>                               &/*flux_basis*/,
    OPERATOR::local_basis_stiffness<dim,2*dim>                         &/*flux_basis_stiffness*/,
    OPERATOR::vol_projection_operator<dim,2*dim>                       &/*soln_basis_projection_oper*/,
    OPERATOR::metric_operators<adtype,dim,2*dim>                       &metric_oper,
    const Physics::PhysicsBase<dim, nstate, adtype>                    &/*pde_physics*/,
    std::vector<adtype>                                                &local_rhs_int_cell)
{
    const unsigned int n_quad_pts  = this->volume_quadrature_collection[poly_degree].size();
    const std::vector<double> &oneD_vol_quad_weights = this->oneD_quadrature_collection[poly_degree].get_weights();


    std::array<std::vector<adtype>,nstate> soln_at_q;
    std::array<std::vector<adtype>,nstate> half_soln_sqr_at_q;
    std::array<std::vector<adtype>,nstate> soln_grad_at_q;
    std::array<std::vector<adtype>,nstate> soln_hess_at_q;
    // Interpolate each state to the quadrature points using sum-factorization
    // with the basis functions in each reference direction.
    for(int istate=0; istate<nstate; istate++){
        soln_at_q[istate].resize(n_quad_pts);
        half_soln_sqr_at_q[istate].resize(n_quad_pts);
        soln_grad_at_q[istate].resize(n_quad_pts);
        soln_hess_at_q[istate].resize(n_quad_pts);
        soln_basis.matrix_vector_mult_1D(soln_coeff[istate], soln_at_q[istate],
                                         soln_basis.oneD_vol_operator);
        soln_basis.matrix_vector_mult_1D(soln_coeff[istate], soln_grad_at_q[istate],
                                         soln_basis.oneD_grad_operator);
        soln_basis.matrix_vector_mult_1D(soln_coeff[istate], soln_hess_at_q[istate],
                                         soln_basis.oneD_second_der_operator);
        for(unsigned int iquad=0; iquad<n_quad_pts; ++iquad)
        {
            half_soln_sqr_at_q[istate][iquad] = pow(soln_at_q[istate][iquad],2)/2.0;
        }
    }

    const double jacdet_double = getValue(metric_oper.det_Jac_vol[0]);
    soln_basis.inner_product_1D(
            soln_at_q[0],
            oneD_vol_quad_weights,
            local_rhs_int_cell,
            soln_basis.oneD_grad_operator,
            true,
            this->c_ks);
    
    soln_basis.inner_product_1D(
            half_soln_sqr_at_q[0],
            oneD_vol_quad_weights,
            local_rhs_int_cell,
            soln_basis.oneD_grad_operator,
            true,
            1.0);
    
    soln_basis.inner_product_1D(
            soln_grad_at_q[0],
            oneD_vol_quad_weights,
            local_rhs_int_cell,
            soln_basis.oneD_grad_operator,
            true,
            1.0/jacdet_double);
    
    soln_basis.inner_product_1D(
            soln_hess_at_q[0],
            oneD_vol_quad_weights,
            local_rhs_int_cell,
            soln_basis.oneD_second_der_operator,
            true,
            -1.0/pow(jacdet_double,3));
}

template <int dim, int nstate, typename real, typename MeshType>
template <typename adtype>
adtype DGStrong<dim,nstate,real,MeshType>::conv_num_flux_advection(
    const adtype &soln_int, 
    const adtype &soln_ext,
    const dealii::Tensor<1,dim,double> &normal_int) const
{
    adtype u_L = 0;
    adtype u_R = 0;
    if(normal_int[0]>0)
    {
        u_L = soln_int;
        u_R = soln_ext;
    }
    else
    {
        u_L = soln_ext;
        u_R = soln_int; 
    }

    if(this->c_ks>=0)
    {
        return this->c_ks*u_L;
    }
    else
    {
        return this->c_ks*u_R;
    }
}

template <int dim, int nstate, typename real, typename MeshType>
template <typename adtype>
adtype DGStrong<dim,nstate,real,MeshType>::conv_num_flux_burgers(
    const adtype &soln_int, 
    const adtype &soln_ext,
    const dealii::Tensor<1,dim,double> &normal_int) const
{
    adtype u_L = 0;
    adtype u_R = 0;
    if(normal_int[0]>0)
    {
        u_L = soln_int;
        u_R = soln_ext;
    }
    else
    {
        u_L = soln_ext;
        u_R = soln_int; 
    }
    
    adtype u_star = 0;

    if(u_L>=u_R)
    {
        const adtype avgval = 0.5*(u_L+u_R);
        if(avgval > 0) {u_star = u_L;}
        else {u_star = u_R;}
    }
    else
    {
        if(0<u_L) {u_star = u_L;}
        else if(u_R<0) {u_star = u_R;}
        else {u_star = 0;}
    }

    return 0.5*pow(u_star,2);
}

template <int dim, int nstate, typename real, typename MeshType>
template <typename adtype>
void DGStrong<dim,nstate,real,MeshType>::form_face_term_ks(
    const unsigned int                            iface, 
    const unsigned int                            /*poly_degree*/,
    OPERATOR::basis_functions<dim,2*dim>          &soln_basis,
    const double                                  jacdet_double,
    const std::vector<double>                     &face_quad_weights,
    const dealii::Tensor<1,dim,double> &normal_face,
    const std::array<std::vector<adtype>,nstate> &c_soln_at_surf_star,
    const std::array<std::vector<adtype>,nstate> &half_sqr_soln_at_surf_star,
    const std::array<std::vector<adtype>,nstate> &soln_grad_avg,
    const std::array<std::vector<adtype>,nstate> &soln_hess_avg,
    const std::array<std::vector<adtype>,nstate> &soln_3rd_der_avg,
    const std::array<std::vector<adtype>,nstate> &soln_jump,
    const std::array<std::vector<adtype>,nstate> &soln_grad_jump,
    const bool is_interior_face,
    std::vector<adtype>                                                &local_rhs_cell) const
{
    const double delta_ip_sipg = 0.0;
    const double sigma_4th_order = 30.0/pow(jacdet_double,3);
    const double tau_4th_order = 20.0/jacdet_double;

    soln_basis.inner_product_surface_1D(iface, c_soln_at_surf_star[0], 
                                    face_quad_weights, local_rhs_cell, 
                                    soln_basis.oneD_surf_operator, 
                                    soln_basis.oneD_vol_operator,
                                    true, -1.0*normal_face[0]);//adding=true, scaled by factor=-1.0 bc subtract it
    
    soln_basis.inner_product_surface_1D(iface, half_sqr_soln_at_surf_star[0], 
                                    face_quad_weights, local_rhs_cell, 
                                    soln_basis.oneD_surf_operator, 
                                    soln_basis.oneD_vol_operator,
                                    true, -1.0*normal_face[0]);//adding=true, scaled by factor=-1.0 bc subtract it
    
    const double factorval = is_interior_face ? 0.5 : 1.0;
    soln_basis.inner_product_surface_1D(iface, soln_jump[0], 
                                    face_quad_weights, local_rhs_cell, 
                                    soln_basis.oneD_surf_grad_operator, 
                                    soln_basis.oneD_vol_operator,
                                    true, -1.0*factorval/jacdet_double);//adding=true, scaled by factor=-1.0 bc subtract it
    if(is_interior_face)
    {
        soln_basis.inner_product_surface_1D(iface, soln_grad_avg[0], 
                                        face_quad_weights, local_rhs_cell, 
                                        soln_basis.oneD_surf_operator, 
                                        soln_basis.oneD_vol_operator,
                                        true, -1.0*normal_face[0]/jacdet_double);//adding=true, scaled by factor=-1.0 bc subtract it
        
        soln_basis.inner_product_surface_1D(iface, soln_jump[0], 
                                        face_quad_weights, local_rhs_cell, 
                                        soln_basis.oneD_surf_operator, 
                                        soln_basis.oneD_vol_operator,
                                        true, 1.0*delta_ip_sipg*normal_face[0]);//adding=true, scaled by factor=-1.0 bc subtract it
    }
    
    soln_basis.inner_product_surface_1D(iface, soln_3rd_der_avg[0], 
                                    face_quad_weights, local_rhs_cell, 
                                    soln_basis.oneD_surf_operator, 
                                    soln_basis.oneD_vol_operator,
                                    true, -1.0*normal_face[0]/pow(jacdet_double,3));//adding=true, scaled by factor=-1.0 bc subtract it
    
    soln_basis.inner_product_surface_1D(iface, soln_jump[0], 
                                    face_quad_weights, local_rhs_cell, 
                                    soln_basis.oneD_surf_third_der_operator, 
                                    soln_basis.oneD_vol_operator,
                                    true, -1.0*factorval/pow(jacdet_double,3));//adding=true, scaled by factor=-1.0 bc subtract it
    
    soln_basis.inner_product_surface_1D(iface, soln_hess_avg[0], 
                                    face_quad_weights, local_rhs_cell, 
                                    soln_basis.oneD_surf_grad_operator, 
                                    soln_basis.oneD_vol_operator,
                                    true, 1.0*normal_face[0]/pow(jacdet_double,3));//adding=true, scaled by factor=-1.0 bc subtract it
    
    soln_basis.inner_product_surface_1D(iface, soln_grad_jump[0],
                                    face_quad_weights, local_rhs_cell, 
                                    soln_basis.oneD_surf_second_der_operator, 
                                    soln_basis.oneD_vol_operator,
                                    true, 1.0*factorval/pow(jacdet_double,3));//adding=true, scaled by factor=-1.0 bc subtract it
    
    soln_basis.inner_product_surface_1D(iface, soln_jump[0],
                                    face_quad_weights, local_rhs_cell, 
                                    soln_basis.oneD_surf_operator, 
                                    soln_basis.oneD_vol_operator,
                                    true, -1.0*sigma_4th_order*normal_face[0]);//adding=true, scaled by factor=-1.0 bc subtract it
    
    soln_basis.inner_product_surface_1D(iface, soln_grad_jump[0],
                                    face_quad_weights, local_rhs_cell, 
                                    soln_basis.oneD_surf_grad_operator, 
                                    soln_basis.oneD_vol_operator,
                                    true, -1.0*tau_4th_order*normal_face[0]/pow(jacdet_double,2));//adding=true, scaled by factor=-1.0 bc subtract it
}

template <int dim, int nstate, typename real, typename MeshType>
template <typename adtype>
void DGStrong<dim,nstate,real,MeshType>::assemble_boundary_term_strong(
    const unsigned int                                                 iface, 
    const dealii::types::global_dof_index                              /*current_cell_index*/,
    const std::array<std::vector<adtype>,nstate>                       &soln_coeff,
    const std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate> &/*aux_soln_coeff*/,
    const unsigned int                                                 /*boundary_id*/,
    const unsigned int                                                 poly_degree, 
    const real                                                         /*penalty*/,
    OPERATOR::basis_functions<dim,2*dim>                               &soln_basis,
    OPERATOR::basis_functions<dim,2*dim>                               &/*flux_basis*/,
    OPERATOR::vol_projection_operator<dim,2*dim>                       &/*soln_basis_projection_oper*/,
    OPERATOR::metric_operators<adtype,dim,2*dim>                       &metric_oper,
    const Physics::PhysicsBase<dim, nstate, adtype>                    &/*pde_physics*/,
    const NumericalFlux::NumericalFluxConvective<dim, nstate, adtype>  &/*conv_num_flux*/,
    const NumericalFlux::NumericalFluxDissipative<dim, nstate, adtype> &/*diss_num_flux*/,
    std::vector<adtype>                                                &local_rhs_cell)
{
    const unsigned int n_face_quad_pts  = this->face_quadrature_collection[poly_degree].size();
    const std::vector<double> &face_quad_weights = this->face_quadrature_collection[poly_degree].get_weights();

    std::array<std::vector<adtype>,nstate> soln_at_surf_q;
    std::array<std::vector<adtype>,nstate> soln_jump;
    std::array<std::vector<adtype>,nstate> soln_grad_jump;
    std::array<std::vector<adtype>,nstate> c_soln_at_surf_star;
    std::array<std::vector<adtype>,nstate> half_sqr_soln_at_surf_star;
    std::array<std::vector<adtype>,nstate> soln_grad_at_surf_q;
    std::array<std::vector<adtype>,nstate> soln_hess_at_surf_q;
    std::array<std::vector<adtype>,nstate> soln_3rd_der_at_surf_q;
    const dealii::Tensor<1,dim,double> unit_ref_normal_int = dealii::GeometryInfo<dim>::unit_normal_vector[iface];
    const double jacdet_double = getValue(metric_oper.det_Jac_vol[0]);
    const adtype soln_ext_at_q = 0.0;
    for(int istate=0; istate<nstate; ++istate){
        //allocate
        soln_at_surf_q[istate].resize(n_face_quad_pts);
        c_soln_at_surf_star[istate].resize(n_face_quad_pts);
        half_sqr_soln_at_surf_star[istate].resize(n_face_quad_pts);
        soln_grad_at_surf_q[istate].resize(n_face_quad_pts);
        soln_hess_at_surf_q[istate].resize(n_face_quad_pts);
        soln_3rd_der_at_surf_q[istate].resize(n_face_quad_pts);
        soln_jump[istate].resize(n_face_quad_pts);
        soln_grad_jump[istate].resize(n_face_quad_pts);
        //solve soln at facet cubature nodes
        soln_basis.matrix_vector_mult_surface_1D(iface,
                                                 soln_coeff[istate], soln_at_surf_q[istate],
                                                 soln_basis.oneD_surf_operator,
                                                 soln_basis.oneD_vol_operator);
        
        soln_basis.matrix_vector_mult_surface_1D(iface,
                                                 soln_coeff[istate], soln_grad_at_surf_q[istate],
                                                 soln_basis.oneD_surf_grad_operator,
                                                 soln_basis.oneD_vol_operator);
        
        soln_basis.matrix_vector_mult_surface_1D(iface,
                                                 soln_coeff[istate], soln_hess_at_surf_q[istate],
                                                 soln_basis.oneD_surf_second_der_operator,
                                                 soln_basis.oneD_vol_operator);
        
        soln_basis.matrix_vector_mult_surface_1D(iface,
                                                 soln_coeff[istate], soln_3rd_der_at_surf_q[istate],
                                                 soln_basis.oneD_surf_third_der_operator,
                                                 soln_basis.oneD_vol_operator);

        for(unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad)
        {
            c_soln_at_surf_star[istate][iquad] = conv_num_flux_advection(soln_at_surf_q[istate][iquad], soln_ext_at_q, unit_ref_normal_int);
            half_sqr_soln_at_surf_star[istate][iquad] = conv_num_flux_burgers(soln_at_surf_q[istate][iquad], soln_ext_at_q, unit_ref_normal_int);
            soln_jump[istate][iquad] = soln_at_surf_q[istate][iquad]*unit_ref_normal_int[0];
            soln_grad_jump[istate][iquad] = soln_grad_at_surf_q[istate][iquad]*unit_ref_normal_int[0];
        }

        const bool is_interior_face = false;
        form_face_term_ks(
            iface, 
            poly_degree,
            soln_basis,
            jacdet_double,
            face_quad_weights,
            unit_ref_normal_int,
            c_soln_at_surf_star,
            half_sqr_soln_at_surf_star,
            soln_grad_at_surf_q,
            soln_hess_at_surf_q,
            soln_3rd_der_at_surf_q,
            soln_jump,
            soln_grad_jump,
            is_interior_face,
            local_rhs_cell); 
    }
}


template <int dim, int nstate, typename real, typename MeshType>
template <typename adtype>
void DGStrong<dim,nstate,real,MeshType>::assemble_face_term_strong(
    const unsigned int                                                 iface, 
    const unsigned int                                                 neighbor_iface, 
    const dealii::types::global_dof_index                              /*current_cell_index*/,
    const dealii::types::global_dof_index                              /*neighbor_cell_index*/,
    const std::array<std::vector<adtype>,nstate>                       &soln_coeff_int,
    const std::array<std::vector<adtype>,nstate>                       &soln_coeff_ext,
    const std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate> &/*aux_soln_coeff_int*/,
    const std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate> &/*aux_soln_coeff_ext*/,
    const unsigned int                                                 poly_degree_int, 
    const unsigned int                                                 poly_degree_ext, 
    const real                                                         /*penalty*/,
    OPERATOR::basis_functions<dim,2*dim>                               &soln_basis_int,
    OPERATOR::basis_functions<dim,2*dim>                               &soln_basis_ext,
    OPERATOR::basis_functions<dim,2*dim>                               &/*flux_basis_int*/,
    OPERATOR::basis_functions<dim,2*dim>                               &/*flux_basis_ext*/,
    OPERATOR::vol_projection_operator<dim,2*dim>                       &/*soln_basis_projection_oper_int*/,
    OPERATOR::vol_projection_operator<dim,2*dim>                       &/*soln_basis_projection_oper_ext*/,
    OPERATOR::metric_operators<adtype,dim,2*dim>                       &metric_oper_int,
    OPERATOR::metric_operators<adtype,dim,2*dim>                       &/*metric_oper_ext*/,
    const Physics::PhysicsBase<dim, nstate, adtype>                    &/*pde_physics*/,
    const NumericalFlux::NumericalFluxConvective<dim, nstate, adtype>  &/*conv_num_flux*/,
    const NumericalFlux::NumericalFluxDissipative<dim, nstate, adtype> &/*diss_num_flux*/,
    std::vector<adtype>                                                  &local_rhs_int_cell,
    std::vector<adtype>                                                  &local_rhs_ext_cell)
{
    const unsigned int n_face_quad_pts = this->face_quadrature_collection[poly_degree_int].size();//assume interior cell does the work
    const std::vector<double> &surf_quad_weights = this->face_quadrature_collection[poly_degree_int].get_weights();
    const dealii::Tensor<1,dim,double> unit_ref_normal_int = dealii::GeometryInfo<dim>::unit_normal_vector[iface];
    const dealii::Tensor<1,dim,double> unit_ref_normal_ext = dealii::GeometryInfo<dim>::unit_normal_vector[neighbor_iface];

    std::array<std::vector<adtype>,nstate> soln_at_surf_q_int;
    std::array<std::vector<adtype>,nstate> soln_at_surf_q_ext;
    std::array<std::vector<adtype>,nstate> soln_jump; 
    std::array<std::vector<adtype>,nstate> soln_grad_jump;
    std::array<std::vector<adtype>,nstate> c_soln_at_surf_star;
    std::array<std::vector<adtype>,nstate> half_sqr_soln_at_surf_star;
    std::array<std::vector<adtype>,nstate> soln_grad_at_surf_q_int;
    std::array<std::vector<adtype>,nstate> soln_grad_at_surf_q_ext;
    std::array<std::vector<adtype>,nstate> soln_grad_avg;
    std::array<std::vector<adtype>,nstate> soln_hess_at_surf_q_int;
    std::array<std::vector<adtype>,nstate> soln_hess_at_surf_q_ext;
    std::array<std::vector<adtype>,nstate> soln_hess_avg;
    std::array<std::vector<adtype>,nstate> soln_3rd_der_at_surf_q_int;
    std::array<std::vector<adtype>,nstate> soln_3rd_der_at_surf_q_ext;
    std::array<std::vector<adtype>,nstate> soln_3rd_der_avg;
    const double jacdet_double = getValue(metric_oper_int.det_Jac_vol[0]);
    for(int istate=0; istate<nstate; ++istate){
        // allocate
        soln_at_surf_q_int[istate].resize(n_face_quad_pts);
        soln_at_surf_q_ext[istate].resize(n_face_quad_pts);
        soln_jump[istate].resize(n_face_quad_pts); 
        soln_grad_jump[istate].resize(n_face_quad_pts);
        c_soln_at_surf_star[istate].resize(n_face_quad_pts);
        half_sqr_soln_at_surf_star[istate].resize(n_face_quad_pts);
        soln_grad_at_surf_q_int[istate].resize(n_face_quad_pts);
        soln_grad_at_surf_q_ext[istate].resize(n_face_quad_pts);
        soln_grad_avg[istate].resize(n_face_quad_pts);
        soln_hess_at_surf_q_int[istate].resize(n_face_quad_pts);
        soln_hess_at_surf_q_ext[istate].resize(n_face_quad_pts);
        soln_hess_avg[istate].resize(n_face_quad_pts);
        soln_3rd_der_at_surf_q_int[istate].resize(n_face_quad_pts);
        soln_3rd_der_at_surf_q_ext[istate].resize(n_face_quad_pts);
        soln_3rd_der_avg[istate].resize(n_face_quad_pts);
        // solve soln at facet cubature nodes
        soln_basis_int.matrix_vector_mult_surface_1D(iface,
                                                     soln_coeff_int[istate], soln_at_surf_q_int[istate],
                                                     soln_basis_int.oneD_surf_operator,
                                                     soln_basis_int.oneD_vol_operator);
        soln_basis_ext.matrix_vector_mult_surface_1D(neighbor_iface,
                                                     soln_coeff_ext[istate], soln_at_surf_q_ext[istate],
                                                     soln_basis_ext.oneD_surf_operator,
                                                     soln_basis_ext.oneD_vol_operator);
        
        soln_basis_int.matrix_vector_mult_surface_1D(iface,
                                                 soln_coeff_int[istate], soln_grad_at_surf_q_int[istate],
                                                 soln_basis_int.oneD_surf_grad_operator,
                                                 soln_basis_int.oneD_vol_operator);
        soln_basis_ext.matrix_vector_mult_surface_1D(neighbor_iface,
                                                 soln_coeff_ext[istate], soln_grad_at_surf_q_ext[istate],
                                                 soln_basis_ext.oneD_surf_grad_operator,
                                                 soln_basis_ext.oneD_vol_operator);
        
        soln_basis_int.matrix_vector_mult_surface_1D(iface,
                                                 soln_coeff_int[istate], soln_hess_at_surf_q_int[istate],
                                                 soln_basis_int.oneD_surf_second_der_operator,
                                                 soln_basis_int.oneD_vol_operator);
        soln_basis_ext.matrix_vector_mult_surface_1D(neighbor_iface,
                                                 soln_coeff_ext[istate], soln_hess_at_surf_q_ext[istate],
                                                 soln_basis_ext.oneD_surf_second_der_operator,
                                                 soln_basis_ext.oneD_vol_operator);
        
        soln_basis_int.matrix_vector_mult_surface_1D(iface,
                                                 soln_coeff_int[istate], soln_3rd_der_at_surf_q_int[istate],
                                                 soln_basis_int.oneD_surf_third_der_operator,
                                                 soln_basis_int.oneD_vol_operator);
        soln_basis_ext.matrix_vector_mult_surface_1D(neighbor_iface,
                                                 soln_coeff_ext[istate], soln_3rd_der_at_surf_q_ext[istate],
                                                 soln_basis_ext.oneD_surf_third_der_operator,
                                                 soln_basis_ext.oneD_vol_operator);

        for(unsigned int iquad = 0; iquad<n_face_quad_pts; ++iquad)
        {
            c_soln_at_surf_star[istate][iquad] = conv_num_flux_advection(soln_at_surf_q_int[istate][iquad], soln_at_surf_q_ext[istate][iquad], unit_ref_normal_int);
            half_sqr_soln_at_surf_star[istate][iquad] = conv_num_flux_burgers(soln_at_surf_q_int[istate][iquad], soln_at_surf_q_ext[istate][iquad], unit_ref_normal_int);
            soln_jump[istate][iquad] = soln_at_surf_q_int[istate][iquad]*unit_ref_normal_int[0] + soln_at_surf_q_ext[istate][iquad]*unit_ref_normal_ext[0]; 
            soln_grad_jump[istate][iquad] = soln_grad_at_surf_q_int[istate][iquad]*unit_ref_normal_int[0] + soln_grad_at_surf_q_ext[istate][iquad]*unit_ref_normal_ext[0]; 
            soln_grad_avg[istate][iquad] = 0.5*(soln_grad_at_surf_q_int[istate][iquad] + soln_grad_at_surf_q_ext[istate][iquad]);
            soln_hess_avg[istate][iquad] = 0.5*(soln_hess_at_surf_q_int[istate][iquad] + soln_hess_at_surf_q_ext[istate][iquad]);
            soln_3rd_der_avg[istate][iquad] = 0.5*(soln_3rd_der_at_surf_q_int[istate][iquad] + soln_3rd_der_at_surf_q_ext[istate][iquad]);
            
            
        }
        const bool is_interior_face = true;
        form_face_term_ks(
            iface, 
            poly_degree_int,
            soln_basis_int,
            jacdet_double,
            surf_quad_weights,
            unit_ref_normal_int,
            c_soln_at_surf_star,
            half_sqr_soln_at_surf_star,
            soln_grad_avg,
            soln_hess_avg,
            soln_3rd_der_avg,
            soln_jump,
            soln_grad_jump,
            is_interior_face,
            local_rhs_int_cell);

        form_face_term_ks(
            neighbor_iface, 
            poly_degree_ext,
            soln_basis_ext,
            jacdet_double,
            surf_quad_weights,
            unit_ref_normal_ext,
            c_soln_at_surf_star,
            half_sqr_soln_at_surf_star,
            soln_grad_avg,
            soln_hess_avg,
            soln_3rd_der_avg,
            soln_jump,
            soln_grad_jump,
            is_interior_face,
            local_rhs_ext_cell);
    }
}

/*******************************************************
 *
 *                   ENTROPY STABLE BR2
 *
 *******************************************************/
template <int dim, int nstate, typename real, typename MeshType>
template <typename adtype>
void DGStrong<dim,nstate,real,MeshType>::apply_K_matrix(
    const std::array<std::vector<adtype>,nstate>                       &entropy_var_at_quads,
    const unsigned int                                                 n_quad_pts,
    const Physics::PhysicsBase<dim, nstate, adtype>                    &pde_physics,
    const std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate>   &T_in,
    std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate>         &T_out) const
{
    //Resize
    for(unsigned int s=0; s<nstate; ++s)
    {
        for(unsigned int d=0; d<dim; ++d)
        {
            T_out[s][d].resize(n_quad_pts);
        }
    }

    for(unsigned int q = 0; q<n_quad_pts; ++q)
    {
        std::array<dealii::Tensor<1,dim,adtype>,nstate> T_in_at_q;
        std::array<adtype,nstate> entropy_var_at_q;
        for(unsigned int s=0; s<nstate; ++s)
        {
            for(unsigned int d=0; d<dim; ++d)
            {
                T_in_at_q[s][d] = T_in[s][d][q];
            }
            entropy_var_at_q[s] = entropy_var_at_quads[s][q];
        }
        std::array<dealii::Tensor<1,dim,adtype>,nstate> T_out_at_q = pde_physics.dissipative_flux_entropy_based(entropy_var_at_q, T_in_at_q);
        for(unsigned int s=0; s<nstate; ++s)
        {
            for(unsigned int d=0; d<dim; ++d)
            {
                T_out[s][d][q] = T_out_at_q[s][d];
            }
        }
    }
}


// Computes \int_k G^h(dw/dx_d)*Kdd2ss2*Td2s2 d\Omega, where T is the quad values of G^h(\nabla entropy_var) or a lift polynomial of size nstate x dim.
template <int dim, int nstate, typename real, typename MeshType>
template <typename adtype>
void DGStrong<dim,nstate,real,MeshType>::entropystable_br2_compute_gradbasis_K_T_vol_integral(
    const std::array<std::vector<adtype>,nstate>                       &entropy_var_at_quads,
    const unsigned int                                                 n_quad_pts,
    const unsigned int                                                 n_dofs_cell,
    OPERATOR::basis_functions<dim,2*dim>                               &soln_basis,
    OPERATOR::metric_operators<adtype,dim,2*dim>                       &metric_oper,
    const std::vector<double>                                          &weight_vect,
    const Physics::PhysicsBase<dim, nstate, adtype>                    &pde_physics,
    const std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate>   &T,
    std::vector<adtype>                                                &integral_val) const
{
    const unsigned int n_shape_fns = n_dofs_cell / nstate; 
    
    // Form L = K*T
    std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate>   L;
    apply_K_matrix(
    entropy_var_at_quads,
    n_quad_pts,
    pde_physics,
    T,
    L);
    
    // Form M = cof(J)^T*L
    std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate>   M;
    for(unsigned int s=0; s<nstate; ++s)
    {
        metric_oper.transform_physical_to_reference_vector(
            L[s],
            metric_oper.metric_cofactor_vol,
            M[s]);
    }

    // Compute \int_k nabla basis * K*T d\Omega
    integral_val.resize(n_dofs_cell);
    std::vector<adtype> integral_val_1state(n_shape_fns);

    for(unsigned int s=0; s<nstate; ++s)
    {
        
        soln_basis.inner_product(
        M[s][0],
        weight_vect,
        integral_val_1state,
        soln_basis.oneD_grad_operator,
        soln_basis.oneD_vol_operator,
        soln_basis.oneD_vol_operator,
        false);
        
        if(dim>=2)
        {
            soln_basis.inner_product(
            M[s][1],
            weight_vect,
            integral_val_1state,
            soln_basis.oneD_vol_operator,
            soln_basis.oneD_grad_operator,
            soln_basis.oneD_vol_operator,
            true);
        }

        if(dim>=3)
        {
            soln_basis.inner_product(
            M[s][2],
            weight_vect,
            integral_val_1state,
            soln_basis.oneD_vol_operator,
            soln_basis.oneD_vol_operator,
            soln_basis.oneD_grad_operator,
            true);
        }
        
        // Put values in integral_val vector.
        const unsigned int start_index = s*n_shape_fns;
        for(unsigned int ishape=0; ishape<n_shape_fns; ++ishape)
        {
            integral_val[start_index + ishape] = integral_val_1state[ishape];
        }
    }

}


template <int dim, int nstate, typename real, typename MeshType>
template <typename adtype>
void DGStrong<dim,nstate,real,MeshType>::compute_physical_grad_entropy_var(
    const std::array<std::vector<adtype>,nstate>                       &entropy_var_coeff,
    OPERATOR::basis_functions<dim,2*dim>                               &soln_basis,
    OPERATOR::metric_operators<adtype,dim,2*dim>                       &metric_oper,
    const unsigned int                                                 n_quad_pts, 
    std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate>       &entropy_var_phys_grad) const
{
    // Entropy grad wrt reference coordinates
    std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate> entropy_var_ref_grad;
    for(unsigned int s=0; s<nstate; ++s)
    {
        for(unsigned int d=0; d<dim; ++d)
        {
            entropy_var_ref_grad[s][d].resize(n_quad_pts);
        }
        soln_basis.gradient_matrix_vector_mult_1D(entropy_var_coeff[s],
                                                  entropy_var_ref_grad[s],
                                                  soln_basis.oneD_vol_operator,
                                                  soln_basis.oneD_grad_operator);
    }
    
    // Entropy grad wrt physical coordinates
    for(unsigned int s=0; s<nstate; ++s)
    {
        metric_oper.transform_reference_to_physical_grad_vector(entropy_var_ref_grad[s],
                                                                metric_oper.metric_cofactor_vol,
                                                                metric_oper.det_Jac_vol,
                                                                entropy_var_phys_grad[s]);
    }
}

template <int dim, int nstate, typename real, typename MeshType>
template <typename adtype>
void DGStrong<dim,nstate,real,MeshType>::compute_lift_polynomial(
    const unsigned int iface,
    const std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate> &phi_at_face,
    const std::vector<adtype> &JxW_face,
    const std::vector<adtype> &JxW_vol,
    OPERATOR::basis_functions<dim,2*dim> &flux_basis,
    const unsigned int /*n_face_quad_pts*/,
    const unsigned int n_vol_quad_pts,
    const bool is_interior_face,
    std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate> &re_out_vol) const
{
    const unsigned int n_shape_fns_flux = n_vol_quad_pts; // collocated
    for(unsigned int s=0; s<nstate; ++s)
    {
        for(unsigned int d=0; d<dim; ++d)
        {
            re_out_vol[s][d].resize(n_shape_fns_flux);
        }
    }
    
    for(unsigned int s=0; s<nstate; ++s)
    {
        for(unsigned int d=0; d<dim; ++d)
        {
            flux_basis.inner_product_surface_1D_JxW(iface, 
                                                phi_at_face[s][d], 
                                                JxW_face, re_out_vol[s][d], 
                                                flux_basis.oneD_surf_operator, 
                                                flux_basis.oneD_vol_operator);
        }
    }
    
    const double mult_factor = is_interior_face ? 0.5 : 1.0;
    for(unsigned int s=0; s<nstate; ++s)
    {
        for(unsigned int d=0; d<dim; ++d)
        {
            for(unsigned int iquad = 0; iquad<n_vol_quad_pts; ++iquad)
            {
                re_out_vol[s][d][iquad]*= -mult_factor/JxW_vol[iquad];
            }
        }
    }
}
    
template <int dim, int nstate, typename real, typename MeshType>
template <typename adtype>
void DGStrong<dim,nstate,real,MeshType>::evaluate_face_integral(
    const unsigned int iface,
    const std::array<std::vector<adtype>,nstate> &sigma_dot_n_at_face,
    const std::vector<adtype> &JxW_face,
    OPERATOR::basis_functions<dim,2*dim> &soln_basis,
    const unsigned int n_dofs_cell,
    std::vector<adtype> &integral_val) const
{
    const unsigned int n_shape_fns = n_dofs_cell/nstate;
    integral_val.resize(n_dofs_cell);

    for(unsigned int s=0; s<nstate; ++s)
    {
        std::vector<adtype> integral_1state(n_shape_fns);
        soln_basis.inner_product_surface_1D_JxW(iface, sigma_dot_n_at_face[s], 
                                            JxW_face, integral_1state, 
                                            soln_basis.oneD_surf_operator, 
                                            soln_basis.oneD_vol_operator);
        const unsigned int start_index = s*n_shape_fns;
        for(unsigned int ishape = 0; ishape<n_shape_fns; ++ishape)
        {
            integral_val[start_index + ishape] = integral_1state[ishape];
        }
    }
}


template <int dim, int nstate, typename real, typename MeshType>
template <typename adtype>
void DGStrong<dim,nstate,real,MeshType>::interpolate_to_face(
    const unsigned int iface,
    const std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate> &T_at_vol,
    OPERATOR::basis_functions<dim,2*dim> &flux_basis,
    const unsigned int n_face_quad_pts,
    std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate> &T_at_face) const
{
    for(unsigned int s=0; s<nstate; ++s)
    {
        for(unsigned int d=0; d<dim; ++d)
        {
            T_at_face[s][d].resize(n_face_quad_pts);
        }
    }

    for(unsigned int s=0; s<nstate; ++s)
    {
        for(unsigned int d=0; d<dim; ++d)
        {
            flux_basis.matrix_vector_mult_surface_1D(iface, 
                                                     T_at_vol[s][d],
                                                     T_at_face[s][d],
                                                     flux_basis.oneD_surf_operator,
                                                     flux_basis.oneD_vol_operator);
        }
    }
}

template <int dim, int nstate, typename real, typename MeshType>
template <typename adtype>
void DGStrong<dim,nstate,real,MeshType>::assemble_volume_term_entropystable_br2(
    const unsigned int                                                 poly_degree,
    const std::array<std::vector<adtype>,nstate>                       &entropy_var_coeff,
    const std::array<std::vector<adtype>,nstate>                       &entropy_var_at_q,
    const unsigned int                                                  n_quad_pts,  
    const unsigned int                                                  n_dofs_cell, 
    OPERATOR::basis_functions<dim,2*dim>                               &soln_basis,
    OPERATOR::basis_functions<dim,2*dim>                               &/*flux_basis*/,
    OPERATOR::metric_operators<adtype,dim,2*dim>                       &metric_oper,
    const Physics::PhysicsBase<dim, nstate, adtype>                    &pde_physics,
    std::vector<adtype>                                                &vol_term) const
{
    const std::vector<double> &weight_vect = this->volume_quadrature_collection[poly_degree].get_weights();
    // Entropy grad wrt physical coordinates
    std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate> entropy_var_phys_grad;
    compute_physical_grad_entropy_var(
       entropy_var_coeff,
       soln_basis,
       metric_oper,
       n_quad_pts,
       entropy_var_phys_grad);

    entropystable_br2_compute_gradbasis_K_T_vol_integral(
    entropy_var_at_q,
    n_quad_pts,
    n_dofs_cell,
    soln_basis,
    metric_oper,
    weight_vect,
    pde_physics,
    entropy_var_phys_grad,
    vol_term);
}

template <int dim, int nstate, typename real, typename MeshType>
template <typename adtype>
void DGStrong<dim,nstate,real,MeshType>::assemble_face_term_entropystable_br2(
    const unsigned int                                                 iface_int,
    const unsigned int                                                 iface_ext,
    const std::array<std::vector<adtype>,nstate>                       &entropy_var_coeff_int,
    const std::array<std::vector<adtype>,nstate>                       &entropy_var_coeff_ext,
    const std::array<std::vector<adtype>,nstate>                       &entropy_var_at_vol_int,
    const std::array<std::vector<adtype>,nstate>                       &entropy_var_at_vol_ext,
    const std::array<std::vector<adtype>,nstate>                       &entropy_var_at_surf_int,
    const std::array<std::vector<adtype>,nstate>                       &entropy_var_at_surf_ext,
    const std::vector<dealii::Tensor<1,dim,adtype>>                    &unit_phys_normal_int,
    const std::vector<adtype>                                          &JxW_face,
    const unsigned int                                                  poly_degree_int,  
    const unsigned int                                                  poly_degree_ext,  
    const unsigned int                                                  n_vol_quad_pts_int,  
    const unsigned int                                                  n_vol_quad_pts_ext,  
    const unsigned int                                                  n_dofs_cell_int, 
    const unsigned int                                                  n_dofs_cell_ext, 
    const unsigned int                                                  n_face_quad_pts, 
    OPERATOR::basis_functions<dim,2*dim>                               &soln_basis_int,
    OPERATOR::basis_functions<dim,2*dim>                               &soln_basis_ext,
    OPERATOR::basis_functions<dim,2*dim>                               &flux_basis_int,
    OPERATOR::basis_functions<dim,2*dim>                               &flux_basis_ext,
    OPERATOR::metric_operators<adtype,dim,2*dim>                       &metric_oper_int,
    OPERATOR::metric_operators<adtype,dim,2*dim>                       &metric_oper_ext,
    const Physics::PhysicsBase<dim, nstate, adtype>                    &pde_physics,
    std::vector<adtype>                                                &face_term_int,
    std::vector<adtype>                                                &face_term_ext) const
{
    face_term_int.resize(n_dofs_cell_int);
    face_term_ext.resize(n_dofs_cell_ext);
    const std::vector<double> &vol_quad_weights_int = this->volume_quadrature_collection[poly_degree_int].get_weights();
    const std::vector<double> &vol_quad_weights_ext = this->volume_quadrature_collection[poly_degree_ext].get_weights();
    std::vector<adtype> JxW_vol_int(n_vol_quad_pts_int);
    std::vector<adtype> JxW_vol_ext(n_vol_quad_pts_ext);
    for(unsigned int iquad=0; iquad<n_vol_quad_pts_int; ++iquad)
    {
        JxW_vol_int[iquad] = metric_oper_int.det_Jac_vol[iquad]*vol_quad_weights_int[iquad];
    }
    for(unsigned int iquad=0; iquad<n_vol_quad_pts_ext; ++iquad)
    {
        JxW_vol_ext[iquad] = metric_oper_ext.det_Jac_vol[iquad]*vol_quad_weights_ext[iquad];
    }

    std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate> entropy_var_phys_grad_int;
    compute_physical_grad_entropy_var(
       entropy_var_coeff_int,
       soln_basis_int,
       metric_oper_int,
       n_vol_quad_pts_int,
       entropy_var_phys_grad_int);
    
    std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate> entropy_var_phys_grad_ext;
    compute_physical_grad_entropy_var(
       entropy_var_coeff_ext,
       soln_basis_ext,
       metric_oper_ext,
       n_vol_quad_pts_ext,
       entropy_var_phys_grad_ext);

    std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate> entropy_var_jump_at_face;
    for(unsigned int s=0; s<nstate; ++s)
    {
        for(unsigned int d=0; d<dim; ++d)
        {
            entropy_var_jump_at_face[s][d].resize(n_face_quad_pts);
        }
    }

    for(unsigned int s=0; s<nstate; ++s)
    {
        for(unsigned int d=0; d<dim; ++d)
        {
            for(unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad)
            {
                entropy_var_jump_at_face[s][d][iquad] = (entropy_var_at_surf_int[s][iquad] - entropy_var_at_surf_ext[s][iquad])*unit_phys_normal_int[iquad][d];
            }
        }
    }
    
    std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate> re_int;
    const bool is_interior_face = true;
    compute_lift_polynomial(
    iface_int,
    entropy_var_jump_at_face,
    JxW_face,
    JxW_vol_int,
    flux_basis_int,
    n_face_quad_pts,
    n_vol_quad_pts_int,
    is_interior_face,
    re_int);
    
    std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate> re_ext;
    compute_lift_polynomial(
    iface_ext,
    entropy_var_jump_at_face,
    JxW_face,
    JxW_vol_ext,
    flux_basis_ext,
    n_face_quad_pts,
    n_vol_quad_pts_ext,
    is_interior_face,
    re_ext);

    const double br2_factor = 2.0*dim + 0.1; // n_faces = 2*dim. br2_factor > n_faces for entropy stability.

    std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate> tensor_int;
    std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate> tensor_ext;
    for(unsigned int s=0; s<nstate; ++s)
    {
        for(unsigned int d=0; d<dim; ++d)
        {
            tensor_int[s][d].resize(n_vol_quad_pts_int);
            tensor_ext[s][d].resize(n_vol_quad_pts_ext);
        }
    }
    
    for(unsigned int q=0; q<n_vol_quad_pts_int; ++q)
    {
        for(unsigned int s=0; s<nstate; ++s)
        {
            for(unsigned int d=0; d<dim; ++d)
            {
                tensor_int[s][d][q] = entropy_var_phys_grad_int[s][d][q] + br2_factor*re_int[s][d][q];
            }
        }       
    }

    for(unsigned int q=0; q<n_vol_quad_pts_ext; ++q)
    {
        for(unsigned int s=0; s<nstate; ++s)
        {
            for(unsigned int d=0; d<dim; ++d)
            {
                tensor_ext[s][d][q] = entropy_var_phys_grad_ext[s][d][q] + br2_factor*re_ext[s][d][q];
            }
        }       
    }

    std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate> K_tensor_int;
    apply_K_matrix(
    entropy_var_at_vol_int,
    n_vol_quad_pts_int,
    pde_physics,
    tensor_int,
    K_tensor_int);
    
    std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate> K_tensor_ext;
    apply_K_matrix(
    entropy_var_at_vol_ext,
    n_vol_quad_pts_ext,
    pde_physics,
    tensor_ext,
    K_tensor_ext);

    std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate> sigma_at_face_int;
    std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate> sigma_at_face_ext;
    std::array<std::vector<adtype>,nstate> sigma_at_face_avg_dot_n_int;
    std::array<std::vector<adtype>,nstate> sigma_at_face_avg_dot_n_ext;

    interpolate_to_face(
    iface_int,
    K_tensor_int,
    flux_basis_int,
    n_face_quad_pts,
    sigma_at_face_int);
    
    interpolate_to_face(
    iface_ext,
    K_tensor_ext,
    flux_basis_ext,
    n_face_quad_pts,
    sigma_at_face_ext);

    // compute sigma_avg_dot_n
    for(unsigned int s=0; s<nstate; ++s)
    {
        sigma_at_face_avg_dot_n_int[s].resize(n_face_quad_pts);
        sigma_at_face_avg_dot_n_ext[s].resize(n_face_quad_pts);
        for(unsigned int q=0; q<n_face_quad_pts; ++q)
        {
            for(unsigned int d=0; d<dim; ++d)
            {
                sigma_at_face_avg_dot_n_int[s][q] += 0.5*(sigma_at_face_int[s][d][q] + sigma_at_face_ext[s][d][q])*unit_phys_normal_int[q][d];
                sigma_at_face_avg_dot_n_ext[s][q] += 0.5*(sigma_at_face_int[s][d][q] + sigma_at_face_ext[s][d][q])*(-unit_phys_normal_int[q][d]);
            }
        }
    }

    std::vector<adtype> int_face_integral_e;
    std::vector<adtype> ext_face_integral_e;
    std::vector<adtype> int_face_integral_k;
    std::vector<adtype> ext_face_integral_k;
    
    evaluate_face_integral(
    iface_int,
    sigma_at_face_avg_dot_n_int,
    JxW_face,
    soln_basis_int,
    n_dofs_cell_int,
    int_face_integral_e);
    
    evaluate_face_integral(
    iface_ext,
    sigma_at_face_avg_dot_n_ext,
    JxW_face,
    soln_basis_ext,
    n_dofs_cell_ext,
    ext_face_integral_e);

    entropystable_br2_compute_gradbasis_K_T_vol_integral(
    entropy_var_at_vol_int,
    n_vol_quad_pts_int,
    n_dofs_cell_int,
    soln_basis_int,
    metric_oper_int,
    vol_quad_weights_int,
    pde_physics,
    re_int,
    int_face_integral_k);
    
    entropystable_br2_compute_gradbasis_K_T_vol_integral(
    entropy_var_at_vol_ext,
    n_vol_quad_pts_ext,
    n_dofs_cell_ext,
    soln_basis_ext,
    metric_oper_ext,
    vol_quad_weights_ext,
    pde_physics,
    re_ext,
    ext_face_integral_k);

    for(unsigned int idof =0; idof<n_dofs_cell_int; ++idof)
    {
        face_term_int[idof] = int_face_integral_k[idof] - int_face_integral_e[idof];
    }

    for(unsigned int idof =0; idof<n_dofs_cell_ext; ++idof)
    {
        face_term_ext[idof] = ext_face_integral_k[idof] - ext_face_integral_e[idof];
    }
}


template <int dim, int nstate, typename real, typename MeshType>
template <typename adtype>
void DGStrong<dim,nstate,real,MeshType>::assemble_boundary_term_entropystable_br2(
    const unsigned int                                                 iface,
    const unsigned int                                                 boundary_id,
    const unsigned int                                                 poly_degree,
    const std::array<std::vector<adtype>,nstate>                       &entropy_var_coeff,
    const std::array<std::vector<adtype>,nstate>                       &entropy_var_at_vol_quads,
    const std::array<std::vector<adtype>,nstate>                       &entropy_var_at_surf_quads,
    const unsigned int                                                  n_vol_quad_pts,  
    const unsigned int                                                  n_face_quad_pts,  
    const unsigned int                                                  n_dofs_cell, 
    OPERATOR::basis_functions<dim,2*dim>                               &soln_basis,
    OPERATOR::basis_functions<dim,2*dim>                               &flux_basis,
    OPERATOR::metric_operators<adtype,dim,2*dim>                       &metric_oper,
    const std::vector<dealii::Tensor<1,dim,adtype>>                    &unit_phys_normal,
    const std::vector<adtype>                                          &JxW_face,
    const Physics::PhysicsBase<dim, nstate, adtype>                    &pde_physics,
    std::vector<adtype>                                                &boundary_term) const
{
    const std::vector<double> &vol_quad_weights = this->volume_quadrature_collection[poly_degree].get_weights();
    std::vector<adtype> JxW_vol(n_vol_quad_pts);
    for(unsigned int iquad=0; iquad<n_vol_quad_pts; ++iquad)
    {
        JxW_vol[iquad] = metric_oper.det_Jac_vol[iquad]*vol_quad_weights[iquad];
    }

    
    std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate>       entropy_var_phys_grad_at_vol;
    compute_physical_grad_entropy_var(
    entropy_var_coeff,
    soln_basis,
    metric_oper,
    n_vol_quad_pts, 
    entropy_var_phys_grad_at_vol);
   
    std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate>       K_nabla_v_at_vol;
    apply_K_matrix(
    entropy_var_at_vol_quads,
    n_vol_quad_pts,
    pde_physics,
    entropy_var_phys_grad_at_vol,
    K_nabla_v_at_vol);
    
    std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate>       poly_K_nabla_v_at_face;
    interpolate_to_face(
    iface,
    K_nabla_v_at_vol,
    flux_basis,
    n_face_quad_pts,
    poly_K_nabla_v_at_face);
    
    std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate>       entropy_var_phys_grad_at_face; // Using the vol entropy grad's projection to polynomial here instead of actually computing phys grad at face quadratures. This should be consistent and should yield correct solutions under mesh refinement.
    interpolate_to_face(
    iface,
    entropy_var_phys_grad_at_vol,
    flux_basis,
    n_face_quad_pts,
    entropy_var_phys_grad_at_face);

    std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate>      jump_entropy_var_face;
    std::array<std::vector<adtype>,nstate> sigma_gamma_dot_n_face;
    for(unsigned int s=0; s<nstate; ++s)
    {
        for(unsigned int d=0; d<dim; ++d)
        {
            jump_entropy_var_face[s][d].resize(n_face_quad_pts);
        }
        sigma_gamma_dot_n_face[s].resize(n_face_quad_pts); // 0 by default
    }

    for(unsigned int q =0; q<n_face_quad_pts; ++q)
    {
        std::array<adtype,nstate> v_int_at_q;
        std::array<dealii::Tensor<1,dim,adtype>,nstate> poly_sigma_at_q;
        std::array<dealii::Tensor<1,dim,adtype>,nstate> entropy_var_phys_grad_face_q;
        for(unsigned int s=0; s<nstate; ++s)
        {
            for(unsigned int d=0; d<dim; ++d)
            {
                poly_sigma_at_q[s][d] = poly_K_nabla_v_at_face[s][d][q];
                entropy_var_phys_grad_face_q[s][d] = entropy_var_phys_grad_at_face[s][d][q];
            }
            v_int_at_q[s] = entropy_var_at_surf_quads[s][q];
        }
        dealii::Point<dim,adtype> surf_flux_node;
        for(int idim=0; idim<dim; idim++){
            surf_flux_node[idim] = metric_oper.flux_nodes_surf[iface][idim][q];
        }
        
        std::array<adtype,nstate> v_bc_at_q;
        std::array<dealii::Tensor<1,dim,adtype>,nstate> sigma_bc_at_q;

        pde_physics.boundary_face_values_entropy_var(surf_flux_node,
                                                     v_int_at_q, 
                                                     poly_sigma_at_q, 
                                                     entropy_var_phys_grad_face_q, 
                                                     v_bc_at_q, 
                                                     sigma_bc_at_q, 
                                                     unit_phys_normal[q],
                                                     boundary_id);

        for(unsigned int s=0; s<nstate; ++s)
        {
            for(unsigned int d=0; d<dim; ++d)
            {
                sigma_gamma_dot_n_face[s][q] += sigma_bc_at_q[s][d]*unit_phys_normal[q][d];
            }
        }

        for(unsigned int s=0; s<nstate; ++s)
        {
            for(unsigned int d=0; d<dim; ++d)
            {
                jump_entropy_var_face[s][d][q] = (v_bc_at_q[s] - v_int_at_q[s])*unit_phys_normal[q][d];
            }
        }
    }

    std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate>      re_jump;
    const bool is_interior_face = false;
    compute_lift_polynomial(
    iface,
    jump_entropy_var_face,
    JxW_face,
    JxW_vol,
    flux_basis,
    n_face_quad_pts,
    n_vol_quad_pts,
    is_interior_face,
    re_jump);

    std::vector<adtype> integral_k;
    std::vector<adtype> integral_e;
    evaluate_face_integral(
    iface,
    sigma_gamma_dot_n_face,
    JxW_face,
    soln_basis,
    n_dofs_cell,
    integral_e);

    entropystable_br2_compute_gradbasis_K_T_vol_integral(
    entropy_var_at_vol_quads,
    n_vol_quad_pts,
    n_dofs_cell,
    soln_basis,
    metric_oper,
    vol_quad_weights,
    pde_physics,
    re_jump,
    integral_k);

    boundary_term.resize(n_dofs_cell);
    for(unsigned int idof=0; idof<n_dofs_cell; ++idof)
    {
        boundary_term[idof] = integral_k[idof] - integral_e[idof];
    }
}


template <int dim, int nstate, typename real, typename MeshType>
template <typename adtype>
void DGStrong<dim,nstate,real,MeshType>::check_same_coords_face_strong(
    const std::array<std::vector<adtype>,dim> &mapping_support_points_int, 
    const std::array<std::vector<adtype>,dim> &mapping_support_points_ext, 
    OPERATOR::mapping_shape_functions<dim,2*dim>  &mapping_basis,
    const unsigned int iface_int, 
    const unsigned int iface_ext, 
    const unsigned int poly_degree_int) const
{
    const unsigned int n_face_quad_pts = this->face_quadrature_collection[poly_degree_int].size();

    std::array<std::vector<adtype>,dim> x_face_int;
    std::array<std::vector<adtype>,dim> x_face_ext;

    for(unsigned int d=0; d<dim; ++d)
    {
        x_face_int[d].resize(n_face_quad_pts);
        x_face_ext[d].resize(n_face_quad_pts);
    }

    for(unsigned int d=0; d<dim; ++d)
    {
        mapping_basis.matrix_vector_mult_surface_1D(iface_int, 
                                                    mapping_support_points_int[d], 
                                                    x_face_int[d], 
                                                    mapping_basis.mapping_shape_functions_flux_nodes.oneD_surf_operator, 
                                                    mapping_basis.mapping_shape_functions_flux_nodes.oneD_vol_operator);
        
        mapping_basis.matrix_vector_mult_surface_1D(iface_ext, 
                                                    mapping_support_points_ext[d], 
                                                    x_face_ext[d], 
                                                    mapping_basis.mapping_shape_functions_flux_nodes.oneD_surf_operator, 
                                                    mapping_basis.mapping_shape_functions_flux_nodes.oneD_vol_operator);
    }

    for(unsigned int iquad = 0; iquad<n_face_quad_pts; ++iquad)
    {
        dealii::Point<dim,adtype> x_int_at_qface;
        dealii::Point<dim,adtype> x_ext_at_qface;
        for(unsigned int d=0; d<dim; ++d)
        {
            x_int_at_qface[d] = x_face_int[d][iquad];
            x_ext_at_qface[d] = x_face_ext[d][iquad];
        }
        adtype dist = 0.0;
        adtype dist_minus_z = 0.0;
        for(unsigned int d=0; d<dim; ++d)
        {
            dist += pow((x_int_at_qface[d]- x_ext_at_qface[d]),2);
            if(d<2)
            {
                dist_minus_z += pow((x_int_at_qface[d]- x_ext_at_qface[d]),2);
            }
        }
        dist = sqrt(dist);
        dist_minus_z = sqrt(dist_minus_z);
        if( dist > 1.0e-10) // Coords not matching
        {
            if ( dist_minus_z < 1.0e-10 && (dist-2)<1.0e-10) // On periodic face
            {
                
            }
            else
            {
                std::cout<<"x_int_at_qface = "<<x_int_at_qface<<std::endl;
                std::cout<<"x_ext_at_qface = "<<x_ext_at_qface<<std::endl;
                std::cout<<"dist = "<<dist<<std::endl;
                std::cout<<"Coords are not the same for strong DG. Aborting..."<<std::endl;
                std::abort();
            }
        }
    }
}

/*******************************************************
 *
 *                     EXPLICIT
 *
 *******************************************************/
template <int dim, int nstate, typename real, typename MeshType>
void DGStrong<dim,nstate,real,MeshType>::assemble_volume_term_explicit(
    typename dealii::DoFHandler<dim>::active_cell_iterator /*cell*/,
    const dealii::types::global_dof_index /*current_cell_index*/,
    const dealii::FEValues<dim,dim> &/*fe_values_vol*/,
    const std::vector<dealii::types::global_dof_index> &/*cell_dofs_indices*/,
    const std::vector<dealii::types::global_dof_index> &/*metric_dof_indices*/,
    const unsigned int /*poly_degree*/,
    const unsigned int /*grid_degree*/,
    dealii::Vector<real> &/*local_rhs_int_cell*/,
    const dealii::FEValues<dim,dim> &/*fe_values_lagrange*/)
{
    //do nothing
}


template <int dim, int nstate, typename real, typename MeshType>
void DGStrong<dim,nstate,real,MeshType>::allocate_dual_vector(const bool compute_d2R)
{
    if(compute_d2R){
        for(unsigned int k=0; k<this->n_duals; ++k)
        {
            this->duals[k].reinit(this->locally_owned_dofs, this->ghost_dofs, this->mpi_communicator);
            this->duals_transpose_dRdW[k].reinit(this->locally_owned_dofs, this->ghost_dofs, this->mpi_communicator);
        }
    }
}

// using default MeshType = Triangulation
// 1D: dealii::Triangulation<dim>;
// Otherwise: dealii::parallel::distributed::Triangulation<dim>;
template class DGStrong <PHILIP_DIM, 1, double, dealii::Triangulation<PHILIP_DIM>>;
template class DGStrong <PHILIP_DIM, 2, double, dealii::Triangulation<PHILIP_DIM>>;
template class DGStrong <PHILIP_DIM, 3, double, dealii::Triangulation<PHILIP_DIM>>;
template class DGStrong <PHILIP_DIM, 4, double, dealii::Triangulation<PHILIP_DIM>>;
template class DGStrong <PHILIP_DIM, 5, double, dealii::Triangulation<PHILIP_DIM>>;
template class DGStrong <PHILIP_DIM, 6, double, dealii::Triangulation<PHILIP_DIM>>;

template class DGStrong <PHILIP_DIM, 1, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class DGStrong <PHILIP_DIM, 2, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class DGStrong <PHILIP_DIM, 3, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class DGStrong <PHILIP_DIM, 4, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class DGStrong <PHILIP_DIM, 5, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class DGStrong <PHILIP_DIM, 6, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;

#if PHILIP_DIM!=1
template class DGStrong <PHILIP_DIM, 1, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class DGStrong <PHILIP_DIM, 2, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class DGStrong <PHILIP_DIM, 3, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class DGStrong <PHILIP_DIM, 4, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class DGStrong <PHILIP_DIM, 5, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class DGStrong <PHILIP_DIM, 6, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif
    
#define POSSIBLE_NSTATE (1)(2)(3)(4)(5)(6)

// Define a macro to instantiate MyTemplate for a specific index
#define INSTANTIATE_DISTRIBUTED(r, data, index) \
    template void DGStrong <PHILIP_DIM, index, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>::assemble_face_term_auxiliary_equation<double>(const unsigned int iface, const unsigned int neighbor_iface, const dealii::types::global_dof_index current_cell_index, const dealii::types::global_dof_index neighbor_cell_index, const std::array<std::vector<double>,index> &soln_coeff_int, const std::array<std::vector<double>,index> &soln_coeff_ext, const unsigned int poly_degree_int,const unsigned int poly_degree_ext, OPERATOR::basis_functions<PHILIP_DIM,2*PHILIP_DIM> &soln_basis_int,OPERATOR::basis_functions<PHILIP_DIM,2*PHILIP_DIM> &soln_basis_ext, OPERATOR::metric_operators<double,PHILIP_DIM,2*PHILIP_DIM> &metric_oper_int, const Physics::PhysicsBase<PHILIP_DIM, index, double> &pde_physics, const NumericalFlux::NumericalFluxDissipative<PHILIP_DIM, index, double> &diss_num_flux, dealii::Tensor<1,PHILIP_DIM,std::vector<double>> &local_auxiliary_RHS_int, dealii::Tensor<1,PHILIP_DIM,std::vector<double>> &local_auxiliary_RHS_ext);\
    template void DGStrong <PHILIP_DIM, index, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>::assemble_face_term_auxiliary_equation<codi_JacobianComputationType>(const unsigned int iface, const unsigned int neighbor_iface, const dealii::types::global_dof_index current_cell_index, const dealii::types::global_dof_index neighbor_cell_index, const std::array<std::vector<codi_JacobianComputationType>,index> &soln_coeff_int, const std::array<std::vector<codi_JacobianComputationType>,index> &soln_coeff_ext, const unsigned int poly_degree_int,const unsigned int poly_degree_ext, OPERATOR::basis_functions<PHILIP_DIM,2*PHILIP_DIM> &soln_basis_int,OPERATOR::basis_functions<PHILIP_DIM,2*PHILIP_DIM> &soln_basis_ext, OPERATOR::metric_operators<codi_JacobianComputationType,PHILIP_DIM,2*PHILIP_DIM> &metric_oper_int, const Physics::PhysicsBase<PHILIP_DIM, index, codi_JacobianComputationType> &pde_physics, const NumericalFlux::NumericalFluxDissipative<PHILIP_DIM, index, codi_JacobianComputationType> &diss_num_flux, dealii::Tensor<1,PHILIP_DIM,std::vector<codi_JacobianComputationType>> &local_auxiliary_RHS_int, dealii::Tensor<1,PHILIP_DIM,std::vector<codi_JacobianComputationType>> &local_auxiliary_RHS_ext);\
    template void DGStrong <PHILIP_DIM, index, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>::assemble_face_term_auxiliary_equation<codi_HessianComputationType>(const unsigned int iface, const unsigned int neighbor_iface, const dealii::types::global_dof_index current_cell_index, const dealii::types::global_dof_index neighbor_cell_index, const std::array<std::vector<codi_HessianComputationType>,index> &soln_coeff_int, const std::array<std::vector<codi_HessianComputationType>,index> &soln_coeff_ext, const unsigned int poly_degree_int,const unsigned int poly_degree_ext, OPERATOR::basis_functions<PHILIP_DIM,2*PHILIP_DIM> &soln_basis_int,OPERATOR::basis_functions<PHILIP_DIM,2*PHILIP_DIM> &soln_basis_ext, OPERATOR::metric_operators<codi_HessianComputationType,PHILIP_DIM,2*PHILIP_DIM> &metric_oper_int, const Physics::PhysicsBase<PHILIP_DIM, index, codi_HessianComputationType> &pde_physics, const NumericalFlux::NumericalFluxDissipative<PHILIP_DIM, index, codi_HessianComputationType> &diss_num_flux, dealii::Tensor<1,PHILIP_DIM,std::vector<codi_HessianComputationType>> &local_auxiliary_RHS_int, dealii::Tensor<1,PHILIP_DIM,std::vector<codi_HessianComputationType>> &local_auxiliary_RHS_ext);


#define INSTANTIATE_SHARED(r, data, index) \
    template void DGStrong <PHILIP_DIM, index, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>::assemble_face_term_auxiliary_equation<double>(const unsigned int iface, const unsigned int neighbor_iface, const dealii::types::global_dof_index current_cell_index, const dealii::types::global_dof_index neighbor_cell_index, const std::array<std::vector<double>,index> &soln_coeff_int, const std::array<std::vector<double>,index> &soln_coeff_ext, const unsigned int poly_degree_int,const unsigned int poly_degree_ext, OPERATOR::basis_functions<PHILIP_DIM,2*PHILIP_DIM> &soln_basis_int,OPERATOR::basis_functions<PHILIP_DIM,2*PHILIP_DIM> &soln_basis_ext, OPERATOR::metric_operators<double,PHILIP_DIM,2*PHILIP_DIM> &metric_oper_int, const Physics::PhysicsBase<PHILIP_DIM, index, double> &pde_physics, const NumericalFlux::NumericalFluxDissipative<PHILIP_DIM, index, double> &diss_num_flux, dealii::Tensor<1,PHILIP_DIM,std::vector<double>> &local_auxiliary_RHS_int, dealii::Tensor<1,PHILIP_DIM,std::vector<double>> &local_auxiliary_RHS_ext);\
    template void DGStrong <PHILIP_DIM, index, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>::assemble_face_term_auxiliary_equation<codi_JacobianComputationType>(const unsigned int iface, const unsigned int neighbor_iface, const dealii::types::global_dof_index current_cell_index, const dealii::types::global_dof_index neighbor_cell_index, const std::array<std::vector<codi_JacobianComputationType>,index> &soln_coeff_int, const std::array<std::vector<codi_JacobianComputationType>,index> &soln_coeff_ext, const unsigned int poly_degree_int,const unsigned int poly_degree_ext, OPERATOR::basis_functions<PHILIP_DIM,2*PHILIP_DIM> &soln_basis_int,OPERATOR::basis_functions<PHILIP_DIM,2*PHILIP_DIM> &soln_basis_ext, OPERATOR::metric_operators<codi_JacobianComputationType,PHILIP_DIM,2*PHILIP_DIM> &metric_oper_int, const Physics::PhysicsBase<PHILIP_DIM, index, codi_JacobianComputationType> &pde_physics, const NumericalFlux::NumericalFluxDissipative<PHILIP_DIM, index, codi_JacobianComputationType> &diss_num_flux, dealii::Tensor<1,PHILIP_DIM,std::vector<codi_JacobianComputationType>> &local_auxiliary_RHS_int, dealii::Tensor<1,PHILIP_DIM,std::vector<codi_JacobianComputationType>> &local_auxiliary_RHS_ext);\
    template void DGStrong <PHILIP_DIM, index, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>::assemble_face_term_auxiliary_equation<codi_HessianComputationType>(const unsigned int iface, const unsigned int neighbor_iface, const dealii::types::global_dof_index current_cell_index, const dealii::types::global_dof_index neighbor_cell_index, const std::array<std::vector<codi_HessianComputationType>,index> &soln_coeff_int, const std::array<std::vector<codi_HessianComputationType>,index> &soln_coeff_ext, const unsigned int poly_degree_int,const unsigned int poly_degree_ext, OPERATOR::basis_functions<PHILIP_DIM,2*PHILIP_DIM> &soln_basis_int,OPERATOR::basis_functions<PHILIP_DIM,2*PHILIP_DIM> &soln_basis_ext, OPERATOR::metric_operators<codi_HessianComputationType,PHILIP_DIM,2*PHILIP_DIM> &metric_oper_int, const Physics::PhysicsBase<PHILIP_DIM, index, codi_HessianComputationType> &pde_physics, const NumericalFlux::NumericalFluxDissipative<PHILIP_DIM, index, codi_HessianComputationType> &diss_num_flux, dealii::Tensor<1,PHILIP_DIM,std::vector<codi_HessianComputationType>> &local_auxiliary_RHS_int, dealii::Tensor<1,PHILIP_DIM,std::vector<codi_HessianComputationType>> &local_auxiliary_RHS_ext);


#define INSTANTIATE_TRIA(r, data, index) \
    template void DGStrong <PHILIP_DIM, index, double, dealii::Triangulation<PHILIP_DIM>>::assemble_face_term_auxiliary_equation<double>(const unsigned int iface, const unsigned int neighbor_iface, const dealii::types::global_dof_index current_cell_index, const dealii::types::global_dof_index neighbor_cell_index, const std::array<std::vector<double>,index> &soln_coeff_int, const std::array<std::vector<double>,index> &soln_coeff_ext, const unsigned int poly_degree_int,const unsigned int poly_degree_ext, OPERATOR::basis_functions<PHILIP_DIM,2*PHILIP_DIM> &soln_basis_int,OPERATOR::basis_functions<PHILIP_DIM,2*PHILIP_DIM> &soln_basis_ext, OPERATOR::metric_operators<double,PHILIP_DIM,2*PHILIP_DIM> &metric_oper_int, const Physics::PhysicsBase<PHILIP_DIM, index, double> &pde_physics, const NumericalFlux::NumericalFluxDissipative<PHILIP_DIM, index, double> &diss_num_flux, dealii::Tensor<1,PHILIP_DIM,std::vector<double>> &local_auxiliary_RHS_int, dealii::Tensor<1,PHILIP_DIM,std::vector<double>> &local_auxiliary_RHS_ext);\
    template void DGStrong <PHILIP_DIM, index, double, dealii::Triangulation<PHILIP_DIM>>::assemble_face_term_auxiliary_equation<codi_JacobianComputationType>(const unsigned int iface, const unsigned int neighbor_iface, const dealii::types::global_dof_index current_cell_index, const dealii::types::global_dof_index neighbor_cell_index, const std::array<std::vector<codi_JacobianComputationType>,index> &soln_coeff_int, const std::array<std::vector<codi_JacobianComputationType>,index> &soln_coeff_ext, const unsigned int poly_degree_int,const unsigned int poly_degree_ext, OPERATOR::basis_functions<PHILIP_DIM,2*PHILIP_DIM> &soln_basis_int,OPERATOR::basis_functions<PHILIP_DIM,2*PHILIP_DIM> &soln_basis_ext, OPERATOR::metric_operators<codi_JacobianComputationType,PHILIP_DIM,2*PHILIP_DIM> &metric_oper_int, const Physics::PhysicsBase<PHILIP_DIM, index, codi_JacobianComputationType> &pde_physics, const NumericalFlux::NumericalFluxDissipative<PHILIP_DIM, index, codi_JacobianComputationType> &diss_num_flux, dealii::Tensor<1,PHILIP_DIM,std::vector<codi_JacobianComputationType>> &local_auxiliary_RHS_int, dealii::Tensor<1,PHILIP_DIM,std::vector<codi_JacobianComputationType>> &local_auxiliary_RHS_ext);\
    template void DGStrong <PHILIP_DIM, index, double, dealii::Triangulation<PHILIP_DIM>>::assemble_face_term_auxiliary_equation<codi_HessianComputationType>(const unsigned int iface, const unsigned int neighbor_iface, const dealii::types::global_dof_index current_cell_index, const dealii::types::global_dof_index neighbor_cell_index, const std::array<std::vector<codi_HessianComputationType>,index> &soln_coeff_int, const std::array<std::vector<codi_HessianComputationType>,index> &soln_coeff_ext, const unsigned int poly_degree_int,const unsigned int poly_degree_ext, OPERATOR::basis_functions<PHILIP_DIM,2*PHILIP_DIM> &soln_basis_int,OPERATOR::basis_functions<PHILIP_DIM,2*PHILIP_DIM> &soln_basis_ext, OPERATOR::metric_operators<codi_HessianComputationType,PHILIP_DIM,2*PHILIP_DIM> &metric_oper_int, const Physics::PhysicsBase<PHILIP_DIM, index, codi_HessianComputationType> &pde_physics, const NumericalFlux::NumericalFluxDissipative<PHILIP_DIM, index, codi_HessianComputationType> &diss_num_flux, dealii::Tensor<1,PHILIP_DIM,std::vector<codi_HessianComputationType>> &local_auxiliary_RHS_int, dealii::Tensor<1,PHILIP_DIM,std::vector<codi_HessianComputationType>> &local_auxiliary_RHS_ext);    


#if PHILIP_DIM!=1
BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_DISTRIBUTED, _, POSSIBLE_NSTATE)
#endif

BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_SHARED, _, POSSIBLE_NSTATE)

BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_TRIA, _, POSSIBLE_NSTATE)
} // PHiLiP namespace
