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
    typename dealii::DoFHandler<dim>::active_cell_iterator             cell,
    const dealii::types::global_dof_index                              current_cell_index,
    const std::array<std::vector<adtype>,nstate>                       &soln_coeff,
    const std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate> &/*aux_soln_coeff*/,
    const unsigned int                                                 poly_degree,
    OPERATOR::basis_functions<dim,2*dim>                               &soln_basis,
    OPERATOR::basis_functions<dim,2*dim>                               &flux_basis,
    OPERATOR::local_basis_stiffness<dim,2*dim>                         &flux_basis_stiffness,
    OPERATOR::vol_projection_operator<dim,2*dim>                       &soln_basis_projection_oper,
    OPERATOR::metric_operators<adtype,dim,2*dim>                       &metric_oper,
    const Physics::PhysicsBase<dim, nstate, adtype>                    &pde_physics,
    std::vector<adtype>                                                &local_rhs_int_cell)
{
    (void) current_cell_index;

    const unsigned int n_quad_pts  = this->volume_quadrature_collection[poly_degree].size();
    const unsigned int n_dofs_cell = this->fe_collection[poly_degree].dofs_per_cell;
    const unsigned int n_shape_fns = n_dofs_cell / nstate; 
    const unsigned int n_quad_pts_1D  = this->oneD_quadrature_collection[poly_degree].size();
    assert(n_quad_pts == pow(n_quad_pts_1D, dim));
    const std::vector<double> &vol_quad_weights = this->volume_quadrature_collection[poly_degree].get_weights();
    const std::vector<double> &oneD_vol_quad_weights = this->oneD_quadrature_collection[poly_degree].get_weights();

    std::array<std::vector<adtype>,nstate> soln_at_q;
    std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate> aux_soln_at_q; //auxiliary sol at flux nodes
    std::vector<std::array<double,nstate>> soln_at_q_for_max_CFL(n_quad_pts);//Need soln written in a different for to use pre-existing max CFL function
    // Interpolate each state to the quadrature points using sum-factorization
    // with the basis functions in each reference direction.
    for(int istate=0; istate<nstate; istate++){
        soln_at_q[istate].resize(n_quad_pts);
        soln_basis.matrix_vector_mult_1D(soln_coeff[istate], soln_at_q[istate],
                                         soln_basis.oneD_vol_operator);
        /*
        for(int idim=0; idim<dim; idim++){
            aux_soln_at_q[istate][idim].resize(n_quad_pts);
            soln_basis.matrix_vector_mult_1D(aux_soln_coeff[istate][idim], aux_soln_at_q[istate][idim],
                                             soln_basis.oneD_vol_operator);
        }
        */
        for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
            soln_at_q_for_max_CFL[iquad][istate] = getValue<adtype>(soln_at_q[istate][iquad]);
        }
    }

    // For pseudotime, we need to compute the time_scaled_solution.
    // Thus, we need to evaluate the max_dt_cell (as previously done in dg/weak_dg.cpp -> assemble_volume_term_explicit)
    // Get max artificial dissipation
    real max_artificial_diss = 0.0;
    const unsigned int n_dofs_arti_diss = this->fe_q_artificial_dissipation.dofs_per_cell;
    typename dealii::DoFHandler<dim>::active_cell_iterator artificial_dissipation_cell(
        this->triangulation.get(), cell->level(), cell->index(), &(this->dof_handler_artificial_dissipation));
    std::vector<dealii::types::global_dof_index> dof_indices_artificial_dissipation(n_dofs_arti_diss);
    artificial_dissipation_cell->get_dof_indices (dof_indices_artificial_dissipation);
    for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
        real artificial_diss_coeff_at_q = 0.0;
        if ( this->all_parameters->artificial_dissipation_param.add_artificial_dissipation ) {
            const dealii::Point<dim,real> point = this->volume_quadrature_collection[poly_degree].point(iquad);
            for (unsigned int idof=0; idof<n_dofs_arti_diss; ++idof) {
                const unsigned int index = dof_indices_artificial_dissipation[idof];
                artificial_diss_coeff_at_q += this->artificial_dissipation_c0[index] * this->fe_q_artificial_dissipation.shape_value(idof, point);
            }
            max_artificial_diss = std::max(artificial_diss_coeff_at_q, max_artificial_diss);
        }
    }
    // Get max_dt_cell for time_scaled_solution with pseudotime
    double cell_volume_estimate = 0.0;
    for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
        cell_volume_estimate += getValue<adtype>(metric_oper.det_Jac_vol[iquad]) * vol_quad_weights[iquad];
    }
    const real cell_volume = cell_volume_estimate;
    const real diameter = cell->diameter();
    const real cell_diameter = cell_volume / std::pow(diameter,dim-1);
    const real cell_radius = 0.5 * cell_diameter;
    this->cell_volume[current_cell_index] = cell_volume;
    this->max_dt_cell[current_cell_index] = this->evaluate_CFL ( soln_at_q_for_max_CFL, max_artificial_diss, cell_radius, poly_degree);

    //get entropy projected variables
    std::array<std::vector<adtype>,nstate> entropy_var_at_q;
    std::array<std::vector<adtype>,nstate> projected_entropy_var_at_q;
    std::array<std::vector<adtype>,nstate> entropy_var_coeffs;
    std::vector<std::array<std::array<real,nstate>,nstate>> dv_du(n_quad_pts);
    if(true){
    //if (this->all_parameters->use_split_form || this->all_parameters->use_curvilinear_split_form){
        for(int istate=0; istate<nstate; istate++){
            entropy_var_at_q[istate].resize(n_quad_pts);
            projected_entropy_var_at_q[istate].resize(n_quad_pts);
            entropy_var_coeffs[istate].resize(n_shape_fns);
        }
        for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
            std::array<adtype,nstate> soln_state;
            for(int istate=0; istate<nstate; istate++){
                soln_state[istate] = soln_at_q[istate][iquad];
            }
            std::array<adtype,nstate> entropy_var;
            entropy_var = pde_physics.compute_entropy_variables(soln_state);
            if constexpr(std::is_same<adtype,double>::value)
            {
                if(this->compute_dRdW_strong)
                {
                    pde_physics.get_d_entropy_var_d_conservative_var(dv_du[iquad],entropy_var); 
                }
            }
            for(int istate=0; istate<nstate; istate++){
                entropy_var_at_q[istate][iquad] = entropy_var[istate];
            }
        }
        for(int istate=0; istate<nstate; istate++){
            //soln_basis_projection_oper.matrix_vector_mult_1D(entropy_var_at_q[istate],
            //                                                 entropy_var_coeffs[istate],
            //                                                 soln_basis_projection_oper.oneD_vol_operator);
            soln_basis_projection_oper.weight_adjusted_vol_projection(entropy_var_at_q[istate],
                                                                      entropy_var_coeffs[istate],
                                                                      n_quad_pts,
                                                                      n_shape_fns,
                                                                      soln_basis,
                                                                      soln_basis_projection_oper,
                                                                      metric_oper.det_Jac_vol);
            soln_basis.matrix_vector_mult_1D(entropy_var_coeffs[istate],
                                             projected_entropy_var_at_q[istate],
                                             soln_basis.oneD_vol_operator);
        }
    }

    std::array<std::array<dealii::FullMatrix<double>,nstate>,nstate> dvtilde_duh;
    xt::xarray<double> H_mat;
    if constexpr(std::is_same<adtype,double>::value)
    {
        if(this->compute_dRdW_strong)
        {
            compute_H_mat(soln_basis_projection_oper.oneD_G_operator_vol, soln_basis.oneD_vol_operator, metric_oper.det_Jac_vol, dv_du, H_mat);
            compute_dvtilde_duh(soln_basis_projection_oper.oneD_G_operator_vol,soln_basis_projection_oper.oneD_G_operator_vol,soln_basis_projection_oper.oneD_G_operator_vol,H_mat, soln_basis.oneD_vol_operator.n(),n_quad_pts_1D,dvtilde_duh);
        }
    }


    //Compute the physical fluxes, then convert them into reference fluxes.
    //From the paper: Cicchino, Alexander, et al. "Provably stable flux reconstruction high-order methods on curvilinear elements." Journal of Computational Physics 463 (2022): 111259.
    //For conservative DG, we compute the reference flux as per Eq. (9), to then recover the second volume integral in Eq. (17).
    //For curvilinear split-form in Eq. (22), we apply a two-pt flux of the metric-cofactor matrix on the matrix operator constructed by the entropy stable/conservtive 2pt flux.
    std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate> conv_ref_flux_at_q;
    //std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate> diffusive_ref_flux_at_q;
    std::array<std::vector<adtype>,nstate> source_at_q;
    std::array<std::vector<adtype>,nstate> physical_source_at_q;

    // The matrix of two-pt fluxes for Hadamard products
    std::array<std::array<std::vector<adtype>,dim>,nstate> conv_ref_2pt_flux_at_q;
    //Hadamard tensor-product sparsity pattern
    std::vector<std::array<unsigned int,dim>> Hadamard_rows_sparsity(n_quad_pts * n_quad_pts_1D);//size n^{d+1}
    std::vector<std::array<unsigned int,dim>> Hadamard_columns_sparsity(n_quad_pts * n_quad_pts_1D);
    //allocate reference 2pt flux for Hadamard product
    if (this->all_parameters->use_split_form || this->all_parameters->use_curvilinear_split_form){
        for(int istate=0; istate<nstate; istate++){
            for(int idim=0; idim<dim; idim++){
                conv_ref_2pt_flux_at_q[istate][idim].resize(n_quad_pts * n_quad_pts_1D);//size n^d x n
            }
        }
        //extract the dof pairs that give non-zero entries for each direction
        //to use the "sum-factorized" Hadamard product.
        flux_basis.sum_factorized_Hadamard_sparsity_pattern(n_quad_pts_1D, n_quad_pts_1D, Hadamard_rows_sparsity, Hadamard_columns_sparsity);
    }

    std::array<std::array<dealii::FullMatrix<double>,nstate>,nstate> dQF_duh;
    if constexpr(std::is_same<adtype,double>::value)
    {
        if(this->compute_dRdW_strong)
        {
            if constexpr (dim==3)
            {
               for(unsigned int s1=0; s1<nstate; ++s1)
               {
                    for(unsigned int s2=0; s2<nstate; ++s2)
                    {
                        dQF_duh[s1][s2].reinit(n_quad_pts,n_shape_fns); 
                        dQF_duh[s1][s2] = 0;
                    }
               }
            }
            else
            {
                std::cout<<"dRdW_strong is not yet implemented for dim<3. Aborting..."<<std::endl;
                std::abort();
            }
        }
    }

    for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
        const int l_ = iquad % n_quad_pts_1D;
        const int m_ = (iquad/n_quad_pts_1D) % n_quad_pts_1D;
        const int n_ = (iquad/n_quad_pts_1D)/n_quad_pts_1D;
        //extract soln and auxiliary soln at quad pt to be used in physics
        std::array<adtype,nstate> soln_state;
        std::array<dealii::Tensor<1,dim,adtype>,nstate> aux_soln_state;
        for(int istate=0; istate<nstate; istate++){
            soln_state[istate] = soln_at_q[istate][iquad];
            /*
            for(int idim=0; idim<dim; idim++){
                aux_soln_state[istate][idim] = aux_soln_at_q[istate][idim][iquad];
            }
            */
        }

        // Copy Metric Cofactor in a way can use for transforming Tensor Blocks to reference space
        // The way it is stored in metric_operators is to use sum-factorization in each direction,
        // but here it is cleaner to apply a reference transformation in each Tensor block returned by physics.
        dealii::Tensor<2,dim,adtype> metric_cofactor;
        for(int idim=0; idim<dim; idim++){
            for(int jdim=0; jdim<dim; jdim++){
                metric_cofactor[idim][jdim] = metric_oper.metric_cofactor_vol[idim][jdim][iquad];
            }
        }

        // Evaluate physical convective flux
        // If 2pt flux, transform to reference at construction to improve performance.
        // We technically use a REFERENCE 2pt flux for all entropy stable schemes.
        std::array<dealii::Tensor<1,dim,adtype>,nstate> conv_phys_flux;
        if (this->all_parameters->use_split_form || this->all_parameters->use_curvilinear_split_form){
            //get the soln for iquad from projected entropy variables
            std::array<adtype,nstate> entropy_var;
            for(int istate=0; istate<nstate; istate++){
                entropy_var[istate] = projected_entropy_var_at_q[istate][iquad];
            }
            soln_state = pde_physics.compute_conservative_variables_from_entropy_variables (entropy_var);
            std::array<std::array<double,nstate>,nstate> dutilde_dvtilde_iquad;
            if constexpr(std::is_same<adtype,double>::value)
            {
                if(this->compute_dRdW_strong)
                {
                    pde_physics.get_d_conservative_var_d_entropy_var(dutilde_dvtilde_iquad, entropy_var);
                }
            }
            
            //loop over all the non-zero entries for "sum-factorized" Hadamard product that corresponds to the iquad.
            for(unsigned int row_index = iquad * n_quad_pts_1D, column_index = 0; 
               // Hadamard_rows_sparsity[row_index][0] == iquad; 
                column_index < n_quad_pts_1D; 
                row_index++, column_index++){

                if(Hadamard_rows_sparsity[row_index][0] != iquad){
                    pcout<<"The volume Hadamard rows sparsity pattern does not match. Aborting..."<<std::endl;
                    std::abort();
                }

                // Copy Metric Cofactor in a way can use for transforming Tensor Blocks to reference space
                // The way it is stored in metric_operators is to use sum-factorization in each direction,
                // but here it is cleaner to apply a reference transformation in each Tensor block returned by physics.

                for(int ref_dim=0; ref_dim<dim; ref_dim++){
                    const unsigned int flux_quad = Hadamard_columns_sparsity[row_index][ref_dim];//extract flux_quad pt that corresponds to a non-zero entry for Hadamard product.

                    dealii::Tensor<2,dim,adtype> metric_cofactor_flux_basis;
                    for(int idim=0; idim<dim; idim++){
                        for(int jdim=0; jdim<dim; jdim++){
                            metric_cofactor_flux_basis[idim][jdim] = metric_oper.metric_cofactor_vol[idim][jdim][flux_quad];
                        }
                    }
                    std::array<adtype,nstate> soln_state_flux_basis;
                    std::array<adtype,nstate> entropy_var_flux_basis;
                    for(int istate=0; istate<nstate; istate++){
                        entropy_var_flux_basis[istate] = projected_entropy_var_at_q[istate][flux_quad];
                    }
                    soln_state_flux_basis = pde_physics.compute_conservative_variables_from_entropy_variables (entropy_var_flux_basis);
                    std::array<std::array<double,nstate>,nstate> dutilde_dvtilde_fluxquad;
                    if constexpr(std::is_same<adtype,double>::value)
                    {
                        if(this->compute_dRdW_strong)
                        {
                            pde_physics.get_d_conservative_var_d_entropy_var(dutilde_dvtilde_fluxquad, entropy_var_flux_basis);
                        }
                    }

                    //Compute the physical flux
                    std::array<dealii::Tensor<1,dim,adtype>,nstate> conv_phys_flux_2pt;
                    conv_phys_flux_2pt = pde_physics.convective_numerical_split_flux(soln_state, soln_state_flux_basis);
                     
                    dealii::Tensor<2,dim,adtype> metric_cofactor_split;
                    for(int idim=0; idim<dim; idim++){
                        for(int jdim=0; jdim<dim; jdim++){
                            metric_cofactor_split[idim][jdim] = 0.5 * (metric_cofactor[idim][jdim] + metric_cofactor_flux_basis[idim][jdim]);
                        }
                    }
                    for(int istate=0; istate<nstate; istate++){
                        dealii::Tensor<1,dim,adtype> conv_ref_flux_2pt;
                        //For each state, transform the physical flux to a reference flux.
                        metric_oper.transform_physical_to_reference(
                            conv_phys_flux_2pt[istate],
                            metric_cofactor_split,
                            conv_ref_flux_2pt);
                        //write into reference Hadamard flux matrix
                        conv_ref_2pt_flux_at_q[istate][ref_dim][iquad * n_quad_pts_1D + column_index] = conv_ref_flux_2pt[ref_dim];
                    }

                    if constexpr(std::is_same<adtype,double>::value)
                    {
                        if(this->compute_dRdW_strong)
                        {
                            dealii::Tensor<1,dim,double> component; 
                            component[ref_dim] = 1.0;
                            std::array<std::array<std::array<double,nstate>,nstate>,2> dFsplit_dutilde = pde_physics.convective_numerical_split_flux_derivative(soln_state, soln_state_flux_basis, metric_cofactor_split, component);
                            double mult_factor = 0;
                            if(ref_dim==0)
                            {
                                const int p_ = flux_quad % n_quad_pts_1D;
                                mult_factor = flux_basis_stiffness.oneD_skew_symm_vol_oper[l_][p_]*oneD_vol_quad_weights[m_]*oneD_vol_quad_weights[n_];
                            }
                            else if(ref_dim==1)
                            {
                                const int q_ = (flux_quad/n_quad_pts_1D) % n_quad_pts_1D;
                                mult_factor = flux_basis_stiffness.oneD_skew_symm_vol_oper[m_][q_]*oneD_vol_quad_weights[l_]*oneD_vol_quad_weights[n_];
                            }
                            else if(dim==3)
                            {
                                const int r_ = (flux_quad/n_quad_pts_1D) / n_quad_pts_1D;
                                mult_factor = flux_basis_stiffness.oneD_skew_symm_vol_oper[n_][r_]*oneD_vol_quad_weights[l_]*oneD_vol_quad_weights[m_];
                            }
                            for(unsigned int s1 = 0; s1<nstate; ++s1)
                            {
                                for(unsigned int s2=0; s2<nstate; ++s2)
                                {
                                    for(unsigned int sk=0; sk<nstate; ++sk)
                                    {
                                        for(unsigned int sl=0; sl<nstate; ++sl)
                                        {
                                            for(unsigned int idof=0; idof<n_shape_fns; ++idof)
                                            {
                                                dQF_duh[s1][s2][iquad][idof] += mult_factor*(dFsplit_dutilde[0][s1][sk]*dutilde_dvtilde_iquad[sk][sl]*dvtilde_duh[sl][s2][iquad][idof] + dFsplit_dutilde[1][s1][sk]*dutilde_dvtilde_fluxquad[sk][sl]*dvtilde_duh[sl][s2][flux_quad][idof]);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        else{
            //Compute the physical flux
            conv_phys_flux = pde_physics.convective_flux (soln_state);
        }

        //Diffusion
        //std::array<dealii::Tensor<1,dim,adtype>,nstate> diffusive_phys_flux;
        //Compute the physical dissipative flux
        //diffusive_phys_flux = pde_physics.dissipative_flux(soln_state, aux_soln_state, current_cell_index);

        // Manufactured source
        std::array<adtype,nstate> manufactured_source;
        if(this->all_parameters->manufactured_convergence_study_param.manufactured_solution_param.use_manufactured_source_term) {
            dealii::Point<dim,adtype> vol_flux_node;
            for(int idim=0; idim<dim; idim++){
                vol_flux_node[idim] = metric_oper.flux_nodes_vol[idim][iquad];
            }
            //compute the manufactured source
            manufactured_source = pde_physics.source_term (vol_flux_node, soln_state, this->current_time, current_cell_index);
        }

        // Physical source
        std::array<adtype,nstate> physical_source;
        if(pde_physics.has_nonzero_physical_source) {
            dealii::Point<dim,adtype> vol_flux_node;
            for(int idim=0; idim<dim; idim++){
                vol_flux_node[idim] = metric_oper.flux_nodes_vol[idim][iquad];
            }
            //compute the physical source
            physical_source = pde_physics.physical_source_term (vol_flux_node, soln_state, aux_soln_state, current_cell_index);
        }

        //Write the values in a way that we can use sum-factorization on.
        for(int istate=0; istate<nstate; istate++){
            dealii::Tensor<1,dim,adtype> conv_ref_flux;
            //dealii::Tensor<1,dim,adtype> diffusive_ref_flux;
            //Trnasform to reference fluxes
            if (this->all_parameters->use_split_form || this->all_parameters->use_curvilinear_split_form){
                //Do Nothing. 
                //I am leaving this block here so the diligent reader
                //remembers that, for entropy stable schemes, we construct
                //a REFERENCE two-point flux at construction, where the physical
                //to reference transformation was done by splitting the metric cofactor.
            }
            else{
                //transform the conservative convective physical flux to reference space
                metric_oper.transform_physical_to_reference(
                    conv_phys_flux[istate],
                    metric_cofactor,
                    conv_ref_flux);
            }
            /*
            //transform the dissipative flux to reference space
            metric_oper.transform_physical_to_reference(
                diffusive_phys_flux[istate],
                metric_cofactor,
                diffusive_ref_flux);
            */

            //Write the data in a way that we can use sum-factorization on.
            //Since sum-factorization improves the speed for matrix-vector multiplications,
            //We need the values to have their inner elements be vectors.
            for(int idim=0; idim<dim; idim++){
                //allocate
                if(iquad == 0){
                    conv_ref_flux_at_q[istate][idim].resize(n_quad_pts);
                    //diffusive_ref_flux_at_q[istate][idim].resize(n_quad_pts);
                }
                //write data
                if (this->all_parameters->use_split_form || this->all_parameters->use_curvilinear_split_form){
                    //Do nothing because written in a Hadamard product sum-factorized form above.
                }
                else{
                    conv_ref_flux_at_q[istate][idim][iquad] = conv_ref_flux[idim];
                }

                //diffusive_ref_flux_at_q[istate][idim][iquad] = diffusive_ref_flux[idim];
            }
            if(this->all_parameters->manufactured_convergence_study_param.manufactured_solution_param.use_manufactured_source_term) {
                if(iquad == 0){
                    source_at_q[istate].resize(n_quad_pts);
                }
                source_at_q[istate][iquad] = manufactured_source[istate];
            }
            if(pde_physics.has_nonzero_physical_source) {
                if(iquad == 0){
                    physical_source_at_q[istate].resize(n_quad_pts);
                }
                physical_source_at_q[istate][iquad] = physical_source[istate];
            }
        }
    }

    // Get a flux basis reference gradient operator in a sum-factorized Hadamard product sparse form. Then apply the divergence.
    std::array<dealii::FullMatrix<real>,dim> flux_basis_stiffness_skew_symm_oper_sparse;
    if (this->all_parameters->use_split_form || this->all_parameters->use_curvilinear_split_form){
        for(int idim=0; idim<dim; idim++){
            flux_basis_stiffness_skew_symm_oper_sparse[idim].reinit(n_quad_pts, n_quad_pts_1D);
        }
        flux_basis.sum_factorized_Hadamard_basis_assembly(n_quad_pts_1D, n_quad_pts_1D, 
                                                          Hadamard_rows_sparsity, Hadamard_columns_sparsity,
                                                          flux_basis_stiffness.oneD_skew_symm_vol_oper, 
                                                          oneD_vol_quad_weights,
                                                          flux_basis_stiffness_skew_symm_oper_sparse);
    }


    //For each state we:
    //  1. Compute reference divergence.
    //  2. Then compute and write the rhs for the given state.
    for(int istate=0; istate<nstate; istate++){

        //Compute reference divergence of the reference fluxes.
        std::vector<adtype> conv_flux_divergence(n_quad_pts); 
        //std::vector<adtype> diffusive_flux_divergence(n_quad_pts); 

        if (this->all_parameters->use_split_form || this->all_parameters->use_curvilinear_split_form){
            //2pt flux Hadamard Product, and then multiply by vector of ones scaled by 1.
            // Same as the volume term in Eq. (15) in Chan, Jesse. "Skew-symmetric entropy stable modal discontinuous Galerkin formulations." Journal of Scientific Computing 81.1 (2019): 459-485. but, 
            // where we use the reference skew-symmetric stiffness operator of the flux basis for the Q operator and the reference two-point flux as to make use of Alex's Hadamard product
            // sum-factorization type algorithm that exploits the structure of the flux basis in the reference space to have O(n^{d+1}).

            for(int ref_dim=0; ref_dim<dim; ref_dim++){
                std::vector<adtype> divergence_ref_flux_Hadamard_product(n_quad_pts * n_quad_pts_1D);
                flux_basis.Hadamard_product_AD_vector(flux_basis_stiffness_skew_symm_oper_sparse[ref_dim], conv_ref_2pt_flux_at_q[istate][ref_dim], divergence_ref_flux_Hadamard_product); 
                //Hadamard product times the vector of ones.
                for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
                    if(ref_dim == 0){
                        conv_flux_divergence[iquad] = 0.0;
                    }
                    for(unsigned int iquad_1D=0; iquad_1D<n_quad_pts_1D; iquad_1D++){
                        conv_flux_divergence[iquad] += divergence_ref_flux_Hadamard_product[iquad * n_quad_pts_1D + iquad_1D];
                    }
                }
            }
            
        }
        else{
            //Reference divergence of the reference convective flux.
            flux_basis.divergence_matrix_vector_mult_1D(conv_ref_flux_at_q[istate], conv_flux_divergence,
                                                        flux_basis.oneD_vol_operator,
                                                        flux_basis.oneD_grad_operator);
        }
        /*
        //Reference divergence of the reference diffusive flux.
        flux_basis.divergence_matrix_vector_mult_1D(diffusive_ref_flux_at_q[istate], diffusive_flux_divergence,
                                                    flux_basis.oneD_vol_operator,
                                                    flux_basis.oneD_grad_operator);
        */

        // Strong form
        // The right-hand side sends all the term to the side of the source term
        // Therefore, 
        // \divergence ( Fconv + Fdiss ) = source 
        // has the right-hand side
        // rhs = - \divergence( Fconv + Fdiss ) + source 
        // Since we have done an integration by parts, the volume term resulting from the divergence of Fconv and Fdiss
        // is negative. Therefore, negative of negative means we add that volume term to the right-hand-side
        std::vector<adtype> rhs(n_shape_fns);

        // Convective
        if (this->all_parameters->use_split_form || this->all_parameters->use_curvilinear_split_form){
            std::vector<real> ones(n_quad_pts, 1.0);
            soln_basis.inner_product_1D(conv_flux_divergence, ones, rhs, soln_basis.oneD_vol_operator, false, -1.0);
        }
        else {
            soln_basis.inner_product_1D(conv_flux_divergence, vol_quad_weights, rhs, soln_basis.oneD_vol_operator, false, -1.0);
        }
/*
        // Diffusive
        // Note that for diffusion, the negative is defined in the physics. Since we used the auxiliary
        // variable, put a negative here.
        soln_basis.inner_product_1D(diffusive_flux_divergence, vol_quad_weights, rhs, soln_basis.oneD_vol_operator, true, -1.0);
*/
        // Manufactured source
        if(this->all_parameters->manufactured_convergence_study_param.manufactured_solution_param.use_manufactured_source_term) {
            std::vector<adtype> JxWxsource(n_quad_pts);
            for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
                JxWxsource[iquad] = vol_quad_weights[iquad] * metric_oper.det_Jac_vol[iquad]*source_at_q[istate][iquad];
            }
            std::vector<real> ones(n_quad_pts, 1.0);
            soln_basis.inner_product_1D(JxWxsource, ones, rhs, soln_basis.oneD_vol_operator, true, 1.0);
        }

        // Physical source
        if(pde_physics.has_nonzero_physical_source) {
            std::vector<adtype> JxWxphys_source(n_quad_pts);
            for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
                JxWxphys_source[iquad] = vol_quad_weights[iquad] * metric_oper.det_Jac_vol[iquad] * physical_source_at_q[istate][iquad];
            }
            std::vector<real> ones(n_quad_pts, 1.0);
            soln_basis.inner_product_1D(JxWxphys_source, ones, rhs, soln_basis.oneD_vol_operator, true, 1.0);
        }

        for(unsigned int ishape=0; ishape<n_shape_fns; ishape++){
            local_rhs_int_cell[istate*n_shape_fns + ishape] += rhs[ishape];
        }

    }
    
    if constexpr(std::is_same<adtype,double>::value)
    {
        if(this->compute_dRdW_strong)
        {
            for(unsigned int s1=0; s1<nstate; ++s1)
            {
                for(unsigned int s2=0; s2<nstate; ++s2)
                {
                    for(unsigned int idof_sol = 0; idof_sol<n_shape_fns; ++idof_sol)
                    {
                        std::vector<adtype> rhs_dRdW(n_shape_fns);
                        std::vector<real> ones(n_quad_pts, 1.0);
                        std::vector<real> dQFduh_at_quads(n_quad_pts);
                        for(unsigned int q_=0; q_<n_quad_pts; ++q_)
                        {
                            dQFduh_at_quads[q_] = dQF_duh[s1][s2][q_][idof_sol];
                        }
                        soln_basis.inner_product_1D(dQFduh_at_quads, ones, rhs_dRdW, soln_basis.oneD_vol_operator, false, -1.0);
                        const unsigned int idof_sol_global = s2*n_shape_fns + idof_sol;
                        for(unsigned int idof_res = 0; idof_res<n_shape_fns; ++idof_res)
                        {
                            const unsigned int idof_res_global = s1*n_shape_fns + idof_res;

                            this->dRdW_vol_cell[idof_res_global][idof_sol_global] = rhs_dRdW[idof_res];
                        }
                        
                    }
                }
            }
        }
    }

    std::vector<adtype> vol_term_br2;
    assemble_volume_term_entropystable_br2(
    poly_degree,
    entropy_var_coeffs,
    projected_entropy_var_at_q,
    n_quad_pts,  
    n_dofs_cell, 
    soln_basis,
    flux_basis,
    metric_oper,
    pde_physics,
    H_mat,
    soln_basis_projection_oper.oneD_G_operator_vol,
    soln_basis_projection_oper.oneD_DG_operator_vol,
    dvtilde_duh,
    soln_basis.oneD_vol_operator.n(),
    soln_basis.oneD_vol_operator.m(),
    vol_term_br2);

    if(this->compute_only_convective_residual)
    {
        //Already computed
    }
    else if(this->compute_only_dissipative_residual)
    {
        for(unsigned int idof = 0; idof<n_dofs_cell; ++idof)
        {
            local_rhs_int_cell[idof] = (-1.0)*vol_term_br2[idof];
        }
    }
    else
    { 
        for(unsigned int idof = 0; idof<n_dofs_cell; ++idof)
        {
            local_rhs_int_cell[idof] += (-1.0)*vol_term_br2[idof];
        }
    }
}

template <int dim, int nstate, typename real, typename MeshType>
template <typename adtype>
void DGStrong<dim,nstate,real,MeshType>::assemble_boundary_term_strong(
    const unsigned int                                                 iface, 
    const dealii::types::global_dof_index                              current_cell_index,
    const std::array<std::vector<adtype>,nstate>                       &soln_coeff,
    const std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate> &/*aux_soln_coeff*/,
    const unsigned int                                                 boundary_id,
    const unsigned int                                                 poly_degree, 
    const real                                                         /*penalty*/,
    OPERATOR::basis_functions<dim,2*dim>                               &soln_basis,
    OPERATOR::basis_functions<dim,2*dim>                               &flux_basis,
    OPERATOR::vol_projection_operator<dim,2*dim>                       &soln_basis_projection_oper,
    OPERATOR::metric_operators<adtype,dim,2*dim>                       &metric_oper,
    const Physics::PhysicsBase<dim, nstate, adtype>                    &pde_physics,
    const NumericalFlux::NumericalFluxConvective<dim, nstate, adtype>  &conv_num_flux,
    const NumericalFlux::NumericalFluxDissipative<dim, nstate, adtype> &/*diss_num_flux*/,
    std::vector<adtype>                                                &local_rhs_cell)
{
    (void) current_cell_index;

    const unsigned int n_face_quad_pts  = this->face_quadrature_collection[poly_degree].size();
    const unsigned int n_quad_pts_vol   = this->volume_quadrature_collection[poly_degree].size();
    const unsigned int n_quad_pts_1D    = this->oneD_quadrature_collection[poly_degree].size();
    const unsigned int n_dofs = this->fe_collection[poly_degree].dofs_per_cell;
    const unsigned int n_shape_fns = n_dofs / nstate; 
    const std::vector<double> &face_quad_weights = this->face_quadrature_collection[poly_degree].get_weights();

    // Interpolate the modal coefficients to the volume cubature nodes.
    std::array<std::vector<adtype>,nstate> soln_at_vol_q;
    std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate> aux_soln_at_vol_q;
    // Interpolate modal soln coefficients to the facet.
    std::array<std::vector<adtype>,nstate> soln_at_surf_q;
    std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate> aux_soln_at_surf_q;
    for(int istate=0; istate<nstate; ++istate){
        //allocate
        soln_at_vol_q[istate].resize(n_quad_pts_vol);
        //solve soln at volume cubature nodes
        soln_basis.matrix_vector_mult_1D(soln_coeff[istate], soln_at_vol_q[istate],
                                         soln_basis.oneD_vol_operator);

        //allocate
        soln_at_surf_q[istate].resize(n_face_quad_pts);
        //solve soln at facet cubature nodes
        soln_basis.matrix_vector_mult_surface_1D(iface,
                                                 soln_coeff[istate], soln_at_surf_q[istate],
                                                 soln_basis.oneD_surf_operator,
                                                 soln_basis.oneD_vol_operator);
/*
        for(int idim=0; idim<dim; idim++){
            //alocate
            aux_soln_at_vol_q[istate][idim].resize(n_quad_pts_vol);
            //solve auxiliary soln at volume cubature nodes
            soln_basis.matrix_vector_mult_1D(aux_soln_coeff[istate][idim], aux_soln_at_vol_q[istate][idim],
                                             soln_basis.oneD_vol_operator);

            //allocate
            aux_soln_at_surf_q[istate][idim].resize(n_face_quad_pts);
            //solve auxiliary soln at facet cubature nodes
            soln_basis.matrix_vector_mult_surface_1D(iface,
                                                     aux_soln_coeff[istate][idim], aux_soln_at_surf_q[istate][idim],
                                                     soln_basis.oneD_surf_operator,
                                                     soln_basis.oneD_vol_operator);
        }
    */
    }

    // Get volume reference fluxes and interpolate them to the facet.
    // Compute reference volume fluxes in both interior and exterior cells.

    // First we do interior.
    std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate> conv_ref_flux_at_vol_q;
    //std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate> diffusive_ref_flux_at_vol_q;
    for (unsigned int iquad=0; iquad<n_quad_pts_vol; ++iquad) {
        // Copy Metric Cofactor in a way can use for transforming Tensor Blocks to reference space
        // The way it is stored in metric_operators is to use sum-factorization in each direction,
        // but here it is cleaner to apply a reference transformation in each Tensor block returned by physics.
        dealii::Tensor<2,dim,adtype> metric_cofactor_vol;
        for(int idim=0; idim<dim; idim++){
            for(int jdim=0; jdim<dim; jdim++){
                metric_cofactor_vol[idim][jdim] = metric_oper.metric_cofactor_vol[idim][jdim][iquad];
            }
        }
        std::array<adtype,nstate> soln_state;
        //std::array<dealii::Tensor<1,dim,adtype>,nstate> aux_soln_state;
        for(int istate=0; istate<nstate; istate++){
            soln_state[istate] = soln_at_vol_q[istate][iquad];
            /*
            for(int idim=0; idim<dim; idim++){
                aux_soln_state[istate][idim] = aux_soln_at_vol_q[istate][idim][iquad];
            }
            */
        }

        // Evaluate physical convective flux
        std::array<dealii::Tensor<1,dim,adtype>,nstate> conv_phys_flux;
        if(!this->all_parameters->use_split_form && !this->all_parameters->use_curvilinear_split_form){
            conv_phys_flux = pde_physics.convective_flux (soln_state);
        }
/*
        // Compute the physical dissipative flux
        std::array<dealii::Tensor<1,dim,adtype>,nstate> diffusive_phys_flux;
        diffusive_phys_flux = pde_physics.dissipative_flux(soln_state, aux_soln_state, current_cell_index);
*/
        // Write the values in a way that we can use sum-factorization on.
        for(int istate=0; istate<nstate; istate++){
            dealii::Tensor<1,dim,adtype> conv_ref_flux;
            //dealii::Tensor<1,dim,adtype> diffusive_ref_flux;
            // transform the conservative convective physical flux to reference space
            if(!this->all_parameters->use_split_form && !this->all_parameters->use_curvilinear_split_form){
                metric_oper.transform_physical_to_reference(
                    conv_phys_flux[istate],
                    metric_cofactor_vol,
                    conv_ref_flux);
            }
            /*
            // transform the dissipative flux to reference space
            metric_oper.transform_physical_to_reference(
                diffusive_phys_flux[istate],
                metric_cofactor_vol,
                diffusive_ref_flux);
            */
            // Write the data in a way that we can use sum-factorization on.
            // Since sum-factorization improves the speed for matrix-vector multiplications,
            // We need the values to have their inner elements be vectors.
            for(int idim=0; idim<dim; idim++){
                //allocate
                if(iquad == 0){
                    conv_ref_flux_at_vol_q[istate][idim].resize(n_quad_pts_vol);
                    //diffusive_ref_flux_at_vol_q[istate][idim].resize(n_quad_pts_vol);
                }
                //write data
                if(!this->all_parameters->use_split_form && !this->all_parameters->use_curvilinear_split_form){
                    conv_ref_flux_at_vol_q[istate][idim][iquad] = conv_ref_flux[idim];
                }

                //diffusive_ref_flux_at_vol_q[istate][idim][iquad] = diffusive_ref_flux[idim];
            }
        }
    }

    // Interpolate the volume reference fluxes to the facet.
    // And do the dot product with the UNIT REFERENCE normal.
    // Since we are computing a dot product with the unit reference normal,
    // we exploit the fact that the unit reference normal has a value of 0 in all reference directions except
    // the outward reference normal dircetion.
    const dealii::Tensor<1,dim,double> unit_ref_normal_int = dealii::GeometryInfo<dim>::unit_normal_vector[iface];
    const int dim_not_zero = iface / 2;//reference direction of face integer division

    std::array<std::vector<adtype>,nstate> conv_int_vol_ref_flux_interp_to_face_dot_ref_normal;
    //std::array<std::vector<adtype>,nstate> diffusive_int_vol_ref_flux_interp_to_face_dot_ref_normal;
    for(int istate=0; istate<nstate; istate++){
        //allocate
        conv_int_vol_ref_flux_interp_to_face_dot_ref_normal[istate].resize(n_face_quad_pts);
        //diffusive_int_vol_ref_flux_interp_to_face_dot_ref_normal[istate].resize(n_face_quad_pts);

        //solve
        //Note, since the normal is zero in all other reference directions, we only have to interpolate one given reference direction to the facet

        //interpolate reference volume convective flux to the facet, and apply unit reference normal as scaled by 1.0 or -1.0
        if(!this->all_parameters->use_split_form && !this->all_parameters->use_curvilinear_split_form){
            flux_basis.matrix_vector_mult_surface_1D(iface, 
                                                     conv_ref_flux_at_vol_q[istate][dim_not_zero],
                                                     conv_int_vol_ref_flux_interp_to_face_dot_ref_normal[istate],
                                                     flux_basis.oneD_surf_operator,//the flux basis interpolates from the flux nodes
                                                     flux_basis.oneD_vol_operator,
                                                     false, unit_ref_normal_int[dim_not_zero]);//don't add to previous value, scale by unit_normal int
        }
/*
        //interpolate reference volume dissipative flux to the facet, and apply unit reference normal as scaled by 1.0 or -1.0
        flux_basis.matrix_vector_mult_surface_1D(iface, 
                                                 diffusive_ref_flux_at_vol_q[istate][dim_not_zero],
                                                 diffusive_int_vol_ref_flux_interp_to_face_dot_ref_normal[istate],
                                                 flux_basis.oneD_surf_operator,
                                                 flux_basis.oneD_vol_operator,
                                                 false, unit_ref_normal_int[dim_not_zero]);
*/
    }

    //Note that for entropy-dissipation and entropy stability, the conservative variables
    //are functions of projected entropy variables. For Euler etc, the transformation is nonlinear
    //so careful attention to what is evaluated where and interpolated to where is needed.
    //For further information, please see Chan, Jesse. "On discretely entropy conservative and entropy stable discontinuous Galerkin methods." Journal of Computational Physics 362 (2018): 346-374.
    //pages 355 (Eq. 57 with text around it) and  page 359 (Eq 86 and text below it).

    // First, transform the volume conservative solution at volume cubature nodes to entropy variables.
    std::array<std::vector<adtype>,nstate> entropy_var_vol;
    std::vector<std::array<std::array<real,nstate>,nstate>> dv_du(n_quad_pts_vol);
    for(unsigned int iquad=0; iquad<n_quad_pts_vol; iquad++){
        std::array<adtype,nstate> soln_state;
        for(int istate=0; istate<nstate; istate++){
            soln_state[istate] = soln_at_vol_q[istate][iquad];
        }
        std::array<adtype,nstate> entropy_var;
        entropy_var = pde_physics.compute_entropy_variables(soln_state);
        if constexpr(std::is_same<adtype,double>::value)
        {
            if(this->compute_dRdW_strong)
            {
                pde_physics.get_d_entropy_var_d_conservative_var(dv_du[iquad],entropy_var); 
            }
        }
        for(int istate=0; istate<nstate; istate++){
            if(iquad==0){
                entropy_var_vol[istate].resize(n_quad_pts_vol);
            }
            entropy_var_vol[istate][iquad] = entropy_var[istate];
        }
    }
    std::array<std::array<dealii::FullMatrix<double>,nstate>,nstate> dvtilde_duh_vol;
    std::array<std::array<dealii::FullMatrix<double>,nstate>,nstate> dvtilde_duh_surf;
    xt::xarray<double> H_mat;
    if constexpr(std::is_same<adtype,double>::value)
    {
        if(this->compute_dRdW_strong)
        {
            compute_H_mat(soln_basis_projection_oper.oneD_G_operator_vol, soln_basis.oneD_vol_operator, metric_oper.det_Jac_vol, dv_du, H_mat);
            compute_dvtilde_duh(soln_basis_projection_oper.oneD_G_operator_vol,soln_basis_projection_oper.oneD_G_operator_vol,soln_basis_projection_oper.oneD_G_operator_vol,H_mat, soln_basis.oneD_vol_operator.n(),n_quad_pts_1D,dvtilde_duh_vol);
            if(iface==0)
            {
                compute_dvtilde_duh(soln_basis_projection_oper.oneD_G_operator_surf[0],soln_basis_projection_oper.oneD_G_operator_vol,soln_basis_projection_oper.oneD_G_operator_vol,H_mat, soln_basis.oneD_vol_operator.n(),n_quad_pts_1D,dvtilde_duh_surf);
            }
            else if(iface==1)
            {
                compute_dvtilde_duh(soln_basis_projection_oper.oneD_G_operator_surf[1],soln_basis_projection_oper.oneD_G_operator_vol,soln_basis_projection_oper.oneD_G_operator_vol,H_mat, soln_basis.oneD_vol_operator.n(),n_quad_pts_1D,dvtilde_duh_surf);
            }
            else if(iface==2)
            {
                compute_dvtilde_duh(soln_basis_projection_oper.oneD_G_operator_vol,soln_basis_projection_oper.oneD_G_operator_surf[0],soln_basis_projection_oper.oneD_G_operator_vol,H_mat, soln_basis.oneD_vol_operator.n(),n_quad_pts_1D,dvtilde_duh_surf);
            }
            else if(iface==3)
            {
                compute_dvtilde_duh(soln_basis_projection_oper.oneD_G_operator_vol,soln_basis_projection_oper.oneD_G_operator_surf[1],soln_basis_projection_oper.oneD_G_operator_vol,H_mat, soln_basis.oneD_vol_operator.n(),n_quad_pts_1D,dvtilde_duh_surf);
            }
            else if(iface==4)
            {
                compute_dvtilde_duh(soln_basis_projection_oper.oneD_G_operator_vol,soln_basis_projection_oper.oneD_G_operator_vol,soln_basis_projection_oper.oneD_G_operator_surf[0],H_mat, soln_basis.oneD_vol_operator.n(),n_quad_pts_1D,dvtilde_duh_surf);
            }
            else if(iface==5)
            {
                compute_dvtilde_duh(soln_basis_projection_oper.oneD_G_operator_vol,soln_basis_projection_oper.oneD_G_operator_vol,soln_basis_projection_oper.oneD_G_operator_surf[1],H_mat, soln_basis.oneD_vol_operator.n(),n_quad_pts_1D,dvtilde_duh_surf);
            }
        }
    }

    //project it onto the solution basis functions and interpolate it
    std::array<std::vector<adtype>,nstate> projected_entropy_var_vol;
    std::array<std::vector<adtype>,nstate> projected_entropy_var_surf;
    std::array<std::vector<adtype>,nstate> entropy_var_coeffs;
    for(int istate=0; istate<nstate; istate++){
        // allocate
        projected_entropy_var_vol[istate].resize(n_quad_pts_vol);
        projected_entropy_var_surf[istate].resize(n_face_quad_pts);
        entropy_var_coeffs[istate].resize(n_shape_fns);

        //interior
        //soln_basis_projection_oper.matrix_vector_mult_1D(entropy_var_vol[istate],
        //                                                 entropy_var_coeffs[istate],
        //                                                 soln_basis_projection_oper.oneD_vol_operator);
        soln_basis_projection_oper.weight_adjusted_vol_projection(entropy_var_vol[istate],
                                                                  entropy_var_coeffs[istate],
                                                                  n_quad_pts_vol,
                                                                  n_shape_fns,
                                                                  soln_basis,
                                                                  soln_basis_projection_oper,
                                                                  metric_oper.det_Jac_vol);
        soln_basis.matrix_vector_mult_1D(entropy_var_coeffs[istate],
                                         projected_entropy_var_vol[istate],
                                         soln_basis.oneD_vol_operator);
        soln_basis.matrix_vector_mult_surface_1D(iface,
                                                 entropy_var_coeffs[istate], 
                                                 projected_entropy_var_surf[istate],
                                                 soln_basis.oneD_surf_operator,
                                                 soln_basis.oneD_vol_operator);
    }

    //get the surface-volume sparsity pattern for a "sum-factorized" Hadamard product only computing terms needed for the operation.
    const unsigned int row_size = n_face_quad_pts * n_quad_pts_1D;
    const unsigned int col_size = n_face_quad_pts * n_quad_pts_1D;
    std::vector<unsigned int> Hadamard_rows_sparsity(row_size);
    std::vector<unsigned int> Hadamard_columns_sparsity(col_size);
    if(this->all_parameters->use_split_form || this->all_parameters->use_curvilinear_split_form){
        flux_basis.sum_factorized_Hadamard_surface_sparsity_pattern(n_face_quad_pts, n_quad_pts_1D, Hadamard_rows_sparsity, Hadamard_columns_sparsity, dim_not_zero);
    }

    std::array<std::array<dealii::FullMatrix<double>,nstate>,nstate> dterm1_duh;
    std::array<std::array<dealii::FullMatrix<double>,nstate>,nstate> dterm2_duh;
    std::array<std::array<dealii::FullMatrix<double>,nstate>,nstate> dterm3_duh;
    std::array<std::array<double,nstate>,nstate> dutilde_dvtilde_face;
    std::array<std::array<double,nstate>,nstate> dutilde_dvtilde_vol;
    if constexpr(std::is_same<adtype,double>::value)
    {
        if(this->compute_dRdW_strong)
        {
            for(unsigned int s1=0; s1<nstate; ++s1)
            {
                for(unsigned int s2=0; s2<nstate; ++s2)
                {
                    dterm1_duh[s1][s2].reinit(n_quad_pts_vol,n_shape_fns);
                    dterm2_duh[s1][s2].reinit(n_face_quad_pts,n_shape_fns);
                    dterm3_duh[s1][s2].reinit(n_face_quad_pts,n_shape_fns);
                }
            }
        }
    }

    std::array<std::vector<adtype>,nstate> surf_vol_ref_2pt_flux_interp_surf;
    std::array<std::vector<adtype>,nstate> surf_vol_ref_2pt_flux_interp_vol;
    if(this->all_parameters->use_split_form || this->all_parameters->use_curvilinear_split_form){
        //get surface-volume hybrid 2pt flux from Eq.(15) in Chan, Jesse. "Skew-symmetric entropy stable modal discontinuous Galerkin formulations." Journal of Scientific Computing 81.1 (2019): 459-485.
        std::array<std::vector<adtype>,nstate> surface_ref_2pt_flux;
        //make use of the sparsity pattern from above to assemble only n^d non-zero entries without ever allocating not computing zeros.
        for(int istate=0; istate<nstate; istate++){
            surface_ref_2pt_flux[istate].resize(n_face_quad_pts * n_quad_pts_1D);
        }
        for(unsigned int iquad_face=0; iquad_face<n_face_quad_pts; iquad_face++){
            dealii::Tensor<2,dim,adtype> metric_cofactor_surf;
            for(int idim=0; idim<dim; idim++){
                for(int jdim=0; jdim<dim; jdim++){
                    metric_cofactor_surf[idim][jdim] = metric_oper.metric_cofactor_surf[idim][jdim][iquad_face];
                }
            }
             
            //Compute the conservative values on the facet from the interpolated entorpy variables.
            std::array<adtype,nstate> entropy_var_face;
            for(int istate=0; istate<nstate; istate++){
                entropy_var_face[istate] = projected_entropy_var_surf[istate][iquad_face];
            }
            std::array<adtype,nstate> soln_state_face;
            soln_state_face= pde_physics.compute_conservative_variables_from_entropy_variables (entropy_var_face);
            if constexpr(std::is_same<adtype,double>::value)
            {
                if(this->compute_dRdW_strong)
                {
                    pde_physics.get_d_conservative_var_d_entropy_var(dutilde_dvtilde_face, entropy_var_face);
                }
            }

            //only do the n_quad_1D vol points that give non-zero entries from Hadamard product.
            for(unsigned int row_index = iquad_face * n_quad_pts_1D, column_index = 0; 
                column_index < n_quad_pts_1D;
                row_index++, column_index++){

                if(Hadamard_rows_sparsity[row_index] != iquad_face){
                    pcout<<"The boundary Hadamard rows sparsity pattern does not match."<<std::endl;
                    std::abort();
                }

                const unsigned int iquad_vol = Hadamard_columns_sparsity[row_index];//extract flux_quad pt that corresponds to a non-zero entry for Hadamard product.
                // Copy Metric Cofactor in a way can use for transforming Tensor Blocks to reference space
                // The way it is stored in metric_operators is to use sum-factorization in each direction,
                // but here it is cleaner to apply a reference transformation in each Tensor block returned by physics.
                dealii::Tensor<2,dim,adtype> metric_cofactor_vol;
                for(int idim=0; idim<dim; idim++){
                    for(int jdim=0; jdim<dim; jdim++){
                        metric_cofactor_vol[idim][jdim] = metric_oper.metric_cofactor_vol[idim][jdim][iquad_vol];
                    }
                }
                dealii::Tensor<2,dim,adtype> metric_cofactor_split;
                for(int idim=0; idim<dim; idim++){
                    for(int jdim=0; jdim<dim; jdim++){
                        metric_cofactor_split[idim][jdim] = 0.5 * (metric_cofactor_surf[idim][jdim] + metric_cofactor_vol[idim][jdim]);
                    }
                }
                std::array<adtype,nstate> entropy_var;
                for(int istate=0; istate<nstate; istate++){
                    entropy_var[istate] = projected_entropy_var_vol[istate][iquad_vol];
                }
                std::array<adtype,nstate> soln_state;
                soln_state = pde_physics.compute_conservative_variables_from_entropy_variables (entropy_var);
                if constexpr(std::is_same<adtype,double>::value)
                {
                    if(this->compute_dRdW_strong)
                    {
                        pde_physics.get_d_conservative_var_d_entropy_var(dutilde_dvtilde_vol, entropy_var);
                    }
                }
                //Note that the flux basis is collocated on the volume cubature set so we don't need to evaluate the entropy variables
                //on the volume set then transform back to the conservative variables since the flux basis volume
                //projection is identity.

                //Compute the physical flux
                std::array<dealii::Tensor<1,dim,adtype>,nstate> conv_phys_flux_2pt;
                conv_phys_flux_2pt = pde_physics.convective_numerical_split_flux(soln_state, soln_state_face);
                for(int istate=0; istate<nstate; istate++){
                    dealii::Tensor<1,dim,adtype> conv_ref_flux_2pt;
                    //For each state, transform the physical flux to a reference flux.
                    metric_oper.transform_physical_to_reference(
                        conv_phys_flux_2pt[istate],
                        metric_cofactor_split,
                        conv_ref_flux_2pt);
                    //only store the dim not zero in reference space bc dot product with unit ref normal later.
                    surface_ref_2pt_flux[istate][iquad_face * n_quad_pts_1D + column_index] = conv_ref_flux_2pt[dim_not_zero];
                }
                if constexpr(std::is_same<adtype,double>::value)
                {
                    if(this->compute_dRdW_strong)
                    {
                        std::array<std::array<std::array<double,nstate>,nstate>,2> dFsplit_dutilde = pde_physics.convective_numerical_split_flux_derivative(soln_state, soln_state_face, metric_cofactor_split, unit_ref_normal_int);
                        double mult_factor=0;
                        const int iface_1D = iface % 2;//the reference face number
                        if(iface==0 || iface==1)
                        {
                            const int L = iquad_vol % n_quad_pts_1D;
                            mult_factor = flux_basis.oneD_surf_operator[iface_1D][0][L]*face_quad_weights[iquad_face];
                        }
                        else if(iface==2 || iface==3)
                        {
                            const int M = (iquad_vol/n_quad_pts_1D) % n_quad_pts_1D;
                            mult_factor = flux_basis.oneD_surf_operator[iface_1D][0][M]*face_quad_weights[iquad_face];
                        }
                        else if(iface==4 || iface==5)
                        {
                            const int N = (iquad_vol/n_quad_pts_1D) / n_quad_pts_1D;
                            mult_factor = flux_basis.oneD_surf_operator[iface_1D][0][N]*face_quad_weights[iquad_face];
                        }
                        // Form term1 and term2
                        for(unsigned int s1=0; s1<nstate; ++s1)
                        {
                            for(unsigned int s2=0; s2<nstate; ++s2)
                            {
                                for(unsigned int idof = 0; idof<n_shape_fns; ++idof)
                                {
                                    dterm1_duh[s1][s2][iquad_vol][idof]=0;
                                    for(unsigned int sk=0; sk<nstate; ++sk)
                                    {
                                        for(unsigned int sl=0; sl<nstate; ++sl)
                                        {
                                            dterm1_duh[s1][s2][iquad_vol][idof]+= mult_factor*(dFsplit_dutilde[0][s1][sk]*dutilde_dvtilde_vol[sk][sl]*dvtilde_duh_vol[sl][s2][iquad_vol][idof] + dFsplit_dutilde[1][s1][sk]*dutilde_dvtilde_face[sk][sl]*dvtilde_duh_surf[sl][s2][iquad_face][idof]);
                                            dterm2_duh[s1][s2][iquad_face][idof] -= mult_factor*(dFsplit_dutilde[0][s1][sk]*dutilde_dvtilde_vol[sk][sl]*dvtilde_duh_vol[sl][s2][iquad_vol][idof] + dFsplit_dutilde[1][s1][sk]*dutilde_dvtilde_face[sk][sl]*dvtilde_duh_surf[sl][s2][iquad_face][idof]);
                                        }
                                    }
                                }
                            }
                        }
                    } // if dRdW_strong
                }
            }
        }
        //get the surface basis operator from Hadamard sparsity pattern
        //to be applied at n^d operations (on the face so n^{d+1-1}=n^d flops)
        //also only allocates n^d terms.
        const int iface_1D = iface % 2;//the reference face number
        const std::vector<double> &oneD_quad_weights_vol= this->oneD_quadrature_collection[poly_degree].get_weights();
        dealii::FullMatrix<real> surf_oper_sparse(n_face_quad_pts, n_quad_pts_1D);
        flux_basis.sum_factorized_Hadamard_surface_basis_assembly(n_face_quad_pts, n_quad_pts_1D, 
                                                                  Hadamard_rows_sparsity, Hadamard_columns_sparsity,
                                                                  flux_basis.oneD_surf_operator[iface_1D], 
                                                                  oneD_quad_weights_vol,
                                                                  surf_oper_sparse,
                                                                  dim_not_zero);

        // Apply the surface Hadamard products and multiply with vector of ones for both off diagonal terms in
        // Eq.(15) in Chan, Jesse. "Skew-symmetric entropy stable modal discontinuous Galerkin formulations." Journal of Scientific Computing 81.1 (2019): 459-485.
        for(int istate=0; istate<nstate; istate++){
            //first apply Hadamard product with the structure made above.
            std::vector<adtype> surface_ref_2pt_flux_int_Hadamard_with_surf_oper(n_face_quad_pts * n_quad_pts_1D);
            flux_basis.Hadamard_product_AD_vector(surf_oper_sparse, 
                                        surface_ref_2pt_flux[istate], 
                                        surface_ref_2pt_flux_int_Hadamard_with_surf_oper);
            //sum with reference unit normal
            surf_vol_ref_2pt_flux_interp_surf[istate].resize(n_face_quad_pts);
            surf_vol_ref_2pt_flux_interp_vol[istate].resize(n_quad_pts_vol);
         
            for(unsigned int iface_quad=0; iface_quad<n_face_quad_pts; iface_quad++){
                for(unsigned int iquad_int=0; iquad_int<n_quad_pts_1D; iquad_int++){
                    surf_vol_ref_2pt_flux_interp_surf[istate][iface_quad] 
                        -= surface_ref_2pt_flux_int_Hadamard_with_surf_oper[iface_quad * n_quad_pts_1D + iquad_int]
                        * unit_ref_normal_int[dim_not_zero];
                    const unsigned int column_index = iface_quad * n_quad_pts_1D + iquad_int;
                    surf_vol_ref_2pt_flux_interp_vol[istate][Hadamard_columns_sparsity[column_index]] 
                        += surface_ref_2pt_flux_int_Hadamard_with_surf_oper[iface_quad * n_quad_pts_1D + iquad_int]
                        * unit_ref_normal_int[dim_not_zero];
                }
            }
        }
    }//end of if split form or curvilinear split form


    //the outward reference normal dircetion.
    std::array<std::vector<adtype>,nstate> conv_flux_dot_normal;
    std::vector<dealii::Tensor<1,dim,adtype>> unit_phys_normals(n_face_quad_pts);
    std::vector<adtype> JxW_face(n_face_quad_pts);
    //std::array<std::vector<adtype>,nstate> diss_flux_dot_normal_diff;
    // Get surface numerical fluxes
    for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) { 
        // Copy Metric Cofactor on the facet in a way can use for transforming Tensor Blocks to reference space
        // The way it is stored in metric_operators is to use sum-factorization in each direction,
        // but here it is cleaner to apply a reference transformation in each Tensor block returned by physics.
        // Note that for a conforming mesh, the facet metric cofactor matrix is the same from either interioir or exterior metric terms. 
        // This is verified for the metric computations in: unit_tests/operator_tests/surface_conforming_test.cpp
        dealii::Tensor<2,dim,adtype> metric_cofactor_surf;
        for(int idim=0; idim<dim; idim++){
            for(int jdim=0; jdim<dim; jdim++){
                metric_cofactor_surf[idim][jdim] = metric_oper.metric_cofactor_surf[idim][jdim][iquad];
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
        JxW_face[iquad] = face_Jac_norm_scaled*face_quad_weights[iquad];
        unit_phys_normal_int /= face_Jac_norm_scaled;//normalize it.
        unit_phys_normals[iquad] = unit_phys_normal_int;

        //get the projected entropy variables, soln, and 
        //auxiliary solution on the surface point.
        std::array<adtype,nstate> entropy_var_face_int;
        std::array<dealii::Tensor<1,dim,adtype>,nstate> aux_soln_state_int;
        std::array<adtype,nstate> soln_interp_to_face_int;
        for(int istate=0; istate<nstate; istate++){
            soln_interp_to_face_int[istate] = soln_at_surf_q[istate][iquad];
            entropy_var_face_int[istate] = projected_entropy_var_surf[istate][iquad];
            /*
            for(int idim=0; idim<dim; idim++){
                aux_soln_state_int[istate][idim] = aux_soln_at_surf_q[istate][idim][iquad];
            }
            */
        }

        //extract solution on surface from projected entropy variables
        std::array<adtype,nstate> soln_state_int;
        soln_state_int = pde_physics.compute_conservative_variables_from_entropy_variables (entropy_var_face_int);
        if constexpr(std::is_same<adtype,double>::value)
        {
            if(this->compute_dRdW_strong)
            {
                pde_physics.get_d_conservative_var_d_entropy_var(dutilde_dvtilde_face, entropy_var_face_int);
            }
        }

        if(!this->all_parameters->use_split_form && !this->all_parameters->use_curvilinear_split_form){
            for(int istate=0; istate<nstate; istate++){
                soln_state_int[istate] = soln_at_surf_q[istate][iquad];
            }
        }

        std::array<adtype,nstate> soln_boundary;
        std::array<dealii::Tensor<1,dim,adtype>,nstate> grad_soln_boundary;
        dealii::Point<dim,adtype> surf_flux_node;
        for(int idim=0; idim<dim; idim++){
            surf_flux_node[idim] = metric_oper.flux_nodes_surf[iface][idim][iquad];
        }
        //I am not sure if BC should be from solution interpolated to face
        //or solution from the projected entropy variables.
        //Now, it uses projected entropy variables for NSFR, and solution
        //interpolated to face for conservative DG.
        const unsigned int boundary_id_passed_conv = ((boundary_id==1001 || boundary_id==1010) && (this->all_parameters->use_split_form || this->all_parameters->use_curvilinear_split_form)) ? 1006 : boundary_id;
        pde_physics.boundary_face_values (boundary_id_passed_conv, surf_flux_node, unit_phys_normal_int, soln_state_int, aux_soln_state_int, soln_boundary, grad_soln_boundary);
        
        // Convective numerical flux.
        std::array<adtype,nstate> conv_num_flux_dot_n_at_q;
        conv_num_flux_dot_n_at_q = conv_num_flux.evaluate_flux(soln_state_int, soln_boundary, unit_phys_normal_int);
        if constexpr(std::is_same<adtype,double>::value)
        {
            if(this->compute_dRdW_strong)
            {
                std::array<std::array<double,nstate>,nstate> d_ubc_d_utilde = pde_physics.compute_d_solnbc_d_u(soln_state_int, unit_phys_normal_int,boundary_id_passed_conv);
                dealii::Tensor<2,dim,double> metric_identity;
                for(unsigned int d=0; d<dim; ++d)
                {
                    metric_identity[d][d]=1.0;
                }
                std::array<std::array<std::array<double,nstate>,nstate>,2> dFsplit_dutilde = pde_physics.convective_numerical_split_flux_derivative(soln_state_int, soln_boundary, metric_identity, unit_phys_normal_int);

                for(unsigned int s1=0; s1<nstate; ++s1)
                {
                    for(unsigned int s2=0; s2<nstate; ++s2)
                    {
                        for(unsigned int idof=0; idof<n_shape_fns; ++idof)
                        {
                            double sum1 = 0;
                            double sum2 = 0;
                            for(unsigned int sk=0; sk<nstate; ++sk)
                            {
                                for(unsigned int sl=0; sl<nstate; ++sl)
                                {
                                    sum1 += dFsplit_dutilde[0][s1][sk]*dutilde_dvtilde_face[sk][sl]*dvtilde_duh_surf[sl][s2][iquad][idof];
                                    for(unsigned int sm=0; sm<nstate; ++sm)
                                    {
                                        sum2 += dFsplit_dutilde[1][s1][sk]*d_ubc_d_utilde[sk][sl]*dutilde_dvtilde_face[sl][sm]*dvtilde_duh_surf[sm][s2][iquad][idof];
                                    }
                                }
                            }
                            dterm3_duh[s1][s2][iquad][idof] = JxW_face[iquad]*(sum1 + sum2);
                        }
                    }
                }
            } // compute_dRdW_strong
        }
       /* 
        // Dissipative numerical flux
        std::array<adtype,nstate> diss_auxi_num_flux_dot_n_at_q;
        diss_auxi_num_flux_dot_n_at_q = diss_num_flux.evaluate_auxiliary_flux(
            current_cell_index, current_cell_index,
            0.0, 0.0,
            soln_interp_to_face_int, soln_boundary,
            aux_soln_state_int, grad_soln_boundary,
            unit_phys_normal_int, penalty, true);
*/
        for(int istate=0; istate<nstate; istate++){
            // allocate
            if(iquad==0){
                conv_flux_dot_normal[istate].resize(n_face_quad_pts);
                //diss_flux_dot_normal_diff[istate].resize(n_face_quad_pts);
            }
            // write data
            conv_flux_dot_normal[istate][iquad] = face_Jac_norm_scaled * conv_num_flux_dot_n_at_q[istate];
            /*
            diss_flux_dot_normal_diff[istate][iquad] = face_Jac_norm_scaled * diss_auxi_num_flux_dot_n_at_q[istate]
                                                     - diffusive_int_vol_ref_flux_interp_to_face_dot_ref_normal[istate][iquad];
            */
        }
    }

    //solve rhs
    for(int istate=0; istate<nstate; istate++){
        std::vector<adtype> rhs(n_shape_fns);
        //Convective flux on the facet
        if(this->all_parameters->use_split_form || this->all_parameters->use_curvilinear_split_form){
            std::vector<real> ones_surf(n_face_quad_pts, 1.0);
            soln_basis.inner_product_surface_1D(iface, 
                                                surf_vol_ref_2pt_flux_interp_surf[istate], 
                                                ones_surf, rhs, 
                                                soln_basis.oneD_surf_operator, 
                                                soln_basis.oneD_vol_operator,
                                                false, -1.0);
            std::vector<real> ones_vol(n_quad_pts_vol, 1.0);
            soln_basis.inner_product_1D(surf_vol_ref_2pt_flux_interp_vol[istate], 
                                            ones_vol, rhs, 
                                            soln_basis.oneD_vol_operator, 
                                            true, -1.0);
        }
        else{
            soln_basis.inner_product_surface_1D(iface, conv_int_vol_ref_flux_interp_to_face_dot_ref_normal[istate], 
                                                face_quad_weights, rhs, 
                                                soln_basis.oneD_surf_operator, 
                                                soln_basis.oneD_vol_operator,
                                                false, 1.0);//adding=false, scaled by factor=-1.0 bc subtract it
        }
        //Convective surface nnumerical flux.
        soln_basis.inner_product_surface_1D(iface, conv_flux_dot_normal[istate], 
                                            face_quad_weights, rhs, 
                                            soln_basis.oneD_surf_operator, 
                                            soln_basis.oneD_vol_operator,
                                            true, -1.0);//adding=true, scaled by factor=-1.0 bc subtract it
        /*
        //Dissipative surface numerical flux.
        soln_basis.inner_product_surface_1D(iface, diss_flux_dot_normal_diff[istate], 
                                            face_quad_weights, rhs, 
                                            soln_basis.oneD_surf_operator, 
                                            soln_basis.oneD_vol_operator,
                                            true, -1.0);//adding=true, scaled by factor=-1.0 bc subtract it
        */

        for(unsigned int ishape=0; ishape<n_shape_fns; ishape++){
            local_rhs_cell[istate*n_shape_fns + ishape] += rhs[ishape];
        }
    }
    
    if constexpr(std::is_same<adtype,double>::value)
    {
        if(this->compute_dRdW_strong)
        {
            std::vector<real> ones_vol(n_quad_pts_vol, 1.0);
            std::vector<real> ones_face(n_face_quad_pts, 1.0);
            for(unsigned int s1=0; s1<nstate; ++s1)
            {
                for(unsigned int s2=0; s2<nstate; ++s2)
                {
                    for(unsigned int idof_sol = 0; idof_sol<n_shape_fns; ++idof_sol)
                    {
                        std::vector<adtype> rhs_dRdW(n_shape_fns);
                        std::vector<real> dterm1_at_quads(n_quad_pts_vol);
                        std::vector<real> dterm23_at_quads(n_face_quad_pts);
                        for(unsigned int q_=0; q_<n_quad_pts_vol; ++q_)
                        {
                            dterm1_at_quads[q_] = dterm1_duh[s1][s2][q_][idof_sol];
                        }
                        for(unsigned int q_=0; q_<n_face_quad_pts; ++q_)
                        {
                            dterm23_at_quads[q_] = dterm2_duh[s1][s2][q_][idof_sol] + dterm3_duh[s1][s2][q_][idof_sol];
                        }
                        soln_basis.inner_product_1D(dterm1_at_quads, ones_vol, rhs_dRdW, soln_basis.oneD_vol_operator, false, -1.0);
                        soln_basis.inner_product_surface_1D(iface, dterm23_at_quads, 
                                                            ones_face, rhs_dRdW, 
                                                            soln_basis.oneD_surf_operator, 
                                                            soln_basis.oneD_vol_operator,
                                                            true, -1.0);//adding=true, scaled by factor=-1.0 bc subtract it
                        const unsigned int idof_sol_global = s2*n_shape_fns + idof_sol;
                        for(unsigned int idof_res = 0; idof_res<n_shape_fns; ++idof_res)
                        {
                            const unsigned int idof_res_global = s1*n_shape_fns + idof_res;
                            this->dRdW_boundary[idof_res_global][idof_sol_global] = rhs_dRdW[idof_res];
                        }                    
                    }
                }
            }
        }
    }

    std::vector<adtype> boundary_term_br2;
    assemble_boundary_term_entropystable_br2(
    iface,
    boundary_id,
    poly_degree,
    entropy_var_coeffs,
    projected_entropy_var_vol,
    projected_entropy_var_surf,
    dvtilde_duh_surf,
    dvtilde_duh_vol,
    H_mat, 
    soln_basis_projection_oper.oneD_G_operator_vol,
    soln_basis_projection_oper.oneD_DG_operator_vol,
    soln_basis.oneD_vol_operator.n(),
    soln_basis.oneD_vol_operator.m(),
    n_quad_pts_vol,  
    n_face_quad_pts,  
    n_dofs, 
    soln_basis,
    flux_basis,
    metric_oper,
    unit_phys_normals,
    JxW_face,
    pde_physics,
    boundary_term_br2);
    
    if(this->compute_only_convective_residual)
    {
        //Already computed
    }
    else if(this->compute_only_dissipative_residual)
    {
        for(unsigned int idof = 0; idof<n_dofs; ++idof)
        {
            local_rhs_cell[idof] = (-1.0)*boundary_term_br2[idof];
        }
    }
    else
    { 
        for(unsigned int idof = 0; idof<n_dofs; ++idof)
        {
            local_rhs_cell[idof] += (-1.0)*boundary_term_br2[idof];
        }
    }

}


template <int dim, int nstate, typename real, typename MeshType>
template <typename adtype>
void DGStrong<dim,nstate,real,MeshType>::assemble_face_term_strong(
    const unsigned int                                                 iface, 
    const unsigned int                                                 neighbor_iface, 
    const dealii::types::global_dof_index                              current_cell_index,
    const dealii::types::global_dof_index                              neighbor_cell_index,
    const std::array<std::vector<adtype>,nstate>                       &soln_coeff_int,
    const std::array<std::vector<adtype>,nstate>                       &soln_coeff_ext,
    const std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate> &/*aux_soln_coeff_int*/,
    const std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate> &/*aux_soln_coeff_ext*/,
    const unsigned int                                                 poly_degree_int, 
    const unsigned int                                                 poly_degree_ext, 
    const real                                                         /*penalty*/,
    OPERATOR::basis_functions<dim,2*dim>                               &soln_basis_int,
    OPERATOR::basis_functions<dim,2*dim>                               &soln_basis_ext,
    OPERATOR::basis_functions<dim,2*dim>                               &flux_basis_int,
    OPERATOR::basis_functions<dim,2*dim>                               &flux_basis_ext,
    OPERATOR::vol_projection_operator<dim,2*dim>                       &soln_basis_projection_oper_int,
    OPERATOR::vol_projection_operator<dim,2*dim>                       &soln_basis_projection_oper_ext,
    OPERATOR::metric_operators<adtype,dim,2*dim>                       &metric_oper_int,
    OPERATOR::metric_operators<adtype,dim,2*dim>                       &metric_oper_ext,
    const Physics::PhysicsBase<dim, nstate, adtype>                    &pde_physics,
    const NumericalFlux::NumericalFluxConvective<dim, nstate, adtype>  &conv_num_flux,
    const NumericalFlux::NumericalFluxDissipative<dim, nstate, adtype> &/*diss_num_flux*/,
    std::vector<adtype>                                                  &local_rhs_int_cell,
    std::vector<adtype>                                                  &local_rhs_ext_cell)
{
    (void) current_cell_index;
    (void) neighbor_cell_index;

    const unsigned int n_face_quad_pts = this->face_quadrature_collection[poly_degree_int].size();//assume interior cell does the work
    const std::vector<double> &surf_quad_weights = this->face_quadrature_collection[poly_degree_int].get_weights();

    const unsigned int n_quad_pts_vol_int  = this->volume_quadrature_collection[poly_degree_int].size();
    const unsigned int n_quad_pts_vol_ext  = this->volume_quadrature_collection[poly_degree_ext].size();
    const unsigned int n_quad_pts_1D_int  = this->oneD_quadrature_collection[poly_degree_int].size();
    const unsigned int n_quad_pts_1D_ext  = this->oneD_quadrature_collection[poly_degree_ext].size();

    const unsigned int n_dofs_int = this->fe_collection[poly_degree_int].dofs_per_cell;
    const unsigned int n_dofs_ext = this->fe_collection[poly_degree_ext].dofs_per_cell;

    const unsigned int n_shape_fns_int = n_dofs_int / nstate;
    const unsigned int n_shape_fns_ext = n_dofs_ext / nstate;

    // Interpolate the modal coefficients to the volume cubature nodes.
    std::array<std::vector<adtype>,nstate> soln_at_vol_q_int;
    std::array<std::vector<adtype>,nstate> soln_at_vol_q_ext;
    //std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate> aux_soln_at_vol_q_int;
    //std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate> aux_soln_at_vol_q_ext;
    // Interpolate modal soln coefficients to the facet.
    std::array<std::vector<adtype>,nstate> soln_at_surf_q_int;
    std::array<std::vector<adtype>,nstate> soln_at_surf_q_ext;
    //std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate> aux_soln_at_surf_q_int;
    //std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate> aux_soln_at_surf_q_ext;
    for(int istate=0; istate<nstate; ++istate){
        // allocate
        soln_at_vol_q_int[istate].resize(n_quad_pts_vol_int);
        soln_at_vol_q_ext[istate].resize(n_quad_pts_vol_ext);
        // solve soln at volume cubature nodes
        soln_basis_int.matrix_vector_mult_1D(soln_coeff_int[istate], soln_at_vol_q_int[istate],
                                             soln_basis_int.oneD_vol_operator);
        soln_basis_ext.matrix_vector_mult_1D(soln_coeff_ext[istate], soln_at_vol_q_ext[istate],
                                             soln_basis_ext.oneD_vol_operator);

        // allocate
        soln_at_surf_q_int[istate].resize(n_face_quad_pts);
        soln_at_surf_q_ext[istate].resize(n_face_quad_pts);
        // solve soln at facet cubature nodes
        soln_basis_int.matrix_vector_mult_surface_1D(iface,
                                                     soln_coeff_int[istate], soln_at_surf_q_int[istate],
                                                     soln_basis_int.oneD_surf_operator,
                                                     soln_basis_int.oneD_vol_operator);
        soln_basis_ext.matrix_vector_mult_surface_1D(neighbor_iface,
                                                     soln_coeff_ext[istate], soln_at_surf_q_ext[istate],
                                                     soln_basis_ext.oneD_surf_operator,
                                                     soln_basis_ext.oneD_vol_operator);
/*
        for(int idim=0; idim<dim; idim++){
            // alocate
            aux_soln_at_vol_q_int[istate][idim].resize(n_quad_pts_vol_int);
            aux_soln_at_vol_q_ext[istate][idim].resize(n_quad_pts_vol_ext);
            // solve auxiliary soln at volume cubature nodes
            soln_basis_int.matrix_vector_mult_1D(aux_soln_coeff_int[istate][idim], aux_soln_at_vol_q_int[istate][idim],
                                                 soln_basis_int.oneD_vol_operator);
            soln_basis_ext.matrix_vector_mult_1D(aux_soln_coeff_ext[istate][idim], aux_soln_at_vol_q_ext[istate][idim],
                                                 soln_basis_ext.oneD_vol_operator);

            // allocate
            aux_soln_at_surf_q_int[istate][idim].resize(n_face_quad_pts);
            aux_soln_at_surf_q_ext[istate][idim].resize(n_face_quad_pts);
            // solve auxiliary soln at facet cubature nodes
            soln_basis_int.matrix_vector_mult_surface_1D(iface,
                                                         aux_soln_coeff_int[istate][idim], aux_soln_at_surf_q_int[istate][idim],
                                                         soln_basis_int.oneD_surf_operator,
                                                         soln_basis_int.oneD_vol_operator);
            soln_basis_ext.matrix_vector_mult_surface_1D(neighbor_iface,
                                                         aux_soln_coeff_ext[istate][idim], aux_soln_at_surf_q_ext[istate][idim],
                                                         soln_basis_ext.oneD_surf_operator,
                                                         soln_basis_ext.oneD_vol_operator);
        }
*/
    }




    // Get volume reference fluxes and interpolate them to the facet.
    // Compute reference volume fluxes in both interior and exterior cells.

    // First we do interior.
    std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate> conv_ref_flux_at_vol_q_int;
    //std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate> diffusive_ref_flux_at_vol_q_int;
    for (unsigned int iquad=0; iquad<n_quad_pts_vol_int; ++iquad) {
        // Copy Metric Cofactor in a way can use for transforming Tensor Blocks to reference space
        // The way it is stored in metric_operators is to use sum-factorization in each direction,
        // but here it is cleaner to apply a reference transformation in each Tensor block returned by physics.
        dealii::Tensor<2,dim,adtype> metric_cofactor_vol_int;
        for(int idim=0; idim<dim; idim++){
            for(int jdim=0; jdim<dim; jdim++){
                metric_cofactor_vol_int[idim][jdim] = metric_oper_int.metric_cofactor_vol[idim][jdim][iquad];
            }
        }
        std::array<adtype,nstate> soln_state;
        //std::array<dealii::Tensor<1,dim,adtype>,nstate> aux_soln_state;
        for(int istate=0; istate<nstate; istate++){
            soln_state[istate] = soln_at_vol_q_int[istate][iquad];
            /*
            for(int idim=0; idim<dim; idim++){
                aux_soln_state[istate][idim] = aux_soln_at_vol_q_int[istate][idim][iquad];
            }
            */
        }

        // Evaluate physical convective flux
        std::array<dealii::Tensor<1,dim,adtype>,nstate> conv_phys_flux;
        //Only for conservtive DG do we interpolate volume fluxes to the facet
        if(!this->all_parameters->use_split_form && !this->all_parameters->use_curvilinear_split_form){
            conv_phys_flux = pde_physics.convective_flux (soln_state);
        }
/*
        // Compute the physical dissipative flux
        std::array<dealii::Tensor<1,dim,adtype>,nstate> diffusive_phys_flux;
        diffusive_phys_flux = pde_physics.dissipative_flux(soln_state, aux_soln_state, current_cell_index);
*/
        // Write the values in a way that we can use sum-factorization on.
        for(int istate=0; istate<nstate; istate++){
            dealii::Tensor<1,dim,adtype> conv_ref_flux;
            //dealii::Tensor<1,dim,adtype> diffusive_ref_flux;
            // transform the conservative convective physical flux to reference space
            if(!this->all_parameters->use_split_form && !this->all_parameters->use_curvilinear_split_form){
                metric_oper_int.transform_physical_to_reference(
                    conv_phys_flux[istate],
                    metric_cofactor_vol_int,
                    conv_ref_flux);
            }
            /*
            // transform the dissipative flux to reference space
            metric_oper_int.transform_physical_to_reference(
                diffusive_phys_flux[istate],
                metric_cofactor_vol_int,
                diffusive_ref_flux);
*/
            // Write the data in a way that we can use sum-factorization on.
            // Since sum-factorization improves the speed for matrix-vector multiplications,
            // We need the values to have their inner elements be vectors.
            for(int idim=0; idim<dim; idim++){
                // allocate
                if(iquad == 0){
                    conv_ref_flux_at_vol_q_int[istate][idim].resize(n_quad_pts_vol_int);
                    //diffusive_ref_flux_at_vol_q_int[istate][idim].resize(n_quad_pts_vol_int);
                }
                // write data
                if(!this->all_parameters->use_split_form && !this->all_parameters->use_curvilinear_split_form){
                    conv_ref_flux_at_vol_q_int[istate][idim][iquad] = conv_ref_flux[idim];
                }
                //diffusive_ref_flux_at_vol_q_int[istate][idim][iquad] = diffusive_ref_flux[idim];
            }
        }
    }

    // Next we do exterior volume reference fluxes.
    // Note we split the quad integrals because the interior and exterior could be of different poly basis
    std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate> conv_ref_flux_at_vol_q_ext;
    //std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate> diffusive_ref_flux_at_vol_q_ext;
    for (unsigned int iquad=0; iquad<n_quad_pts_vol_ext; ++iquad) {

        // Extract exterior volume metric cofactor matrix at given volume cubature node.
        dealii::Tensor<2,dim,adtype> metric_cofactor_vol_ext;
        for(int idim=0; idim<dim; idim++){
            for(int jdim=0; jdim<dim; jdim++){
                metric_cofactor_vol_ext[idim][jdim] = metric_oper_ext.metric_cofactor_vol[idim][jdim][iquad];
            }
        }

        std::array<adtype,nstate> soln_state;
        //std::array<dealii::Tensor<1,dim,adtype>,nstate> aux_soln_state;
        for(int istate=0; istate<nstate; istate++){
            soln_state[istate] = soln_at_vol_q_ext[istate][iquad];
            /*
            for(int idim=0; idim<dim; idim++){
                aux_soln_state[istate][idim] = aux_soln_at_vol_q_ext[istate][idim][iquad];
            }
            */
        }

        // Evaluate physical convective flux
        std::array<dealii::Tensor<1,dim,adtype>,nstate> conv_phys_flux;
        if(!this->all_parameters->use_split_form && !this->all_parameters->use_curvilinear_split_form){
            conv_phys_flux = pde_physics.convective_flux (soln_state);
        }
/*
        // Compute the physical dissipative flux
        std::array<dealii::Tensor<1,dim,adtype>,nstate> diffusive_phys_flux;
        diffusive_phys_flux = pde_physics.dissipative_flux(soln_state, aux_soln_state, neighbor_cell_index);
*/
        // Write the values in a way that we can use sum-factorization on.
        for(int istate=0; istate<nstate; istate++){
            dealii::Tensor<1,dim,adtype> conv_ref_flux;
            //dealii::Tensor<1,dim,adtype> diffusive_ref_flux;
            // transform the conservative convective physical flux to reference space
            if(!this->all_parameters->use_split_form && !this->all_parameters->use_curvilinear_split_form){
                metric_oper_ext.transform_physical_to_reference(
                    conv_phys_flux[istate],
                    metric_cofactor_vol_ext,
                    conv_ref_flux);
            }
            /*
            // transform the dissipative flux to reference space
            metric_oper_ext.transform_physical_to_reference(
                diffusive_phys_flux[istate],
                metric_cofactor_vol_ext,
                diffusive_ref_flux);
                */

            // Write the data in a way that we can use sum-factorization on.
            // Since sum-factorization improves the speed for matrix-vector multiplications,
            // We need the values to have their inner elements be vectors.
            for(int idim=0; idim<dim; idim++){
                // allocate
                if(iquad == 0){
                    conv_ref_flux_at_vol_q_ext[istate][idim].resize(n_quad_pts_vol_ext);
                    //diffusive_ref_flux_at_vol_q_ext[istate][idim].resize(n_quad_pts_vol_ext);
                }
                // write data
                if(!this->all_parameters->use_split_form && !this->all_parameters->use_curvilinear_split_form){
                    conv_ref_flux_at_vol_q_ext[istate][idim][iquad] = conv_ref_flux[idim];
                }
                //diffusive_ref_flux_at_vol_q_ext[istate][idim][iquad] = diffusive_ref_flux[idim];
            }
        }
    }

    // Interpolate the volume reference fluxes to the facet.
    // And do the dot product with the UNIT REFERENCE normal.
    // Since we are computing a dot product with the unit reference normal,
    // we exploit the fact that the unit reference normal has a value of 0 in all reference directions except
    // the outward reference normal dircetion.
    const dealii::Tensor<1,dim,double> unit_ref_normal_int = dealii::GeometryInfo<dim>::unit_normal_vector[iface];
    const dealii::Tensor<1,dim,double> unit_ref_normal_ext = dealii::GeometryInfo<dim>::unit_normal_vector[neighbor_iface];
    // Extract the reference direction that is outward facing on the facet.
    const int dim_not_zero_int = iface / 2;//reference direction of face integer division
    const int dim_not_zero_ext = neighbor_iface / 2;//reference direction of face integer division

    std::array<std::vector<adtype>,nstate> conv_int_vol_ref_flux_interp_to_face_dot_ref_normal;
    std::array<std::vector<adtype>,nstate> conv_ext_vol_ref_flux_interp_to_face_dot_ref_normal;
    //std::array<std::vector<adtype>,nstate> diffusive_int_vol_ref_flux_interp_to_face_dot_ref_normal;
    //std::array<std::vector<adtype>,nstate> diffusive_ext_vol_ref_flux_interp_to_face_dot_ref_normal;
    for(int istate=0; istate<nstate; istate++){
        //allocate
        conv_int_vol_ref_flux_interp_to_face_dot_ref_normal[istate].resize(n_face_quad_pts);
        conv_ext_vol_ref_flux_interp_to_face_dot_ref_normal[istate].resize(n_face_quad_pts);
        //diffusive_int_vol_ref_flux_interp_to_face_dot_ref_normal[istate].resize(n_face_quad_pts);
        //diffusive_ext_vol_ref_flux_interp_to_face_dot_ref_normal[istate].resize(n_face_quad_pts);

        // solve
        // Note, since the normal is zero in all other reference directions, we only have to interpolate one given reference direction to the facet
        
        // interpolate reference volume convective flux to the facet, and apply unit reference normal as scaled by 1.0 or -1.0
        if(!this->all_parameters->use_split_form && !this->all_parameters->use_curvilinear_split_form){
            flux_basis_int.matrix_vector_mult_surface_1D(iface, 
                                                         conv_ref_flux_at_vol_q_int[istate][dim_not_zero_int],
                                                         conv_int_vol_ref_flux_interp_to_face_dot_ref_normal[istate],
                                                         flux_basis_int.oneD_surf_operator,//the flux basis interpolates from the flux nodes
                                                         flux_basis_int.oneD_vol_operator,
                                                         false, unit_ref_normal_int[dim_not_zero_int]);//don't add to previous value, scale by unit_normal int
            flux_basis_ext.matrix_vector_mult_surface_1D(neighbor_iface, 
                                                         conv_ref_flux_at_vol_q_ext[istate][dim_not_zero_ext],
                                                         conv_ext_vol_ref_flux_interp_to_face_dot_ref_normal[istate],
                                                         flux_basis_ext.oneD_surf_operator,
                                                         flux_basis_ext.oneD_vol_operator,
                                                         false, unit_ref_normal_ext[dim_not_zero_ext]);//don't add to previous value, unit_normal ext is -unit normal int
        }
/*
        // interpolate reference volume dissipative flux to the facet, and apply unit reference normal as scaled by 1.0 or -1.0
        flux_basis_int.matrix_vector_mult_surface_1D(iface, 
                                                     diffusive_ref_flux_at_vol_q_int[istate][dim_not_zero_int],
                                                     diffusive_int_vol_ref_flux_interp_to_face_dot_ref_normal[istate],
                                                     flux_basis_int.oneD_surf_operator,
                                                     flux_basis_int.oneD_vol_operator,
                                                     false, unit_ref_normal_int[dim_not_zero_int]);
        flux_basis_ext.matrix_vector_mult_surface_1D(neighbor_iface, 
                                                     diffusive_ref_flux_at_vol_q_ext[istate][dim_not_zero_ext],
                                                     diffusive_ext_vol_ref_flux_interp_to_face_dot_ref_normal[istate],
                                                     flux_basis_ext.oneD_surf_operator,
                                                     flux_basis_ext.oneD_vol_operator,
                                                     false, unit_ref_normal_ext[dim_not_zero_ext]);
  */
    }


    //Note that for entropy-dissipation and entropy stability, the conservative variables
    //are functions of projected entropy variables. For Euler etc, the transformation is nonlinear
    //so careful attention to what is evaluated where and interpolated to where is needed.
    //For further information, please see Chan, Jesse. "On discretely entropy conservative and entropy stable discontinuous Galerkin methods." Journal of Computational Physics 362 (2018): 346-374.
    //pages 355 (Eq. 57 with text around it) and  page 359 (Eq 86 and text below it).

    // First, transform the volume conservative solution at volume cubature nodes to entropy variables.
    std::array<std::vector<adtype>,nstate> entropy_var_vol_int;
    std::vector<std::array<std::array<real,nstate>,nstate>> dv_du_int(n_quad_pts_vol_int);
    for(unsigned int iquad=0; iquad<n_quad_pts_vol_int; iquad++){
        std::array<adtype,nstate> soln_state;
        for(int istate=0; istate<nstate; istate++){
            soln_state[istate] = soln_at_vol_q_int[istate][iquad];
        }
        std::array<adtype,nstate> entropy_var;
        entropy_var = pde_physics.compute_entropy_variables(soln_state);
        if constexpr(std::is_same<adtype,double>::value)
        {
            if(this->compute_dRdW_strong)
            {
                pde_physics.get_d_entropy_var_d_conservative_var(dv_du_int[iquad],entropy_var); 
            }
        }
        for(int istate=0; istate<nstate; istate++){
            if(iquad==0){
                entropy_var_vol_int[istate].resize(n_quad_pts_vol_int);
            }
            entropy_var_vol_int[istate][iquad] = entropy_var[istate];
        }
    }
    std::array<std::vector<adtype>,nstate> entropy_var_vol_ext;
    std::vector<std::array<std::array<real,nstate>,nstate>> dv_du_ext(n_quad_pts_vol_ext);
    for(unsigned int iquad=0; iquad<n_quad_pts_vol_ext; iquad++){
        std::array<adtype,nstate> soln_state;
        for(int istate=0; istate<nstate; istate++){
            soln_state[istate] = soln_at_vol_q_ext[istate][iquad];
        }
        std::array<adtype,nstate> entropy_var;
        entropy_var = pde_physics.compute_entropy_variables(soln_state);
        if constexpr(std::is_same<adtype,double>::value)
        {
            if(this->compute_dRdW_strong)
            {
                pde_physics.get_d_entropy_var_d_conservative_var(dv_du_ext[iquad],entropy_var); 
            }
        }
        for(int istate=0; istate<nstate; istate++){
            if(iquad==0){
                entropy_var_vol_ext[istate].resize(n_quad_pts_vol_ext);
            }
            entropy_var_vol_ext[istate][iquad] = entropy_var[istate];
        }
    }
    
    std::array<std::array<dealii::FullMatrix<double>,nstate>,nstate> dvtilde_duh_vol_int;
    std::array<std::array<dealii::FullMatrix<double>,nstate>,nstate> dvtilde_duh_surf_int;
    std::array<std::array<dealii::FullMatrix<double>,nstate>,nstate> dvtilde_duh_vol_ext;
    std::array<std::array<dealii::FullMatrix<double>,nstate>,nstate> dvtilde_duh_surf_ext;
    xt::xarray<double> H_mat_int;
    xt::xarray<double> H_mat_ext;
    if constexpr(std::is_same<adtype,double>::value)
    {
        if(this->compute_dRdW_strong)
        {
            compute_H_mat(soln_basis_projection_oper_int.oneD_G_operator_vol, soln_basis_int.oneD_vol_operator, metric_oper_int.det_Jac_vol, dv_du_int, H_mat_int);
            compute_H_mat(soln_basis_projection_oper_ext.oneD_G_operator_vol, soln_basis_ext.oneD_vol_operator, metric_oper_ext.det_Jac_vol, dv_du_ext, H_mat_ext);
            compute_dvtilde_duh(soln_basis_projection_oper_int.oneD_G_operator_vol,soln_basis_projection_oper_int.oneD_G_operator_vol,soln_basis_projection_oper_int.oneD_G_operator_vol,H_mat_int, soln_basis_int.oneD_vol_operator.n(),n_quad_pts_1D_int,dvtilde_duh_vol_int);
            compute_dvtilde_duh(soln_basis_projection_oper_ext.oneD_G_operator_vol,soln_basis_projection_oper_ext.oneD_G_operator_vol,soln_basis_projection_oper_ext.oneD_G_operator_vol,H_mat_ext, soln_basis_ext.oneD_vol_operator.n(),n_quad_pts_1D_ext,dvtilde_duh_vol_ext);
            if(iface==0)
            {
                compute_dvtilde_duh(soln_basis_projection_oper_int.oneD_G_operator_surf[0],soln_basis_projection_oper_int.oneD_G_operator_vol,soln_basis_projection_oper_int.oneD_G_operator_vol,H_mat_int, soln_basis_int.oneD_vol_operator.n(),n_quad_pts_1D_int,dvtilde_duh_surf_int);
            }
            else if(iface==1)
            {
                compute_dvtilde_duh(soln_basis_projection_oper_int.oneD_G_operator_surf[1],soln_basis_projection_oper_int.oneD_G_operator_vol,soln_basis_projection_oper_int.oneD_G_operator_vol,H_mat_int, soln_basis_int.oneD_vol_operator.n(),n_quad_pts_1D_int,dvtilde_duh_surf_int);
            }
            else if(iface==2)
            {
                compute_dvtilde_duh(soln_basis_projection_oper_int.oneD_G_operator_vol,soln_basis_projection_oper_int.oneD_G_operator_surf[0],soln_basis_projection_oper_int.oneD_G_operator_vol,H_mat_int, soln_basis_int.oneD_vol_operator.n(),n_quad_pts_1D_int,dvtilde_duh_surf_int);
            }
            else if(iface==3)
            {
                compute_dvtilde_duh(soln_basis_projection_oper_int.oneD_G_operator_vol,soln_basis_projection_oper_int.oneD_G_operator_surf[1],soln_basis_projection_oper_int.oneD_G_operator_vol,H_mat_int, soln_basis_int.oneD_vol_operator.n(),n_quad_pts_1D_int,dvtilde_duh_surf_int);
            }
            else if(iface==4)
            {
                compute_dvtilde_duh(soln_basis_projection_oper_int.oneD_G_operator_vol,soln_basis_projection_oper_int.oneD_G_operator_vol,soln_basis_projection_oper_int.oneD_G_operator_surf[0],H_mat_int, soln_basis_int.oneD_vol_operator.n(),n_quad_pts_1D_int,dvtilde_duh_surf_int);
            }
            else if(iface==5)
            {
                compute_dvtilde_duh(soln_basis_projection_oper_int.oneD_G_operator_vol,soln_basis_projection_oper_int.oneD_G_operator_vol,soln_basis_projection_oper_int.oneD_G_operator_surf[1],H_mat_int, soln_basis_int.oneD_vol_operator.n(),n_quad_pts_1D_int,dvtilde_duh_surf_int);
            }
            if(neighbor_iface==0)
            {
                compute_dvtilde_duh(soln_basis_projection_oper_ext.oneD_G_operator_surf[0],soln_basis_projection_oper_ext.oneD_G_operator_vol,soln_basis_projection_oper_ext.oneD_G_operator_vol,H_mat_ext, soln_basis_ext.oneD_vol_operator.n(),n_quad_pts_1D_ext,dvtilde_duh_surf_ext);
            }
            else if(neighbor_iface==1)
            {
                compute_dvtilde_duh(soln_basis_projection_oper_ext.oneD_G_operator_surf[1],soln_basis_projection_oper_ext.oneD_G_operator_vol,soln_basis_projection_oper_ext.oneD_G_operator_vol,H_mat_ext, soln_basis_ext.oneD_vol_operator.n(),n_quad_pts_1D_ext,dvtilde_duh_surf_ext);
            }
            else if(neighbor_iface==2)
            {
                compute_dvtilde_duh(soln_basis_projection_oper_ext.oneD_G_operator_vol,soln_basis_projection_oper_ext.oneD_G_operator_surf[0],soln_basis_projection_oper_ext.oneD_G_operator_vol,H_mat_ext, soln_basis_ext.oneD_vol_operator.n(),n_quad_pts_1D_ext,dvtilde_duh_surf_ext);
            }
            else if(neighbor_iface==3)
            {
                compute_dvtilde_duh(soln_basis_projection_oper_ext.oneD_G_operator_vol,soln_basis_projection_oper_ext.oneD_G_operator_surf[1],soln_basis_projection_oper_ext.oneD_G_operator_vol,H_mat_ext, soln_basis_ext.oneD_vol_operator.n(),n_quad_pts_1D_ext,dvtilde_duh_surf_ext);
            }
            else if(neighbor_iface==4)
            {
                compute_dvtilde_duh(soln_basis_projection_oper_ext.oneD_G_operator_vol,soln_basis_projection_oper_ext.oneD_G_operator_vol,soln_basis_projection_oper_ext.oneD_G_operator_surf[0],H_mat_ext, soln_basis_ext.oneD_vol_operator.n(),n_quad_pts_1D_ext,dvtilde_duh_surf_ext);
            }
            else if(neighbor_iface==5)
            {
                compute_dvtilde_duh(soln_basis_projection_oper_ext.oneD_G_operator_vol,soln_basis_projection_oper_ext.oneD_G_operator_vol,soln_basis_projection_oper_ext.oneD_G_operator_surf[1],H_mat_ext, soln_basis_ext.oneD_vol_operator.n(),n_quad_pts_1D_ext,dvtilde_duh_surf_ext);
            }
        }
    }

    //project it onto the solution basis functions and interpolate it
    std::array<std::vector<adtype>,nstate> projected_entropy_var_vol_int;
    std::array<std::vector<adtype>,nstate> projected_entropy_var_vol_ext;
    std::array<std::vector<adtype>,nstate> projected_entropy_var_surf_int;
    std::array<std::vector<adtype>,nstate> projected_entropy_var_surf_ext;
    std::array<std::vector<adtype>,nstate> entropy_var_coeffs_int;
    std::array<std::vector<adtype>,nstate> entropy_var_coeffs_ext;
    for(int istate=0; istate<nstate; istate++){
        // allocate
        projected_entropy_var_vol_int[istate].resize(n_quad_pts_vol_int);
        projected_entropy_var_vol_ext[istate].resize(n_quad_pts_vol_ext);
        projected_entropy_var_surf_int[istate].resize(n_face_quad_pts);
        projected_entropy_var_surf_ext[istate].resize(n_face_quad_pts);
        entropy_var_coeffs_int[istate].resize(n_shape_fns_int);
        entropy_var_coeffs_ext[istate].resize(n_shape_fns_ext);

        //interior
        //soln_basis_projection_oper_int.matrix_vector_mult_1D(entropy_var_vol_int[istate],
        //                                                     entropy_var_coeffs_int[istate],
        //                                                     soln_basis_projection_oper_int.oneD_vol_operator);
        soln_basis_projection_oper_int.weight_adjusted_vol_projection(entropy_var_vol_int[istate],
                                                                  entropy_var_coeffs_int[istate],
                                                                  n_quad_pts_vol_int,
                                                                  n_shape_fns_int,
                                                                  soln_basis_int,
                                                                  soln_basis_projection_oper_int,
                                                                  metric_oper_int.det_Jac_vol);
        soln_basis_int.matrix_vector_mult_1D(entropy_var_coeffs_int[istate],
                                             projected_entropy_var_vol_int[istate],
                                             soln_basis_int.oneD_vol_operator);
        soln_basis_int.matrix_vector_mult_surface_1D(iface,
                                                     entropy_var_coeffs_int[istate], 
                                                     projected_entropy_var_surf_int[istate],
                                                     soln_basis_int.oneD_surf_operator,
                                                     soln_basis_int.oneD_vol_operator);

        //exterior
        //soln_basis_projection_oper_ext.matrix_vector_mult_1D(entropy_var_vol_ext[istate],
        //                                                     entropy_var_coeffs_ext[istate],
        //                                                     soln_basis_projection_oper_ext.oneD_vol_operator);
        soln_basis_projection_oper_ext.weight_adjusted_vol_projection(entropy_var_vol_ext[istate],
                                                                  entropy_var_coeffs_ext[istate],
                                                                  n_quad_pts_vol_ext,
                                                                  n_shape_fns_ext,
                                                                  soln_basis_ext,
                                                                  soln_basis_projection_oper_ext,
                                                                  metric_oper_ext.det_Jac_vol);

        soln_basis_ext.matrix_vector_mult_1D(entropy_var_coeffs_ext[istate],
                                             projected_entropy_var_vol_ext[istate],
                                             soln_basis_ext.oneD_vol_operator);
        soln_basis_ext.matrix_vector_mult_surface_1D(neighbor_iface,
                                                     entropy_var_coeffs_ext[istate], 
                                                     projected_entropy_var_surf_ext[istate],
                                                     soln_basis_ext.oneD_surf_operator,
                                                     soln_basis_ext.oneD_vol_operator);
    }

    //get the surface-volume sparsity pattern for a "sum-factorized" Hadamard product only computing terms needed for the operation.
    const unsigned int row_size_int = n_face_quad_pts * n_quad_pts_1D_int;
    const unsigned int col_size_int = n_face_quad_pts * n_quad_pts_1D_int;
    std::vector<unsigned int> Hadamard_rows_sparsity_int(row_size_int);
    std::vector<unsigned int> Hadamard_columns_sparsity_int(col_size_int);
    const unsigned int row_size_ext = n_face_quad_pts * n_quad_pts_1D_ext;
    const unsigned int col_size_ext = n_face_quad_pts * n_quad_pts_1D_ext;
    std::vector<unsigned int> Hadamard_rows_sparsity_ext(row_size_ext);
    std::vector<unsigned int> Hadamard_columns_sparsity_ext(col_size_ext);
    if(this->all_parameters->use_split_form || this->all_parameters->use_curvilinear_split_form){
        flux_basis_int.sum_factorized_Hadamard_surface_sparsity_pattern(n_face_quad_pts, n_quad_pts_1D_int, Hadamard_rows_sparsity_int, Hadamard_columns_sparsity_int, dim_not_zero_int);
        flux_basis_ext.sum_factorized_Hadamard_surface_sparsity_pattern(n_face_quad_pts, n_quad_pts_1D_ext, Hadamard_rows_sparsity_ext, Hadamard_columns_sparsity_ext, dim_not_zero_ext);
    }
    
    std::array<std::array<dealii::FullMatrix<double>,nstate>,nstate> dterm1int_duhint;
    std::array<std::array<dealii::FullMatrix<double>,nstate>,nstate> dterm2int_duhint;
    std::array<std::array<dealii::FullMatrix<double>,nstate>,nstate> dterm3int_duhint;
    std::array<std::array<dealii::FullMatrix<double>,nstate>,nstate> dterm3int_duhext;
    std::array<std::array<dealii::FullMatrix<double>,nstate>,nstate> dterm1ext_duhext;
    std::array<std::array<dealii::FullMatrix<double>,nstate>,nstate> dterm2ext_duhext;
    std::array<std::array<dealii::FullMatrix<double>,nstate>,nstate> dterm3ext_duhext;
    std::array<std::array<dealii::FullMatrix<double>,nstate>,nstate> dterm3ext_duhint;
    std::array<std::array<double,nstate>,nstate> dutilde_dvtilde_face_int;
    std::array<std::array<double,nstate>,nstate> dutilde_dvtilde_vol_int;
    std::array<std::array<double,nstate>,nstate> dutilde_dvtilde_face_ext;
    std::array<std::array<double,nstate>,nstate> dutilde_dvtilde_vol_ext;
    if constexpr(std::is_same<adtype,double>::value)
    {
        if(this->compute_dRdW_strong)
        {
            for(unsigned int s1=0; s1<nstate; ++s1)
            {
                for(unsigned int s2=0; s2<nstate; ++s2)
                {
                    dterm1int_duhint[s1][s2].reinit(n_quad_pts_vol_int,n_shape_fns_int);
                    dterm1ext_duhext[s1][s2].reinit(n_quad_pts_vol_ext,n_shape_fns_ext);
                    dterm2int_duhint[s1][s2].reinit(n_face_quad_pts,n_shape_fns_int);
                    dterm2ext_duhext[s1][s2].reinit(n_face_quad_pts,n_shape_fns_ext);
                    dterm3int_duhint[s1][s2].reinit(n_face_quad_pts,n_shape_fns_int);
                    dterm3int_duhext[s1][s2].reinit(n_face_quad_pts,n_shape_fns_ext);
                    dterm3ext_duhint[s1][s2].reinit(n_face_quad_pts,n_shape_fns_int);
                    dterm3ext_duhext[s1][s2].reinit(n_face_quad_pts,n_shape_fns_ext);
                }
            }
        }
    }

    std::array<std::vector<adtype>,nstate> surf_vol_ref_2pt_flux_interp_surf_int;
    std::array<std::vector<adtype>,nstate> surf_vol_ref_2pt_flux_interp_surf_ext;
    std::array<std::vector<adtype>,nstate> surf_vol_ref_2pt_flux_interp_vol_int;
    std::array<std::vector<adtype>,nstate> surf_vol_ref_2pt_flux_interp_vol_ext;
    if(this->all_parameters->use_split_form || this->all_parameters->use_curvilinear_split_form){
        //get surface-volume hybrid 2pt flux from Eq.(15) in Chan, Jesse. "Skew-symmetric entropy stable modal discontinuous Galerkin formulations." Journal of Scientific Computing 81.1 (2019): 459-485.
        std::array<std::vector<adtype>,nstate> surface_ref_2pt_flux_int;
        std::array<std::vector<adtype>,nstate> surface_ref_2pt_flux_ext;
        //make use of the sparsity pattern from above to assemble only n^d non-zero entries without ever allocating not computing zeros.
        for(int istate=0; istate<nstate; istate++){
            surface_ref_2pt_flux_int[istate].resize(n_face_quad_pts * n_quad_pts_1D_int);
            surface_ref_2pt_flux_ext[istate].resize(n_face_quad_pts * n_quad_pts_1D_ext);
        }
        for(unsigned int iquad_face=0; iquad_face<n_face_quad_pts; iquad_face++){
            dealii::Tensor<2,dim,adtype> metric_cofactor_surf;
            for(int idim=0; idim<dim; idim++){
                for(int jdim=0; jdim<dim; jdim++){
                    metric_cofactor_surf[idim][jdim] = metric_oper_int.metric_cofactor_surf[idim][jdim][iquad_face];
                }
            }
             
            //Compute the conservative values on the facet from the interpolated entorpy variables.
            std::array<adtype,nstate> entropy_var_face_int;
            std::array<adtype,nstate> entropy_var_face_ext;
            for(int istate=0; istate<nstate; istate++){
                entropy_var_face_int[istate] = projected_entropy_var_surf_int[istate][iquad_face];
                entropy_var_face_ext[istate] = projected_entropy_var_surf_ext[istate][iquad_face];
            }
            std::array<adtype,nstate> soln_state_face_int;
            soln_state_face_int = pde_physics.compute_conservative_variables_from_entropy_variables (entropy_var_face_int);
            std::array<adtype,nstate> soln_state_face_ext;
            soln_state_face_ext = pde_physics.compute_conservative_variables_from_entropy_variables (entropy_var_face_ext);
            if constexpr(std::is_same<adtype,double>::value)
            {
                if(this->compute_dRdW_strong)
                {
                    pde_physics.get_d_conservative_var_d_entropy_var(dutilde_dvtilde_face_int, entropy_var_face_int);
                    pde_physics.get_d_conservative_var_d_entropy_var(dutilde_dvtilde_face_ext, entropy_var_face_ext);
                }
            }

            //only do the n_quad_1D vol points that give non-zero entries from Hadamard product.
            for(unsigned int row_index = iquad_face * n_quad_pts_1D_int, column_index = 0; 
                column_index < n_quad_pts_1D_int;
                row_index++, column_index++){

                if(Hadamard_rows_sparsity_int[row_index] != iquad_face){
                    pcout<<"The interior Hadamard rows sparsity pattern does not match."<<std::endl;
                    std::abort();
                }

                const unsigned int iquad_vol = Hadamard_columns_sparsity_int[row_index];//extract flux_quad pt that corresponds to a non-zero entry for Hadamard product.
                // Copy Metric Cofactor in a way can use for transforming Tensor Blocks to reference space
                // The way it is stored in metric_operators is to use sum-factorization in each direction,
                // but here it is cleaner to apply a reference transformation in each Tensor block returned by physics.
                dealii::Tensor<2,dim,adtype> metric_cofactor_vol_int;
                for(int idim=0; idim<dim; idim++){
                    for(int jdim=0; jdim<dim; jdim++){
                        metric_cofactor_vol_int[idim][jdim] = metric_oper_int.metric_cofactor_vol[idim][jdim][iquad_vol];
                    }
                }
                std::array<adtype,nstate> entropy_var;
                for(int istate=0; istate<nstate; istate++){
                    entropy_var[istate] = projected_entropy_var_vol_int[istate][iquad_vol];
                }
                std::array<adtype,nstate> soln_state;
                soln_state = pde_physics.compute_conservative_variables_from_entropy_variables (entropy_var);
                if constexpr(std::is_same<adtype,double>::value)
                {
                    if(this->compute_dRdW_strong)
                    {
                        pde_physics.get_d_conservative_var_d_entropy_var(dutilde_dvtilde_vol_int, entropy_var);
                    }
                }
                //Note that the flux basis is collocated on the volume cubature set so we don't need to evaluate the entropy variables
                //on the volume set then transform back to the conservative variables since the flux basis volume
                //projection is identity.

                //Compute the physical flux
                dealii::Tensor<2,dim,adtype> metric_cofactor_split;
                for(int idim=0; idim<dim; idim++){
                    for(int jdim=0; jdim<dim; jdim++){
                        metric_cofactor_split[idim][jdim] = 0.5 * (metric_cofactor_surf[idim][jdim] + metric_cofactor_vol_int[idim][jdim]);
                    }
                }
                std::array<dealii::Tensor<1,dim,adtype>,nstate> conv_phys_flux_2pt;
                conv_phys_flux_2pt = pde_physics.convective_numerical_split_flux(soln_state, soln_state_face_int);
                for(int istate=0; istate<nstate; istate++){
                    dealii::Tensor<1,dim,adtype> conv_ref_flux_2pt;
                    //For each state, transform the physical flux to a reference flux.
                    //For each state, transform the physical flux to a reference flux.
                    metric_oper_int.transform_physical_to_reference(
                        conv_phys_flux_2pt[istate],
                        metric_cofactor_split,
                        conv_ref_flux_2pt);
                    //only store the dim not zero in reference space bc dot product with unit ref normal later.
                    surface_ref_2pt_flux_int[istate][iquad_face * n_quad_pts_1D_int + column_index] = conv_ref_flux_2pt[dim_not_zero_int];
                }
                if constexpr(std::is_same<adtype,double>::value)
                {
                    if(this->compute_dRdW_strong)
                    {
                        std::array<std::array<std::array<double,nstate>,nstate>,2> dFsplit_dutilde = pde_physics.convective_numerical_split_flux_derivative(soln_state, soln_state_face_int, metric_cofactor_split, unit_ref_normal_int);
                        double mult_factor=0;
                        const int iface_1D = iface % 2;//the reference face number
                        if(iface==0 || iface==1)
                        {
                            const int L = iquad_vol % n_quad_pts_1D_int;
                            mult_factor = flux_basis_int.oneD_surf_operator[iface_1D][0][L]*surf_quad_weights[iquad_face];
                        }
                        else if(iface==2 || iface==3)
                        {
                            const int M = (iquad_vol/n_quad_pts_1D_int) % n_quad_pts_1D_int;
                            mult_factor = flux_basis_int.oneD_surf_operator[iface_1D][0][M]*surf_quad_weights[iquad_face];
                        }
                        else if(iface==4 || iface==5)
                        {
                            const int N = (iquad_vol/n_quad_pts_1D_int) / n_quad_pts_1D_int;
                            mult_factor = flux_basis_int.oneD_surf_operator[iface_1D][0][N]*surf_quad_weights[iquad_face];
                        }
                        // Form term1 and term2
                        for(unsigned int s1=0; s1<nstate; ++s1)
                        {
                            for(unsigned int s2=0; s2<nstate; ++s2)
                            {
                                for(unsigned int idof = 0; idof<n_shape_fns_int; ++idof)
                                {
                                    dterm1int_duhint[s1][s2][iquad_vol][idof]=0;
                                    for(unsigned int sk=0; sk<nstate; ++sk)
                                    {
                                        for(unsigned int sl=0; sl<nstate; ++sl)
                                        {
                                            dterm1int_duhint[s1][s2][iquad_vol][idof]+= mult_factor*(dFsplit_dutilde[0][s1][sk]*dutilde_dvtilde_vol_int[sk][sl]*dvtilde_duh_vol_int[sl][s2][iquad_vol][idof] + dFsplit_dutilde[1][s1][sk]*dutilde_dvtilde_face_int[sk][sl]*dvtilde_duh_surf_int[sl][s2][iquad_face][idof]);
                                            dterm2int_duhint[s1][s2][iquad_face][idof] -= mult_factor*(dFsplit_dutilde[0][s1][sk]*dutilde_dvtilde_vol_int[sk][sl]*dvtilde_duh_vol_int[sl][s2][iquad_vol][idof] + dFsplit_dutilde[1][s1][sk]*dutilde_dvtilde_face_int[sk][sl]*dvtilde_duh_surf_int[sl][s2][iquad_face][idof]);
                                        }
                                    }
                                }
                            }
                        }
                    } // if dRdW_strong
                }
            }
            for(unsigned int row_index = iquad_face * n_quad_pts_1D_ext, column_index = 0; 
                column_index < n_quad_pts_1D_ext;
                row_index++, column_index++){

                if(Hadamard_rows_sparsity_ext[row_index] != iquad_face){
                    pcout<<"The exterior Hadamard rows sparsity pattern does not match."<<std::endl;
                    std::abort();
                }

                const unsigned int iquad_vol = Hadamard_columns_sparsity_ext[row_index];//extract flux_quad pt that corresponds to a non-zero entry for Hadamard product.
                // Copy Metric Cofactor in a way can use for transforming Tensor Blocks to reference space
                // The way it is stored in metric_operators is to use sum-factorization in each direction,
                // but here it is cleaner to apply a reference transformation in each Tensor block returned by physics.
                dealii::Tensor<2,dim,adtype> metric_cofactor_vol_ext;
                for(int idim=0; idim<dim; idim++){
                    for(int jdim=0; jdim<dim; jdim++){
                        metric_cofactor_vol_ext[idim][jdim] = metric_oper_ext.metric_cofactor_vol[idim][jdim][iquad_vol];
                    }
                }
                dealii::Tensor<2,dim,adtype> metric_cofactor_split;
                for(int idim=0; idim<dim; idim++){
                    for(int jdim=0; jdim<dim; jdim++){
                        metric_cofactor_split[idim][jdim] = 0.5 * (metric_cofactor_surf[idim][jdim] + metric_cofactor_vol_ext[idim][jdim]);
                    }
                }
                std::array<adtype,nstate> entropy_var;
                for(int istate=0; istate<nstate; istate++){
                    entropy_var[istate] = projected_entropy_var_vol_ext[istate][iquad_vol];
                }
                std::array<adtype,nstate> soln_state;
                soln_state = pde_physics.compute_conservative_variables_from_entropy_variables (entropy_var);
                if constexpr(std::is_same<adtype,double>::value)
                {
                    if(this->compute_dRdW_strong)
                    {
                        pde_physics.get_d_conservative_var_d_entropy_var(dutilde_dvtilde_vol_ext, entropy_var);
                    }
                }
                //Compute the physical flux
                std::array<dealii::Tensor<1,dim,adtype>,nstate> conv_phys_flux_2pt;
                conv_phys_flux_2pt = pde_physics.convective_numerical_split_flux(soln_state, soln_state_face_ext);
                for(int istate=0; istate<nstate; istate++){
                    dealii::Tensor<1,dim,adtype> conv_ref_flux_2pt;
                    //For each state, transform the physical flux to a reference flux.
                    metric_oper_ext.transform_physical_to_reference(
                        conv_phys_flux_2pt[istate],
                        metric_cofactor_split,
                        conv_ref_flux_2pt);
                    //only store the dim not zero in reference space bc dot product with unit ref normal later.
                    surface_ref_2pt_flux_ext[istate][iquad_face * n_quad_pts_1D_ext + column_index] = conv_ref_flux_2pt[dim_not_zero_ext];
                }
                if constexpr(std::is_same<adtype,double>::value)
                {
                    if(this->compute_dRdW_strong)
                    {
                        std::array<std::array<std::array<double,nstate>,nstate>,2> dFsplit_dutilde = pde_physics.convective_numerical_split_flux_derivative(soln_state, soln_state_face_ext, metric_cofactor_split, unit_ref_normal_ext);
                        double mult_factor=0;
                        const int iface_1D = neighbor_iface % 2;//the reference face number
                        if(iface==0 || iface==1)
                        {
                            const int L = iquad_vol % n_quad_pts_1D_ext;
                            mult_factor = flux_basis_ext.oneD_surf_operator[iface_1D][0][L]*surf_quad_weights[iquad_face];
                        }
                        else if(iface==2 || iface==3)
                        {
                            const int M = (iquad_vol/n_quad_pts_1D_ext) % n_quad_pts_1D_ext;
                            mult_factor = flux_basis_ext.oneD_surf_operator[iface_1D][0][M]*surf_quad_weights[iquad_face];
                        }
                        else if(iface==4 || iface==5)
                        {
                            const int N = (iquad_vol/n_quad_pts_1D_ext) / n_quad_pts_1D_ext;
                            mult_factor = flux_basis_ext.oneD_surf_operator[iface_1D][0][N]*surf_quad_weights[iquad_face];
                        }
                        // Form term1 and term2
                        for(unsigned int s1=0; s1<nstate; ++s1)
                        {
                            for(unsigned int s2=0; s2<nstate; ++s2)
                            {
                                for(unsigned int idof = 0; idof<n_shape_fns_ext; ++idof)
                                {
                                    dterm1ext_duhext[s1][s2][iquad_vol][idof]=0;
                                    for(unsigned int sk=0; sk<nstate; ++sk)
                                    {
                                        for(unsigned int sl=0; sl<nstate; ++sl)
                                        {
                                            dterm1ext_duhext[s1][s2][iquad_vol][idof]+= mult_factor*(dFsplit_dutilde[0][s1][sk]*dutilde_dvtilde_vol_ext[sk][sl]*dvtilde_duh_vol_ext[sl][s2][iquad_vol][idof] + dFsplit_dutilde[1][s1][sk]*dutilde_dvtilde_face_ext[sk][sl]*dvtilde_duh_surf_ext[sl][s2][iquad_face][idof]);
                                            dterm2ext_duhext[s1][s2][iquad_face][idof] -= mult_factor*(dFsplit_dutilde[0][s1][sk]*dutilde_dvtilde_vol_ext[sk][sl]*dvtilde_duh_vol_ext[sl][s2][iquad_vol][idof] + dFsplit_dutilde[1][s1][sk]*dutilde_dvtilde_face_ext[sk][sl]*dvtilde_duh_surf_ext[sl][s2][iquad_face][idof]);
                                        }
                                    }
                                }
                            }
                        }
                    } // if dRdW_strong
                }
            }
        }

        //get the surface basis operator from Hadamard sparsity pattern
        //to be applied at n^d operations (on the face so n^{d+1-1}=n^d flops)
        //also only allocates n^d terms.
        const int iface_1D = iface % 2;//the reference face number
        const std::vector<double> &oneD_quad_weights_vol_int = this->oneD_quadrature_collection[poly_degree_int].get_weights();
        dealii::FullMatrix<real> surf_oper_sparse_int(n_face_quad_pts, n_quad_pts_1D_int);
        flux_basis_int.sum_factorized_Hadamard_surface_basis_assembly(n_face_quad_pts, n_quad_pts_1D_int, 
                                                                      Hadamard_rows_sparsity_int, Hadamard_columns_sparsity_int,
                                                                      flux_basis_int.oneD_surf_operator[iface_1D], 
                                                                      oneD_quad_weights_vol_int,
                                                                      surf_oper_sparse_int,
                                                                      dim_not_zero_int);
        const int neighbor_iface_1D = neighbor_iface % 2;//the reference neighbour face number
        const std::vector<double> &oneD_quad_weights_vol_ext = this->oneD_quadrature_collection[poly_degree_ext].get_weights();
        dealii::FullMatrix<real> surf_oper_sparse_ext(n_face_quad_pts, n_quad_pts_1D_ext);
        flux_basis_ext.sum_factorized_Hadamard_surface_basis_assembly(n_face_quad_pts, n_quad_pts_1D_ext, 
                                                                      Hadamard_rows_sparsity_ext, Hadamard_columns_sparsity_ext,
                                                                      flux_basis_ext.oneD_surf_operator[neighbor_iface_1D], 
                                                                      oneD_quad_weights_vol_ext,
                                                                      surf_oper_sparse_ext,
                                                                      dim_not_zero_ext);

        // Apply the surface Hadamard products and multiply with vector of ones for both off diagonal terms in
        // Eq.(15) in Chan, Jesse. "Skew-symmetric entropy stable modal discontinuous Galerkin formulations." Journal of Scientific Computing 81.1 (2019): 459-485.
        for(int istate=0; istate<nstate; istate++){
            //first apply Hadamard product with the structure made above.
            std::vector<adtype> surface_ref_2pt_flux_int_Hadamard_with_surf_oper(n_face_quad_pts * n_quad_pts_1D_int);
            flux_basis_int.Hadamard_product_AD_vector(surf_oper_sparse_int, 
                                            surface_ref_2pt_flux_int[istate], 
                                            surface_ref_2pt_flux_int_Hadamard_with_surf_oper);
            std::vector<adtype> surface_ref_2pt_flux_ext_Hadamard_with_surf_oper(n_face_quad_pts * n_quad_pts_1D_ext);
            flux_basis_ext.Hadamard_product_AD_vector(surf_oper_sparse_ext, 
                                            surface_ref_2pt_flux_ext[istate], 
                                            surface_ref_2pt_flux_ext_Hadamard_with_surf_oper);
            //sum with reference unit normal
            surf_vol_ref_2pt_flux_interp_surf_int[istate].resize(n_face_quad_pts);
            surf_vol_ref_2pt_flux_interp_surf_ext[istate].resize(n_face_quad_pts);
            surf_vol_ref_2pt_flux_interp_vol_int[istate].resize(n_quad_pts_vol_int);
            surf_vol_ref_2pt_flux_interp_vol_ext[istate].resize(n_quad_pts_vol_ext);
         
            for(unsigned int iface_quad=0; iface_quad<n_face_quad_pts; iface_quad++){
                for(unsigned int iquad_int=0; iquad_int<n_quad_pts_1D_int; iquad_int++){
                    surf_vol_ref_2pt_flux_interp_surf_int[istate][iface_quad] 
                        -= surface_ref_2pt_flux_int_Hadamard_with_surf_oper[iface_quad * n_quad_pts_1D_int + iquad_int]
                        * unit_ref_normal_int[dim_not_zero_int];
                    const unsigned int column_index = iface_quad * n_quad_pts_1D_int + iquad_int;
                    surf_vol_ref_2pt_flux_interp_vol_int[istate][Hadamard_columns_sparsity_int[column_index]] 
                        += surface_ref_2pt_flux_int_Hadamard_with_surf_oper[iface_quad * n_quad_pts_1D_int + iquad_int]
                        * unit_ref_normal_int[dim_not_zero_int];
                }
                for(unsigned int iquad_ext=0; iquad_ext<n_quad_pts_1D_ext; iquad_ext++){
                    surf_vol_ref_2pt_flux_interp_surf_ext[istate][iface_quad] 
                        -= surface_ref_2pt_flux_ext_Hadamard_with_surf_oper[iface_quad * n_quad_pts_1D_ext + iquad_ext]
                        * (unit_ref_normal_ext[dim_not_zero_ext]);
                    const unsigned int column_index = iface_quad * n_quad_pts_1D_ext + iquad_ext;
                    surf_vol_ref_2pt_flux_interp_vol_ext[istate][Hadamard_columns_sparsity_ext[column_index]] 
                        += surface_ref_2pt_flux_ext_Hadamard_with_surf_oper[iface_quad * n_quad_pts_1D_ext + iquad_ext]
                        * (unit_ref_normal_ext[dim_not_zero_ext]);
                }
            }
        }
    }//end of if split form or curvilinear split form



    // Evaluate reference numerical fluxes.
    
    std::array<std::vector<adtype>,nstate> conv_num_flux_dot_n;
    std::vector<dealii::Tensor<1,dim,adtype>> unit_phys_normals_int(n_face_quad_pts);
    std::vector<adtype> JxW_face(n_face_quad_pts);
    //std::array<std::vector<adtype>,nstate> diss_auxi_num_flux_dot_n;
    for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {
        // Copy Metric Cofactor on the facet in a way can use for transforming Tensor Blocks to reference space
        // The way it is stored in metric_operators is to use sum-factorization in each direction,
        // but here it is cleaner to apply a reference transformation in each Tensor block returned by physics.
        // Note that for a conforming mesh, the facet metric cofactor matrix is the same from either interioir or exterior metric terms. 
        // This is verified for the metric computations in: unit_tests/operator_tests/surface_conforming_test.cpp
        dealii::Tensor<2,dim,adtype> metric_cofactor_surf;
        for(int idim=0; idim<dim; idim++){
            for(int jdim=0; jdim<dim; jdim++){
                metric_cofactor_surf[idim][jdim] = metric_oper_int.metric_cofactor_surf[idim][jdim][iquad];
            }
        }

        std::array<adtype,nstate> entropy_var_face_int;
        std::array<adtype,nstate> entropy_var_face_ext;
        //std::array<dealii::Tensor<1,dim,adtype>,nstate> aux_soln_state_int;
        //std::array<dealii::Tensor<1,dim,adtype>,nstate> aux_soln_state_ext;
        std::array<adtype,nstate> soln_interp_to_face_int;
        std::array<adtype,nstate> soln_interp_to_face_ext;
        for(int istate=0; istate<nstate; istate++){
            soln_interp_to_face_int[istate] = soln_at_surf_q_int[istate][iquad];
            soln_interp_to_face_ext[istate] = soln_at_surf_q_ext[istate][iquad];
            entropy_var_face_int[istate] = projected_entropy_var_surf_int[istate][iquad];
            entropy_var_face_ext[istate] = projected_entropy_var_surf_ext[istate][iquad];
            /*
            for(int idim=0; idim<dim; idim++){
                aux_soln_state_int[istate][idim] = aux_soln_at_surf_q_int[istate][idim][iquad];
                aux_soln_state_ext[istate][idim] = aux_soln_at_surf_q_ext[istate][idim][iquad];
            }
            */
        }

        std::array<adtype,nstate> soln_state_int;
        soln_state_int = pde_physics.compute_conservative_variables_from_entropy_variables (entropy_var_face_int);
        std::array<adtype,nstate> soln_state_ext;
        soln_state_ext = pde_physics.compute_conservative_variables_from_entropy_variables (entropy_var_face_ext);
        if constexpr(std::is_same<adtype,double>::value)
        {
            if(this->compute_dRdW_strong)
            {
                pde_physics.get_d_conservative_var_d_entropy_var(dutilde_dvtilde_face_int, entropy_var_face_int);
                pde_physics.get_d_conservative_var_d_entropy_var(dutilde_dvtilde_face_ext, entropy_var_face_ext);
            }
        }

        if(!this->all_parameters->use_split_form && !this->all_parameters->use_curvilinear_split_form){
            for(int istate=0; istate<nstate; istate++){
                soln_state_int[istate] = soln_at_surf_q_int[istate][iquad];
                soln_state_ext[istate] = soln_at_surf_q_ext[istate][iquad];
            }
        }

        // numerical fluxes
        dealii::Tensor<1,dim,adtype> unit_phys_normal_int;
        metric_oper_int.transform_reference_to_physical(unit_ref_normal_int,
                                                        metric_cofactor_surf,
                                                        unit_phys_normal_int);
        adtype face_Jac_norm_scaled = 0.0;
        for(int idim=0; idim<dim; idim++){
            face_Jac_norm_scaled += unit_phys_normal_int[idim] * unit_phys_normal_int[idim];
        }
        face_Jac_norm_scaled = sqrt(face_Jac_norm_scaled);
        JxW_face[iquad] = face_Jac_norm_scaled*surf_quad_weights[iquad];
        unit_phys_normal_int /= face_Jac_norm_scaled;//normalize it. 
        unit_phys_normals_int[iquad] = unit_phys_normal_int;
        // Note that the facet determinant of metric jacobian is the above norm multiplied by the determinant of the metric Jacobian evaluated on the facet.
        // Since the determinant of the metric Jacobian evaluated on the face cancels off, we can just scale the numerical flux by the norm.

        std::array<adtype,nstate> conv_num_flux_dot_n_at_q;
        //std::array<adtype,nstate> diss_auxi_num_flux_dot_n_at_q;
        // Convective numerical flux. 
        conv_num_flux_dot_n_at_q = conv_num_flux.evaluate_flux(soln_state_int, soln_state_ext, unit_phys_normal_int);
        // dissipative numerical flux
        /*
        diss_auxi_num_flux_dot_n_at_q = diss_num_flux.evaluate_auxiliary_flux(
            current_cell_index, neighbor_cell_index,
            0.0, 0.0,
            soln_interp_to_face_int, soln_interp_to_face_ext,
            aux_soln_state_int, aux_soln_state_ext,
            unit_phys_normal_int, penalty, false);
        */

        // Write the values in a way that we can use sum-factorization on.
        for(int istate=0; istate<nstate; istate++){
            // Write the data in a way that we can use sum-factorization on.
            // Since sum-factorization improves the speed for matrix-vector multiplications,
            // We need the values to have their inner elements be vectors of n_face_quad_pts.

            // allocate
            if(iquad == 0){
                conv_num_flux_dot_n[istate].resize(n_face_quad_pts);
         //       diss_auxi_num_flux_dot_n[istate].resize(n_face_quad_pts);
            }

            // write data
            conv_num_flux_dot_n[istate][iquad] = face_Jac_norm_scaled * conv_num_flux_dot_n_at_q[istate];
           // diss_auxi_num_flux_dot_n[istate][iquad] = face_Jac_norm_scaled * diss_auxi_num_flux_dot_n_at_q[istate];
        }
        if constexpr(std::is_same<adtype,double>::value)
        {
            if(this->compute_dRdW_strong)
            {
                dealii::Tensor<2,dim,double> metric_identity;
                for(unsigned int d=0; d<dim; ++d)
                {
                    metric_identity[d][d]=1.0;
                }
                std::array<std::array<std::array<double,nstate>,nstate>,2> dFsplit_dutilde = pde_physics.convective_numerical_split_flux_derivative(soln_state_int, soln_state_ext, metric_identity, unit_phys_normal_int);

                for(unsigned int s1=0; s1<nstate; ++s1)
                {
                    for(unsigned int s2=0; s2<nstate; ++s2)
                    {
                        for(unsigned int idof=0; idof<n_shape_fns_int; ++idof)
                        {
                            double sum_int = 0;
                            for(unsigned int sk=0; sk<nstate; ++sk)
                            {
                                for(unsigned int sl=0; sl<nstate; ++sl)
                                {
                                    sum_int += dFsplit_dutilde[0][s1][sk]*dutilde_dvtilde_face_int[sk][sl]*dvtilde_duh_surf_int[sl][s2][iquad][idof];
                                }
                            }
                            dterm3int_duhint[s1][s2][iquad][idof] = JxW_face[iquad]*sum_int;
                            dterm3ext_duhint[s1][s2][iquad][idof] = -dterm3int_duhint[s1][s2][iquad][idof];
                        }
                        for(unsigned int idof=0; idof<n_shape_fns_ext; ++idof)
                        {
                            double sum_ext = 0;
                            for(unsigned int sk=0; sk<nstate; ++sk)
                            {
                                for(unsigned int sl=0; sl<nstate; ++sl)
                                {
                                    sum_ext += dFsplit_dutilde[1][s1][sk]*dutilde_dvtilde_face_ext[sk][sl]*dvtilde_duh_surf_ext[sl][s2][iquad][idof];
                                }
                            }
                            dterm3int_duhext[s1][s2][iquad][idof] = JxW_face[iquad]*sum_ext;
                            dterm3ext_duhext[s1][s2][iquad][idof] = -dterm3int_duhext[s1][s2][iquad][idof];
                        }
                    }
                }
            } // compute_dRdW_strong
        }
    }

    // Compute RHS
    for(int istate=0; istate<nstate; istate++){
        // interior RHS
        std::vector<adtype> rhs_int(n_shape_fns_int);

        // convective flux
        if(this->all_parameters->use_split_form || this->all_parameters->use_curvilinear_split_form){
            std::vector<real> ones_surf(n_face_quad_pts, 1.0);
            soln_basis_int.inner_product_surface_1D(iface, 
                                                    surf_vol_ref_2pt_flux_interp_surf_int[istate], 
                                                    ones_surf, rhs_int, 
                                                    soln_basis_int.oneD_surf_operator, 
                                                    soln_basis_int.oneD_vol_operator,
                                                    false, -1.0);
            std::vector<real> ones_vol(n_quad_pts_vol_int, 1.0);
            soln_basis_int.inner_product_1D(surf_vol_ref_2pt_flux_interp_vol_int[istate], 
                                            ones_vol, rhs_int, 
                                            soln_basis_int.oneD_vol_operator, 
                                            true, -1.0);
        }
        else 
        {
            soln_basis_int.inner_product_surface_1D(iface, 
                                                    conv_int_vol_ref_flux_interp_to_face_dot_ref_normal[istate], 
                                                    surf_quad_weights, rhs_int, 
                                                    soln_basis_int.oneD_surf_operator, 
                                                    soln_basis_int.oneD_vol_operator,
                                                    false, 1.0);
        }
        /*
        // dissipative flux
        soln_basis_int.inner_product_surface_1D(iface, 
                                                diffusive_int_vol_ref_flux_interp_to_face_dot_ref_normal[istate], 
                                                surf_quad_weights, rhs_int, 
                                                soln_basis_int.oneD_surf_operator, 
                                                soln_basis_int.oneD_vol_operator,
                                                true, 1.0);//adding=true, subtract the negative so add it
        */
        // convective numerical flux
        soln_basis_int.inner_product_surface_1D(iface, conv_num_flux_dot_n[istate], 
                                                surf_quad_weights, rhs_int, 
                                                soln_basis_int.oneD_surf_operator, 
                                                soln_basis_int.oneD_vol_operator,
                                                true, -1.0);//adding=true, scaled by factor=-1.0 bc subtract it
        /*
        // dissipative numerical flux
        soln_basis_int.inner_product_surface_1D(iface, diss_auxi_num_flux_dot_n[istate], 
                                                surf_quad_weights, rhs_int, 
                                                soln_basis_int.oneD_surf_operator, 
                                                soln_basis_int.oneD_vol_operator,
                                                true, -1.0);//adding=true, scaled by factor=-1.0 bc subtract it
*/

        for(unsigned int ishape=0; ishape<n_shape_fns_int; ishape++){
            local_rhs_int_cell[istate*n_shape_fns_int + ishape] += rhs_int[ishape];
        }

        // exterior RHS
        std::vector<adtype> rhs_ext(n_shape_fns_ext);

        // convective flux
        if(this->all_parameters->use_split_form || this->all_parameters->use_curvilinear_split_form){
            std::vector<real> ones_surf(n_face_quad_pts, 1.0);
            soln_basis_ext.inner_product_surface_1D(neighbor_iface, 
                                                    surf_vol_ref_2pt_flux_interp_surf_ext[istate], 
                                                    ones_surf, rhs_ext, 
                                                    soln_basis_ext.oneD_surf_operator, 
                                                    soln_basis_ext.oneD_vol_operator,
                                                    false, -1.0);//the negative sign is bc the surface Hadamard function computes it on the otherside.
                                                    //to satisfy the unit test that checks consistency with Jesse Chan's formulation.
            std::vector<real> ones_vol(n_quad_pts_vol_ext, 1.0);
            soln_basis_ext.inner_product_1D(surf_vol_ref_2pt_flux_interp_vol_ext[istate], 
                                            ones_vol, rhs_ext, 
                                            soln_basis_ext.oneD_vol_operator, 
                                            true, -1.0);
        }
        else 
        {
            soln_basis_ext.inner_product_surface_1D(neighbor_iface, 
                                                    conv_ext_vol_ref_flux_interp_to_face_dot_ref_normal[istate], 
                                                    surf_quad_weights, rhs_ext, 
                                                    soln_basis_ext.oneD_surf_operator, 
                                                    soln_basis_ext.oneD_vol_operator,
                                                    false, 1.0);//adding false
        }
        /*
        // dissipative flux
        soln_basis_ext.inner_product_surface_1D(neighbor_iface, 
                                                diffusive_ext_vol_ref_flux_interp_to_face_dot_ref_normal[istate], 
                                                surf_quad_weights, rhs_ext, 
                                                soln_basis_ext.oneD_surf_operator, 
                                                soln_basis_ext.oneD_vol_operator,
                                                true, 1.0);//adding=true
        */
        // convective numerical flux
        soln_basis_ext.inner_product_surface_1D(neighbor_iface, conv_num_flux_dot_n[istate], 
                                                surf_quad_weights, rhs_ext, 
                                                soln_basis_ext.oneD_surf_operator, 
                                                soln_basis_ext.oneD_vol_operator,
                                                true, 1.0);//adding=true, scaled by factor=1.0 because negative numerical flux and subtract it
        /*
        // dissipative numerical flux
        soln_basis_ext.inner_product_surface_1D(neighbor_iface, diss_auxi_num_flux_dot_n[istate], 
                                                surf_quad_weights, rhs_ext, 
                                                soln_basis_ext.oneD_surf_operator, 
                                                soln_basis_ext.oneD_vol_operator,
                                                true, 1.0);//adding=true, scaled by factor=1.0 because negative numerical flux and subtract it

*/
        for(unsigned int ishape=0; ishape<n_shape_fns_ext; ishape++){
            local_rhs_ext_cell[istate*n_shape_fns_ext + ishape] += rhs_ext[ishape];
        }
    }
    
    if constexpr(std::is_same<adtype,double>::value)
    {
        if(this->compute_dRdW_strong)
        {
            std::vector<real> ones_vol_int(n_quad_pts_vol_int, 1.0);
            std::vector<real> ones_vol_ext(n_quad_pts_vol_ext, 1.0);
            std::vector<real> ones_face(n_face_quad_pts, 1.0);
            for(unsigned int s1=0; s1<nstate; ++s1)
            {
                for(unsigned int s2=0; s2<nstate; ++s2)
                {
                    // Form dRint_dWint
                    for(unsigned int idof_sol = 0; idof_sol<n_shape_fns_int; ++idof_sol)
                    {
                        std::vector<adtype> rhs_dRdW(n_shape_fns_int);
                        std::vector<real> dterm1_at_quads(n_quad_pts_vol_int);
                        std::vector<real> dterm2_at_quads(n_face_quad_pts);
                        std::vector<real> dterm3_at_quads(n_face_quad_pts);
                        for(unsigned int q_=0; q_<n_quad_pts_vol_int; ++q_)
                        {
                            dterm1_at_quads[q_] = dterm1int_duhint[s1][s2][q_][idof_sol];
                        }
                        for(unsigned int q_=0; q_<n_face_quad_pts; ++q_)
                        {
                            dterm2_at_quads[q_] = dterm2int_duhint[s1][s2][q_][idof_sol];
                            dterm3_at_quads[q_] = dterm3int_duhint[s1][s2][q_][idof_sol];
                        }
                        soln_basis_int.inner_product_1D(dterm1_at_quads, ones_vol_int, rhs_dRdW, soln_basis_int.oneD_vol_operator, false, -1.0);
                        soln_basis_int.inner_product_surface_1D(iface, dterm2_at_quads, 
                                                            ones_face, rhs_dRdW, 
                                                            soln_basis_int.oneD_surf_operator, 
                                                            soln_basis_int.oneD_vol_operator,
                                                            true, -1.0);//adding=true, scaled by factor=-1.0 bc subtract it
                        soln_basis_int.inner_product_surface_1D(iface, dterm3_at_quads, 
                                                            ones_face, rhs_dRdW, 
                                                            soln_basis_int.oneD_surf_operator, 
                                                            soln_basis_int.oneD_vol_operator,
                                                            true, -1.0);//adding=true, scaled by factor=-1.0 bc subtract it
                        const unsigned int idof_sol_global = s2*n_shape_fns_int + idof_sol;
                        for(unsigned int idof_res = 0; idof_res<n_shape_fns_int; ++idof_res)
                        {
                            const unsigned int idof_res_global = s1*n_shape_fns_int + idof_res;
                            this->dRint_dWint_face[idof_res_global][idof_sol_global] = rhs_dRdW[idof_res];
                        }                    
                    }
                    // Form dRint_dWext
                    for(unsigned int idof_sol = 0; idof_sol<n_shape_fns_ext; ++idof_sol)
                    {
                        std::vector<adtype> rhs_dRdW(n_shape_fns_int);
                        std::vector<real> dterm3_at_quads(n_face_quad_pts);
                        for(unsigned int q_=0; q_<n_face_quad_pts; ++q_)
                        {
                            dterm3_at_quads[q_] = dterm3int_duhext[s1][s2][q_][idof_sol];
                        }
                        soln_basis_int.inner_product_surface_1D(iface, dterm3_at_quads, 
                                                            ones_face, rhs_dRdW, 
                                                            soln_basis_int.oneD_surf_operator, 
                                                            soln_basis_int.oneD_vol_operator,
                                                            false, -1.0);//adding=true, scaled by factor=-1.0 bc subtract it
                        const unsigned int idof_sol_global = s2*n_shape_fns_ext + idof_sol;
                        for(unsigned int idof_res = 0; idof_res<n_shape_fns_int; ++idof_res)
                        {
                            const unsigned int idof_res_global = s1*n_shape_fns_int + idof_res;
                            this->dRint_dWext_face[idof_res_global][idof_sol_global] = rhs_dRdW[idof_res];
                        }                    
                    }
                    

                    // Form dRext_dWext
                    for(unsigned int idof_sol = 0; idof_sol<n_shape_fns_ext; ++idof_sol)
                    {
                        std::vector<adtype> rhs_dRdW(n_shape_fns_ext);
                        std::vector<real> dterm1_at_quads(n_quad_pts_vol_ext);
                        std::vector<real> dterm2_at_quads(n_face_quad_pts);
                        std::vector<real> dterm3_at_quads(n_face_quad_pts);
                        for(unsigned int q_=0; q_<n_quad_pts_vol_ext; ++q_)
                        {
                            dterm1_at_quads[q_] = dterm1ext_duhext[s1][s2][q_][idof_sol];
                        }
                        for(unsigned int q_=0; q_<n_face_quad_pts; ++q_)
                        {
                            dterm2_at_quads[q_] = dterm2ext_duhext[s1][s2][q_][idof_sol];
                            dterm3_at_quads[q_] = dterm3ext_duhext[s1][s2][q_][idof_sol];
                        }
                        soln_basis_ext.inner_product_1D(dterm1_at_quads, ones_vol_ext, rhs_dRdW, soln_basis_ext.oneD_vol_operator, false, -1.0);
                        soln_basis_ext.inner_product_surface_1D(neighbor_iface, dterm2_at_quads, 
                                                            ones_face, rhs_dRdW, 
                                                            soln_basis_ext.oneD_surf_operator, 
                                                            soln_basis_ext.oneD_vol_operator,
                                                            true, -1.0);//adding=true, scaled by factor=-1.0 bc subtract it
                        soln_basis_ext.inner_product_surface_1D(neighbor_iface, dterm3_at_quads, 
                                                            ones_face, rhs_dRdW, 
                                                            soln_basis_ext.oneD_surf_operator, 
                                                            soln_basis_ext.oneD_vol_operator,
                                                            true, -1.0);//adding=true, scaled by factor=-1.0 bc subtract it
                        const unsigned int idof_sol_global = s2*n_shape_fns_ext + idof_sol;
                        for(unsigned int idof_res = 0; idof_res<n_shape_fns_ext; ++idof_res)
                        {
                            const unsigned int idof_res_global = s1*n_shape_fns_ext + idof_res;
                            this->dRext_dWext_face[idof_res_global][idof_sol_global] = rhs_dRdW[idof_res];
                        }                    
                    }
                    // Form dRext_dWint
                    for(unsigned int idof_sol = 0; idof_sol<n_shape_fns_int; ++idof_sol)
                    {
                        std::vector<adtype> rhs_dRdW(n_shape_fns_ext);
                        std::vector<real> dterm3_at_quads(n_face_quad_pts);
                        for(unsigned int q_=0; q_<n_face_quad_pts; ++q_)
                        {
                            dterm3_at_quads[q_] = dterm3ext_duhint[s1][s2][q_][idof_sol];
                        }
                        soln_basis_ext.inner_product_surface_1D(neighbor_iface, dterm3_at_quads, 
                                                            ones_face, rhs_dRdW, 
                                                            soln_basis_ext.oneD_surf_operator, 
                                                            soln_basis_ext.oneD_vol_operator,
                                                            false, -1.0);
                        const unsigned int idof_sol_global = s2*n_shape_fns_int + idof_sol;
                        for(unsigned int idof_res = 0; idof_res<n_shape_fns_ext; ++idof_res)
                        {
                            const unsigned int idof_res_global = s1*n_shape_fns_ext + idof_res;
                            this->dRext_dWint_face[idof_res_global][idof_sol_global] = rhs_dRdW[idof_res];
                        }                    
                    }
                }
            }
        }
    }

    std::vector<adtype> face_term_br2_int;
    std::vector<adtype> face_term_br2_ext;
    assemble_face_term_entropystable_br2(
    iface,
    neighbor_iface,
    entropy_var_coeffs_int,
    entropy_var_coeffs_ext,
    projected_entropy_var_vol_int,
    projected_entropy_var_vol_ext,
    projected_entropy_var_surf_int,
    projected_entropy_var_surf_ext,
    dvtilde_duh_vol_int,
    dvtilde_duh_vol_ext,
    dvtilde_duh_surf_int,
    dvtilde_duh_surf_ext,
    unit_phys_normals_int,
    JxW_face,
    poly_degree_int,  
    poly_degree_ext,  
    n_quad_pts_vol_int,  
    n_quad_pts_vol_ext,  
    n_dofs_int, 
    n_dofs_ext, 
    n_face_quad_pts,
    soln_basis_int.oneD_vol_operator.m(),
    soln_basis_int.oneD_vol_operator.n(),
    soln_basis_int,
    soln_basis_ext,
    flux_basis_int,
    flux_basis_ext,
    metric_oper_int,
    metric_oper_ext,
    H_mat_int, 
    H_mat_ext, 
    soln_basis_projection_oper_int.oneD_G_operator_vol,
    soln_basis_projection_oper_int.oneD_DG_operator_vol,
    soln_basis_projection_oper_ext.oneD_G_operator_vol,
    soln_basis_projection_oper_ext.oneD_DG_operator_vol,
    pde_physics,
    face_term_br2_int,
    face_term_br2_ext);

    if(this->compute_only_convective_residual)
    {
        //Already computed
    }
    else if(this->compute_only_dissipative_residual)
    {
        for(unsigned int idof = 0; idof<n_dofs_int; ++idof)
        {
            local_rhs_int_cell[idof] = (-1.0)*face_term_br2_int[idof];
        }
        for(unsigned int idof = 0; idof<n_dofs_ext; ++idof)
        {
            local_rhs_ext_cell[idof] = (-1.0)*face_term_br2_ext[idof];
        }
    }
    else
    { 
        for(unsigned int idof = 0; idof<n_dofs_int; ++idof)
        {
            local_rhs_int_cell[idof] += (-1.0)*face_term_br2_int[idof];
        }
        for(unsigned int idof = 0; idof<n_dofs_ext; ++idof)
        {
            local_rhs_ext_cell[idof] += (-1.0)*face_term_br2_ext[idof];
        }
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

// Computes \int_k G^h(dws/dx_d)*Tds d\Omega, where T is the quad values of G^h(\nabla entropy_var) or a lift polynomial of size nstate x dim.
template <int dim, int nstate, typename real, typename MeshType>
template <typename adtype>
void DGStrong<dim,nstate,real,MeshType>::compute_gradbasis_T_vol_integral(
    const unsigned int                                                 /*n_quad_pts*/,
    const unsigned int                                                 n_dofs_cell,
    OPERATOR::basis_functions<dim,2*dim>                               &soln_basis,
    OPERATOR::metric_operators<adtype,dim,2*dim>                       &metric_oper,
    const std::vector<double>                                          &weight_vect,
    const std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate>   &T,
    std::vector<adtype>                                                &integral_val) const
{
    const unsigned int n_shape_fns = n_dofs_cell / nstate; 
    
    // Form M = cof(J)^T*T
    std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate>   M;
    for(unsigned int s=0; s<nstate; ++s)
    {
        metric_oper.transform_physical_to_reference_vector(
            T[s],
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
    // Form L = K*T
    std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate>   L;
    apply_K_matrix(
    entropy_var_at_quads,
    n_quad_pts,
    pde_physics,
    T,
    L);

    compute_gradbasis_T_vol_integral(
    n_quad_pts,
    n_dofs_cell,
    soln_basis,
    metric_oper,
    weight_vect,
    L,
    integral_val);
}

template <int dim, int nstate, typename real, typename MeshType>
template <typename adtype>
void DGStrong<dim,nstate,real,MeshType>::compute_gradbasis_dT_duh_vol_integral(
    const unsigned int                                                 n_quad_pts_vol,
    const unsigned int                                                 n_dofs_cell,
    OPERATOR::basis_functions<dim,2*dim>                               &soln_basis,
    OPERATOR::metric_operators<adtype,dim,2*dim>                       &metric_oper,
    const std::vector<double>                                          &weight_vect,
    const std::array<std::array<std::array<dealii::FullMatrix<double>,nstate>,nstate>,dim> &dT_duh_vol,
    dealii::FullMatrix<double> &integral_val) const
{
    const unsigned int n_shape_fns = n_dofs_cell/nstate;
    integral_val.reinit(n_dofs_cell,n_dofs_cell);

    for(unsigned int s2=0; s2<nstate; ++s2)
    {
        for(unsigned int idof=0; idof<n_shape_fns; ++idof)
        {
            std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate>   dT_duhs2idof;
            for(unsigned int s1=0; s1<nstate; ++s1)
            {
                for(unsigned int d=0; d<dim; ++d)
                {
                    dT_duhs2idof[s1][d].resize(n_quad_pts_vol);
                    for(unsigned int iquad=0; iquad<n_quad_pts_vol; ++iquad)
                    {
                        dT_duhs2idof[s1][d][iquad] = dT_duh_vol[d][s1][s2][iquad][idof];
                    }
                }
            }
            std::vector<adtype> integral_val_s2_idof;
            compute_gradbasis_T_vol_integral(
                n_quad_pts_vol,
                n_dofs_cell,
                soln_basis,
                metric_oper,
                weight_vect,
                dT_duhs2idof,
                integral_val_s2_idof);

            const unsigned int col = s2*n_shape_fns + idof;
            for(unsigned int row = 0; row<n_dofs_cell; ++row)
            {
                integral_val[row][col] = integral_val_s2_idof[row];
            }            
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
void DGStrong<dim,nstate,real,MeshType>::evaluate_face_integral_dsigma_duh(
    const unsigned int iface,
    const std::array<std::array<std::array<dealii::FullMatrix<double>,nstate>,nstate>,dim> &dsigma_duh_face,
    const std::vector<adtype> &JxW_face,
    const std::vector<dealii::Tensor<1,dim,adtype>> &unit_phys_normal,
    OPERATOR::basis_functions<dim,2*dim> &soln_basis,
    const unsigned int n_dofs_cell,
    const unsigned int n_face_quad_pts,
    dealii::FullMatrix<double> &integral_val) const
{
    const unsigned int n_shape_fns = n_dofs_cell/nstate;
    integral_val.reinit(n_dofs_cell,n_dofs_cell);


    std::array<std::array<dealii::FullMatrix<double>,nstate>,nstate> dsigma_dot_n_duh_face;

    for(unsigned int s1=0; s1<nstate; ++s1)
    {
        for(unsigned int s2=0; s2<nstate; ++s2)
        {
            dsigma_dot_n_duh_face[s1][s2].reinit(n_face_quad_pts,n_shape_fns);
            for(unsigned int idof=0; idof<n_shape_fns; ++idof)
            {
                for(unsigned int iquad = 0; iquad<n_face_quad_pts; ++iquad)
                {
                    dsigma_dot_n_duh_face[s1][s2][iquad][idof]=0;
                    for(unsigned int d=0; d<dim; ++d)
                    {
                        dsigma_dot_n_duh_face[s1][s2][iquad][idof] += dsigma_duh_face[d][s1][s2][iquad][idof]*unit_phys_normal[iquad][d]; 
                    }
                }
            }
        }
    }

    for(unsigned int s2=0; s2<nstate; ++s2)
    {
        for(unsigned int idof = 0; idof<n_shape_fns; ++idof)
        {
            std::array<std::vector<adtype>,nstate> dsigma_dot_n_duhs2idof;
            for(unsigned int s1=0; s1<nstate; ++s1)
            {
                dsigma_dot_n_duhs2idof[s1].resize(n_face_quad_pts);
                for(unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad)
                {
                    dsigma_dot_n_duhs2idof[s1][iquad] = dsigma_dot_n_duh_face[s1][s2][iquad][idof];
                }
            }
            std::vector<adtype> integral_val_s2idof;

            evaluate_face_integral(
                iface,
                dsigma_dot_n_duhs2idof,
                JxW_face,
                soln_basis,
                n_dofs_cell,
                integral_val_s2idof);

            const unsigned int col = s2*n_shape_fns + idof;
            for(unsigned int row=0; row<n_dofs_cell; ++row)
            {
                integral_val[row][col] = integral_val_s2idof[row];
            }
        }
    }
}

template <int dim, int nstate, typename real, typename MeshType>
void DGStrong<dim,nstate,real,MeshType>::sum_dRduh( 
    const double a,
    const dealii::FullMatrix<double> &dRduh1,
    const double b,
    const dealii::FullMatrix<double> &dRduh2,
    dealii::FullMatrix<double> &dRduh_sum) const
{
    dRduh_sum.reinit(dRduh1.m(),dRduh1.n());
    for(unsigned int i=0; i<dRduh1.m(); ++i)
    {
        for(unsigned int j=0; j<dRduh1.n(); ++j)
        {
            dRduh_sum[i][j] = a*dRduh1[i][j] + b*dRduh2[i][j];
        }
    }
}

template <int dim, int nstate, typename real, typename MeshType>
void DGStrong<dim,nstate,real,MeshType>::sum_dTduh( 
    const double a,
    const std::array<std::array<std::array<dealii::FullMatrix<double>,nstate>,nstate>,dim> &dTduh1,
    const double b,
    const std::array<std::array<std::array<dealii::FullMatrix<double>,nstate>,nstate>,dim> &dTduh2,
    std::array<std::array<std::array<dealii::FullMatrix<double>,nstate>,nstate>,dim>  &dTduh_sum) const
{
    const unsigned int row_length = dTduh1[0][0][0].m(); 
    const unsigned int col_length = dTduh1[0][0][0].n();
    for(unsigned int d=0; d<dim; ++d)
    {
        for(unsigned int s1=0; s1<nstate; ++s1)
        {
            for(unsigned int s2=0; s2<nstate; ++s2)
            {
                dTduh_sum[d][s1][s2].reinit(row_length,col_length);
                for(unsigned int i=0; i<row_length; ++i)
                {
                    for(unsigned int j=0; j<col_length; ++j)
                    {
                        dTduh_sum[d][s1][s2][i][j] = a*dTduh1[d][s1][s2][i][j] + b*dTduh2[d][s1][s2][i][j];
                    }
                }
            }
        }
    }
}

template <int dim, int nstate, typename real, typename MeshType>
void DGStrong<dim,nstate,real,MeshType>::interpolate_dT_duh_to_face(
    const unsigned int iface,
    const std::array<std::array<std::array<dealii::FullMatrix<double>,nstate>,nstate>,dim> &dT_duh_at_vol,
    OPERATOR::basis_functions<dim,2*dim> &flux_basis,
    const unsigned int n_face_quad_pts,
    const unsigned int n_vol_quad_pts,
    const unsigned int n_dofs_cell,
    std::array<std::array<std::array<dealii::FullMatrix<double>,nstate>,nstate>,dim> &dT_duh_at_face) const
{
    const unsigned int n_shape_fns = n_dofs_cell/nstate;

    for(unsigned int d=0; d<dim; ++d)
    {
        for(unsigned int s1=0; s1<nstate; ++s1)
        {
            for(unsigned int s2=0; s2<nstate; ++s2)
            {
                dT_duh_at_face[d][s1][s2].reinit(n_face_quad_pts,n_shape_fns);
            }
        }
    }

    for(unsigned int s2=0; s2<nstate; ++s2)
    {
        for(unsigned int idof=0; idof<n_shape_fns; ++idof)
        {
            std::array<dealii::Tensor<1,dim,std::vector<double>>,nstate> dT_duhs2idof_at_vol;
            for(unsigned int s1=0; s1<nstate; ++s1)
            {
                for(unsigned int d=0; d<dim; ++d)
                {
                    dT_duhs2idof_at_vol[s1][d].resize(n_vol_quad_pts);
                    for(unsigned int iquad=0; iquad<n_vol_quad_pts; ++iquad)
                    {
                        dT_duhs2idof_at_vol[s1][d][iquad] = dT_duh_at_vol[d][s1][s2][iquad][idof];
                    }
                }
            }
            std::array<dealii::Tensor<1,dim,std::vector<double>>,nstate> dT_duhs2idof_at_face;
            interpolate_to_face(
                iface,
                dT_duhs2idof_at_vol,
                flux_basis,
                n_face_quad_pts,
                dT_duhs2idof_at_face);
            for(unsigned int s1=0; s1<nstate; ++s1)
            {
                for(unsigned int d=0; d<dim; ++d)
                {
                    for(unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad)
                    {
                        dT_duh_at_face[d][s1][s2][iquad][idof] = dT_duhs2idof_at_face[s1][d][iquad];
                    }
                }
            }
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
void DGStrong<dim,nstate,real,MeshType>::compute_dKreT_duh(
    std::array<std::array<std::array<dealii::FullMatrix<double>,nstate>,nstate>,dim> &dKreT_duh,
    const std::vector<dealii::Tensor<1,dim,adtype>>                    &unit_phys_normal,
    const std::array<std::array<dealii::FullMatrix<double>,nstate>,nstate>    &dT_duh_face,
    const std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate>   &re_T,
    OPERATOR::basis_functions<dim,2*dim> &flux_basis,
    const Physics::PhysicsBase<dim, nstate, adtype>                    &pde_physics,
    const std::array<std::vector<adtype>,nstate>                       &entropy_var_at_vol_quads,
    const std::array<std::array<dealii::FullMatrix<double>,nstate>,nstate> &dvtilde_duh_vol,
    const unsigned int n_face_quad_pts,
    const unsigned int n_vol_quads_1D,
    const unsigned int n_shape_fns_1D,
    const bool is_interior_face,
    const std::vector<adtype> &JxW_face,
    const std::vector<adtype> &JxW_vol,
    const unsigned int iface,
    const bool compute_derivative_K) const
{
    const unsigned int n_vol_quad_pts = n_vol_quads_1D*n_vol_quads_1D*n_vol_quads_1D;
    const unsigned int n_shape_fns = n_shape_fns_1D*n_shape_fns_1D*n_shape_fns_1D;
    std::array<std::array<std::array<dealii::FullMatrix<double>,nstate>,nstate>,dim> dT_duh_face_times_normal;
    std::array<std::array<std::array<dealii::FullMatrix<double>,nstate>,nstate>,dim> re_dT_duh_face_times_normal;
    for(unsigned int d=0; d<dim; ++d)
    {
        for(unsigned int s1=0; s1<nstate; ++s1)
        {
            for(unsigned int s2=0; s2<nstate; ++s2)
            {
                dT_duh_face_times_normal[d][s1][s2].reinit(n_face_quad_pts,n_shape_fns);
                dT_duh_face_times_normal[d][s1][s2]=0;
                re_dT_duh_face_times_normal[d][s1][s2].reinit(n_vol_quad_pts,n_shape_fns);
                re_dT_duh_face_times_normal[d][s1][s2]=0;
            }
        }
    }

    for(unsigned int d=0; d<dim; ++d)
    {
        for(unsigned int s1=0; s1<nstate; ++s1)
        {
            for(unsigned int s2=0; s2<nstate; ++s2)
            {
                for(unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad)
                {
                    for(unsigned int idof=0; idof<n_shape_fns; ++idof)
                    {
                        dT_duh_face_times_normal[d][s1][s2][iquad][idof] = dT_duh_face[s1][s2][iquad][idof]*unit_phys_normal[iquad][d];
                    }
                }
            }
        }
    }

    for(unsigned int s2=0; s2<nstate; ++s2)
    {
        for(unsigned int idof=0; idof<n_shape_fns; ++idof)
        {
            std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate> phi_at_face;
            for(unsigned int s1=0; s1<nstate; ++s1)
            {
                for(unsigned int d=0; d<dim; ++d)
                {
                    phi_at_face[s1][d].resize(n_face_quad_pts);
                    for(unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad)
                    {
                        phi_at_face[s1][d][iquad] = dT_duh_face_times_normal[d][s1][s2][iquad][idof];
                    }
                }
            }
            std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate> re_out_vol;
            compute_lift_polynomial(
                iface,
                phi_at_face,
                JxW_face,
                JxW_vol,
                flux_basis,
                n_face_quad_pts,
                n_vol_quad_pts,
                is_interior_face,
                re_out_vol);

            for(unsigned int s1=0; s1<nstate; ++s1)
            {
                for(unsigned int d=0; d<dim; ++d)
                {
                    for(unsigned int iquad=0; iquad<n_vol_quad_pts; ++iquad)
                    {
                        re_dT_duh_face_times_normal[d][s1][s2][iquad][idof] = re_out_vol[s1][d][iquad];
                    }
                }
            }
        }
    }

    compute_dKT_duh_vol(
    pde_physics,
    entropy_var_at_vol_quads,
    re_T,
    dvtilde_duh_vol,
    re_dT_duh_face_times_normal,
    dKreT_duh,
    n_vol_quads_1D,
    n_shape_fns_1D,
    compute_derivative_K);
}

template <int dim, int nstate, typename real, typename MeshType>
template <typename adtype>
void DGStrong<dim,nstate,real,MeshType>::compute_dK_gradientropyvar_duh_vol(
    const xt::xarray<double> &H_mat, 
    const Physics::PhysicsBase<dim, nstate, adtype>                    &pde_physics,
    const std::array<std::vector<adtype>,nstate>                       &entropy_var_at_q,
    const std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate>   &grad_entropy_var_at_q,
    const std::array<std::array<dealii::FullMatrix<double>,nstate>,nstate> &dvtilde_duh_vol,
    const dealii::FullMatrix<double> & G_oper,
    const dealii::FullMatrix<double> & DG_oper,
    const unsigned int n_shape_fns_1D,
    const unsigned int n_vol_quads_1D,
    OPERATOR::metric_operators<adtype,dim,2*dim>                       &metric_oper,
    std::array<std::array<std::array<dealii::FullMatrix<double>,nstate>,nstate>,dim> &d_K_nablavtilde_d_uh) const
{
    std::array<std::array<std::array<dealii::FullMatrix<double>,nstate>,nstate>,dim> d_nablavtilde_d_uh_vol;
    compute_dgradientropyvar_duh(
        H_mat, 
        G_oper,
        DG_oper,
        n_shape_fns_1D,
        n_vol_quads_1D,
        metric_oper,
        d_nablavtilde_d_uh_vol);

    compute_dKT_duh_vol(
        pde_physics,
        entropy_var_at_q,
        grad_entropy_var_at_q,
        dvtilde_duh_vol,
        d_nablavtilde_d_uh_vol,
        d_K_nablavtilde_d_uh,
        n_vol_quads_1D,
        n_shape_fns_1D,
        true);
}


template <int dim, int nstate, typename real, typename MeshType>
template <typename adtype>
void DGStrong<dim,nstate,real,MeshType>::compute_dgradientropyvar_duh(
    const xt::xarray<double> &H_mat, 
    const dealii::FullMatrix<double> & G_oper,
    const dealii::FullMatrix<double> & DG_oper,
    const unsigned int n_shape_fns_1D,
    const unsigned int n_vol_quads_1D,
    OPERATOR::metric_operators<adtype,dim,2*dim>                       &metric_oper,
    std::array<std::array<std::array<dealii::FullMatrix<double>,nstate>,nstate>,dim> &d_nablavtilde_d_uh) const
{
    std::array<std::array<std::array<dealii::FullMatrix<double>,nstate>,nstate>,dim> d_nablavtilde_d_uh_ref;
    const unsigned int n_quad_pts = n_vol_quads_1D*n_vol_quads_1D*n_vol_quads_1D;
    const unsigned int n_shape_fns = n_shape_fns_1D*n_shape_fns_1D*n_shape_fns_1D;
    for(unsigned int d=0; d<dim; ++d)
    {
        for(unsigned int s1=0; s1<nstate; ++s1)
        {
            for(unsigned int s2=0; s2<nstate; ++s2)
            {
                d_nablavtilde_d_uh[d][s1][s2].reinit(n_vol_quads_1D*n_vol_quads_1D*n_vol_quads_1D,n_shape_fns_1D*n_shape_fns_1D*n_shape_fns_1D);
                d_nablavtilde_d_uh_ref[d][s1][s2].reinit(n_vol_quads_1D*n_vol_quads_1D*n_vol_quads_1D,n_shape_fns_1D*n_shape_fns_1D*n_shape_fns_1D);
            }
        }
    }
    
    xt::xarray<double> temp1  = xt::xarray<double>::from_shape({n_vol_quads_1D,n_shape_fns_1D,n_shape_fns_1D,n_shape_fns_1D,n_vol_quads_1D,n_vol_quads_1D}); // L,i,j,k,Mtilde,Ntilde    
    xt::xarray<double> temp2  = xt::xarray<double>::from_shape({n_vol_quads_1D,n_vol_quads_1D,n_shape_fns_1D,n_shape_fns_1D,n_shape_fns_1D,n_vol_quads_1D}); // M,L,i,j,k,Ntilde
    const unsigned int n_quad_pts_1D = n_vol_quads_1D;
    for(unsigned int d=0; d<dim; ++d)
    {
        dealii::FullMatrix<double> operL = G_oper;
        dealii::FullMatrix<double> operM = G_oper;
        dealii::FullMatrix<double> operN = G_oper;
    
        if(d==0)
        {
            operL = DG_oper;
        }
        else if(d==1)
        {
            operM = DG_oper;
        }
        else if(d==2)
        {
            operN = DG_oper;
        }

        for(unsigned int s1=0; s1<nstate; ++s1)
        {
            for(unsigned int s2=0; s2<nstate; ++s2)
            {
                // Form temp1
                for(unsigned int L = 0; L<n_vol_quads_1D; ++L)
                {
                    for(unsigned int i=0; i<n_shape_fns_1D; ++i)
                    {
                        for(unsigned int j=0; j<n_shape_fns_1D; ++j)
                        {
                            for(unsigned int k=0; k<n_shape_fns_1D; ++k)
                            {
                                for(unsigned int Mtilde = 0; Mtilde<n_quad_pts_1D; ++Mtilde)
                                {
                                    for(unsigned int Ntilde = 0; Ntilde<n_quad_pts_1D; ++Ntilde)
                                    {
                                        temp1(L,i,j,k,Mtilde,Ntilde) = 0;
                                        for(unsigned int Ltilde=0; Ltilde<n_quad_pts_1D; ++Ltilde)
                                        {
                                            temp1(L,i,j,k,Mtilde,Ntilde) += operL[L][Ltilde]*H_mat(s1,s2,i,j,k,Ltilde,Mtilde,Ntilde);
                                        }
                                    }
                                }
                                
                            }
                        }
                    }
                }
                
            // Form temp2
            for(unsigned int M = 0; M<n_quad_pts_1D; ++M)
            {
                for(unsigned int L = 0; L<n_quad_pts_1D; ++L)
                {
                    for(unsigned int i=0; i<n_shape_fns_1D; ++i)
                    {
                        for(unsigned int j=0; j<n_shape_fns_1D; ++j)
                        {
                            for(unsigned int k=0; k<n_shape_fns_1D; ++k)
                            {
                                for(unsigned int Ntilde = 0; Ntilde<n_quad_pts_1D; ++Ntilde)
                                {
                                    temp2(M,L,i,j,k,Ntilde) = 0;
                                    for(unsigned int Mtilde=0; Mtilde<n_quad_pts_1D; ++Mtilde)
                                    {
                                        temp2(M,L,i,j,k,Ntilde) += operM[M][Mtilde]*temp1(L,i,j,k,Mtilde,Ntilde);
                                    }
                                } 
                            }
                        }
                    }
                }
            }

            // Form the term
            for(unsigned int i=0; i<n_shape_fns_1D; ++i)
            {
                for(unsigned int j=0; j<n_shape_fns_1D; ++j)
                {
                    for(unsigned int k=0; k<n_shape_fns_1D; ++k)
                    {
                        const unsigned int idof = i + j*n_shape_fns_1D + k*n_shape_fns_1D*n_shape_fns_1D;
                        for(unsigned int L = 0; L<n_quad_pts_1D; ++L)
                        {
                            for(unsigned int M = 0; M<n_quad_pts_1D; ++M)
                            {
                                for(unsigned int N = 0; N<n_quad_pts_1D; ++N)
                                {
                                    const unsigned int iquad = L + M*n_quad_pts_1D + N*n_quad_pts_1D*n_quad_pts_1D;
                                    d_nablavtilde_d_uh_ref[d][s1][s2][iquad][idof]=0;
                                    for(unsigned int Ntilde = 0; Ntilde<n_quad_pts_1D; ++Ntilde)
                                    {
                                        d_nablavtilde_d_uh_ref[d][s1][s2][iquad][idof] += operN[N][Ntilde]*temp2(M,L,i,j,k,Ntilde);
                                    } 
                                }
                            }
                        }
                    }
                }
            }

        }
    }
    }


    for(unsigned int s1=0; s1<nstate; ++s1)
    {
        for(unsigned int s2=0; s2<nstate; ++s2)
        {
            for(unsigned int iquad = 0; iquad<n_quad_pts; ++iquad)
            {
                for(unsigned int idof = 0; idof<n_shape_fns; ++idof)
                {
                    for(unsigned int d=0; d<dim; ++d)
                    {
                        d_nablavtilde_d_uh[d][s1][s2][iquad][idof] = 0;
                        for(unsigned int l=0; l<dim; ++l)
                        {
                            d_nablavtilde_d_uh[d][s1][s2][iquad][idof] += d_nablavtilde_d_uh_ref[l][s1][s2][iquad][idof]*metric_oper.metric_cofactor_vol[d][l][iquad]/metric_oper.det_Jac_vol[iquad];
                        }
                    }
                }
            }
        }
    }

}

template <int dim, int nstate, typename real, typename MeshType>
template <typename adtype>
void DGStrong<dim,nstate,real,MeshType>::compute_dKT_duh_vol(
    const Physics::PhysicsBase<dim, nstate, adtype>                    &pde_physics,
    const std::array<std::vector<adtype>,nstate>                       &entropy_var_at_q,
    const std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate>   &T,
    const std::array<std::array<dealii::FullMatrix<double>,nstate>,nstate> &dvtilde_duh_vol,
    const std::array<std::array<std::array<dealii::FullMatrix<double>,nstate>,nstate>,dim> &dT_duh_vol,
    std::array<std::array<std::array<dealii::FullMatrix<double>,nstate>,nstate>,dim> &dKT_duh,
    const unsigned int n_quad_pts_1D,
    const unsigned int n_shape_fns_1D,
    bool compute_derivative_K) const
{
    const unsigned int n_quad_pts = n_quad_pts_1D*n_quad_pts_1D*n_quad_pts_1D;
    const unsigned int n_shape_fns = n_shape_fns_1D*n_shape_fns_1D*n_shape_fns_1D;
    
    std::vector<std::array<std::array<std::array<double,nstate>,nstate>,dim>> dKT_dvtilde(n_quad_pts);
    if(compute_derivative_K)
    {
        for(unsigned int iquad=0; iquad<n_quad_pts; ++iquad)
        {
            std::array<double,nstate> entropy_var;
            std::array<dealii::Tensor<1,dim,double>,nstate> T_q;
            for(unsigned int s=0; s<nstate; ++s)
            {
                entropy_var[s] = entropy_var_at_q[s][iquad];
                for(unsigned int d=0; d<dim; ++d)
                {
                    T_q[s][d] = T[s][d][iquad];
                }
            }
            pde_physics.compute_dKT_dvtilde(dKT_dvtilde[iquad],entropy_var,T_q);
        }
    }

    // Compute dKT_duh
    for(unsigned int d=0; d<dim; ++d)
    {
        for(unsigned int s1=0; s1<nstate; ++s1)
        {
            for(unsigned int s2=0; s2<nstate; ++s2)
            {
                dKT_duh[d][s1][s2].reinit(n_quad_pts,n_shape_fns);
                dKT_duh[d][s1][s2] = 0;
            }
        }
    }

    for(unsigned int s2=0; s2<nstate; ++s2)
    {
        for(unsigned int idof=0; idof<n_shape_fns; ++idof)
        {
            // Extract dT_duhs2idof
            std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate> dT_duhs2idof;
            for(unsigned int s=0; s<nstate; ++s)
            {
                for(unsigned int d=0; d<dim; ++d)
                {
                    dT_duhs2idof[s][d].resize(n_quad_pts);
                    for(unsigned int iquad=0; iquad<n_quad_pts; ++iquad)
                    {
                        dT_duhs2idof[s][d][iquad] = dT_duh_vol[d][s][s2][iquad][idof];
                    }
                }
            }
            std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate> KdT_duhs2idof;

            apply_K_matrix(
                entropy_var_at_q,
                n_quad_pts,
                pde_physics,
                dT_duhs2idof,
                KdT_duhs2idof);

            for(unsigned int d=0; d<dim; ++d)
            {
                for(unsigned int s1=0; s1<nstate; ++s1)
                {
                    for(unsigned int iquad=0; iquad<n_quad_pts; ++iquad)
                    {
                        dKT_duh[d][s1][s2][iquad][idof] = KdT_duhs2idof[s1][d][iquad];

                        if(compute_derivative_K)
                        {
                            for(unsigned int sk=0; sk<nstate; ++sk)
                            {
                                dKT_duh[d][s1][s2][iquad][idof] += dKT_dvtilde[iquad][d][s1][sk]*dvtilde_duh_vol[sk][s2][iquad][idof];
                            }
                        }
                    }
                }
            }               
        } // idof
    } // s2
}

template <int dim, int nstate, typename real, typename MeshType>
template <typename adtype>
void DGStrong<dim,nstate,real,MeshType>::compute_dvbc_duh_and_dsigmabc_duh(
    const unsigned int iface,
    const unsigned int boundary_id,
    const std::vector<dealii::Tensor<1,dim,adtype>>                                 &normal_int,
    const Physics::PhysicsBase<dim, nstate, adtype>                    &pde_physics,
    const std::array<std::array<dealii::FullMatrix<double>,nstate>,nstate> &dvtilde_duh_face,
    const std::array<std::array<dealii::FullMatrix<double>,nstate>,nstate> &dvtilde_duh_vol,
    const unsigned int                                                 n_face_quad_pts,  
    const unsigned int                                                 n_dofs_cell,
    const std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate>       &entropy_var_phys_grad_at_vol,
    const std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate>       &poly_K_nabla_v_at_face,
    const std::array<dealii::Tensor<1,dim,std::vector<adtype>>,nstate>       &entropy_var_phys_grad_at_face,
    const std::array<std::vector<adtype>,nstate>                       &entropy_var_at_vol_quads,
    const std::array<std::vector<adtype>,nstate>                       &entropy_var_at_surf_quads,
    const xt::xarray<double> &H_mat, 
    const dealii::FullMatrix<double> & G_oper_vol,
    const dealii::FullMatrix<double> & DG_oper_vol,
    const unsigned int n_shape_fns_1D,
    const unsigned int n_vol_quads_1D,
    OPERATOR::basis_functions<dim,2*dim> &flux_basis,
    OPERATOR::metric_operators<adtype,dim,2*dim>                       &metric_oper,
    std::array<std::array<std::array<dealii::FullMatrix<double>,nstate>,nstate>,dim> &dsigmabc_duh,
    std::array<std::array<dealii::FullMatrix<double>,nstate>,nstate> &dvbc_duh) const
{
    const unsigned int n_shape_fns = n_dofs_cell/nstate;
    const unsigned int n_vol_quad_pts = n_vol_quads_1D*n_vol_quads_1D*n_vol_quads_1D;
    for(unsigned int s1=0; s1<nstate; ++s1)
    {
        for(unsigned int s2=0; s2<nstate; ++s2)
        {
            dvbc_duh[s1][s2].reinit(n_face_quad_pts,n_shape_fns);
            for(unsigned int d=0; d<dim; ++d)
            {
                dsigmabc_duh[d][s1][s2].reinit(n_face_quad_pts,n_shape_fns);
            }
        }
    }
    

    // Compute dgradv_duh_vol
    std::array<std::array<std::array<dealii::FullMatrix<double>,nstate>,nstate>,dim> d_gradv_duh_vol;
    compute_dgradientropyvar_duh(
    H_mat, 
    G_oper_vol,
    DG_oper_vol,
    n_shape_fns_1D,
    n_vol_quads_1D,
    metric_oper,
    d_gradv_duh_vol);
    

    // Compute d K\nablav/duh
    std::array<std::array<std::array<dealii::FullMatrix<double>,nstate>,nstate>,dim> dK_gradv_duh_vol;
    compute_dKT_duh_vol(
    pde_physics,
    entropy_var_at_vol_quads,
    entropy_var_phys_grad_at_vol,
    dvtilde_duh_vol,
    d_gradv_duh_vol,
    dK_gradv_duh_vol,
    n_vol_quads_1D,
    n_shape_fns_1D,
    true);

    // Compute G^h(dgradv_duh_vol)
    std::array<std::array<std::array<dealii::FullMatrix<double>,nstate>,nstate>,dim> d_gradv_duh_face;
    interpolate_dT_duh_to_face(
    iface,
    d_gradv_duh_vol,
    flux_basis,
    n_face_quad_pts,
    n_vol_quad_pts,
    n_dofs_cell,
    d_gradv_duh_face);
    
    // Compute G^h(dK_gradv_duh_vol)
    std::array<std::array<std::array<dealii::FullMatrix<double>,nstate>,nstate>,dim> dK_gradv_duh_face;
    interpolate_dT_duh_to_face(
    iface,
    dK_gradv_duh_vol,
    flux_basis,
    n_face_quad_pts,
    n_vol_quad_pts,
    n_dofs_cell,
    dK_gradv_duh_face);


    
    for(unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad)
    {
        std::array<double,nstate> v_at_q;
        std::array<dealii::Tensor<1,dim,double> ,nstate> sigma_h_at_q;
        std::array<dealii::Tensor<1,dim,double> ,nstate> grad_v_at_q;
        for(unsigned int s=0; s<nstate; ++s)
        {
            v_at_q[s] = entropy_var_at_surf_quads[s][iquad];
            for(unsigned int d=0; d<dim; ++d)
            {
                sigma_h_at_q[s][d] = poly_K_nabla_v_at_face[s][d][iquad];
                grad_v_at_q[s][d] = entropy_var_phys_grad_at_face[s][d][iquad];
            }
        }

        std::array<std::array<std::array<std::array<double,dim>,nstate>,dim>,nstate> d_sigmabc_d_sigmah;
        std::array<std::array<std::array<std::array<double,dim>,nstate>,dim>,nstate> d_sigmabc_d_gradv;
        std::array<std::array<std::array<double,nstate>,nstate>,dim> d_sigmabc_dvh;
        std::array<std::array<double,nstate>,nstate> d_vbc_dvh;

        pde_physics.compute_dsigmabc_and_dvbc_derivatives(v_at_q,normal_int[iquad],sigma_h_at_q, grad_v_at_q, d_sigmabc_d_sigmah, d_sigmabc_d_gradv, d_sigmabc_dvh, d_vbc_dvh, boundary_id);
        
        for(unsigned int s2=0; s2<nstate; ++s2)
        {
            for(unsigned int idof=0; idof<n_shape_fns; ++idof)
            {
                // Compute dvbc_duh
                for(unsigned int s1=0; s1<nstate; ++s1)
                {
                    dvbc_duh[s1][s2][iquad][idof] = 0;
                    for(unsigned int sk=0; sk<nstate; ++sk)
                    {
                        dvbc_duh[s1][s2][iquad][idof] += d_vbc_dvh[s1][sk]*dvtilde_duh_face[sk][s2][iquad][idof];
                    }
                }

                // Compute dsigmabc_duh
                for(unsigned int d=0; d<dim; ++d)
                {
                    for(unsigned int s1=0; s1<nstate; ++s1)
                    {
                        dsigmabc_duh[d][s1][s2][iquad][idof]=0;

                        for(unsigned int sk=0; sk<nstate; ++sk)
                        {
                            dsigmabc_duh[d][s1][s2][iquad][idof] += d_sigmabc_dvh[d][s1][sk]*dvtilde_duh_face[sk][s2][iquad][idof];
                            for(unsigned int dk=0; dk<dim; ++dk)
                            {
                               dsigmabc_duh[d][s1][s2][iquad][idof] += d_sigmabc_d_sigmah[s1][d][sk][dk]*dK_gradv_duh_face[dk][sk][s2][iquad][idof]
                                                                     + d_sigmabc_d_gradv[s1][d][sk][dk]*d_gradv_duh_face[dk][sk][s2][iquad][idof];
                            }
                        }
                    }
                }
            }
        }
    }

}


template <int dim, int nstate, typename real, typename MeshType>
template <typename adtype>
void DGStrong<dim,nstate,real,MeshType>::assemble_volume_term_entropystable_br2(
    const unsigned int                                                 poly_degree,
    const std::array<std::vector<adtype>,nstate>                       &entropy_var_coeff,
    const std::array<std::vector<adtype>,nstate>                       &entropy_var_at_q,
    const unsigned int                                                 n_quad_pts,  
    const unsigned int                                                 n_dofs_cell, 
    OPERATOR::basis_functions<dim,2*dim>                               &soln_basis,
    OPERATOR::basis_functions<dim,2*dim>                               &/*flux_basis*/,
    OPERATOR::metric_operators<adtype,dim,2*dim>                       &metric_oper,
    const Physics::PhysicsBase<dim, nstate, adtype>                    &pde_physics,
    const xt::xarray<double>                                           &H_mat, 
    const dealii::FullMatrix<double> & G_oper,
    const dealii::FullMatrix<double> & DG_oper,
    const std::array<std::array<dealii::FullMatrix<double>,nstate>,nstate> &dvtilde_duh_vol,
    const unsigned int n_shape_fns_1D,
    const unsigned int n_vol_quads_1D,
    std::vector<adtype>                                                &vol_term) 
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

    if constexpr(std::is_same<adtype,double>::value)
    {
        if(this->compute_dRdW_strong)
        {
            std::array<std::array<std::array<dealii::FullMatrix<double>,nstate>,nstate>,dim> d_K_gradv_d_uh;

            compute_dK_gradientropyvar_duh_vol(
                H_mat, 
                pde_physics,
                entropy_var_at_q,
                entropy_var_phys_grad,
                dvtilde_duh_vol,
                G_oper,
                DG_oper,
                n_shape_fns_1D,
                n_vol_quads_1D,
                metric_oper,
                d_K_gradv_d_uh);

            dealii::FullMatrix<double> voldRdW;
            compute_gradbasis_dT_duh_vol_integral(
                n_quad_pts,
                n_dofs_cell,
                soln_basis,
                metric_oper,
                weight_vect,
                d_K_gradv_d_uh,
                voldRdW);

            this->dRdW_vol_cell.add(-1.0,voldRdW); 
        }
    }
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
    const std::array<std::array<dealii::FullMatrix<double>,nstate>,nstate> &dvtilde_duh_vol_int,
    const std::array<std::array<dealii::FullMatrix<double>,nstate>,nstate> &dvtilde_duh_vol_ext,
    const std::array<std::array<dealii::FullMatrix<double>,nstate>,nstate> &dvtilde_duh_surf_int,
    const std::array<std::array<dealii::FullMatrix<double>,nstate>,nstate> &dvtilde_duh_surf_ext,
    const std::vector<dealii::Tensor<1,dim,adtype>>                    &unit_phys_normal_int,
    const std::vector<adtype>                                          &JxW_face,
    const unsigned int                                                  poly_degree_int,  
    const unsigned int                                                  poly_degree_ext,  
    const unsigned int                                                  n_vol_quad_pts_int,  
    const unsigned int                                                  n_vol_quad_pts_ext,  
    const unsigned int                                                  n_dofs_cell_int, 
    const unsigned int                                                  n_dofs_cell_ext, 
    const unsigned int                                                  n_face_quad_pts, 
    const unsigned int n_vol_quads_1D,
    const unsigned int n_shape_fns_1D,
    OPERATOR::basis_functions<dim,2*dim>                               &soln_basis_int,
    OPERATOR::basis_functions<dim,2*dim>                               &soln_basis_ext,
    OPERATOR::basis_functions<dim,2*dim>                               &flux_basis_int,
    OPERATOR::basis_functions<dim,2*dim>                               &flux_basis_ext,
    OPERATOR::metric_operators<adtype,dim,2*dim>                       &metric_oper_int,
    OPERATOR::metric_operators<adtype,dim,2*dim>                       &metric_oper_ext,
    const xt::xarray<double> &H_mat_int, 
    const xt::xarray<double> &H_mat_ext, 
    const dealii::FullMatrix<double> & G_oper_int,
    const dealii::FullMatrix<double> & DG_oper_int,
    const dealii::FullMatrix<double> & G_oper_ext,
    const dealii::FullMatrix<double> & DG_oper_ext,
    const Physics::PhysicsBase<dim, nstate, adtype>                    &pde_physics,
    std::vector<adtype>                                                &face_term_int,
    std::vector<adtype>                                                &face_term_ext) 
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

    std::vector<dealii::Tensor<1,dim,adtype>> unit_phys_normal_ext(n_face_quad_pts);
    // compute sigma_avg_dot_n
    for(unsigned int s=0; s<nstate; ++s)
    {
        sigma_at_face_avg_dot_n_int[s].resize(n_face_quad_pts);
        sigma_at_face_avg_dot_n_ext[s].resize(n_face_quad_pts);
        for(unsigned int q=0; q<n_face_quad_pts; ++q)
        {
            for(unsigned int d=0; d<dim; ++d)
            {
                unit_phys_normal_ext[q][d] = -unit_phys_normal_int[q][d];
                sigma_at_face_avg_dot_n_int[s][q] += 0.5*(sigma_at_face_int[s][d][q] + sigma_at_face_ext[s][d][q])*unit_phys_normal_int[q][d];
                sigma_at_face_avg_dot_n_ext[s][q] += 0.5*(sigma_at_face_int[s][d][q] + sigma_at_face_ext[s][d][q])*unit_phys_normal_ext[q][d];
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

    if constexpr(std::is_same<adtype,double>::value)
    {
        if(this->compute_dRdW_strong)
        {
            std::array<std::array<std::array<dealii::FullMatrix<double>,nstate>,nstate>,dim> dKreint_duhint_vol;
            compute_dKreT_duh(
                dKreint_duhint_vol,
                unit_phys_normal_int,
                dvtilde_duh_surf_int,
                re_int,
                flux_basis_int,
                pde_physics,
                entropy_var_at_vol_int,
                dvtilde_duh_vol_int,
                n_face_quad_pts,
                n_vol_quads_1D,
                n_shape_fns_1D,
                is_interior_face,
                JxW_face,
                JxW_vol_int,
                iface_int,
                true);
            
            std::array<std::array<std::array<dealii::FullMatrix<double>,nstate>,nstate>,dim> dKreint_duhext_vol;
            compute_dKreT_duh(
                dKreint_duhext_vol,
                unit_phys_normal_ext,
                dvtilde_duh_surf_ext,
                re_int,
                flux_basis_int,
                pde_physics,
                entropy_var_at_vol_int,
                dvtilde_duh_vol_int,
                n_face_quad_pts,
                n_vol_quads_1D,
                n_shape_fns_1D,
                is_interior_face,
                JxW_face,
                JxW_vol_int,
                iface_int,
                false);
            
            std::array<std::array<std::array<dealii::FullMatrix<double>,nstate>,nstate>,dim> dKreext_duhint_vol;
            compute_dKreT_duh(
                dKreext_duhint_vol,
                unit_phys_normal_int,
                dvtilde_duh_surf_int,
                re_ext,
                flux_basis_ext,
                pde_physics,
                entropy_var_at_vol_ext,
                dvtilde_duh_vol_ext,
                n_face_quad_pts,
                n_vol_quads_1D,
                n_shape_fns_1D,
                is_interior_face,
                JxW_face,
                JxW_vol_ext,
                iface_ext,
                false);
            
            std::array<std::array<std::array<dealii::FullMatrix<double>,nstate>,nstate>,dim> dKreext_duhext_vol;
            compute_dKreT_duh(
                dKreext_duhext_vol,
                unit_phys_normal_ext,
                dvtilde_duh_surf_ext,
                re_ext,
                flux_basis_ext,
                pde_physics,
                entropy_var_at_vol_ext,
                dvtilde_duh_vol_ext,
                n_face_quad_pts,
                n_vol_quads_1D,
                n_shape_fns_1D,
                is_interior_face,
                JxW_face,
                JxW_vol_ext,
                iface_ext,
                true);

            // Form volume k terms
            dealii::FullMatrix<double> dRint_dWint_k;
            compute_gradbasis_dT_duh_vol_integral(
                n_vol_quad_pts_int,
                n_dofs_cell_int,
                soln_basis_int,
                metric_oper_int,
                vol_quad_weights_int,
                dKreint_duhint_vol,
                dRint_dWint_k);

            dealii::FullMatrix<double> dRint_dWext_k;
            compute_gradbasis_dT_duh_vol_integral(
                n_vol_quad_pts_int,
                n_dofs_cell_int,
                soln_basis_int,
                metric_oper_int,
                vol_quad_weights_int,
                dKreint_duhext_vol,
                dRint_dWext_k);
            
            dealii::FullMatrix<double> dRext_dWint_k;
            compute_gradbasis_dT_duh_vol_integral(
                n_vol_quad_pts_ext,
                n_dofs_cell_ext,
                soln_basis_ext,
                metric_oper_ext,
                vol_quad_weights_ext,
                dKreext_duhint_vol,
                dRext_dWint_k);
            
            dealii::FullMatrix<double> dRext_dWext_k;
            compute_gradbasis_dT_duh_vol_integral(
                n_vol_quad_pts_ext,
                n_dofs_cell_ext,
                soln_basis_ext,
                metric_oper_ext,
                vol_quad_weights_ext,
                dKreext_duhext_vol,
                dRext_dWext_k);

            this->dRint_dWint_face.add(-1.0,dRint_dWint_k);
            this->dRint_dWext_face.add(-1.0,dRint_dWext_k);
            this->dRext_dWint_face.add(-1.0,dRext_dWint_k);
            this->dRext_dWext_face.add(-1.0,dRext_dWext_k);

            // Form sigma terms
            std::array<std::array<std::array<dealii::FullMatrix<double>,nstate>,nstate>,dim> d_Kgradv_d_uh_int;
            compute_dK_gradientropyvar_duh_vol(
                H_mat_int, 
                pde_physics,
                entropy_var_at_vol_int,
                entropy_var_phys_grad_int,
                dvtilde_duh_vol_int,
                G_oper_int,
                DG_oper_int,
                n_shape_fns_1D,
                n_vol_quads_1D,
                metric_oper_int,
                d_Kgradv_d_uh_int);

            std::array<std::array<std::array<dealii::FullMatrix<double>,nstate>,nstate>,dim> d_Kgradv_d_uh_ext;
            compute_dK_gradientropyvar_duh_vol(
                H_mat_ext, 
                pde_physics,
                entropy_var_at_vol_ext,
                entropy_var_phys_grad_ext,
                dvtilde_duh_vol_ext,
                G_oper_ext,
                DG_oper_ext,
                n_shape_fns_1D,
                n_vol_quads_1D,
                metric_oper_ext,
                d_Kgradv_d_uh_ext);

            std::array<std::array<std::array<dealii::FullMatrix<double>,nstate>,nstate>,dim> t1vol;
            std::array<std::array<std::array<dealii::FullMatrix<double>,nstate>,nstate>,dim> t2vol;
            std::array<std::array<std::array<dealii::FullMatrix<double>,nstate>,nstate>,dim> t3vol;
            std::array<std::array<std::array<dealii::FullMatrix<double>,nstate>,nstate>,dim> t4vol;
            std::array<std::array<std::array<dealii::FullMatrix<double>,nstate>,nstate>,dim> t1surf;
            std::array<std::array<std::array<dealii::FullMatrix<double>,nstate>,nstate>,dim> t2surf;
            std::array<std::array<std::array<dealii::FullMatrix<double>,nstate>,nstate>,dim> t3surf;
            std::array<std::array<std::array<dealii::FullMatrix<double>,nstate>,nstate>,dim> t4surf;

            for(unsigned int d=0; d<dim; ++d)
            {
                for(unsigned int s1=0; s1<nstate; ++s1)
                {
                    for(unsigned int s2=0; s2<nstate; ++s2)
                    {
                        t1vol[d][s1][s2] = d_Kgradv_d_uh_int[d][s1][s2];
                        t1vol[d][s1][s2].add(br2_factor,dKreint_duhint_vol[d][s1][s2]);

                        t2vol[d][s1][s2] = dKreext_duhint_vol[d][s1][s2];
                        t2vol[d][s1][s2]*=br2_factor;

                        t3vol[d][s1][s2] = dKreint_duhext_vol[d][s1][s2];
                        t3vol[d][s1][s2]*=br2_factor;

                        t4vol[d][s1][s2] = d_Kgradv_d_uh_ext[d][s1][s2];
                        t4vol[d][s1][s2].add(br2_factor,dKreext_duhext_vol[d][s1][s2]);
                    }
                }
            }

            interpolate_dT_duh_to_face(
                iface_int,
                t1vol,
                flux_basis_int,
                n_face_quad_pts,
                n_vol_quad_pts_int,
                n_dofs_cell_int,
                t1surf);
            
            interpolate_dT_duh_to_face(
                iface_ext,
                t2vol,
                flux_basis_ext,
                n_face_quad_pts,
                n_vol_quad_pts_ext,
                n_dofs_cell_ext,
                t2surf);
            
            interpolate_dT_duh_to_face(
                iface_int,
                t3vol,
                flux_basis_int,
                n_face_quad_pts,
                n_vol_quad_pts_int,
                n_dofs_cell_int,
                t3surf);
            
            interpolate_dT_duh_to_face(
                iface_ext,
                t4vol,
                flux_basis_ext,
                n_face_quad_pts,
                n_vol_quad_pts_ext,
                n_dofs_cell_ext,
                t4surf);

            std::array<std::array<std::array<dealii::FullMatrix<double>,nstate>,nstate>,dim> d_sigmastar_duhint;
            std::array<std::array<std::array<dealii::FullMatrix<double>,nstate>,nstate>,dim> d_sigmastar_duhext;

            sum_dTduh( 
                0.5,
                t1surf,
                0.5,
                t2surf,
                d_sigmastar_duhint);
            
            sum_dTduh( 
                0.5,
                t3surf,
                0.5,
                t4surf,
                d_sigmastar_duhext);

            dealii::FullMatrix<double> dRint_dWint_e;
            dealii::FullMatrix<double> dRint_dWext_e;
            dealii::FullMatrix<double> dRext_dWint_e;
            dealii::FullMatrix<double> dRext_dWext_e;

            evaluate_face_integral_dsigma_duh(
                iface_int,
                d_sigmastar_duhint,
                JxW_face,
                unit_phys_normal_int,
                soln_basis_int,
                n_dofs_cell_int,
                n_face_quad_pts,
                dRint_dWint_e);
            
            evaluate_face_integral_dsigma_duh(
                iface_int,
                d_sigmastar_duhext,
                JxW_face,
                unit_phys_normal_int,
                soln_basis_int,
                n_dofs_cell_int,
                n_face_quad_pts,
                dRint_dWext_e);
            
            evaluate_face_integral_dsigma_duh(
                iface_ext,
                d_sigmastar_duhint,
                JxW_face,
                unit_phys_normal_ext,
                soln_basis_ext,
                n_dofs_cell_ext,
                n_face_quad_pts,
                dRext_dWint_e);
            
            evaluate_face_integral_dsigma_duh(
                iface_ext,
                d_sigmastar_duhext,
                JxW_face,
                unit_phys_normal_ext,
                soln_basis_ext,
                n_dofs_cell_ext,
                n_face_quad_pts,
                dRext_dWext_e);

            this->dRint_dWint_face.add(1.0,dRint_dWint_e);
            this->dRint_dWext_face.add(1.0,dRint_dWext_e);
            this->dRext_dWint_face.add(1.0,dRext_dWint_e);
            this->dRext_dWext_face.add(1.0,dRext_dWext_e);
        }
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
    const std::array<std::array<dealii::FullMatrix<double>,nstate>,nstate> &dvtilde_duh_face,
    const std::array<std::array<dealii::FullMatrix<double>,nstate>,nstate> &dvtilde_duh_vol,
    const xt::xarray<double> &H_mat, 
    const dealii::FullMatrix<double> & G_oper_vol,
    const dealii::FullMatrix<double> & DG_oper_vol,
    const unsigned int n_shape_fns_1D,
    const unsigned int n_vol_quads_1D,
    const unsigned int                                                  n_vol_quad_pts,  
    const unsigned int                                                  n_face_quad_pts,  
    const unsigned int                                                  n_dofs_cell, 
    OPERATOR::basis_functions<dim,2*dim>                               &soln_basis,
    OPERATOR::basis_functions<dim,2*dim>                               &flux_basis,
    OPERATOR::metric_operators<adtype,dim,2*dim>                       &metric_oper,
    const std::vector<dealii::Tensor<1,dim,adtype>>                    &unit_phys_normal,
    const std::vector<adtype>                                          &JxW_face,
    const Physics::PhysicsBase<dim, nstate, adtype>                    &pde_physics,
    std::vector<adtype>                                                &boundary_term) 
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

    if constexpr(std::is_same<adtype,double>::value)
    {
        if(this->compute_dRdW_strong)
        {
            std::array<std::array<std::array<dealii::FullMatrix<double>,nstate>,nstate>,dim> dsigmabc_duh;
            std::array<std::array<dealii::FullMatrix<double>,nstate>,nstate> dvbc_duh;
            compute_dvbc_duh_and_dsigmabc_duh(
                iface,
                boundary_id,
                unit_phys_normal,
                pde_physics,
                dvtilde_duh_face,
                dvtilde_duh_vol,
                n_face_quad_pts,  
                n_dofs_cell,
                entropy_var_phys_grad_at_vol,
                poly_K_nabla_v_at_face,
                entropy_var_phys_grad_at_face,
                entropy_var_at_vol_quads,
                entropy_var_at_surf_quads,
                H_mat, 
                G_oper_vol,
                DG_oper_vol,
                n_shape_fns_1D,
                n_vol_quads_1D,
                flux_basis,
                metric_oper,
                dsigmabc_duh,
                dvbc_duh);

            dealii::FullMatrix<double> dRdW_e;
            evaluate_face_integral_dsigma_duh(
            iface,
            dsigmabc_duh,
            JxW_face,
            unit_phys_normal,
            soln_basis,
            n_dofs_cell,
            n_face_quad_pts,
            dRdW_e);

            std::array<std::array<dealii::FullMatrix<double>,nstate>,nstate> d_jump_duh_face;
            const unsigned int n_shape_fns = n_dofs_cell/nstate;
            for(unsigned int s1=0; s1<nstate; ++s1)
            {
                for(unsigned int s2=0; s2<nstate; ++s2)
                {
                    d_jump_duh_face[s1][s2].reinit(n_face_quad_pts,n_shape_fns);
                    d_jump_duh_face[s1][s2] = dvbc_duh[s1][s2];
                    d_jump_duh_face[s1][s2].add(-1.0,dvtilde_duh_face[s1][s2]);
                }
            }


            std::array<std::array<std::array<dealii::FullMatrix<double>,nstate>,nstate>,dim> dKrejump_duh_vol;
            compute_dKreT_duh(
                dKrejump_duh_vol,
                unit_phys_normal,
                d_jump_duh_face,
                re_jump,
                flux_basis,
                pde_physics,
                entropy_var_at_vol_quads,
                dvtilde_duh_vol,
                n_face_quad_pts,
                n_vol_quads_1D,
                n_shape_fns_1D,
                is_interior_face,
                JxW_face,
                JxW_vol,
                iface,
                true);
                
            dealii::FullMatrix<double> dRdW_k;
            compute_gradbasis_dT_duh_vol_integral(
            n_vol_quad_pts,
            n_dofs_cell,
            soln_basis,
            metric_oper,
            vol_quad_weights,
            dKrejump_duh_vol,
            dRdW_k);

            this->dRdW_boundary.add(-1.0, dRdW_k);
            this->dRdW_boundary.add(1.0,dRdW_e);    
        }
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


/***********************************************************
    // Additional operators for computing dRdW
*************************************************************/
template <int dim, int nstate, typename real, typename MeshType>
void DGStrong<dim,nstate,real,MeshType>::compute_H_mat(
    const dealii::FullMatrix<double> & G_oper, 
    const dealii::FullMatrix<double> & basis, 
    const std::vector<double> &jac_det_vol,
    const std::vector<std::array<std::array<real,nstate>,nstate>> &dv_du, 
    xt::xarray<double> &H) const // H[s1][s2][i][j][k][Ltilde][Mtilde][Ntilde]
{
    const unsigned int n_quad_pts_1D = basis.m();
    const unsigned int n_shape_fns_1D = basis.n();

    xt::xarray<double> temp1  = xt::xarray<double>::from_shape({n_quad_pts_1D,n_shape_fns_1D,n_quad_pts_1D,n_quad_pts_1D}); // Ltilde,i,m,n    
    xt::xarray<double> temp2  = xt::xarray<double>::from_shape({n_quad_pts_1D,n_shape_fns_1D,n_quad_pts_1D,n_shape_fns_1D,n_quad_pts_1D}); // Mtilde,j,Ltilde,i,n    
    H  = xt::xarray<double>::from_shape({nstate,nstate,n_shape_fns_1D,n_shape_fns_1D,n_shape_fns_1D,n_quad_pts_1D,n_quad_pts_1D,n_quad_pts_1D}); // s1,s2,i,j,k,Ltilde,Mtilde,Ntilde
    for(unsigned int s1=0; s1<nstate; ++s1)
    {
        for(unsigned int s2=0; s2<nstate; ++s2)
        {
            // Form temp1
            for(unsigned int Ltilde=0; Ltilde<n_quad_pts_1D; ++Ltilde)
            {
                for(unsigned int i=0; i<n_shape_fns_1D; ++i)
                {
                    for(unsigned int m=0; m<n_quad_pts_1D; ++m)
                    {
                        for(unsigned int n=0; n<n_quad_pts_1D; ++n)
                        {
                            temp1(Ltilde,i,m,n) = 0;
                            for(unsigned int l=0; l<n_quad_pts_1D; ++l)
                            {
                                const unsigned int iquad = l + m*n_quad_pts_1D + n*n_quad_pts_1D*n_quad_pts_1D;
                                temp1(Ltilde,i,m,n) += G_oper[Ltilde][l]*basis[l][i]*jac_det_vol[iquad]*dv_du[iquad][s1][s2];
                            }
                        }
                    }
                }
            }

            // Form temp2
            for(unsigned int Mtilde = 0; Mtilde < n_quad_pts_1D; ++Mtilde)
            {
                for(unsigned int j =0; j<n_shape_fns_1D; ++j)
                {
                    for(unsigned int Ltilde=0; Ltilde<n_quad_pts_1D; ++Ltilde)
                    {
                        for(unsigned int i=0; i<n_shape_fns_1D; ++i)
                        {
                            for(unsigned int n=0; n<n_quad_pts_1D; ++n)
                            {
                                temp2(Mtilde,j,Ltilde,i,n)=0;
                                for(unsigned int m=0; m<n_quad_pts_1D; ++m)
                                {
                                    temp2(Mtilde,j,Ltilde,i,n) += G_oper[Mtilde][m]*basis[m][j]*temp1(Ltilde,i,m,n);
                                }
                            }
                        }
                    }
                }
            }

            // Form H
            for(unsigned int i=0; i<n_shape_fns_1D; ++i)
            {
                for(unsigned int j=0; j<n_shape_fns_1D; ++j)
                {
                    for(unsigned int k=0; k<n_shape_fns_1D; ++k)
                    {
                        for(unsigned int Ltilde=0; Ltilde<n_quad_pts_1D; ++Ltilde)
                        {
                            for(unsigned int Mtilde=0; Mtilde<n_quad_pts_1D; ++Mtilde)
                            {
                                for(unsigned int Ntilde=0; Ntilde<n_quad_pts_1D; ++Ntilde)
                                {
                                    H(s1,s2,i,j,k,Ltilde,Mtilde,Ntilde) = 0;
                                    for(unsigned int n=0; n<n_quad_pts_1D; ++n)
                                    {
                                        H(s1,s2,i,j,k,Ltilde,Mtilde,Ntilde) += G_oper[Ntilde][n]*basis[n][k]*temp2(Mtilde,j,Ltilde,i,n);
                                    }
                                    H(s1,s2,i,j,k,Ltilde,Mtilde,Ntilde) /= jac_det_vol[Ltilde + Mtilde*n_quad_pts_1D + Ntilde*n_quad_pts_1D*n_quad_pts_1D];
                                }
                            }
                        }
                    }
                }
            }

        }
    }
}

template <int dim, int nstate, typename real, typename MeshType>
void DGStrong<dim,nstate,real,MeshType>::compute_dvtilde_duh(
    const dealii::FullMatrix<double> & G_operl, 
    const dealii::FullMatrix<double> & G_operm, 
    const dealii::FullMatrix<double> & G_opern, 
    const xt::xarray<double> &H,
    const unsigned int n_shape_fns_1D,
    const unsigned int n_vol_quads_1D,
    std::array<std::array<dealii::FullMatrix<double>,nstate>,nstate> &dvtilde_duh) const
{
    const unsigned int sizel = G_operl.m();
    const unsigned int sizem = G_operm.m();
    const unsigned int sizen = G_opern.m();
    for(unsigned int s1=0; s1<nstate; ++s1)
    {
        for(unsigned int s2=0; s2<nstate; ++s2)
        {
            dvtilde_duh[s1][s2].reinit(sizel*sizem*sizen,n_shape_fns_1D*n_shape_fns_1D*n_shape_fns_1D);
        }
    }
    

    xt::xarray<double> temp1  = xt::xarray<double>::from_shape({sizel,n_shape_fns_1D,n_shape_fns_1D,n_shape_fns_1D,n_vol_quads_1D,n_vol_quads_1D}); // l,i,j,k,Mtilde,Ntilde
    xt::xarray<double> temp2  = xt::xarray<double>::from_shape({sizem,sizel,n_shape_fns_1D,n_shape_fns_1D,n_shape_fns_1D,n_vol_quads_1D}); // m,l,i,j,k,Ntilde

    for(unsigned int s1=0; s1<nstate; ++s1)
    {
        for(unsigned int s2=0; s2<nstate; ++s2)
        {
            // Form temp1
            for(unsigned int l=0; l<sizel; ++l)
            {
                for(unsigned int i=0; i<n_shape_fns_1D; ++i)
                {
                    for(unsigned int j=0; j<n_shape_fns_1D; ++j)
                    {
                        for(unsigned int k=0; k<n_shape_fns_1D; ++k)
                        {
                            for(unsigned int Mtilde=0; Mtilde<n_vol_quads_1D; ++Mtilde)
                            {
                                for(unsigned int Ntilde=0; Ntilde<n_vol_quads_1D; ++Ntilde)
                                {
                                    temp1(l,i,j,k,Mtilde,Ntilde) = 0;
                                    for(unsigned int Ltilde=0; Ltilde<n_vol_quads_1D; ++Ltilde)
                                    {
                                        temp1(l,i,j,k,Mtilde,Ntilde) += G_operl[l][Ltilde]*H(s1,s2,i,j,k,Ltilde,Mtilde,Ntilde);
                                    }
                                }
                            }
                        }
                    }
                }

            }

            // Form temp2
            for(unsigned int m=0; m<sizem; ++m)
            {
                for(unsigned int l=0; l<sizel; ++l)
                {
                    for(unsigned int i=0; i<n_shape_fns_1D; ++i)
                    {
                        for(unsigned int j=0; j<n_shape_fns_1D; ++j)
                        {
                            for(unsigned int k=0; k<n_shape_fns_1D; ++k)
                            {
                                for(unsigned int Ntilde=0; Ntilde<n_vol_quads_1D; ++Ntilde)
                                {
                                    temp2(m,l,i,j,k,Ntilde) = 0;
                                    for(unsigned int Mtilde=0; Mtilde<n_vol_quads_1D; ++Mtilde)
                                    {
                                        temp2(m,l,i,j,k,Ntilde) += G_operm[m][Mtilde]*temp1(l,i,j,k,Mtilde,Ntilde);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            
            //Form dvtilde_duh
            for(unsigned int l=0; l<sizel; ++l)
            {
                for(unsigned int m=0; m<sizem; ++m)
                {
                    for(unsigned int n=0; n<sizen; ++n)
                    {
                        const unsigned int iquad = l + m*sizel + n*sizel*sizem;
                        for(unsigned int i=0; i<n_shape_fns_1D; ++i)
                        {
                            for(unsigned int j=0; j<n_shape_fns_1D; ++j)
                            {
                                for(unsigned int k=0; k<n_shape_fns_1D; ++k)
                                {
                                    const unsigned int idof = i + j*n_shape_fns_1D + k*n_shape_fns_1D*n_shape_fns_1D;
                                    dvtilde_duh[s1][s2][iquad][idof] = 0;
                                    for(unsigned int Ntilde=0; Ntilde<n_vol_quads_1D; ++Ntilde)
                                    {
                                        dvtilde_duh[s1][s2][iquad][idof] += G_opern[n][Ntilde]*temp2(m,l,i,j,k,Ntilde);
                                    }
                                }
                            }
                        }
                    }
                }
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
        this->dual.reinit(this->locally_owned_dofs, this->ghost_dofs, this->mpi_communicator);
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
