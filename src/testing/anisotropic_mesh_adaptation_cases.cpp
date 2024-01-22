#include <stdlib.h>
#include <iostream>
#include "physics/euler.h"
#include "anisotropic_mesh_adaptation_cases.h"
#include "flow_solver/flow_solver_factory.h"
#include "mesh/mesh_adaptation/anisotropic_mesh_adaptation.h"
#include "mesh/mesh_adaptation/fe_values_shape_hessian.h"
#include "mesh/mesh_adaptation/mesh_error_estimate.h"
#include "mesh/mesh_adaptation/mesh_optimizer.hpp"
#include "mesh/mesh_adaptation/mesh_adaptation.h"
#include <deal.II/grid/grid_in.h>
#include <deal.II/base/convergence_table.h>

namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
AnisotropicMeshAdaptationCases<dim, nstate> :: AnisotropicMeshAdaptationCases(
    const Parameters::AllParameters *const parameters_input,
    const dealii::ParameterHandler &parameter_handler_input)
    : TestsBase::TestsBase(parameters_input)
    , parameter_handler(parameter_handler_input)
{}

template<int dim, int nstate>
void AnisotropicMeshAdaptationCases<dim,nstate>::increase_grid_degree_and_interpolate_solution(std::shared_ptr<DGBase<dim,double>> dg) const
{
    const unsigned int grid_degree_updated = 2;
    dg->high_order_grid->set_q_degree(grid_degree_updated, true);

    const unsigned int poly_degree_updated = dg->all_parameters->flow_solver_param.max_poly_degree_for_adaptation - 1;
    dg->set_p_degree_and_interpolate_solution(poly_degree_updated);
}

template<int dim, int nstate>
void AnisotropicMeshAdaptationCases<dim,nstate>::evaluate_regularization_matrix(
    dealii::TrilinosWrappers::SparseMatrix &regularization_matrix, 
    std::shared_ptr<DGBase<dim,double>> dg) const
{
    // Get volume of smallest element.
    const dealii::Quadrature<dim> &volume_quadrature = dg->volume_quadrature_collection[dg->high_order_grid->grid_degree];
    const dealii::Mapping<dim> &mapping = (*(dg->high_order_grid->mapping_fe_field));
    dealii::FEValues<dim,dim> fe_values_vol(mapping, dg->high_order_grid->fe_metric_collection[dg->high_order_grid->grid_degree], volume_quadrature,
                    dealii::update_values | dealii::update_gradients | dealii::update_JxW_values);
    const unsigned int n_quad_pts = fe_values_vol.n_quadrature_points;
    const unsigned int dofs_per_cell = fe_values_vol.dofs_per_cell;
    std::vector<dealii::types::global_dof_index> dofs_indices (fe_values_vol.dofs_per_cell);
    
    double min_cell_volume_local = 1.0e6;
    for(const auto &cell : dg->high_order_grid->dof_handler_grid.active_cell_iterators())
    {
        if (!cell->is_locally_owned()) continue;
        double cell_vol = 0.0;
        fe_values_vol.reinit (cell);

        for(unsigned int q=0; q<n_quad_pts; ++q)
        {
            cell_vol += fe_values_vol.JxW(q);
        }

        if(cell_vol < min_cell_volume_local)
        {
            min_cell_volume_local = cell_vol;
        }
    }

    const double min_cell_vol = dealii::Utilities::MPI::min(min_cell_volume_local, mpi_communicator);

    // Set sparsity pattern
    dealii::AffineConstraints<double> hanging_node_constraints;
    hanging_node_constraints.clear();
    dealii::DoFTools::make_hanging_node_constraints(dg->high_order_grid->dof_handler_grid,
                                            hanging_node_constraints);
    hanging_node_constraints.close();

    dealii::DynamicSparsityPattern dsp(dg->high_order_grid->dof_handler_grid.n_dofs(), dg->high_order_grid->dof_handler_grid.n_dofs());
    dealii::DoFTools::make_sparsity_pattern(dg->high_order_grid->dof_handler_grid, dsp, hanging_node_constraints);
    const dealii::IndexSet &locally_owned_dofs = dg->high_order_grid->locally_owned_dofs_grid;
    regularization_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp, this->mpi_communicator);

    // Set elements.
    dealii::FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    for(const auto &cell : dg->high_order_grid->dof_handler_grid.active_cell_iterators())
    {
        if (!cell->is_locally_owned()) continue;
        fe_values_vol.reinit (cell);
        cell->get_dof_indices(dofs_indices);
        cell_matrix = 0;
        
        double cell_vol = 0.0;
        for(unsigned int q=0; q<n_quad_pts; ++q)
        {
            cell_vol += fe_values_vol.JxW(q);
        }
        const double omega_k = min_cell_vol/cell_vol;

        for(unsigned int i=0; i<dofs_per_cell; ++i)
        {
            const unsigned int icomp = fe_values_vol.get_fe().system_to_component_index(i).first;
            for(unsigned int j=0; j<dofs_per_cell; ++j)
            {
                const unsigned int jcomp = fe_values_vol.get_fe().system_to_component_index(j).first;
                double val_ij = 0.0;

                if(icomp == jcomp)
                {
                    for(unsigned int q=0; q<n_quad_pts; ++q)
                    {
                        val_ij += omega_k*fe_values_vol.shape_grad(i,q)*fe_values_vol.shape_grad(j,q)*fe_values_vol.JxW(q);
                    }
                }
                cell_matrix(i,j) = val_ij;
            }
        }
        hanging_node_constraints.distribute_local_to_global(cell_matrix, dofs_indices, regularization_matrix); 
    } // cell loop ends
    regularization_matrix.compress(dealii::VectorOperation::add);
}

template <int dim, int nstate>
void AnisotropicMeshAdaptationCases<dim,nstate> :: verify_fe_values_shape_hessian(const DGBase<dim, double> &dg) const
{
    const auto mapping = (*(dg.high_order_grid->mapping_fe_field));
    dealii::hp::MappingCollection<dim> mapping_collection(mapping);
    const dealii::UpdateFlags update_flags = dealii::update_jacobian_pushed_forward_grads | dealii::update_inverse_jacobians;
    dealii::hp::FEValues<dim,dim>   fe_values_collection_volume (mapping_collection, dg.fe_collection, dg.volume_quadrature_collection, update_flags);
    
    dealii::MappingQGeneric<dim, dim> mapping2(dg.high_order_grid->get_current_fe_system().degree);
    dealii::hp::MappingCollection<dim> mapping_collection2(mapping2);
    dealii::hp::FEValues<dim,dim>   fe_values_collection_volume2 (mapping_collection2, dg.fe_collection, dg.volume_quadrature_collection, dealii::update_hessians);
    
    PHiLiP::FEValuesShapeHessian<dim> fe_values_shape_hessian;
    for(const auto &cell : dg.dof_handler.active_cell_iterators())
    {
        if(! cell->is_locally_owned()) {continue;}
        
        const unsigned int i_fele = cell->active_fe_index();
        const unsigned int i_quad = i_fele;
        const unsigned int i_mapp = 0;
        fe_values_collection_volume.reinit(cell, i_quad, i_mapp, i_fele);
        fe_values_collection_volume2.reinit(cell, i_quad, i_mapp, i_fele);
        const dealii::FEValues<dim,dim> &fe_values_volume = fe_values_collection_volume.get_present_fe_values();
        const dealii::FEValues<dim,dim> &fe_values_volume2 = fe_values_collection_volume2.get_present_fe_values();
        
        const unsigned int n_dofs_cell = fe_values_volume.dofs_per_cell;
        const unsigned int n_quad_pts = fe_values_volume.n_quadrature_points;
        for(unsigned int iquad = 0; iquad < n_quad_pts; ++iquad)
        {
            fe_values_shape_hessian.reinit(fe_values_volume, iquad);
            
            for(unsigned int idof = 0; idof < n_dofs_cell; ++idof)
            {
                const unsigned int istate = fe_values_volume.get_fe().system_to_component_index(idof).first;
                dealii::Tensor<2,dim,double> shape_hessian_dealii = fe_values_volume2.shape_hessian_component(idof, iquad, istate);
                
                dealii::Tensor<2,dim,double> shape_hessian_philip = fe_values_shape_hessian.shape_hessian_component(idof, iquad, istate, fe_values_volume.get_fe());

                dealii::Tensor<2,dim,double> shape_hessian_diff = shape_hessian_dealii;
                shape_hessian_diff -= shape_hessian_philip;

                if(shape_hessian_diff.norm() > 1.0e-8)
                {
                    std::cout<<"Dealii's FEValues shape_hessian = "<<shape_hessian_dealii<<std::endl;
                    std::cout<<"PHiLiP's FEValues shape_hessian = "<<shape_hessian_philip<<std::endl;
                    std::cout<<"Frobenius norm of diff = "<<shape_hessian_diff.norm()<<std::endl;
                    std::cout<<"Aborting.."<<std::endl<<std::flush;
                    std::abort();
                }
            } // idof
        } // iquad
    } // cell loop ends

    pcout<<"PHiLiP's physical shape hessian matches that computed by dealii."<<std::endl;
}

template <int dim, int nstate>
double AnisotropicMeshAdaptationCases<dim,nstate> :: output_vtk_files(std::shared_ptr<DGBase<dim,double>> dg, const int countval) const
{
    const int outputval = 7000 + countval;
    dg->output_results_vtk(outputval);
/*
    std::unique_ptr<DualWeightedResidualError<dim, nstate , double>> dwr_error_val = std::make_unique<DualWeightedResidualError<dim, nstate , double>>(dg);
    const double abs_dwr_error = dwr_error_val->total_dual_weighted_residual_error();
    return abs_dwr_error;
*/
    return 0;
}

template <int dim, int nstate>
double AnisotropicMeshAdaptationCases<dim,nstate> :: evaluate_functional_error(std::shared_ptr<DGBase<dim,double>> dg) const
{
/*
//    const double functional_exact = 0.1615892748498965;
    const double mach_inf = dg->all_parameters->euler_param.mach_inf;
    const double desnity_inf = 1.0;
    const double gam = 1.4;
    const double pressure_inf = 1.0/(gam * pow(mach_inf,2)); 
    const double tot_energy = pressure_inf/(gam - 1.0) + 0.5*desnity_inf;
    const double enthalpy_inf = (tot_energy + pressure_inf)/desnity_inf;

    const double domain_length = 1.4;
    double functional_exact = enthalpy_inf*domain_length;
*/
    const double functional_exact = 0.0;


    std::shared_ptr< Functional<dim, nstate, double> > functional
                                = FunctionalFactory<dim,nstate,double>::create_Functional(dg->all_parameters->functional_param, dg);
    const double functional_val = functional->evaluate_functional();
    const double error_val = abs(functional_val - functional_exact);
    return error_val;
}

template <int dim, int nstate>
double AnisotropicMeshAdaptationCases<dim,nstate> :: evaluate_abs_dwr_error(std::shared_ptr<DGBase<dim,double>> dg) const
{
    std::unique_ptr<DualWeightedResidualError<dim, nstate , double>> dwr_error_val = std::make_unique<DualWeightedResidualError<dim, nstate , double>>(dg);
    dwr_error_val->total_dual_weighted_residual_error();
    return abs(dwr_error_val->net_functional_error);
}

template <int dim, int nstate>
std::tuple<double,double,double,double> AnisotropicMeshAdaptationCases<dim,nstate> 
    :: evaluate_enthalpy_entropy_pressure_density_error(std::shared_ptr<DGBase<dim,double>> dg) const
{
if constexpr (nstate==dim+2)
{
    Physics::Euler<dim,nstate,double> euler_physics_double
        = Physics::Euler<dim, nstate, double>(
                dg->all_parameters->euler_param.ref_length,
                dg->all_parameters->euler_param.gamma_gas,
                dg->all_parameters->euler_param.mach_inf,
                dg->all_parameters->euler_param.angle_of_attack,
                dg->all_parameters->euler_param.side_slip_angle);
    
    int overintegrate = 10;
    const unsigned int poly_degree = dg->get_min_fe_degree();
    dealii::QGauss<dim> quad_extra(poly_degree+1+overintegrate);
    const dealii::Mapping<dim> &mapping = (*(dg->high_order_grid->mapping_fe_field));
    dealii::FEValues<dim,dim> fe_values_extra(mapping, dg->fe_collection[poly_degree], quad_extra, 
            dealii::update_values | dealii::update_JxW_values| dealii::update_quadrature_points);
    const unsigned int n_quad_pts = fe_values_extra.n_quadrature_points;
    const unsigned int n_dofs_cell = fe_values_extra.dofs_per_cell;
    std::array<double,nstate> soln_at_q;

    double l2error_enthalpy = 0;
    double l2error_entropy = 0;
    double l2error_pressure = 0;
    double l2error_density = 0;

    std::vector<dealii::types::global_dof_index> dofs_indices (n_dofs_cell);

    // Integrate solution error and output error
    for (const auto &cell : dg->dof_handler.active_cell_iterators()) 
    {
        if (!cell->is_locally_owned()) continue;
        fe_values_extra.reinit (cell);
        cell->get_dof_indices (dofs_indices);

        for(unsigned int iquad = 0; iquad < n_quad_pts; ++iquad)
        {
            std::fill(soln_at_q.begin(), soln_at_q.end(), 0.0);
            for(unsigned int idof = 0; idof < n_dofs_cell; ++idof)
            {
                const unsigned int istate = fe_values_extra.get_fe().system_to_component_index(idof).first;
                soln_at_q[istate] += dg->solution(dofs_indices[idof])*fe_values_extra.shape_value_component(idof,iquad,istate); 
            }
            
            const double pressure = euler_physics_double.compute_pressure(soln_at_q);
            const double enthalpy_at_q = euler_physics_double.compute_specific_enthalpy(soln_at_q,pressure);
            l2error_enthalpy += pow((enthalpy_at_q - euler_physics_double.enthalpy_inf),2) * fe_values_extra.JxW(iquad);
            l2error_entropy += pow(euler_physics_double.compute_entropy_measure(soln_at_q) - euler_physics_double.entropy_inf,2) * fe_values_extra.JxW(iquad);
            // Evaluate exact pressure and density.
            const dealii::Point<dim> point_val = fe_values_extra.quadrature_point(iquad);
            const double rval = point_val.norm();
            const double density_exact = euler_physics_double.density_inf*
               pow(1.0 + euler_physics_double.gamm1/2.0*euler_physics_double.mach_inf_sqr*(1.0-1.0/pow(rval,2)), 1.0/euler_physics_double.gamm1);
            const double pressure_exact = pow(density_exact,euler_physics_double.gam)/euler_physics_double.gam;

            l2error_pressure += pow(pressure - pressure_exact,2)*fe_values_extra.JxW(iquad);
            l2error_density += pow(soln_at_q[0] - density_exact,2)*fe_values_extra.JxW(iquad);
        }
    } // cell loop ends
    const double l2error_enthalpy_global = sqrt(dealii::Utilities::MPI::sum(l2error_enthalpy, MPI_COMM_WORLD));
    const double l2error_entropy_global = sqrt(dealii::Utilities::MPI::sum(l2error_entropy, MPI_COMM_WORLD));
    const double l2error_pressure_global = sqrt(dealii::Utilities::MPI::sum(l2error_pressure, MPI_COMM_WORLD));
    const double l2error_density_global = sqrt(dealii::Utilities::MPI::sum(l2error_density, MPI_COMM_WORLD));
    const std::tuple<double,double,double,double> enthalpy_entropy_pressure_density_error 
        (std::make_tuple(l2error_enthalpy_global, l2error_entropy_global, l2error_pressure_global, l2error_density_global));
    return enthalpy_entropy_pressure_density_error;
}
std::abort();
return std::make_tuple<double,double,double,double>(0,0,0,0);
}

template <int dim, int nstate>
int AnisotropicMeshAdaptationCases<dim, nstate> :: run_test () const
{
    int output_val = 0;
    const Parameters::AllParameters param = *(TestsBase::all_parameters);
    const bool run_fixedfraction_mesh_adaptation = param.mesh_adaptation_param.total_mesh_adaptation_cycles > 0;
    
    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(&param, parameter_handler);
    flow_solver->run();
    output_vtk_files(flow_solver->dg, output_val++);
    //return 0;
    flow_solver->use_polynomial_ramping = false;

    dealii::ConvergenceTable convergence_table;
    std::tuple<double,double,double,double> enthalpy_entropy_pressure_density_error = 
        evaluate_enthalpy_entropy_pressure_density_error(flow_solver->dg);
    convergence_table.add_value("n_cells", flow_solver->dg->triangulation->n_global_active_cells());
    convergence_table.add_value("enthalpy_error",std::get<0>(enthalpy_entropy_pressure_density_error));
    convergence_table.add_value("entropy_error",std::get<1>(enthalpy_entropy_pressure_density_error));
    convergence_table.add_value("pressure_error",std::get<2>(enthalpy_entropy_pressure_density_error));
    convergence_table.add_value("density_error",std::get<3>(enthalpy_entropy_pressure_density_error));
    
    if(run_fixedfraction_mesh_adaptation)
    {
        const unsigned int n_adaptation_cycles = param.mesh_adaptation_param.total_mesh_adaptation_cycles;

        std::unique_ptr<MeshAdaptation<dim,double>> meshadaptation =
        std::make_unique<MeshAdaptation<dim,double>>(flow_solver->dg, &(param.mesh_adaptation_param));

        for(unsigned int icycle = 0; icycle < n_adaptation_cycles; ++icycle)
        {
            meshadaptation->adapt_mesh();
            flow_solver->run();

            enthalpy_entropy_pressure_density_error = evaluate_enthalpy_entropy_pressure_density_error(flow_solver->dg);
            convergence_table.add_value("n_cells", flow_solver->dg->triangulation->n_global_active_cells());
            convergence_table.add_value("enthalpy_error",std::get<0>(enthalpy_entropy_pressure_density_error));
            convergence_table.add_value("entropy_error",std::get<1>(enthalpy_entropy_pressure_density_error));
            convergence_table.add_value("pressure_error",std::get<2>(enthalpy_entropy_pressure_density_error));
            convergence_table.add_value("density_error",std::get<3>(enthalpy_entropy_pressure_density_error));
            output_vtk_files(flow_solver->dg, output_val++);
        }
    }

    convergence_table.evaluate_convergence_rates("enthalpy_error", "n_cells", dealii::ConvergenceTable::reduction_rate_log2, dim);
    convergence_table.evaluate_convergence_rates("entropy_error", "n_cells", dealii::ConvergenceTable::reduction_rate_log2, dim);
    convergence_table.evaluate_convergence_rates("pressure_error", "n_cells", dealii::ConvergenceTable::reduction_rate_log2, dim);
    convergence_table.evaluate_convergence_rates("density_error", "n_cells", dealii::ConvergenceTable::reduction_rate_log2, dim);
    convergence_table.set_scientific("enthalpy_error", true);
    convergence_table.set_scientific("entropy_error", true);
    convergence_table.set_scientific("pressure_error", true);
    convergence_table.set_scientific("density_error", true);

    pcout << std::endl << std::endl << std::endl << std::endl;
    pcout << " ********************************************" << std::endl;
    pcout << " Convergence summary for errors" << std::endl;
    pcout << " ********************************************" << std::endl;
    if(pcout.is_active()) {convergence_table.write_text(pcout.get_stream());}
    
return 0;
}

#if PHILIP_DIM==2
template class AnisotropicMeshAdaptationCases <PHILIP_DIM, 1>;
template class AnisotropicMeshAdaptationCases <PHILIP_DIM, PHILIP_DIM + 2>;
#endif
} // namespace Tests
} // namespace PHiLiP 
    
