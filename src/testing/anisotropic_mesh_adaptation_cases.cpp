#include <stdlib.h>
#include <iostream>
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

template <int dim, int nstate>
void AnisotropicMeshAdaptationCases<dim,nstate> :: verify_fe_values_shape_hessian(const DGBase<dim, double> &dg) const
{
    const auto mapping = (*(dg.high_order_grid->mapping_fe_field));
    dealii::hp::MappingCollection<dim> mapping_collection(mapping);
    const dealii::UpdateFlags update_flags = dealii::update_jacobian_pushed_forward_grads | dealii::update_inverse_jacobians;
    dealii::hp::FEValues<dim,dim>   fe_values_collection_volume (mapping_collection, dg.fe_collection, dg.volume_quadrature_collection, update_flags);
    
    dealii::MappingQGeneric<dim, dim> mapping2(dg.high_order_grid->dof_handler_grid.get_fe().degree);
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
double AnisotropicMeshAdaptationCases<dim,nstate> :: output_vtk_files(std::shared_ptr<DGBase<dim,double>> dg) const
{
    dg->output_results_vtk(98765);

    std::unique_ptr<DualWeightedResidualError<dim, nstate , double>> dwr_error_val = std::make_unique<DualWeightedResidualError<dim, nstate , double>>(dg);
    const double abs_dwr_error = dwr_error_val->total_dual_weighted_residual_error();
    return abs_dwr_error;

    return 0;
}

template <int dim, int nstate>
double AnisotropicMeshAdaptationCases<dim,nstate> :: evaluate_functional_error(std::shared_ptr<DGBase<dim,double>> dg) const
{
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
    const double abs_dwr_error = dwr_error_val->total_dual_weighted_residual_error();
    return abs_dwr_error;
}

template <int dim, int nstate>
double AnisotropicMeshAdaptationCases<dim,nstate> :: evaluate_solution_error(std::shared_ptr<DGBase<dim,double>> dg) const
{
    int overintegrate = 10;
    const unsigned int poly_degree = dg->get_min_fe_degree();
    dealii::QGauss<dim> quad_extra(poly_degree+1+overintegrate);
    const dealii::Mapping<dim> &mapping = (*(dg->high_order_grid->mapping_fe_field));
    dealii::FEValues<dim,dim> fe_values_extra(mapping, dg->fe_collection[poly_degree], quad_extra, 
            dealii::update_values | dealii::update_JxW_values | dealii::update_quadrature_points);
    const unsigned int n_quad_pts = fe_values_extra.n_quadrature_points;
    const unsigned int n_dofs_cell = fe_values_extra.dofs_per_cell;
    std::array<double,nstate> soln_at_q;

    double l2error = 0;

    std::vector<dealii::types::global_dof_index> dofs_indices (n_dofs_cell);

    // Integrate solution error and output error
    for (const auto &cell : dg->dof_handler.active_cell_iterators()) 
    {
        if (!cell->is_locally_owned()) continue;
        fe_values_extra.reinit (cell);
        cell->get_dof_indices (dofs_indices);

        for(unsigned int iquad = 0; iquad < n_quad_pts; ++iquad)
        {
            const dealii::Point<dim> &phys_point = fe_values_extra.quadrature_point(iquad);
            const std::array<double,nstate> soln_exact_at_q = evaluate_soln_exact(phys_point);
            
            std::fill(soln_at_q.begin(), soln_at_q.end(), 0.0);
            for(unsigned int idof = 0; idof < n_dofs_cell; ++idof)
            {
                const unsigned int istate = fe_values_extra.get_fe().system_to_component_index(idof).first;
                soln_at_q[istate] += dg->solution(dofs_indices[idof])*fe_values_extra.shape_value_component(idof,iquad,istate); 
            }
/*
            std::cout<<"(x,y) = ("<<phys_point[0]<<", "<<phys_point[1]
            <<");  soln_at_q[0] = "<<soln_at_q[0]<<";  soln_exact_at_q[0] = "<<soln_exact_at_q[0]<<std::endl;
            
            std::cout<<"(x,y) = ("<<phys_point[0]<<", "<<phys_point[1]
            <<");  soln_at_q[1] = "<<soln_at_q[1]<<";  soln_exact_at_q[1] = "<<soln_exact_at_q[1]<<std::endl;
*/
            double error_norm_squared = 0;
            for(unsigned int istate = 0; istate < nstate; ++istate)
            {
                error_norm_squared += pow(soln_exact_at_q[istate] - soln_at_q[istate],2);
            }
            l2error += error_norm_squared * fe_values_extra.JxW(iquad);
        }

    } // cell loop ends
    const double l2error_global = sqrt(dealii::Utilities::MPI::sum(l2error, MPI_COMM_WORLD));


    return l2error_global;
}

template <int dim, int nstate>
std::array<double,nstate> AnisotropicMeshAdaptationCases<dim,nstate> :: evaluate_soln_exact(const dealii::Point<dim> &point) const
{
    std::array<double, nstate> soln_exact;
    const double x = point[0];
    const double y = point[1];
    const double a = -0.4;
    const double b = 0.4;
    const double c = 3.0/8.0;
    const double x_on_curve = a*pow(y,2) + b*y + c;

    // Get u0 exact
    soln_exact[0] = 0.0;
    if(x <= x_on_curve)
    {
        soln_exact[0] = 1.0;
    }

    // Get u1 exact
    const bool region_1 = (y > (1.0-x)) && (x <=x_on_curve);
    const bool region_2 = (y <= (1.0-x)) && (x <= x_on_curve);
    const bool region_3 = (y <= (1.0-x)) && (x > x_on_curve);
    const bool region_4 = (x > x_on_curve) && (y < (11.0/8.0 - x)) && (y > (1.0-x));
    const bool region_5 = y >= (11.0/8.0 - x);
    const double y_tilde = (-(b+1.0) + sqrt(pow(b+1.0,2) - 4.0*a*(c-x-y)))/(2.0*a);
    const double x_tilde = a*pow(y_tilde,2) + b*y_tilde + c;
    if(region_1)
    {
        soln_exact[1] = 0.3 + (1.0-y);
    }
    else if(region_2)
    {
        soln_exact[1] = 0.3 + x;
    }
    else if(region_3)
    {
        soln_exact[1] = 0.3 + x_tilde;
    }
    else if(region_4)
    {
        soln_exact[1] = 0.3 + (1.0 - y_tilde);
    }
    else if(region_5)
    {
        soln_exact[1] = 0.3;
    }
    else
    {
        std::cout<<"The domain is completely covered by regions 1 to 5. Shouldn't have reached here. Aborting.."<<std::endl;
        std::abort();
    }
    return soln_exact;
}

template <int dim, int nstate>
int AnisotropicMeshAdaptationCases<dim, nstate> :: run_test () const
{
    const Parameters::AllParameters param = *(TestsBase::all_parameters);
    const bool run_mesh_optimizer = true;
    const bool run_fixedfraction_mesh_adaptation = true;
    
    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(&param, parameter_handler);
    
    flow_solver->run();

    std::vector<double> solution_error_vector;
    std::vector<unsigned int> n_cycle_vector;
    std::vector<unsigned int> n_dofs_vector;

    const double solution_error_initial = evaluate_solution_error(flow_solver->dg);
    solution_error_vector.push_back(solution_error_initial);
    n_dofs_vector.push_back(flow_solver->dg->n_dofs());
    unsigned int current_cycle = 0;
    n_cycle_vector.push_back(current_cycle++);
    dealii::ConvergenceTable convergence_table;

    if(run_mesh_optimizer)
    {
        std::unique_ptr<MeshOptimizer<dim,nstate>> mesh_optimizer = std::make_unique<MeshOptimizer<dim,nstate>> (flow_solver->dg,&param, true);
        mesh_optimizer->run_full_space_optimizer();

        const double solution_error = evaluate_solution_error(flow_solver->dg);
        solution_error_vector.push_back(solution_error);
        n_dofs_vector.push_back(flow_solver->dg->n_dofs());
        n_cycle_vector.push_back(current_cycle++);
        
        convergence_table.add_value("cells", flow_solver->dg->triangulation->n_global_active_cells());
        convergence_table.add_value("soln_error_l2",solution_error);
    }

    if(run_fixedfraction_mesh_adaptation)
    {
        const unsigned int n_adaptation_cycles = param.mesh_adaptation_param.total_mesh_adaptation_cycles;

        std::unique_ptr<MeshAdaptation<dim,double>> meshadaptation =
        std::make_unique<MeshAdaptation<dim,double>>(flow_solver->dg, &(param.mesh_adaptation_param));

        for(unsigned int icycle = 0; icycle < n_adaptation_cycles; ++icycle)
        {
            meshadaptation->adapt_mesh();
            flow_solver->run();

            const double solution_error = evaluate_solution_error(flow_solver->dg);
            solution_error_vector.push_back(solution_error);
            n_dofs_vector.push_back(flow_solver->dg->n_dofs());
            n_cycle_vector.push_back(current_cycle++);
            
            convergence_table.add_value("cells", flow_solver->dg->triangulation->n_global_active_cells());
            convergence_table.add_value("soln_error_l2",solution_error);
        }
    }

    output_vtk_files(flow_solver->dg);

    // output error vals
    pcout<<"\n cycles = [";
    for(long unsigned int i=0; i<n_cycle_vector.size(); ++i)
    {
        if(i!=0) {pcout<<", ";}
        pcout<<n_cycle_vector[i];
    }
    pcout<<"];"<<std::endl;

    pcout<<"\n n_dofs = [";
    for(long unsigned int i=0; i<n_dofs_vector.size(); ++i)
    {
        if(i!=0) {pcout<<", ";}
        pcout<<n_dofs_vector[i];
    }
    pcout<<"];"<<std::endl;

    std::string functional_type = "solution_error";
    pcout<<"\n "<<functional_type<<" = [";
    for(long unsigned int i=0; i<solution_error_vector.size(); ++i)
    {
        if(i!=0) {pcout<<", ";}
        pcout<<solution_error_vector[i];
    }
    pcout<<"];"<<std::endl;

    
    convergence_table.evaluate_convergence_rates("soln_error_l2", "cells", dealii::ConvergenceTable::reduction_rate_log2, dim);
    convergence_table.set_scientific("soln_error_l2", true);

    pcout << std::endl << std::endl << std::endl << std::endl;
    pcout << " ********************************************" << std::endl;
    pcout << " Convergence summary" << std::endl;
    pcout << " ********************************************" << std::endl;
    if(pcout.is_active()) {convergence_table.write_text(pcout.get_stream());}
    return 0;
}

#if PHILIP_DIM==2
template class AnisotropicMeshAdaptationCases <PHILIP_DIM, 1>;
template class AnisotropicMeshAdaptationCases <PHILIP_DIM, 2>;
template class AnisotropicMeshAdaptationCases <PHILIP_DIM, PHILIP_DIM + 2>;
#endif
} // namespace Tests
} // namespace PHiLiP 
    
