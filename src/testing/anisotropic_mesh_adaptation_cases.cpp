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
#include <deal.II/base/timer.h>

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

template<int dim, int nstate>
void AnisotropicMeshAdaptationCases<dim,nstate>::increase_grid_degree_and_interpolate_solution(std::shared_ptr<DGBase<dim,double>> dg) const
{
    const unsigned int grid_degree_updated = 2;
    dg->high_order_grid->set_q_degree(grid_degree_updated, true);

    const unsigned int poly_degree_updated = dg->all_parameters->flow_solver_param.max_poly_degree_for_adaptation - 1;
    dg->set_p_degree_and_interpolate_solution(poly_degree_updated);
}

template <int dim, int nstate>
void AnisotropicMeshAdaptationCases<dim,nstate> :: move_nodes_to_shock(std::shared_ptr<DGBase<dim,double>> dg) const
{
    const dealii::IndexSet &volume_range = dg->high_order_grid->volume_nodes.get_partitioner()->locally_owned_range();
    const unsigned int n_vol_nodes = dg->high_order_grid->volume_nodes.size();

    for (unsigned int i = 0; i<n_vol_nodes; ++i)
    {
        if(! volume_range.is_element(i)) {continue;}

        if(i % dim == 0)
        {
            const double x = dg->high_order_grid->volume_nodes(i);

            const double y = dg->high_order_grid->volume_nodes(i+1);

            const double x_curve = -0.4*pow(y,2) + 0.4*y + 0.375;

            const double a1 = 0.1875, a2 = 28125, a3 = 0.4875;

            if(x == 0.375)
            {
                dg->high_order_grid->volume_nodes(i) = x_curve;
            }
            else if (x == a3)
            {
                dg->high_order_grid->volume_nodes(i) = (0.6 + x_curve)/2.0;
            }
            else if (x==a2)
            {
                dg->high_order_grid->volume_nodes(i) = (a1 + x_curve)/2.0;
            }
        }
    } // ivol for loop ends
    dg->high_order_grid->volume_nodes.update_ghost_values();
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

    const double functional_exact = 0.1512447195285363; 
    std::shared_ptr< Functional<dim, nstate, double> > functional
                                = FunctionalFactory<dim,nstate,double>::create_Functional(dg->all_parameters->functional_param, dg);
    functional->overintegrate_functional = true;
    const double functional_val = functional->evaluate_functional();
    const double error_val = abs(functional_exact - functional_val);
    return error_val;

/*
    int overintegrate = 500;
    const unsigned int poly_degree = dg->get_min_fe_degree();
    dealii::QGauss<dim-1> face_quad_extra(overintegrate);
    const dealii::Mapping<dim> &mapping = (*(dg->high_order_grid->mapping_fe_field));
    dealii::FEFaceValues<dim,dim> fe_face_values_extra(mapping, dg->fe_collection[poly_degree], face_quad_extra, 
            dealii::update_values | dealii::update_JxW_values | dealii::update_quadrature_points);
    const unsigned int n_face_quad_pts = fe_face_values_extra.n_quadrature_points;
    
    double functional_local = 0.0;

    // Integrate solution error and output error
    for (const auto &cell : dg->dof_handler.active_cell_iterators()) 
    {
        if (!cell->is_locally_owned()) continue;

        for(unsigned int iface = 0; iface < dealii::GeometryInfo<dim>::faces_per_cell; ++iface)
        {
            auto face = cell->face(iface);
            if(face->at_boundary())
            {
                const unsigned int boundary_id = face->boundary_id();
                if(boundary_id == 1)
                {
                    fe_face_values_extra.reinit (cell, iface);

                    for(unsigned int iquad = 0; iquad < n_face_quad_pts; ++iquad)
                    {
                        const dealii::Point<dim> &phys_point = fe_face_values_extra.quadrature_point(iquad);
                        const std::array<double,nstate> soln_exact_at_q = evaluate_soln_exact(phys_point);
                        if( abs(phys_point[0] - 1.0) > 1.0e-15)
                        {
                            std::cout<<"Not at right boundary. Aborting..."<<std::endl;
                            std::cout<<"x = "<<std::setprecision(16)<<phys_point[0]<<std::endl;
                            std::cout<<"error in x = "<<std::setprecision(16)<<abs(1.0-phys_point[0])<<std::endl;
                            std::abort();
                        }
                        
                        //=======================================================================
                        // Evaluate continuous logistic heaviside.
                        const double y = phys_point[1];
                        const double yc = 0.05;
                        const double heaviside_min = 1.0e-5; // heaviside at y=0
                        const double logterm = log(1/heaviside_min - 1.0);
                        const double epsilon_val = yc/logterm;
                        const double heaviside_at_y = 1.0/(1.0 + exp(-(y-yc)/epsilon_val));
                        //=======================================================================
                        const double integrand = heaviside_at_y * pow(soln_exact_at_q[1],2);
                        functional_local += integrand*fe_face_values_extra.JxW(iquad);                        
                    } // iquad ends
                    
                } // if (boundary_id==1) ends
            } // if (face->at_boundary()) ends
        } // face loop ends
    } // cell loop ends

    const double functional_global = dealii::Utilities::MPI::sum(functional_local, MPI_COMM_WORLD);
    return functional_global;
*/ 
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
    const double u1_bc = 0.3;
    if(region_1)
    {
        soln_exact[1] = u1_bc + (1.0-y);
    }
    else if(region_2)
    {
        soln_exact[1] = u1_bc + x;
    }
    else if(region_3)
    {
        soln_exact[1] = u1_bc + x_tilde;
    }
    else if(region_4)
    {
        soln_exact[1] = u1_bc + (1.0 - y_tilde);
    }
    else if(region_5)
    {
        soln_exact[1] = u1_bc;
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
    const bool run_fixedfraction_mesh_adaptation = false;
    
    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(&param, parameter_handler);
    flow_solver->dg->use_smooth_upwind_flux = param.mesh_adaptation_param.use_goal_oriented_mesh_adaptation ? false : true; 
    dealii::Timer timer(this->mpi_communicator, true);    
    flow_solver->run();

    std::vector<double> functional_error_vector;
    std::vector<unsigned int> n_cycle_vector;
    std::vector<unsigned int> n_dofs_vector;
    std::vector<double> elapsed_time_vector;

    const double functional_error_initial = evaluate_functional_error(flow_solver->dg);
    timer.stop();
    functional_error_vector.push_back(functional_error_initial);
    n_dofs_vector.push_back(flow_solver->dg->n_dofs());
    elapsed_time_vector.push_back(timer.wall_time());
    unsigned int current_cycle = 0;
    n_cycle_vector.push_back(current_cycle++);
    dealii::ConvergenceTable convergence_table;
    timer.reset();
    output_vtk_files(flow_solver->dg);

    if(run_mesh_optimizer)
    {
    
        double mesh_weight = param.optimization_param.mesh_weight_factor;
        Parameters::AllParameters param2 = *(TestsBase::all_parameters);
        timer.start();
        for(unsigned int i=0; i<2; ++i)
        {
            std::unique_ptr<MeshOptimizer<dim,nstate>> mesh_optimizer = 
                                                std::make_unique<MeshOptimizer<dim,nstate>> (flow_solver->dg, &param2, mesh_weight, true);
            dealii::TrilinosWrappers::SparseMatrix regularization_matrix_poisson;
            evaluate_regularization_matrix(regularization_matrix_poisson, flow_solver->dg);
            mesh_optimizer->run_full_space_optimizer(regularization_matrix_poisson);
            increase_grid_degree_and_interpolate_solution(flow_solver->dg); 
            param2.optimization_param.max_design_cycles = 70;
        }
        timer.stop();
    
        const double functional_error = evaluate_functional_error(flow_solver->dg);
        functional_error_vector.push_back(functional_error);
        elapsed_time_vector.push_back(timer.wall_time());
        n_dofs_vector.push_back(flow_solver->dg->n_dofs());
        n_cycle_vector.push_back(current_cycle++);
        pcout<<"Current cycle = "<<(current_cycle-1)<<";  Functional error = "<<functional_error<<std::endl;
        
        convergence_table.add_value("cells", flow_solver->dg->triangulation->n_global_active_cells());
        convergence_table.add_value("functional_error",functional_error);
        timer.reset();
    }

    if(run_fixedfraction_mesh_adaptation)
    {
        const unsigned int n_adaptation_cycles = param.mesh_adaptation_param.total_mesh_adaptation_cycles;

        std::unique_ptr<MeshAdaptation<dim,double>> meshadaptation =
        std::make_unique<MeshAdaptation<dim,double>>(flow_solver->dg, &(param.mesh_adaptation_param));

        for(unsigned int icycle = 0; icycle < n_adaptation_cycles; ++icycle)
        {
            timer.start();
            meshadaptation->adapt_mesh();
            flow_solver->run();
            timer.stop();

            const double functional_error = evaluate_functional_error(flow_solver->dg);
            functional_error_vector.push_back(functional_error);
            elapsed_time_vector.push_back(timer.wall_time());
            n_dofs_vector.push_back(flow_solver->dg->n_dofs());
            n_cycle_vector.push_back(current_cycle++);
            pcout<<"Current cycle = "<<(current_cycle-1)<<";  Functional error = "<<functional_error<<std::endl;
            
            convergence_table.add_value("cells", flow_solver->dg->triangulation->n_global_active_cells());
            convergence_table.add_value("functional_error",functional_error);
            timer.reset();
        }
    }

    output_vtk_files(flow_solver->dg);

    // output error vals
    pcout<<"\n cycles = [";
    for(long unsigned int i=0; i<n_cycle_vector.size(); ++i)
    {
        pcout<<n_cycle_vector[i];
        if(i!=(n_cycle_vector.size()-1)) {pcout<<", ";}
    }
    pcout<<"];"<<std::endl;

    pcout<<"\n n_dofs = [";
    for(long unsigned int i=0; i<n_dofs_vector.size(); ++i)
    {
        pcout<<n_dofs_vector[i];
        if(i!=(n_cycle_vector.size()-1)) {pcout<<", ";}
    }
    pcout<<"];"<<std::endl;

    std::string functional_type = "functional_error";
    pcout<<"\n "<<functional_type<<" = [";
    for(long unsigned int i=0; i<functional_error_vector.size(); ++i)
    {
        pcout<<functional_error_vector[i];
        if(i!=(n_cycle_vector.size()-1)) {pcout<<", ";}
    }
    pcout<<"];"<<std::endl;

    pcout<<"\n elapsed_time = [";
    for(long unsigned int i=0; i<elapsed_time_vector.size(); ++i)
    {
        pcout<<elapsed_time_vector[i];
        if(i!=(n_cycle_vector.size()-1)) {pcout<<", ";}
    }
    pcout<<"];"<<std::endl;

    
    convergence_table.evaluate_convergence_rates("functional_error", "cells", dealii::ConvergenceTable::reduction_rate_log2, dim);
    convergence_table.set_scientific("functional_error", true);

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
    
