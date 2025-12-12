#include <stdlib.h>     /* srand, rand */
#include <iostream>
#include <deal.II/grid/grid_refinement.h>
#include "physics/manufactured_solution.h"
#include "euler_naca0012.hpp"
#include "flow_solver/flow_solver_factory.h"
#include <deal.II/base/convergence_table.h>
#include "functional/adjoint_march.h"

namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
EulerNACA0012<dim,nstate>::EulerNACA0012(const Parameters::AllParameters *const parameters_input,
                                         const dealii::ParameterHandler &parameter_handler_input)
    :
    TestsBase::TestsBase(parameters_input)
    , parameter_handler(parameter_handler_input)
{}

template<int dim, int nstate>
int EulerNACA0012<dim,nstate>
::run_test () const
{

    // Code to compute R, b, d and h vecs and store in file
    Parameters::AllParameters param = *(TestsBase::all_parameters);
    param.ode_solver_param.allocate_matrix_dRdW = true; 
    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(&param, parameter_handler);
    /*
    const double dt = param.flow_solver_param.constant_time_step;
    const double delT = 500*dt;//2*dt;
    const double T = 66*delT;//6*dt;
    const double T_extra = 26*delT;//4*dt;
    std::unique_ptr<AdjointMarch<dim, nstate, 12>> adjoint_march = std::make_unique<AdjointMarch<dim, nstate, 12>>(flow_solver->dg,66177,dt,delT,T,T_extra);  

    adjoint_march->compute_R_b_d_h_vecs();
    */
    /*
{
    for(int i=3; i<8; ++i)
    {
        const double perturbation = std::pow(10.0,-i);
        std::cout<<"perturbation = "<<perturbation<<std::endl;
        std::unique_ptr<AdjointMarch<dim, nstate, 12>> adjoint_march = std::make_unique<AdjointMarch<dim, nstate, 12>>(flow_solver->dg,17730,dt,delT,T,T_extra, perturbation);  

        adjoint_march->load_solution_at_time(T+T_extra);
        dealii::LinearAlgebra::distributed::Vector<double> f_c(flow_solver->dg->solution);
        double J_c;
        adjoint_march->compute_df_dc_and_dJ_dc(f_c, J_c);
        this->pcout<<"perturbation = "<<perturbation<<"   dRdc_norm = "<<f_c.l2_norm()<<"   J_c = "<<J_c<<std::endl;
        
    }

}
*/

// Time the residual
    const double dt = param.flow_solver_param.constant_time_step;
    const double delT = 2*dt;
    const double T = 6*dt;
    const double T_extra = 4*dt;
    std::unique_ptr<AdjointMarch<dim, nstate, 12>> adjoint_march = std::make_unique<AdjointMarch<dim, nstate, 12>>(flow_solver->dg,17730,dt,delT,T,T_extra);  
    /*
    adjoint_march->load_solution_at_time(param.flow_solver_param.constant_time_step*10);
    {
        dealii::Timer timer;
        timer.start();
        flow_solver->dg->assemble_residual();
        timer.stop();
        this->pcout<<"Wall time to assemble usual residual = "<<timer.wall_time()<<std::endl;
    }
    */
    double wall_time_avg = 0;
    int countval = 0;
    {
        for(unsigned int k=0; k<flow_solver->dg->n_duals; ++k)
        {
            flow_solver->dg->duals[k] =0;
            flow_solver->dg->duals[k] =1;
            /*
            if(flow_solver->dg->duals[k].get_partitioner()->in_local_range(k))
            {
                flow_solver->dg->duals[k][k] = 1.0;
            }
            */
            flow_solver->dg->duals[k].update_ghost_values();
        }
        for (unsigned int i=0; i<10; ++i)
        {
            adjoint_march->load_solution_at_time(param.flow_solver_param.constant_time_step*(10-i));
            dealii::Timer timer;
            timer.start();
            flow_solver->dg->assemble_residual(true);
            timer.stop();
            wall_time_avg += timer.wall_time();
            countval++;
            pcout<<"Iteration "<<i<<" :"<<std::endl;
            pcout<<"Dual norm = "<<std::endl;
            for(unsigned int k=0; k<flow_solver->dg->n_duals; ++k)
            {
                pcout<<flow_solver->dg->duals[k].l2_norm()<<", ";
            }
            pcout<<std::endl;
            pcout<<"dRdW_transposed*dual norm = "<<std::endl;
            for(unsigned int k=0; k<flow_solver->dg->n_duals; ++k)
            {
                pcout<<flow_solver->dg->duals_transpose_dRdW[k].l2_norm()<<", ";
            }
            pcout<<std::endl<<"===================================================================="<<std::endl;
        }
    }
    wall_time_avg/=countval;
    this->pcout<<"Average wall time to assemble AD residual = "<<wall_time_avg<<std::endl;
/*
    // General code to run flow solver over the cylinder
    // CHANGE grid, initial_condition, the below code for other runs
    Parameters::AllParameters param = *(TestsBase::all_parameters);
    const double grid_size = 0.1/pow(2.0,param.flow_solver_param.number_of_mesh_refinements); 
    const double max_wave_speed = (1.0 + 1.0/param.euler_param.mach_inf)*1.5;
    const double dt = 1.0/(2.0*param.flow_solver_param.poly_degree+1.0) * grid_size/max_wave_speed; // From N. Chalmers, L. Krivodonova, A robust CFL condition for the discontinuous Galerkin method on triangular meshes, JCP 2020.
    pcout<<"Time step limit due to the grid size and wave speed: dt = "<<dt<<std::endl;

    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(&param, parameter_handler);
    flow_solver->run();
*/





/*
//=============================================================================================================================
    // Testing lid driven cavity
    Parameters::AllParameters param = *(TestsBase::all_parameters);
    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(&param, parameter_handler);
    const double finalTime = param.flow_solver_param.final_time;

    pcout << " number dofs " << flow_solver->dg->dof_handler.n_dofs()<<std::endl;
    pcout << "preparing to advance solution in time" << std::endl;

    flow_solver->ode_solver->current_iteration = 0;
    flow_solver->ode_solver->allocate_ode_system();

    //loop over time
    while(flow_solver->ode_solver->current_time < finalTime){
        //get timestep
        const double time_step =  param.flow_solver_param.constant_time_step;
        if(flow_solver->ode_solver->current_iteration%param.ode_solver_param.print_iteration_modulo==0)
            pcout<<"time step "<<time_step<<" current time "<<flow_solver->ode_solver->current_time<<std::endl;
        //take the minimum timestep from all processors.
        const double dt = time_step;
        //integrate in time
        flow_solver->ode_solver->step_in_time(dt, false);
        flow_solver->ode_solver->current_iteration += 1;

        //get the change in entropy
        const std::array<double,2> current_change_entropy = compute_change_in_entropy(flow_solver->dg, param.flow_solver_param.poly_degree);
        const double current_change_entropy_mpi = dealii::Utilities::MPI::sum(current_change_entropy[0], mpi_communicator);
        pcout << "M plus K norm Change in Entropy at time " << flow_solver->ode_solver->current_time << " is " << current_change_entropy_mpi<< std::endl;
        //check if change in entropy is conserved at machine precision
        if(current_change_entropy[0] > 1e-12 && (flow_solver->dg->all_parameters->two_point_num_flux_type == Parameters::AllParameters::TwoPointNumericalFlux::IR || flow_solver->dg->all_parameters->two_point_num_flux_type == Parameters::AllParameters::TwoPointNumericalFlux::CH || flow_solver->dg->all_parameters->two_point_num_flux_type == Parameters::AllParameters::TwoPointNumericalFlux::Ra)){
          pcout << " Change in entropy was not monotonically conserved." << std::endl;
          return 1;
        }
    }
    flow_solver->dg->output_results_vtk(7000);
*/
/*
//=========================================================================================================================
    // Testing p+1 convergence orders of wall BC
    const unsigned int n_grids_input       =  (*TestsBase::all_parameters).manufactured_convergence_study_param.number_of_grids;
    dealii::ConvergenceTable convergence_table;
    const unsigned int p_start             = (*TestsBase::all_parameters).manufactured_convergence_study_param.degree_start;
    const unsigned int p_end               = (*TestsBase::all_parameters).manufactured_convergence_study_param.degree_end;
    std::vector<dealii::ConvergenceTable> convergence_table_vector;

    for(unsigned int poly_degree = p_start; poly_degree<=p_end; ++poly_degree)
    {
        for(unsigned int igrid = 0; igrid<n_grids_input; ++igrid)
        {
            Parameters::AllParameters param = *(TestsBase::all_parameters);
            param.flow_solver_param.number_of_mesh_refinements += igrid; 
            param.flow_solver_param.constant_time_step /= pow(2.0,2*igrid); 
            param.flow_solver_param.poly_degree = poly_degree;
            param.flow_solver_param.max_poly_degree_for_adaptation = poly_degree;
        
            std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(&param, parameter_handler);
            flow_solver->dg->check_same_coords_strongdg = false;
            flow_solver->run();

            // Compute error at the wall
            int overintegrate = 10;
            dealii::QGauss<dim-1> facequad_extra(flow_solver->dg->max_degree+1+overintegrate);
            const dealii::Mapping<dim> &mapping = (*(flow_solver->dg->high_order_grid->mapping_fe_field));
            std::array<double,nstate> soln_at_q;
            dealii::FEFaceValues<dim,dim>    fe_face_values (mapping, flow_solver->dg->fe_collection[poly_degree], facequad_extra, dealii::update_values | dealii::update_JxW_values);
            const unsigned int n_face_quad_pts = fe_face_values.n_quadrature_points;
            std::vector<dealii::types::global_dof_index> dofs_indices (fe_face_values.dofs_per_cell);
            double l2error = 0.0;
            for (auto cell = flow_solver->dg->dof_handler.begin_active(); cell!=flow_solver->dg->dof_handler.end(); ++cell) {

                if (!cell->is_locally_owned()) continue;
                for (unsigned int iface=0; iface<dealii::GeometryInfo<dim>::faces_per_cell; ++iface) {
                    if (cell->face(iface)->at_boundary()) {
                        if(cell->face(iface)->boundary_id()==1001)
                        {
                            fe_face_values.reinit (cell,iface);
                            cell->get_dof_indices (dofs_indices);

                            for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {

                                std::fill(soln_at_q.begin(), soln_at_q.end(), 0);
                                for (unsigned int idof=0; idof<fe_face_values.dofs_per_cell; ++idof) {
                                    const unsigned int istate = fe_face_values.get_fe().system_to_component_index(idof).first;
                                    soln_at_q[istate] += flow_solver->dg->solution[dofs_indices[idof]] * fe_face_values.shape_value_component(idof, iquad, istate);
                                }

                                const double u1 =  soln_at_q[1]/ soln_at_q[0];
                                const double u2 =  soln_at_q[2]/ soln_at_q[0];
                                l2error += (u1*u1+u2*u2)*fe_face_values.JxW(iquad);

                            }
                        }
                    }
                }
            }
                const double l2error_mpi_sum = std::sqrt(dealii::Utilities::MPI::sum(l2error, mpi_communicator));
                const unsigned int n_dofs = flow_solver->dg->dof_handler.n_dofs();
                const unsigned int n_global_active_cells = flow_solver->dg->triangulation->n_global_active_cells();
                double dx = 1.0/pow(n_dofs,(1.0/dim));
                pcout<<"l2 error = "<<l2error_mpi_sum<<std::endl;
                convergence_table.add_value("p", poly_degree);
                convergence_table.add_value("cells", n_global_active_cells);
                convergence_table.add_value("DoFs", n_dofs);
                convergence_table.add_value("dx", dx);
                convergence_table.add_value("L2error", l2error_mpi_sum);
        }
        pcout << " ********************************************" << std::endl
             << " Convergence rates for p = " << poly_degree << std::endl
             << " ********************************************" << std::endl;
        convergence_table.evaluate_convergence_rates("L2error", "cells", dealii::ConvergenceTable::reduction_rate_log2, dim);
        convergence_table.set_scientific("dx", true);
        convergence_table.set_scientific("L2error", true);
        //convergence_table.set_scientific("L2_error", true);
        if (pcout.is_active()) convergence_table.write_text(pcout.get_stream());
        
        convergence_table_vector.push_back(convergence_table);
    }
    pcout << std::endl << std::endl << std::endl << std::endl;
    pcout << " ********************************************" << std::endl;
    pcout << " Convergence summary" << std::endl;
    pcout << " ********************************************" << std::endl;
    for (auto conv = convergence_table_vector.begin(); conv!=convergence_table_vector.end(); conv++) {
        if (pcout.is_active()) conv->write_text(pcout.get_stream());
        pcout << " ********************************************" << std::endl;
    }
//=========================================================================================================================
*/
    return 0;
}

template<int dim, int nstate>
std::array<double,2> EulerNACA0012<dim, nstate>::compute_change_in_entropy(const std::shared_ptr < DGBase<dim, double> > &dg, unsigned int poly_degree) const
{
    const unsigned int n_dofs_cell = dg->fe_collection[poly_degree].dofs_per_cell;
    const unsigned int n_quad_pts = dg->volume_quadrature_collection[poly_degree].size();
    const unsigned int n_shape_fns = n_dofs_cell / nstate;
    //We have to project the vector of entropy variables because the mass matrix has an interpolation from solution nodes built into it.
    OPERATOR::vol_projection_operator<dim,2*dim> vol_projection(1, poly_degree, dg->max_grid_degree);
    vol_projection.build_1D_volume_operator(dg->oneD_fe_collection_1state[poly_degree], dg->oneD_quadrature_collection[poly_degree]);

    OPERATOR::basis_functions<dim,2*dim> soln_basis(1, poly_degree, dg->max_grid_degree);
    soln_basis.build_1D_volume_operator(dg->oneD_fe_collection_1state[poly_degree], dg->oneD_quadrature_collection[poly_degree]);

    dealii::LinearAlgebra::distributed::Vector<double> entropy_var_hat_global(dg->right_hand_side);
    dealii::LinearAlgebra::distributed::Vector<double> energy_var_hat_global(dg->right_hand_side);
    std::vector<dealii::types::global_dof_index> dofs_indices (n_dofs_cell);

    std::shared_ptr < Physics::Euler<dim, nstate, double > > euler_double  = std::dynamic_pointer_cast<Physics::Euler<dim,dim+2,double>>(PHiLiP::Physics::PhysicsFactory<dim,nstate,double>::create_Physics(dg->all_parameters));

    for (auto cell = dg->dof_handler.begin_active(); cell!=dg->dof_handler.end(); ++cell) {
        if (!cell->is_locally_owned()) continue;
        cell->get_dof_indices (dofs_indices);

        //get solution modal coeff
        std::array<std::vector<double>,nstate> soln_coeff;
        for(unsigned int idof=0; idof<n_dofs_cell; idof++){
            const unsigned int istate = dg->fe_collection[poly_degree].system_to_component_index(idof).first;
            const unsigned int ishape = dg->fe_collection[poly_degree].system_to_component_index(idof).second;
            if(ishape == 0)
                soln_coeff[istate].resize(n_shape_fns);
            soln_coeff[istate][ishape] = dg->solution(dofs_indices[idof]);
        }

        //interpolate solution to quadrature points
        std::array<std::vector<double>,nstate> soln_at_q;
        for(int istate=0; istate<nstate; istate++){
            soln_at_q[istate].resize(n_quad_pts);
            soln_basis.matrix_vector_mult_1D(soln_coeff[istate], soln_at_q[istate],
                                             soln_basis.oneD_vol_operator);
        }
        //compute entropy and kinetic energy "entropy" variables at quad points
        std::array<std::vector<double>,nstate> entropy_var_at_q;
        std::array<std::vector<double>,nstate> energy_var_at_q;
        for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
            std::array<double,nstate> soln_state;
            for(int istate=0; istate<nstate; istate++){
                soln_state[istate] = soln_at_q[istate][iquad];
            }
            std::array<double,nstate> entropy_var_state = euler_double->compute_entropy_variables(soln_state);
            std::array<double,nstate> kin_energy_state = euler_double->compute_kinetic_energy_variables(soln_state);
            for(int istate=0; istate<nstate; istate++){
                if(iquad==0){
                    entropy_var_at_q[istate].resize(n_quad_pts);
                    energy_var_at_q[istate].resize(n_quad_pts);
                }
                energy_var_at_q[istate][iquad] = kin_energy_state[istate];
                entropy_var_at_q[istate][iquad] = entropy_var_state[istate];
            }
        }
        //project the enrtopy and KE var to modal coefficients
        //then write it into a global vector
        for(int istate=0; istate<nstate; istate++){
            //Projected vector of entropy variables.
            std::vector<double> entropy_var_hat(n_shape_fns);
            vol_projection.matrix_vector_mult_1D(entropy_var_at_q[istate], entropy_var_hat,
                                                 vol_projection.oneD_vol_operator);
            std::vector<double> energy_var_hat(n_shape_fns);
            vol_projection.matrix_vector_mult_1D(energy_var_at_q[istate], energy_var_hat,
                                                 vol_projection.oneD_vol_operator);

            for(unsigned int ishape=0; ishape<n_shape_fns; ishape++){
                const unsigned int idof = istate * n_shape_fns + ishape;
                entropy_var_hat_global[dofs_indices[idof]] = entropy_var_hat[ishape];
                energy_var_hat_global[dofs_indices[idof]] = energy_var_hat[ishape];
            }
        }
    }
    entropy_var_hat_global.update_ghost_values();;

    //evaluate the change in entropy and change in KE
    dg->assemble_residual();
    std::array<double,2> change_entropy_and_energy;
    change_entropy_and_energy[0] = entropy_var_hat_global * dg->right_hand_side;
    change_entropy_and_energy[1] = energy_var_hat_global * dg->right_hand_side;

    //compute changes in only one residual
    dg->compute_only_convective_residual = true;
    dg->compute_only_dissipative_residual = false;
    dg->assemble_residual();
    const double entropy_var_times_res_conv = dealii::Utilities::MPI::sum((entropy_var_hat_global*dg->right_hand_side),mpi_communicator);
    
    dg->compute_only_convective_residual = false;
    dg->compute_only_dissipative_residual = true;
    dg->assemble_residual();
    const double entropy_var_times_res_dissip = dealii::Utilities::MPI::sum((entropy_var_hat_global*dg->right_hand_side),mpi_communicator);

    if(abs(entropy_var_times_res_conv) > 1.0e-12) 
    {
        std::cout<<"Entropy var times convective residual is non-zero. Aborting.."<<std::endl;
        std::cout<<"entropy_var_times_res_conv = "<<entropy_var_times_res_conv<<std::endl;
        std::abort();
    }
    if(entropy_var_times_res_dissip > 1.0e-12) 
    {
        std::cout<<"Entropy var times dissipative residual is positive. Aborting.."<<std::endl;
        std::cout<<"entropy_var_times_res_dissip = "<<entropy_var_times_res_dissip<<std::endl;
        std::abort();
    }
    


    // reset to false
    dg->compute_only_convective_residual = false;
    dg->compute_only_dissipative_residual = false;
    return change_entropy_and_energy;
}


#if PHILIP_DIM!=1
    template class EulerNACA0012 <PHILIP_DIM,PHILIP_DIM+2>;
#endif

} // Tests namespace
} // PHiLiP namespace


