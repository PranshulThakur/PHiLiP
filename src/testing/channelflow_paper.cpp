#include <stdlib.h>
#include <iostream>
#include <deal.II/grid/grid_refinement.h>
#include "physics/manufactured_solution.h"
#include "channelflow_paper.hpp"
#include "flow_solver/flow_solver_factory.h"
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/base/convergence_table.h>

namespace PHiLiP {
namespace Tests {

template <int dim, int nspecies, int nstate>
ChannelFlowPaper<dim,nspecies,nstate>::ChannelFlowPaper(const Parameters::AllParameters *const parameters_input,
                                                        const dealii::ParameterHandler &parameter_handler_input)
    :
    TestsBase::TestsBase(parameters_input)
    , parameter_handler(parameter_handler_input)
{}

template<int dim, int nspecies, int nstate>
int ChannelFlowPaper<dim,nspecies,nstate>
::run_test () const
{
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
        //=====================================================================================================================
        // Grid for testing the p+1 convergence orders of wall BC
        std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation>(
            #if PHILIP_DIM!=1
            this->mpi_communicator
            #endif
        );
        const int n_refinements = param.flow_solver_param.number_of_mesh_refinements;
        dealii::Point<dim> p1;
        dealii::Point<dim> p2;
        p1[0] = -2.0; p1[1] = -1.0;
        p2[0] = 2.0; p2[1] = 1.0;
        const double pi = 3.141592653589793238462643383279502884e+00;
        std::vector<unsigned int> repetitions(2);
        repetitions[0] = 2;
        repetitions[1] = 1;
        dealii::GridGenerator::subdivided_hyper_rectangle	( *grid, repetitions, p1, p2,true );

        for (typename dealii::parallel::distributed::Triangulation<dim>::active_cell_iterator cell = grid->begin_active(); cell != grid->end(); ++cell) {
            for (unsigned int face=0; face<dealii::GeometryInfo<dim>::faces_per_cell; ++face) {
                if (cell->face(face)->at_boundary()) {
                    unsigned int current_id = cell->face(face)->boundary_id();
                    if (current_id == 2 || current_id == 3 ) cell->face(face)->set_boundary_id (1001); // top and bottom
                }
            }
        }
        std::vector<dealii::GridTools::PeriodicFacePair<typename dealii::Triangulation<dim>::cell_iterator> > matched_pairs;
        dealii::GridTools::collect_periodic_faces(*grid,0,1,0,matched_pairs);
        grid->add_periodicity(matched_pairs);
        grid->refine_global(n_refinements);
        std::vector<double> y_expected(pow(2,n_refinements)-1);
        for(unsigned int i=0; i<y_expected.size(); ++i)
        {
            y_expected[i] = -1.0 + 2.0/(y_expected.size()+1)*(i+1.0);
        }
        for (typename dealii::parallel::distributed::Triangulation<dim>::active_cell_iterator cell = grid->begin_active(); cell != grid->end(); ++cell) {
            for(unsigned int ivertex = 0; ivertex < 4; ++ivertex)
            {
                for(unsigned int i=0; i<y_expected.size(); ++i)
                {
                    if(abs(cell->vertex(ivertex)[1] - y_expected[i])<1.0e-6)
                    {
                        cell->vertex(ivertex)[1] = y_expected[i] + 0.25*sin(pi*y_expected[i])+1.0e-5;
                    }
                }
            }
        }
    //=====================================================================================================================
    //============================================================================================================
        // Create and run flow solver
        std::unique_ptr<FlowSolver::FlowSolver<dim,nspecies,nstate>> flow_solver = FlowSolver::FlowSolverFactory<dim,nspecies,nstate>::select_flow_case(&param, parameter_handler);
        std::shared_ptr<HighOrderGrid<dim,double>> channel_grid = std::make_shared<HighOrderGrid<dim,double>>(1,grid,true,true,false);
        flow_solver->dg->set_high_order_grid(channel_grid);
        flow_solver->dg->allocate_system(false,false,false);
        // Initialize solution
        Physics::Euler<dim,nspecies,nstate,double> euler_physics_double = Physics::Euler<dim, nspecies, nstate, double>(
                &param,
                param.euler_param.ref_length,
                param.euler_param.gamma_gas,
                param.euler_param.mach_inf,
                param.euler_param.angle_of_attack,
                param.euler_param.side_slip_angle);
        auto initial_condition_function = std::make_shared<InitialConditionChannelFlowPaper<dim,nspecies,nstate>>(euler_physics_double);
        SetInitialCondition<dim,nspecies,nstate,double>::set_initial_condition(initial_condition_function, flow_solver->dg, &param);
        flow_solver->dg->solution.update_ghost_values();
        
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
        }
    //============================================================================================================
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

    return 0;
}

#if PHILIP_DIM>=2 && PHILIP_SPECIES==1
    template class ChannelFlowPaper <PHILIP_DIM, PHILIP_SPECIES,PHILIP_DIM+2>;
#endif

} // Tests namespace
} // PHiLiP namespace


