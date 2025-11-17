#include <stdlib.h>     /* srand, rand */
#include <iostream>
#include <deal.II/grid/grid_refinement.h>
#include "physics/manufactured_solution.h"
#include "euler_naca0012.hpp"
#include "flow_solver/flow_solver_factory.h"
#include <deal.II/base/convergence_table.h>


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
/*
    Parameters::AllParameters param = *(TestsBase::all_parameters);
    const unsigned int p_start             = param.manufactured_convergence_study_param.degree_start;
    const unsigned int p_end               = param.manufactured_convergence_study_param.degree_end;
    const unsigned int n_grids_input       = param.manufactured_convergence_study_param.number_of_grids;

    for (unsigned int poly_degree = p_start; poly_degree <= p_end; ++poly_degree) {
        for (unsigned int igrid=0; igrid<n_grids_input; ++igrid) {
            param.flow_solver_param.poly_degree = poly_degree;
            param.flow_solver_param.max_poly_degree_for_adaptation = poly_degree;
            param.flow_solver_param.number_of_mesh_refinements = igrid;
            std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(&param, parameter_handler);
            flow_solver->run();
        }
    }
*/
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

            // CHANGE grid, initial_condition, the below code for other runs
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

    return 0;
}


#if PHILIP_DIM!=1
    template class EulerNACA0012 <PHILIP_DIM,PHILIP_DIM+2>;
#endif

} // Tests namespace
} // PHiLiP namespace


