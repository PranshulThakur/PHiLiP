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
/*
    // Code to compute R, b, d and h vecs and store in file
    Parameters::AllParameters param = *(TestsBase::all_parameters);
    param.ode_solver_param.allocate_matrix_dRdW = true; 
    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(&param, parameter_handler);
    
    const double dt = param.flow_solver_param.constant_time_step;
    const double delT = 100*dt/10;
    const double T = 800*delT*10;
    const double T_extra = 100*delT*10;
    std::unique_ptr<AdjointMarch<dim, nstate, 20>> adjoint_march = std::make_unique<AdjointMarch<dim, nstate, 20>>(flow_solver->dg,10002,dt,delT,T,T_extra);  

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
/*
// Time the residual
    const double dt = param.flow_solver_param.constant_time_step;
    const double delT = 2*dt;
    const double T = 6*dt;
    const double T_extra = 4*dt;
    std::unique_ptr<AdjointMarch<dim, nstate, 12>> adjoint_march = std::make_unique<AdjointMarch<dim, nstate, 12>>(flow_solver->dg,17730,dt,delT,T,T_extra);  
    double wall_time_avg = 0;
    int countval = 0;
    {
        for(unsigned int k=0; k<flow_solver->dg->n_duals; ++k)
        {
            flow_solver->dg->duals[k] =0;
            flow_solver->dg->duals[k] =1;
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
*/

    // General code to run flow solver over the cylinder
    // CHANGE grid, initial_condition, the below code for other runs
    Parameters::AllParameters param = *(TestsBase::all_parameters);
    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(&param, parameter_handler);
    flow_solver->run();

    return 0;
}

#if PHILIP_DIM==1
    template class EulerNACA0012 <PHILIP_DIM,PHILIP_DIM>;
#endif 
#if PHILIP_DIM!=1
    template class EulerNACA0012 <PHILIP_DIM,PHILIP_DIM+2>;
#endif

} // Tests namespace
} // PHiLiP namespace


