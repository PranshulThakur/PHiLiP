#include <stdlib.h> 
#include <iostream>
#include "goal_oriented_mesh_optimization.h"
#include "flow_solver/flow_solver_factory.h"

#include "optimization/design_parameterization/inner_vol_parameterization.hpp"
#include "optimization/rol_to_dealii_vector.hpp"
#include "optimization/flow_constraints.hpp"
#include "optimization/rol_objective.hpp"
#include "functional/dual_weighted_residual_obj_func.h"
#include "optimization/full_space_step.hpp"

#include "Teuchos_GlobalMPISession.hpp"
#include "ROL_Algorithm.hpp"
#include "ROL_Reduced_Objective_SimOpt.hpp"
#include "ROL_OptimizationSolver.hpp"
#include "ROL_LineSearchStep.hpp"
#include "ROL_StatusTest.hpp"

#include <deal.II/optimization/rol/vector_adaptor.h>

namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
GoalOrientedMeshOptimization<dim, nstate> :: GoalOrientedMeshOptimization(
    const Parameters::AllParameters *const parameters_input,
    const dealii::ParameterHandler &parameter_handler_input)
    : TestsBase::TestsBase(parameters_input)
    , parameter_handler(parameter_handler_input)
{}

template <int dim, int nstate>
int GoalOrientedMeshOptimization<dim, nstate> :: run_test () const
{
    int test_error = 0;
    const Parameters::AllParameters all_param = *(TestsBase::all_parameters);
    using OptiParam = Parameters::OptimizationParam;

    std::string optimization_output_name;
    if(all_param.optimization_param.optimization_type == OptiParam::OptimizationType::full_space)
    {
        optimization_output_name = "full_space";
    }
    else if(all_param.optimization_param.optimization_type == OptiParam::OptimizationType::reduced_space)
    {
      optimization_output_name  = "reduced_space_newton";
    }
    else
    {
        pcout<<"Invalid optimization type. Aborting.."<<std::endl;
        std::abort();
    }
    
    ROL::nullstream null_stream; // outputs nothing
    std::filebuf filebuffer;
    if (this->mpi_rank == 0) filebuffer.open ("optimization_"+optimization_output_name+".log", std::ios::out);
    std::ostream std_outstream(&filebuffer);

    Teuchos::RCP<std::ostream> rcp_outstream;
    if (this->mpi_rank == 0) {rcp_outstream = ROL::makePtrFromRef(std_outstream);} // processor #0 outputs in file
    else if (this->mpi_rank == 1) {rcp_outstream = ROL::makePtrFromRef(std::cout);} // processor #1 outputs on screen
    else rcp_outstream = ROL::makePtrFromRef(null_stream);

    using DealiiVector = dealii::LinearAlgebra::distributed::Vector<double>;
    using VectorAdaptor = dealii::Rol::VectorAdaptor<DealiiVector>;
    
    AssertDimension(dim, all_param.dimension);    
    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(&all_param, parameter_handler);

    flow_solver->run(); // Solves steady state

    flow_solver->dg->output_results_vtk(99999); // Outputs initial solution and grid.
    flow_solver->dg->set_dual(flow_solver->dg->solution);

    DealiiVector initial_design_variables;
    
    std::shared_ptr<BaseParameterization<dim>> design_parameterization = 
                        std::make_shared<InnerVolParameterization<dim>>(flow_solver->dg->high_order_grid);

    design_parameterization->initialize_design_variables(initial_design_variables); // get inner volume nodes
    pcout<<"Initialized design variables."<<std::endl;


    // Copy vectors to be used by optimizer into ROL vectors.
    DealiiVector simulation_variables = flow_solver->dg->solution;
    DealiiVector design_variables = initial_design_variables;
    DealiiVector adjoint_variables = flow_solver->dg->dual; // Adjoint of optmization, not error indicator.

    const bool has_ownership = false;
    VectorAdaptor simulation_variables_rol(Teuchos::rcp(&simulation_variables, has_ownership));
    VectorAdaptor design_variables_rol(Teuchos::rcp(&design_variables, has_ownership));
    VectorAdaptor adjoint_variables_rol(Teuchos::rcp(&adjoint_variables, has_ownership));

    ROL::Ptr<ROL::Vector<double>> simulation_variables_rol_ptr = ROL::makePtr<VectorAdaptor>(simulation_variables_rol);
    ROL::Ptr<ROL::Vector<double>> design_variables_rol_ptr = ROL::makePtr<VectorAdaptor>(design_variables_rol);
    ROL::Ptr<ROL::Vector<double>> adjoint_variables_rol_ptr = ROL::makePtr<VectorAdaptor>(adjoint_variables_rol);
    const bool use_coarse_residual = false;
    DualWeightedResidualObjFunc<dim, nstate, double> dwr_obj_function(flow_solver->dg, true, false, use_coarse_residual);

    auto objective_function = ROL::makePtr<ROLObjectiveSimOpt<dim,nstate>>(dwr_obj_function, design_parameterization); 
    auto flow_constraints  = ROL::makePtr<FlowConstraints<dim>>(flow_solver->dg, design_parameterization); // Constraints of Residual = 0

    ROL::OptimizationProblem<double> optimization_problem;
    ROL::Ptr< const ROL::AlgorithmState <double> > algo_state;
    Teuchos::ParameterList parlist;

    const double timing_start = MPI_Wtime();
    auto all_variables_rol_ptr = ROL::makePtr<ROL::Vector_SimOpt<double>>(simulation_variables_rol_ptr, design_variables_rol_ptr);
    
    // ROL set optimization parameters.
    parlist.sublist("General").set("Print Verbosity", 1);
    parlist.sublist("Status Test").set("Gradient Tolerance", all_param.optimization_param.gradient_tolerance);
    parlist.sublist("Status Test").set("Iteration Limit", all_param.optimization_param.max_design_cycles);

    parlist.sublist("Step").sublist("Line Search").set("User Defined Initial Step Size",true);
    parlist.sublist("Step").sublist("Line Search").set("Initial Step Size", all_param.optimization_param.initial_step_size);
    parlist.sublist("Step").sublist("Line Search").set("Function Evaluation Limit", all_param.optimization_param.functional_evaluation_limit); // 0.5^30 ~  1e-10
    parlist.sublist("Step").sublist("Line Search").set("Accept Linesearch Minimizer",true);
    parlist.sublist("Step").sublist("Line Search").sublist("Line-Search Method").set("Type", all_param.optimization_param.line_search_method);
    parlist.sublist("Step").sublist("Line Search").sublist("Curvature Condition").set("Type", all_param.optimization_param.line_search_curvature);


    parlist.sublist("General").sublist("Secant").set("Type","Limited-Memory BFGS");
    parlist.sublist("General").sublist("Secant").set("Maximum Storage", all_param.optimization_param.max_design_cycles);

    parlist.sublist("Full Space").set("Preconditioner", all_param.optimization_param.full_space_preconditioner);
    
    if(all_param.optimization_param.optimization_type == OptiParam::OptimizationType::reduced_space)
    {
        // Reduced space Newton
        const bool storage = true;
        const bool useFDHessian = false;
        auto reduced_objective = ROL::makePtr<ROL::Reduced_Objective_SimOpt<double>>(
                                                                    objective_function,
                                                                    flow_constraints,
                                                                    simulation_variables_rol_ptr,
                                                                    design_variables_rol_ptr,
                                                                    adjoint_variables_rol_ptr,
                                                                    storage,
                                                                    useFDHessian);
        optimization_problem = ROL::OptimizationProblem<double>(reduced_objective, design_variables_rol_ptr);
        ROL::EProblem problemType = optimization_problem.getProblemType();
        std::cout << ROL::EProblemToString(problemType) << std::endl;

        parlist.sublist("Step").set("Type","Line Search");
        parlist.sublist("Step").sublist("Line Search").sublist("Descent Method").set("Type", all_param.optimization_param.reduced_space_descent_method);

        if (all_param.optimization_param.reduced_space_descent_method == "Newton-Krylov") {
            parlist.sublist("General").sublist("Secant").set("Use as Preconditioner", true);
            //parlist.sublist("General").sublist("Krylov").set("Type","Conjugate Gradients");
            parlist.sublist("General").sublist("Krylov").set("Type","GMRES");
            parlist.sublist("General").sublist("Krylov").set("Absolute Tolerance", 1.0e-8);
            parlist.sublist("General").sublist("Krylov").set("Relative Tolerance", 1.0e-4);
            parlist.sublist("General").sublist("Krylov").set("Iteration Limit", all_param.optimization_param.linear_iteration_limit);
            parlist.sublist("General").set("Inexact Hessian-Times-A-Vector",false);
        }
        
        *rcp_outstream << "Starting Reduced Space mesh optimization..."<<std::endl;
        ROL::OptimizationSolver<double> solver(optimization_problem, parlist);
        solver.solve(*rcp_outstream);
        algo_state = solver.getAlgorithmState();
    }
    else if(all_param.optimization_param.optimization_type == OptiParam::OptimizationType::full_space)
    {
        // Full space Newton
        *rcp_outstream << "Starting Full Space mesh optimization..."<<std::endl;
        auto full_space_step = ROL::makePtr<ROL::FullSpace_BirosGhattas<double>>(parlist);
        auto status_test = ROL::makePtr<ROL::StatusTest<double>>(parlist);
        const bool printHeader = true;
        ROL::Algorithm<double> algorithm(full_space_step, status_test, printHeader);
        const bool print  = true;
        algorithm.run(*all_variables_rol_ptr, 
                      *adjoint_variables_rol_ptr, 
                      *objective_function, 
                      *flow_constraints, 
                      print, 
                      *rcp_outstream);
        algo_state = algorithm.getState();
    }

    const double timing_end = MPI_Wtime();

    *rcp_outstream << "The process took "<<timing_end - timing_start << " seconds to run."<<std::endl;
    test_error += algo_state->statusFlag;

    
    filebuffer.close();
    return 0;
}

template class GoalOrientedMeshOptimization <PHILIP_DIM, 1>;

} // namespace Tests
} // namespace PHiLiP 
    
