#include <stdlib.h> 
#include <iostream>
#include "goal_oriented_mesh_optimization.h"
#include "flow_solver/flow_solver_factory.h"
#include "mesh/mesh_error_estimate.h"

#include "optimization/design_parameterization/inner_vol_parameterization.hpp"
#include "optimization/design_parameterization/unit_vector_parameterization.hpp"
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

//====================================================================================================================================
    std::unique_ptr<DualWeightedResidualError<dim, nstate , double>> dwr_error_val = std::make_unique<DualWeightedResidualError<dim, nstate , double>>(flow_solver->dg);
    const double abs_error_initial = dwr_error_val->total_dual_weighted_residual_error();
    const double actual_error_initial = dwr_error_val->net_functional_error;
    //================== Evaluate exact functional error=============================================
    std::shared_ptr< Functional<dim, nstate, double> > functional = FunctionalFactory<dim,nstate,double>::create_Functional(flow_solver->dg->all_parameters->functional_param, flow_solver->dg);
    flow_solver->dg->change_cells_fe_degree_by_deltadegree_and_interpolate_solution(1);
    flow_solver->dg->assemble_residual();
    const double fine_residual_norm_initial = sqrt(flow_solver->dg->right_hand_side*flow_solver->dg->right_hand_side);
    double functional_val_coarse = functional->evaluate_functional();
    flow_solver->run();
    double functional_val_fine = functional->evaluate_functional();
    flow_solver->dg->change_cells_fe_degree_by_deltadegree_and_interpolate_solution(-1);
    const double exact_functional_error_initial = functional_val_fine - functional_val_coarse;
//====================================================================================================================================

    if(all_param.optimization_param.use_fine_solution == false)
    {
        pcout<<"Using coarse solution as the initial guess."<<std::endl;
        flow_solver->run(); // Solves steady state
    }
    else
    {
        pcout<<"Using fine solution interpolated to the coarse mesh as initial guess."<<std::endl;
    }

    flow_solver->dg->set_dual(flow_solver->dg->solution);

    DealiiVector initial_design_variables;
    
    std::shared_ptr<BaseParameterization<dim>> design_parameterization = 
                        //std::make_shared<InnerVolParameterization<dim>>(flow_solver->dg->high_order_grid);
                        std::make_shared<UnitVectorParameterization<dim>>(flow_solver->dg->high_order_grid);

    design_parameterization->initialize_design_variables(initial_design_variables); // get inner volume nodes
    pcout<<"Initialized design variables."<<std::endl;
    const double initial_control_var_norm = design_parameterization->control_var_norm();


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
    DualWeightedResidualObjFunc<dim, nstate, double> dwr_obj_function(flow_solver->dg, true, false, all_param.optimization_param.use_coarse_residual);

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

//============================== Check objective function and constraint gradients/Hessians ==================================================================
    const bool check_derivatives = false;
    if(check_derivatives)
    {
        ROL::Ptr<ROL::Vector<double>> d_sim = simulation_variables_rol_ptr->clone(); d_sim->randomize(-1.0, 1.0);
        ROL::Ptr<ROL::Vector<double>> d_control = design_variables_rol_ptr->clone(); d_control->randomize(-1.0, 1.0);
        ROL::Vector_SimOpt<double> direction(d_sim, d_control);

        *rcp_outstream << "objective_function->checkGradient_1..." << std::endl;
        objective_function->checkGradient_1(*simulation_variables_rol_ptr, *design_variables_rol_ptr, *d_sim, true, *rcp_outstream);
        *rcp_outstream << "objective_function->checkGradient_2..." << std::endl;
        objective_function->checkGradient_2(*simulation_variables_rol_ptr, *design_variables_rol_ptr, *d_control, true, *rcp_outstream);
        *rcp_outstream << "objective_function->checkHessVec_11..." << std::endl;
        objective_function->checkHessVec_11(*simulation_variables_rol_ptr, *design_variables_rol_ptr, *d_sim, true, *rcp_outstream);
        *rcp_outstream << "objective_function->checkHessVec_12..." << std::endl;
        objective_function->checkHessVec_12(*simulation_variables_rol_ptr, *design_variables_rol_ptr, *d_control, true, *rcp_outstream);
        *rcp_outstream << "objective_function->checkHessVec_21..." << std::endl;
        objective_function->checkHessVec_21(*simulation_variables_rol_ptr, *design_variables_rol_ptr, *d_sim, true, *rcp_outstream);
        *rcp_outstream << "objective_function->checkHessVec_22..." << std::endl;
        objective_function->checkHessVec_22(*simulation_variables_rol_ptr, *design_variables_rol_ptr, *d_control, true, *rcp_outstream);

        *rcp_outstream << "objective_function->checkGradient..." << std::endl;
        objective_function->checkGradient(*all_variables_rol_ptr, direction, true, *rcp_outstream);
        *rcp_outstream << "objective_function->checkHessVec..." << std::endl;
        objective_function->checkHessVec(*all_variables_rol_ptr, direction, true, *rcp_outstream);

        // Some additional checks
        *rcp_outstream << "flow_constraints->checkSolve..." << std::endl;
        flow_constraints->checkSolve(*simulation_variables_rol_ptr, *design_variables_rol_ptr, *adjoint_variables_rol_ptr, true, *rcp_outstream);
        *rcp_outstream << "flow_constraints->checkApplyAdjointHessian_11..." << std::endl;
        flow_constraints->checkApplyAdjointHessian_11(*simulation_variables_rol_ptr, *design_variables_rol_ptr, *adjoint_variables_rol_ptr, *d_sim, *adjoint_variables_rol_ptr, true, *rcp_outstream);
        *rcp_outstream << "flow_constraints->checkApplyAdjointHessian_12..." << std::endl;
        flow_constraints->checkApplyAdjointHessian_12(*simulation_variables_rol_ptr, *design_variables_rol_ptr, *adjoint_variables_rol_ptr, *d_sim, *design_variables_rol_ptr, true, *rcp_outstream);
        *rcp_outstream << "flow_constraints->checkApplyAdjointHessian_21..." << std::endl;
        flow_constraints->checkApplyAdjointHessian_21(*simulation_variables_rol_ptr, *design_variables_rol_ptr, *adjoint_variables_rol_ptr, *d_control, *adjoint_variables_rol_ptr, true, *rcp_outstream);
        *rcp_outstream << "flow_constraints->checkApplyAdjointHessian_22..." << std::endl;
        flow_constraints->checkApplyAdjointHessian_22(*simulation_variables_rol_ptr, *design_variables_rol_ptr, *adjoint_variables_rol_ptr, *d_control, *design_variables_rol_ptr, true, *rcp_outstream);

        *rcp_outstream << "flow_constraints->checkApplyJacobian_1..." << std::endl;
        flow_constraints->checkApplyJacobian_1(*simulation_variables_rol_ptr, *design_variables_rol_ptr, *d_sim, *simulation_variables_rol_ptr, true, *rcp_outstream);
        *rcp_outstream << "flow_constraints->checkApplyJacobian_2..." << std::endl;
        flow_constraints->checkApplyJacobian_2(*simulation_variables_rol_ptr, *design_variables_rol_ptr, *d_control, *simulation_variables_rol_ptr, true, *rcp_outstream);
        *rcp_outstream << "flow_constraints->checkApplyJacobian..." << std::endl;
        flow_constraints->checkApplyJacobian(*all_variables_rol_ptr, direction, *simulation_variables_rol_ptr, true, *rcp_outstream);
        *rcp_outstream << "flow_constraints->checkApplyAdjointHessian..." << std::endl;
        flow_constraints->checkApplyAdjointHessian(*all_variables_rol_ptr, *d_sim, direction, *all_variables_rol_ptr, true, *rcp_outstream);
        *rcp_outstream << "flow_constraints->checkAdjointConsistencyJacobian..." << std::endl;
        flow_constraints->checkAdjointConsistencyJacobian(*d_sim, direction, *all_variables_rol_ptr, true, *rcp_outstream);
        *rcp_outstream << "flow_constraints->checkInverseJacobian_1..." << std::endl;
        flow_constraints->checkInverseJacobian_1(*simulation_variables_rol_ptr, *simulation_variables_rol_ptr, *simulation_variables_rol_ptr, *design_variables_rol_ptr, true, *rcp_outstream);
        *rcp_outstream << "flow_constraints->checkInverseAdjointJacobian_1..." << std::endl;
        flow_constraints->checkInverseAdjointJacobian_1(*simulation_variables_rol_ptr, *simulation_variables_rol_ptr, *simulation_variables_rol_ptr, *design_variables_rol_ptr, true, *rcp_outstream);
        auto robj = ROL::makePtr<ROL::Reduced_Objective_SimOpt<double>>(
                                                                    objective_function,
                                                                    flow_constraints,
                                                                    simulation_variables_rol_ptr,
                                                                    design_variables_rol_ptr,
                                                                    adjoint_variables_rol_ptr,
                                                                    true,
                                                                    false);
        *rcp_outstream << "robj->checkGradient..." << std::endl;
        robj->checkGradient(*design_variables_rol_ptr, *d_control, true, *rcp_outstream);
        *rcp_outstream << "robj->checkHessVec..." << std::endl;
        robj->checkHessVec(*design_variables_rol_ptr, *d_control, true, *rcp_outstream);

        return 0;
    }
//============================== Check objective function and constraint gradients/Hessians ==================================================================





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
        parlist.sublist("Full Space").set("Preconditioner", all_param.optimization_param.full_space_preconditioner);
        parlist.sublist("Full Space").set("Linear iteration Limit", all_param.optimization_param.linear_iteration_limit); 
        parlist.sublist("Full Space").set("regularization_parameter", all_param.optimization_param.regularization_parameter);
        parlist.sublist("Full Space").set("regularization_scaling", all_param.optimization_param.regularization_scaling);
        parlist.sublist("Full Space").set("regularization_tol_low", all_param.optimization_param.regularization_tol_low);
        parlist.sublist("Full Space").set("regularization_tol_high", all_param.optimization_param.regularization_tol_high);

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

//================================================================================================================================================ 
    const double abs_error_final = dwr_error_val->total_dual_weighted_residual_error();
    const double actual_error_final = dwr_error_val->net_functional_error;
    //================== Evaluate exact functional error=============================================
    flow_solver->run();
    flow_solver->dg->assemble_residual();
    const double coarse_resiudual_norm = flow_solver->dg->right_hand_side.l2_norm();
    flow_solver->dg->change_cells_fe_degree_by_deltadegree_and_interpolate_solution(1);
    flow_solver->dg->assemble_residual(true);
    functional_val_coarse = functional->evaluate_functional(true, true, true);
    dealii::LinearAlgebra::distributed::Vector<double> delU (flow_solver->dg->solution);
    solve_linear(flow_solver->dg->system_matrix, flow_solver->dg->right_hand_side, delU, flow_solver->dg->all_parameters->linear_solver_param);
    delU *= -1.0;
    delU.update_ghost_values();
    flow_solver->dg->assemble_residual();
    const double residual_fine_norm = sqrt(flow_solver->dg->right_hand_side*flow_solver->dg->right_hand_side);
    const double first_order_term = functional->dIdw * delU;
    dealii::LinearAlgebra::distributed::Vector<double> intermediate_vector (flow_solver->dg->solution);
    functional->d2IdWdW->vmult(intermediate_vector, delU);
    intermediate_vector.update_ghost_values();
    const double second_order_error = intermediate_vector * delU;
    flow_solver->dg->solution += delU;
    flow_solver->dg->solution.update_ghost_values();
    functional_val_fine = functional->evaluate_functional();
    flow_solver->dg->change_cells_fe_degree_by_deltadegree_and_interpolate_solution(-1);
    const double exact_functional_error_final = functional_val_fine - functional_val_coarse;
    //============================================================================


    pcout<<"Initial absolute dwr error = "<<abs_error_initial<<std::endl;
    pcout<<"Final absolute dwr error = "<<abs_error_final<<std::endl;

    pcout<<"\nInitial dwr error = "<<actual_error_initial<<std::endl;
    pcout<<"Final dwr error = "<<actual_error_final<<std::endl;

    pcout<<"\nExact functional error initial = "<<exact_functional_error_initial<<std::endl;
    pcout<<"Exact functional error final = "<<exact_functional_error_final<<std::endl<<std::endl;

    pcout<<"\nResidual coarse sqrt(r^T * r) = "<<coarse_resiudual_norm<<std::endl;
    pcout<<"Initial residual fine sqrt(R^T * R) = "<<fine_residual_norm_initial<<std::endl;
    pcout<<"Residual fine sqrt(R^T * R) = "<<residual_fine_norm<<std::endl;
    pcout<<"\nsqrt(delU^T * delU) = "<<delU.l2_norm()<<std::endl;
    pcout<<"First order term (J_u^T * delU) = "<<first_order_term<<std::endl; 
    pcout<<"Second order term (delU^T * J_uu * delU) = "<<second_order_error<<std::endl;
//================================================================================================================================================ 
   
    const double final_control_var_norm = design_parameterization->control_var_norm();
    pcout<<"Initial control var norm = "<<initial_control_var_norm<<"     Final control var norm = "<<final_control_var_norm<<std::endl;
    
    return 0;
}

#if PHILIP_DIM != 3
    template class GoalOrientedMeshOptimization <PHILIP_DIM, 1>;
#endif

} // namespace Tests
} // namespace PHiLiP 
    
