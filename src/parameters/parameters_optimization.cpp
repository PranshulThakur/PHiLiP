#include "parameters/parameters_optimization.h"

namespace PHiLiP {
namespace Parameters {

OptimizationParam::OptimizationParam() {}

void OptimizationParam::declare_parameters (dealii::ParameterHandler &prm)
{
    prm.enter_subsection("optimization");
    {
        prm.declare_entry("optimization_type", "reduced_space",
                           dealii::Patterns::Selection(
                           " reduced_space | "
                           " full_space "
                           ),
                           "Optimization type to be used. "
                           "Choices are "
                           " reduced_space | "
                           " full_space ");

        prm.declare_entry("max_design_cycles", "20",
                           dealii::Patterns::Integer(0, 1000),
                           "Maximum number of optimization design cycles");
        
        prm.declare_entry("linear_iteration_limit", "200",
                           dealii::Patterns::Integer(0, 2000),
                           "Maximum number of iterations to solve a linear system.");

        prm.declare_entry("gradient_tolerance", "1.0e-9",
                           dealii::Patterns::Double(0.0, 1.0e-2),
                           "Value of the gradient at which we terminate optimization.");
        
        prm.declare_entry("functional_evaluation_limit", "200",
                           dealii::Patterns::Integer(0, 2000),
                           "Maximum computations of functional during backtracking.");
        
        prm.declare_entry("initial_step_size", "1.0",
                           dealii::Patterns::Double(0.0, 1.0),
                           "Initial step size for backtracking.");
        
        prm.declare_entry("mesh_weight_factor", "1.0e-5",
                           dealii::Patterns::Double(0.0, 1.0),
                           "Weight of mesh distortion indicator added to the objective function.");
        
        prm.declare_entry("mesh_volume_power", "-2",
                           dealii::Patterns::Integer(),
                           "Power to which cell volume is raised in mesh distrotion indicator.");
        
        prm.declare_entry("full_space_preconditioner", "identity",
                          dealii::Patterns::Selection(
                          " P2 | "
                          " P2A | "
                          " P4 | "
                          " P4A | "
                          " identity "
                          ),
                          "Preconditioner to be used for full space. Choices are "
                          " P2 | "
                          " P2A | "
                          " P4 | "
                          " P4A | "
                          " identity ");

        prm.declare_entry("line_search_method", "Backtracking",
                          dealii::Patterns::Selection(
                          " Backtracking | "
                          " Cubic Interpolation | "
                          " Iteration Scaling "
                          ),
                          "Line search method used to get step length. Choices are "
                          " Backtracking | "
                          " Cubic Interpolation | "
                          " Iteration Scaling "
                          );
        
        prm.declare_entry("line_search_curvature", "Null Curvature Condition",
                          dealii::Patterns::Selection(
                          " Null Curvature Condition | "
                          " Goldstein Conditions | "
                          " Strong Wolfe Conditions "
                          ),
                          "Curvature condition used to get step length. Choices are "
                          " Null Curvature Condition | "
                          " Goldstein Conditions | "
                          " Strong Wolfe Conditions "
                          );

        prm.declare_entry("reduced_space_descent_method", "Newton-Krylov",
                           dealii::Patterns::Selection(
                           " Newton-Krylov | "
                           " Quasi-Newton Method "
                           ),
                           "Descent method for reduced space. Choices are "
                           " Newton-Krylov | "
                           " Quasi-Newton Method "
                           );
    }

    prm.leave_subsection();
}

void OptimizationParam::parse_parameters (dealii::ParameterHandler &prm)
{
    prm.enter_subsection("optimization");
    {
        const std::string optimization_type_string = prm.get("optimization_type");
        if(optimization_type_string == "reduced_space")     {optimization_type = OptimizationType::reduced_space;}
        else if(optimization_type_string == "full_space")   {optimization_type = OptimizationType::full_space;}

        max_design_cycles = prm.get_integer("max_design_cycles");
        linear_iteration_limit = prm.get_integer("linear_iteration_limit");
        gradient_tolerance = prm.get_double("gradient_tolerance");
        functional_evaluation_limit = prm.get_integer("functional_evaluation_limit");
        initial_step_size = prm.get_double("initial_step_size");
        mesh_weight_factor = prm.get_double("mesh_weight_factor");
        mesh_volume_power = prm.get_integer("mesh_volume_power");
        full_space_preconditioner = prm.get("full_space_preconditioner");
        line_search_method = prm.get("line_search_method");
        line_search_curvature = prm.get("line_search_curvature");
        reduced_space_descent_method = prm.get("reduced_space_descent_method");
    }

    prm.leave_subsection();
}

} // namespace Parameters
} // namespace PHiLiP
