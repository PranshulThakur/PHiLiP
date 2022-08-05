#include "reduced_space_optimization.h"

namespace PHiLiP
{
template <int dim, int nstate, typename real, typename MeshType>
ReducedSpaceOptimization<dim, nstate, real, MeshType>::ReducedSpaceOptimization(int refinement_level_input, int polynomial_order_input, const Parameters::AllParameters *const parameters_input)
    : refinement_level(refinement_level_input)
    , polynomial_order(polynomial_order_input)
    , all_param(*parameters_input)
{
    int no_of_elements = pow(2, refinement_level);
    n_inner_vertices = no_of_elements - 1;
    metric.reinit(n_inner_vertices);

    // Initialize metric with equidistributed vertices
    double h = pow(2,-refinement_level);
    for(unsigned int i=0; i<metric.size(); i++)
    {
        metric(i) = (i+1)*h;
    }

    GenerateTriangulation<dim, nstate, real, MeshType> triang(metric, refinement_level);
    const unsigned int grid_degree = 1;
    std::shared_ptr <DGBase<dim, real> > dg = DGFactory<dim, real>::create_discontinuous_galerkin(&all_param, polynomial_order, polynomial_order+1, grid_degree, triang.triangulation);
    dg->allocate_system();
    dg->solution *= 0.0;
    std::shared_ptr<ODE::ODESolverBase<dim, double>> ode_solver = ODE::ODESolverFactory<dim, double>::create_ODESolver(dg);
    ode_solver->steady_state();
    
    // Initialize all variables.
    n_dofs = dg->n_dofs();
    n_total = n_inner_vertices; // Since we are in reduced space
    solution_coeffs.reinit(n_dofs);
    solution_coeffs = dg->solution;

    global_variables.reinit(n_total);
    gradient.reinit(n_total);
    search_direction.reinit(n_total);
    update_gradient_and_hessian();
}

template <int dim, int nstate, typename real, typename MeshType>
void ReducedSpaceOptimization<dim, nstate, real, MeshType>::update_gradient_and_hessian()
{
    std::cout<<"Updating gradient and hessian."<<std::endl;
    GenerateTriangulation<dim, nstate, real, MeshType> triang(metric, refinement_level);
    const unsigned int grid_degree = 1;
    std::shared_ptr <DGBase<dim, real> > dg = DGFactory<dim, real>::create_discontinuous_galerkin(&all_param, polynomial_order, polynomial_order+1, grid_degree, triang.triangulation);
    dg->allocate_system();
    dg->solution = solution_coeffs;
    //================================ Solve for solution coeff begins (required for reduced space) =========================================================================
    std::shared_ptr<ODE::ODESolverBase<dim, double>> ode_solver = ODE::ODESolverFactory<dim, double>::create_ODESolver(dg);
    ode_solver->steady_state();
    solution_coeffs = dg->solution;
    //================================ Solve for solution coeff ends =========================================================================
    bool evaluate_derivatives = true;
    TotalDerivativeObjfunc<dim, nstate, double, MeshType> derivatives_of_objective_function(dg, evaluate_derivatives);
    AssertDimension(gradient.size(), derivatives_of_objective_function.dF_dX_total.size());
    gradient = derivatives_of_objective_function.dF_dX_total;
    hessian_sparse.copy_from(derivatives_of_objective_function.Hessian_sparse);
    residual_norm = derivatives_of_objective_function.residual_norm;
    std::cout<<"Updated gradient and hessian."<<std::endl;
}

template <int dim, int nstate, typename real, typename MeshType>
real ReducedSpaceOptimization<dim, nstate, real, MeshType>::evaluate_function_val(VectorType &metric_modified)
{
    GenerateTriangulation<dim, nstate, real, MeshType> triang(metric_modified, refinement_level);
    const unsigned int grid_degree = 1;
    std::shared_ptr <DGBase<dim, real> > dg = DGFactory<dim, real>::create_discontinuous_galerkin(&all_param, polynomial_order, polynomial_order+1, grid_degree, triang.triangulation);
    dg->allocate_system();
    dg->solution = solution_coeffs;
    //================================ Solve for solution coeff begins (required for reduced space) =========================================================================
    std::shared_ptr<ODE::ODESolverBase<dim, double>> ode_solver = ODE::ODESolverFactory<dim, double>::create_ODESolver(dg);
    ode_solver->steady_state();
    solution_coeffs = dg->solution;
    //================================ Solve for solution coeff ends =========================================================================
    bool evaluate_derivatives = false;
    TotalDerivativeObjfunc<dim, nstate, double, MeshType> derivatives_of_objective_function(dg, evaluate_derivatives);
    return derivatives_of_objective_function.objective_function_val;
}

template <int dim, int nstate, typename real, typename MeshType>
real ReducedSpaceOptimization<dim, nstate, real, MeshType>::evaluate_backtracking_alpha()
{
    double alpha = 0.5, c = 0.1, rho=0.5;
    VectorType metric_modified  = metric;
    metric_modified.add(alpha, search_direction);
    check_metric(metric_modified);
    double dwr_original = evaluate_function_val(metric);
    while (evaluate_function_val(metric_modified) >= (dwr_original + c*alpha*(gradient*search_direction)))
    {
        alpha *= rho;
        metric_modified = metric;
        metric_modified.add(alpha,search_direction);

        if(alpha < 1.0e-15)
        {
            std::cout<<"Backtracking alpha is too small"<<std::endl;
            return 1.0e-1;
        }
    }
    std::cout<<"Backtracking alpha = "<<alpha<<std::endl;
    return alpha;
}

template <int dim, int nstate, typename real, typename MeshType>
void ReducedSpaceOptimization<dim, nstate, real, MeshType>::get_search_direction_from_hessian_gradient_system()
{
    solve_linear(hessian_sparse, gradient, search_direction, all_param.linear_solver_param);
    search_direction *= -1.0;
}


template <int dim, int nstate, typename real, typename MeshType>
void ReducedSpaceOptimization<dim, nstate, real, MeshType>::solve_optimization_problem()
{
    double step_length = 0.1;
    int iterations = 0;
    std::ofstream myfile_gradient, myfile_error, myfile_residual, myfile_time;
    myfile_gradient.open("Reduced_space_gradient_convergence.txt");
    myfile_error.open("Reduced_space_error_convergence.txt");
    myfile_residual.open("Reduced_space_residual_convergence.txt");
    myfile_time.open("Reduced_space_time.txt");
    std::clock_t c_start = std::clock();
    double time_elapsed = 0;
    while (gradient.l2_norm() > 1.5e-14)
    {
        std::cout<<"Magnitude of the gradient before = "<<gradient.l2_norm()<<std::endl;
        iterations++;
        if(iterations > 50) break;
        
        std::cout<<"Update 1: Obtaining search direction."<<std::endl;
        get_search_direction_from_hessian_gradient_system();
        std::cout<<"Update 1: Obtained search direction."<<std::endl;

        std::cout<<"Update 2: Backtracking."<<std::endl;
        step_length = evaluate_backtracking_alpha();
        std::cout<<"Update 2: Finished backtracking."<<std::endl;
        metric.add(step_length, search_direction);
        check_metric(metric);
        std::cout<<"Update 3: Evaluating gradient and hessian."<<std::endl;
        update_gradient_and_hessian();
        std::cout<<"Update 3: Evaluated gradient and hessian."<<std::endl;
        std::cout<<"Magnitude of the gradient = "<<gradient.l2_norm()<<std::endl;
        myfile_gradient<<gradient.l2_norm()<<std::endl;
        myfile_error<<evaluate_function_val(metric)<<"\n";
        myfile_residual<<residual_norm<<"\n";
        std::clock_t c_end = std::clock();
        time_elapsed = 1000.0*(c_end - c_start)/CLOCKS_PER_SEC;
        myfile_time<<time_elapsed/1000<<"\n";
    }
    myfile_gradient.close();
    myfile_error.close();
    myfile_residual.close();
    myfile_time.close();
}

template <int dim, int nstate, typename real, typename MeshType>
void ReducedSpaceOptimization<dim, nstate, real, MeshType>::check_metric(VectorType &metric_modified)
{
    for(unsigned int i=0; i<metric_modified.size(); i++)
    {
        bool is_metric_good = true;
        if(metric_modified(i) < 0.0 || metric_modified(i) > 1.0) is_metric_good = false;
        if(i < (metric_modified.size()-1))
        {
            if(metric_modified(i) > metric_modified(i+1)) is_metric_good = false;
        }

        if(!is_metric_good)
        {
            std::cout<<"Vertices have overlapped. Aborting..."<<std::endl;
            std::abort();
        }

    }
}


#if PHILIP_DIM == 1
template class ReducedSpaceOptimization<PHILIP_DIM, 1, double, dealii::Triangulation<PHILIP_DIM>>;
template class ReducedSpaceOptimization<PHILIP_DIM, 2, double, dealii::Triangulation<PHILIP_DIM>>;
template class ReducedSpaceOptimization<PHILIP_DIM, 3, double, dealii::Triangulation<PHILIP_DIM>>;
template class ReducedSpaceOptimization<PHILIP_DIM, 4, double, dealii::Triangulation<PHILIP_DIM>>;
template class ReducedSpaceOptimization<PHILIP_DIM, 5, double, dealii::Triangulation<PHILIP_DIM>>;
#endif

} // PHiLiP namespace
