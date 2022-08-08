#include "full_space_optimization.h"

namespace PHiLiP
{
template <int dim, int nstate, typename real, typename MeshType>
FullSpaceOptimization<dim, nstate, real, MeshType>::FullSpaceOptimization(unsigned int refinement_level_input, unsigned int polynomial_order_input, const Parameters::AllParameters *const parameters_input)
    : refinement_level(refinement_level_input)
    , polynomial_order(polynomial_order_input)
    , all_param(*parameters_input)
{
    unsigned int no_of_elements = pow(2, refinement_level);
    n_inner_vertices = no_of_elements - 1;
    metric.reinit(n_inner_vertices);

    // Initialize metric with equidistributed vertices
    double h = pow(2.0,-refinement_level);
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
    n_total = n_inner_vertices + n_dofs; // Since we are in reduced space
    solution_coeffs.reinit(n_dofs);
    solution_coeffs = dg->solution;

    global_variables.reinit(n_total);
    gradient.reinit(n_total);
    search_direction.reinit(n_total);
    hessian_full.reinit(n_total, n_total);
    update_global_variables_from_metric_solution();
    update_gradient_and_hessian();
}

template <int dim, int nstate, typename real, typename MeshType>
void FullSpaceOptimization<dim, nstate, real, MeshType>::update_global_variables_from_metric_solution()
{
    unsigned int count_index = 0;
    for(unsigned int i=0; i<n_inner_vertices; ++i)
    {
        global_variables(count_index) = metric(i);
        ++count_index;
    }
    
    for(unsigned int i=0; i<n_dofs; ++i)
    {
        global_variables(count_index) = solution_coeffs(i);
        ++count_index;
    }
}

template <int dim, int nstate, typename real, typename MeshType>
void FullSpaceOptimization<dim, nstate, real, MeshType>::update_metric_solution_from_global_variables()
{
    unsigned int count_index = 0;
    for(unsigned int i=0; i<n_inner_vertices; ++i)
    {
        metric(i) = global_variables(count_index);
        ++count_index;
    }
    
    for(unsigned int i=0; i<n_dofs; ++i)
    {
        solution_coeffs(i) = global_variables(count_index);
        ++count_index;
    }
}

template <int dim, int nstate, typename real, typename MeshType>
void FullSpaceOptimization<dim, nstate, real, MeshType>::update_gradient_and_hessian()
{
    update_metric_solution_from_global_variables();
    check_metric(metric);
    std::cout<<"Updating gradient and hessian."<<std::endl;
    GenerateTriangulation<dim, nstate, real, MeshType> triang(metric, refinement_level);
    const unsigned int grid_degree = 1;
    std::shared_ptr <DGBase<dim, real> > dg = DGFactory<dim, real>::create_discontinuous_galerkin(&all_param, polynomial_order, polynomial_order+1, grid_degree, triang.triangulation);
    dg->allocate_system();
    dg->solution = solution_coeffs;
    
    bool evaluate_derivatives = true;
    TotalDerivativeObjfunc<dim, nstate, double, MeshType> derivatives_of_objective_function(dg, evaluate_derivatives);
    // Form vector of derivatives
    AssertDimension(n_inner_vertices, derivatives_of_objective_function.dF_dX_total.size());
    AssertDimension(n_dofs, derivatives_of_objective_function.residual.size());
    
    unsigned int count1 = 0;
    for(unsigned int i=0; i<n_inner_vertices; i++)
    {
        gradient(count1) = derivatives_of_objective_function.dF_dX_total(i);
        ++count1;
    }
    
    for(unsigned int i=0; i<n_dofs; i++)
    {
        gradient(count1) = derivatives_of_objective_function.residual(i);
        ++count1;
    }
    
    // Form total hessian
    AssertDimension(n_inner_vertices, derivatives_of_objective_function.Hessian_sparse.m());
    AssertDimension(n_inner_vertices, derivatives_of_objective_function.Hessian_sparse.n());
    
    // Include d2F_dXdX
    for(unsigned int i=0; i<n_inner_vertices; ++i)
    {
        for(unsigned int j=0; j<n_inner_vertices; ++j)
        {
            hessian_full(i,j) = derivatives_of_objective_function.Hessian_sparse.el(i,j);
        }
    }

    // Include Rx
    unsigned int i_local, j_local;
    for(unsigned int i=n_inner_vertices; i<n_total; ++i)
    {
        i_local = i - n_inner_vertices;
        for(unsigned int j=0; j<n_inner_vertices; ++j)
        {
            j_local = j;
            hessian_full(i,j) = derivatives_of_objective_function.r_x_initial.el(i_local,j_local);
        }
    }

    // Include Ru
    for(unsigned int i=n_inner_vertices; i<n_total; ++i)
    {
        i_local = i - n_inner_vertices;
        for(unsigned int j=n_inner_vertices; j<n_total; ++j)
        {
            j_local = j - n_inner_vertices;
            hessian_full(i,j) = derivatives_of_objective_function.r_u.el(i_local,j_local);
        }
    }

    residual_norm = derivatives_of_objective_function.residual_norm;
    std::cout<<"Updated gradient and hessian."<<std::endl;
}

template <int dim, int nstate, typename real, typename MeshType>
real FullSpaceOptimization<dim, nstate, real, MeshType>::evaluate_function_val(VectorType &global_variables_modified)
{
    VectorType metric_modified(n_inner_vertices);
    VectorType solution_coeffs_modified(n_dofs);
    AssertDimension(global_variables_modified.size(), n_total);
    unsigned int count1 = 0;
    for(unsigned int i=0; i<n_inner_vertices; ++i)
    {
        metric_modified(i) = global_variables_modified(count1);
        ++count1;
    }
    
    for(unsigned int i=0; i<n_dofs; ++i)
    {
        solution_coeffs_modified(i) = global_variables_modified(count1);
        ++count1;
    }

    check_metric(metric_modified);
    GenerateTriangulation<dim, nstate, real, MeshType> triang(metric_modified, refinement_level);
    const unsigned int grid_degree = 1;
    std::shared_ptr <DGBase<dim, real> > dg = DGFactory<dim, real>::create_discontinuous_galerkin(&all_param, polynomial_order, polynomial_order+1, grid_degree, triang.triangulation);
    dg->allocate_system();
    AssertDimension(dg->solution.size(), solution_coeffs_modified.size());
    dg->solution = solution_coeffs_modified;
    
    bool evaluate_derivatives = false;
    TotalDerivativeObjfunc<dim, nstate, double, MeshType> derivatives_of_objective_function(dg, evaluate_derivatives);
    return derivatives_of_objective_function.objective_function_val;
}

template <int dim, int nstate, typename real, typename MeshType>
real FullSpaceOptimization<dim, nstate, real, MeshType>::evaluate_backtracking_alpha()
{
    //return 0.5;
    //return 0.1;
    double alpha = 0.5, c = 0.0, rho=0.1;
    VectorType global_variables_modified  = global_variables;
    global_variables_modified.add(alpha, search_direction);
    double dwr_original = evaluate_function_val(global_variables);
    bool is_metric_good = check_metric_bool(global_variables_modified);
    double dwr_modified;
    if(!is_metric_good) // If metric is not good, make sure dwr_modified > dwr_original so that alpha can be reduced.
    {
        dwr_modified = dwr_original + 1.0e10;
    }
    else
    {
        dwr_modified = evaluate_function_val(global_variables_modified);
    }
        

    while (dwr_modified >= (dwr_original + c*alpha*(gradient*search_direction)))
    {
        alpha *= rho;
        global_variables_modified = global_variables;
        global_variables_modified.add(alpha,search_direction);
        is_metric_good = check_metric_bool(global_variables_modified);
        if(!is_metric_good)
        {
            dwr_modified = dwr_original + 1.0e10;
        }
        else
        {
            dwr_modified = evaluate_function_val(global_variables_modified);
        }


        if(alpha < 1.0e-7)
        {
            std::cout<<"Backtracking alpha is too small"<<std::endl;
            return 0.0;
        }
    }
    std::cout<<"Backtracking alpha = "<<alpha<<std::endl;
    return alpha;
}

template <int dim, int nstate, typename real, typename MeshType>
void FullSpaceOptimization<dim, nstate, real, MeshType>::get_search_direction_from_hessian_gradient_system()
{
    dealii::TrilinosWrappers::SparseMatrix hessian_sparse;
    dealii::SparsityPattern hessian_sparsity_pattern;

    hessian_sparsity_pattern.copy_from(hessian_full);
    hessian_sparse.reinit(hessian_sparsity_pattern);

    for(unsigned int i=0; i<hessian_full.m(); i++)
    {
        for(unsigned int j=0; j<hessian_full.n(); j++)
        {
            if(hessian_full(i,j)==0) continue;

            hessian_sparse.add(i,j,hessian_full(i,j));
        }
    }

    solve_linear(hessian_sparse, gradient, search_direction, all_param.linear_solver_param);
    search_direction *= -1.0;
    std::cout<<"Norm of search direction = "<<search_direction.l2_norm()<<std::endl;
}


template <int dim, int nstate, typename real, typename MeshType>
void FullSpaceOptimization<dim, nstate, real, MeshType>::solve_optimization_problem()
{
    double step_length = 0.1;
    int iterations = 0;
    std::ofstream myfile_gradient, myfile_error, myfile_residual, myfile_time;
    myfile_gradient.open("Full_space_gradient_convergence.txt");
    myfile_error.open("Full_space_error_convergence.txt");
    myfile_residual.open("Full_space_residual_convergence.txt");
    myfile_time.open("Full_space_time.txt");
    std::clock_t c_start = std::clock();
    double time_elapsed = 0;
    double error_value = 0;
    while (gradient.l2_norm() > 1.0e-10)
    {
        std::cout<<"Magnitude of the gradient before = "<<gradient.l2_norm()<<std::endl;
        iterations++;
        if(iterations > 50) break;
        std::cout<<"================================================================="<<std::endl;
        std::cout<<"Nonlinear Newton iteration # : "<<iterations<<std::endl; 
        std::cout<<"================================================================="<<std::endl;
        std::cout<<"Update 1: Obtaining search direction."<<std::endl;
        get_search_direction_from_hessian_gradient_system();
        std::cout<<"Update 1: Obtained search direction."<<std::endl;

        std::cout<<"Update 2: Backtracking."<<std::endl;
        step_length = evaluate_backtracking_alpha();
        std::cout<<"Update 2: Finished backtracking."<<std::endl;
        if(step_length == 0.0)
        {
            std::cout<<"Cannot reduce the functional any further."<<std::endl;
            break;
        }
        global_variables.add(step_length, search_direction);
        std::cout<<"Update 3: Evaluating gradient and hessian."<<std::endl;
        update_gradient_and_hessian();
        std::cout<<"Update 3: Evaluated gradient and hessian."<<std::endl;
        std::cout<<"Magnitude of the gradient = "<<gradient.l2_norm()<<std::endl;
        myfile_gradient<<gradient.l2_norm()<<std::endl;
        error_value = evaluate_function_val(global_variables);
        myfile_error<<error_value<<"\n";
        std::cout<<"Error value = "<<error_value<<std::endl;
        myfile_residual<<residual_norm<<"\n";
        std::clock_t c_end = std::clock();
        time_elapsed = 1000.0*(c_end - c_start)/CLOCKS_PER_SEC;
        myfile_time<<time_elapsed/1000<<"\n";
    }
    myfile_gradient.close();
    myfile_error.close();
    myfile_residual.close();
    myfile_time.close();

    // Output converged metric
    GenerateTriangulation<dim, nstate, real, MeshType> triang(metric, refinement_level, true);
    
}

template <int dim, int nstate, typename real, typename MeshType>
void FullSpaceOptimization<dim, nstate, real, MeshType>::check_metric(VectorType &metric_modified)
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
            std::cout<<"Vertices have overlapped or are outside [0,1]. Aborting..."<<std::endl;
            std::abort();
        }

    }
}

template <int dim, int nstate, typename real, typename MeshType>
bool FullSpaceOptimization<dim, nstate, real, MeshType>::check_metric_bool(VectorType &global_variables_modified)
{
    bool is_metric_good = true;
    VectorType metric_modified = metric;
    for(unsigned int i=0; i<n_inner_vertices; i++) // loop only over x
    {
        metric_modified(i) = global_variables_modified(i);
    }

    for(unsigned int i=0; i<n_inner_vertices; i++) // loop only over x
    {
        if(metric_modified(i) < 0.0 || metric_modified(i) > 1.0) 
        {
            is_metric_good = false;
            return is_metric_good;
        }
        if(i < (metric_modified.size()-1))
        {
            if(metric_modified(i) > metric_modified(i+1)) 
            {
                is_metric_good = false;
                return is_metric_good;
            }
        }
    }
    return is_metric_good;
}


#if PHILIP_DIM == 1
template class FullSpaceOptimization<PHILIP_DIM, 1, double, dealii::Triangulation<PHILIP_DIM>>;
template class FullSpaceOptimization<PHILIP_DIM, 2, double, dealii::Triangulation<PHILIP_DIM>>;
template class FullSpaceOptimization<PHILIP_DIM, 3, double, dealii::Triangulation<PHILIP_DIM>>;
template class FullSpaceOptimization<PHILIP_DIM, 4, double, dealii::Triangulation<PHILIP_DIM>>;
template class FullSpaceOptimization<PHILIP_DIM, 5, double, dealii::Triangulation<PHILIP_DIM>>;
#endif

} // PHiLiP namespace
