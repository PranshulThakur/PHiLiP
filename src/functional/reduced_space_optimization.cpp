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
    unsigned int grid_degree = 1;
    std::shared_ptr <DGBase<dim, real> > dg = DGFactory<dim, real>::create_discontinuous_galerkin(&all_param, polynomial_order, polynomial_order+1, grid_degree, triang.triangulation);
    dg->allocate_system();
    dg->solution *= 0.0;
    std::shared_ptr<ODE::ODESolverBase<dim, double>> ode_solver = ODE::ODESolverFactory<dim, double>::create_ODESolver(dg);
    ode_solver->steady_state();

    n_dofs = dg->n_dofs();
    n_total = n_inner_vertices; // Since we are in reduced space
    solution_coeffs.reinit(n_dofs);
    solution_coeffs = dg->solution;

    global_variables.reinit(n_total);
    hessian.reinit(n_total, n_total);
    gradient.reinit(n_total);
    search_direction.reinit(n_total);
    update_gradient_and_hessian();
}





#if PHILIP_DIM == 1
template class ReducedSpaceOptimization<PHILIP_DIM, 1, double, dealii::Triangulation<PHILIP_DIM>>;
template class ReducedSpaceOptimization<PHILIP_DIM, 2, double, dealii::Triangulation<PHILIP_DIM>>;
template class ReducedSpaceOptimization<PHILIP_DIM, 3, double, dealii::Triangulation<PHILIP_DIM>>;
template class ReducedSpaceOptimization<PHILIP_DIM, 4, double, dealii::Triangulation<PHILIP_DIM>>;
template class ReducedSpaceOptimization<PHILIP_DIM, 5, double, dealii::Triangulation<PHILIP_DIM>>;
#endif

} // PHiLiP namespace
