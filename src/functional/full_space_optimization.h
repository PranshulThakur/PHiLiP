#ifndef __FULL_SPACE_OPTIMIZATION__
#define __FULL_SPACE_OPTIMIZATION__

#include "generate_triangulation.h"
#include "parameters/all_parameters.h"
#include "dg/dg.h"
#include "dg/dg_factory.hpp"
#include "ode_solver/ode_solver_factory.h"

namespace PHiLiP {

#if PHILIP_DIM==1
template <int dim, int nstate, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, int nstate, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif

class FullSpaceOptimization
{
    using VectorType = typename dealii::LinearAlgebra::distributed::Vector<real>;
    VectorType metric;
    VectorType solution_coeffs;
    VectorType global_variables; ///< Just x, since we are in reduced space.
    VectorType gradient;
    VectorType search_direction;
    dealii::FullMatrix<real> hessian_full;

    const int refinement_level;
    const int polynomial_order;
    const Parameters::AllParameters all_param;
    
    unsigned int n_inner_vertices; ///< No. of inner vertices i.e. excluding the vertices at boundary. 
    unsigned int n_total;
    unsigned int n_dofs;
    real residual_norm;

public:
    FullSpaceOptimization(unsigned int refienement_level_input, unsigned int polynomial_order_input, const Parameters::AllParameters *const parameters_input);
    ~FullSpaceOptimization() {};
    real evaluate_function_val(VectorType &modified_global_variables); ///< Required for backtracking.
    void update_gradient_and_hessian();
    void solve_for_reduced_solution_coeff();
    real evaluate_backtracking_alpha();
    
    void form_block_hessian();
    void update_global_variables_from_metric_solution();
    void update_metric_solution_from_global_variables();
    
    void solve_optimization_problem();
    void get_search_direction_from_hessian_gradient_system();
    void check_metric(VectorType &metric_modified);
};

} // namespace PHiLiP


#endif
