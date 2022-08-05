#ifndef __REDUCED_SPACE_OPTIMIZATION__
#define __REDUCED_SPACE_OPTIMIZATION__

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

class ReducedSpaceOptimization
{
    using VectorType = typename dealii::LinearAlgebra::distributed::Vector<real>;
    dealii::Vector<real> metric;
    VectorType solution_coeffs;
    VectorType global_variables; ///< Just x, since we are in reduced space.
    VectorType gradient;
    VectorType search_direction;
    dealii::FullMatrix<real> hessian;

    const int refinement_level;
    const int polynomial_order;
    const Parameters::AllParameters all_param;
    
    int n_inner_vertices; ///< No. of inner vertices i.e. excluding the vertices at boundary. 
    int n_total;
    int n_dofs;

public:
    ReducedSpaceOptimization(int refienement_level_input, int polynomial_order_input, const Parameters::AllParameters *const parameters_input);
    ~ReducedSpaceOptimization() {};
    real evaluate_function_val(VectorType &modified_global_variables);
    void update_gradient_and_hessian();
    real evaluate_backtracking_alpha();
   // void form_block_hessian();
    //void update_global_variables_from_metric_solution_lambda();
    //void update_metric_solution_lambda_from_global_variables();
    
    void solve_optimization_problem();
};

} // namespace PHiLiP


#endif
