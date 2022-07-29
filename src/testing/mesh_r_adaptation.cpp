#include "mesh_r_adaptation.h"
#include "ode_solver/ode_solver_factory.h"
#include "functional/total_derivatives_of_objective_function.h"
namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
MeshRAdaptation<dim, nstate>::MeshRAdaptation(const Parameters::AllParameters *const parameters_input)
: TestsBase::TestsBase(parameters_input)
{}

template <int dim, int nstate>
int MeshRAdaptation<dim, nstate>::run_test() const
{
    Parameters::AllParameters param = *(TestsBase::all_parameters);
    Assert(dim == param.dimension, dealii::ExcDimensionMismatch(dim, param.dimension));
    Assert(dim == 1, dealii::ExcDimensionMismatch(dim, param.dimension));
    using MeshType = dealii::Triangulation<dim>;
    std::shared_ptr<MeshType> grid = std::make_shared<MeshType>();
    const bool colorize = true;
    dealii::GridGenerator::hyper_cube(*grid, 0, 1, colorize);
    grid->refine_global(param.manufactured_convergence_study_param.initial_grid_size);
    unsigned int poly_degree = param.manufactured_convergence_study_param.degree_start;
    std::shared_ptr <DGBase<dim, double> > dg = DGFactory<dim,double>::create_discontinuous_galerkin(&param, poly_degree, poly_degree+1, grid);
    dg->allocate_system();
    dg->solution*=0.0;
    std::cout<<"Created and allocated DG."<<std::endl;

    std::shared_ptr<ODE::ODESolverBase<dim, double>> ode_solver = ODE::ODESolverFactory<dim, double>::create_ODESolver(dg);
    std::cout<<"Created ODE solver."<<std::endl;
    ode_solver->steady_state();
    std::cout<<"Solved steady state."<<std::endl;
    
    std::cout<<"Now computing total derivative..."<<std::endl;
    TotalDerivativeObjfunc<dim, nstate, double, MeshType> totder(dg);
    return 0;
}

#if PHILIP_DIM==1
    template class MeshRAdaptation<PHILIP_DIM,PHILIP_DIM>;
#endif
} // namespace Tests
} // namespace PHiLiP

