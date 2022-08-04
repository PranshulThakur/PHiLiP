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
    std::shared_ptr <DGBase<dim, double> > dg = DGFactory<dim,double>::create_discontinuous_galerkin(&param, poly_degree, poly_degree+1, 1, grid);
    dg->allocate_system();
    dg->solution*=0.0;
    std::cout<<"Created and allocated DG."<<std::endl;

    std::shared_ptr<ODE::ODESolverBase<dim, double>> ode_solver = ODE::ODESolverFactory<dim, double>::create_ODESolver(dg);
    std::cout<<"Created ODE solver."<<std::endl;
    ode_solver->steady_state();
    std::cout<<"Solved steady state."<<std::endl;
    
    std::cout<<"Now computing total derivative..."<<std::endl;
    TotalDerivativeObjfunc<dim, nstate, double, MeshType> totder(dg);
   
    std::cout<<"Checking with finite difference..."<<std::endl;
    auto cell = grid->begin_active();
    double step_size = 1.0e-6;
    cell->vertex(1)[0] += step_size;
    std::shared_ptr <DGBase<dim, double> > dg2 = DGFactory<dim,double>::create_discontinuous_galerkin(&param, poly_degree, poly_degree+1, 1, grid);
    dg2->allocate_system();
    dg2->solution*=0.0;
    std::shared_ptr<ODE::ODESolverBase<dim, double>> ode_solver2 = ODE::ODESolverFactory<dim, double>::create_ODESolver(dg2);
    ode_solver2->steady_state();
    TotalDerivativeObjfunc<dim, nstate, double, MeshType> totder2(dg2);
    
    std::cout<<"dF_dX_total = "<<std::endl;
    totder.dF_dX_total.print(std::cout, 3, true, false);
    std::cout<<"dF_dX_FD = "<<(totder2.objective_function_val - totder.objective_function_val)/step_size<<std::endl;
    
    std::cout<<"Hessian_total = "<<std::endl;
    totder.Hessian_total.print(std::cout,10,1);

    dealii::LinearAlgebra::distributed::Vector<double> H_FD = totder2.dF_dX_total;
    H_FD -= totder.dF_dX_total;
    H_FD /= step_size;
    std::cout<<"Hessian FD = "<<std::endl;
    H_FD.print(std::cout, 3, true, false);


    return 0;
}

#if PHILIP_DIM==1
    template class MeshRAdaptation<PHILIP_DIM,PHILIP_DIM>;
#endif
} // namespace Tests
} // namespace PHiLiP

