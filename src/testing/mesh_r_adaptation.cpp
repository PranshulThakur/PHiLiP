#include "mesh_r_adaptation.h"
#include "ode_solver/ode_solver_factory.h"
#include "functional/reduced_space_optimization.h"
#include "functional/full_space_optimization.h"

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
    unsigned int poly_degree = param.manufactured_convergence_study_param.degree_start;
    //unsigned int refinement_level = param.manufactured_convergence_study_param.initial_grid_size;
    
    std::shared_ptr<MeshType> grid = std::make_shared<MeshType>();
    const bool colorize = true;
    dealii::GridGenerator::hyper_cube(*grid, 0, 1, colorize);
    grid->refine_global(param.manufactured_convergence_study_param.initial_grid_size);
    unsigned int grid_degree = 1;
    std::shared_ptr <DGBase<dim, double> > dg = DGFactory<dim,double>::create_discontinuous_galerkin(&param, poly_degree, poly_degree+1, grid_degree, grid);
    dg->allocate_system();
    dg->solution*=0.0;
    std::cout<<"Created and allocated DG."<<std::endl;

    std::shared_ptr<ODE::ODESolverBase<dim, double>> ode_solver = ODE::ODESolverFactory<dim, double>::create_ODESolver(dg);
    std::cout<<"Created ODE solver."<<std::endl;
    ode_solver->steady_state();
    std::cout<<"Solved steady state."<<std::endl;
    std::shared_ptr< Functional<dim, nstate, double, MeshType> > functional = FunctionalFactory<dim,nstate,double,MeshType>::create_Functional(dg->all_parameters->functional_param, dg);
    const double functional_value_coarse = functional->evaluate_functional();

    // Adjoint based error indicator
    std::unique_ptr <MeshErrorEstimateBase <dim, double, MeshType>> mesh_error_adjoint = std::make_unique<DualWeightedResidualError<dim, nstate, double, MeshType>>(dg);
    dealii::Vector<double> cellwise_errors = mesh_error_adjoint->compute_cellwise_errors();
    
    double adjoint_functional_error = 0.0;
    for(unsigned int i=0; i<cellwise_errors.size(); ++i)
    {
        adjoint_functional_error += cellwise_errors(i);
    }

    // Direct functional error
    TotalDerivativeObjfunc<dim, nstate, double, MeshType> totder(dg, false);
    const double direct_taylor_expansion_error = totder.objective_function_val;
    // Exact functional error
    std::shared_ptr <DGBase<dim, double> > dg_fine = DGFactory<dim,double>::create_discontinuous_galerkin(&param, poly_degree+1, poly_degree+2, grid_degree, grid);
    dg_fine->allocate_system();
    dg_fine->solution*=0.0;
    std::cout<<"Created and allocated DG."<<std::endl;

    std::shared_ptr<ODE::ODESolverBase<dim, double>> ode_solver_fine = ODE::ODESolverFactory<dim, double>::create_ODESolver(dg_fine);
    std::cout<<"Created ODE solver."<<std::endl;
    ode_solver_fine->steady_state();
    std::cout<<"Solved steady state."<<std::endl;
    
    std::shared_ptr< Functional<dim, nstate, double, MeshType> > functional_fine = FunctionalFactory<dim,nstate,double,MeshType>::create_Functional(dg_fine->all_parameters->functional_param, dg_fine);
    const double functional_value_fine = functional_fine->evaluate_functional();
    const double exact_functional_error = functional_value_fine - functional_value_coarse;

    std::cout<<"Results of the analysis: "<<std::endl;
    std::cout<<"Adjoint based functional error = "<<adjoint_functional_error<<std::endl;
    std::cout<<"Direct taylor expanded functional error = "<<direct_taylor_expansion_error<<std::endl;
    std::cout<<"Exact functional error = "<<exact_functional_error<<std::endl;
    std::cout<<"N_dofs = "<<dg->n_dofs()<<std::endl;





    /*
//==============================================================================================================================================================
                    // Run optimization algorithm.
//==============================================================================================================================================================
    //ReducedSpaceOptimization<dim, nstate, double, MeshType> optimizer(refinement_level, poly_degree, &param);
    FullSpaceOptimization<dim, nstate, double, MeshType> optimizer(refinement_level, poly_degree, &param);
    optimizer.solve_optimization_problem();
    */
/*
//==============================================================================================================================================================
                    // Check total derivative and hessian of the objective function
//==============================================================================================================================================================
    std::shared_ptr<MeshType> grid = std::make_shared<MeshType>();
    const bool colorize = true;
    dealii::GridGenerator::hyper_cube(*grid, 0, 1, colorize);
    grid->refine_global(param.manufactured_convergence_study_param.initial_grid_size);
    std::shared_ptr <DGBase<dim, double> > dg = DGFactory<dim,double>::create_discontinuous_galerkin(&param, poly_degree, poly_degree+1, 1, grid);
    dg->allocate_system();
    dg->solution*=0.0;
    std::cout<<"Created and allocated DG."<<std::endl;

    std::shared_ptr<ODE::ODESolverBase<dim, double>> ode_solver = ODE::ODESolverFactory<dim, double>::create_ODESolver(dg);
    std::cout<<"Created ODE solver."<<std::endl;
    ode_solver->steady_state();
    std::cout<<"Solved steady state."<<std::endl;
    
    std::cout<<"Now computing total derivative..."<<std::endl;
    TotalDerivativeObjfunc<dim, nstate, double, MeshType> totder(dg, true);
   
    std::cout<<"Checking with finite difference..."<<std::endl;
    auto cell = grid->begin_active();
    double step_size = 1.0e-6;
    //cell++;
    cell->vertex(1)[0] += step_size;
    std::shared_ptr <DGBase<dim, double> > dg2 = DGFactory<dim,double>::create_discontinuous_galerkin(&param, poly_degree, poly_degree+1, 1, grid);
    dg2->allocate_system();
    //dg2->solution*=0.0;
    dg2->solution = dg->solution;
    std::shared_ptr<ODE::ODESolverBase<dim, double>> ode_solver2 = ODE::ODESolverFactory<dim, double>::create_ODESolver(dg2);
    ode_solver2->steady_state();
    TotalDerivativeObjfunc<dim, nstate, double, MeshType> totder2(dg2, true);
    
    std::cout<<"dF_dX_total = "<<std::endl;
    totder.dF_dX_total.print(std::cout, 3, true, false);
    std::cout<<"dF_dX_FD = "<<(totder2.objective_function_val - totder.objective_function_val)/step_size<<std::endl;
    
    std::cout<<"Hessian_sparse = "<<std::endl;
    dealii::FullMatrix<double> Hessian_sparse_full;
    Hessian_sparse_full.copy_from(totder.Hessian_sparse);
    Hessian_sparse_full.print(std::cout,10,1);

    dealii::LinearAlgebra::distributed::Vector<double> H_FD = totder2.dF_dX_total;
    H_FD -= totder.dF_dX_total;
    H_FD /= step_size;
    std::cout<<"Hessian FD = "<<std::endl;
    H_FD.print(std::cout, 3, true, false);
    

    std::cout<<"================================================================================================================"<<std::endl;
    std::cout<<"Checking vertex and solution.."<<std::endl;
    std::cout<<"================================================================================================================"<<std::endl;
    
    std::cout<<"vertex_positions = "<<"[0, "; 
    for(const auto &cell : dg->triangulation->active_cell_iterators())
    {
        std::cout<<cell->vertex(1)[0]<<", ";
    }
    std::cout<<"]"<<std::endl;


   std::vector<double> solution_at_quad_pts;
   std::vector<double> x_quad_pts;
   dealii::QGauss<dim> quad(poly_degree + 1);
   const dealii::Mapping<dim> &mapping = (*(dg->high_order_grid->mapping_fe_field));
   dealii::FEValues<dim,dim> fe_values_volume(mapping, dg->fe_collection[poly_degree], quad, dealii::update_values | dealii::update_JxW_values | dealii::update_quadrature_points);
   std::vector<dealii::types::global_dof_index> dof_indices(fe_values_volume.dofs_per_cell);
   const unsigned int n_quad_pts = fe_values_volume.n_quadrature_points;
   const unsigned int n_dofs_cell = fe_values_volume.dofs_per_cell;

    for (const auto &cell : dg->dof_handler.active_cell_iterators())
    {
         if (!(cell->is_locally_owned() || cell->is_ghost())) continue;
        // Get FEValues of of the current cell.
         fe_values_volume.reinit(cell);
         cell->get_dof_indices(dof_indices);
        
        for (unsigned int iquad = 0; iquad < n_quad_pts; ++iquad)
        {
            x_quad_pts.push_back(fe_values_volume.quadrature_point(iquad)[0]); // get physical quadrature
            double soln_at_q = 0;
            for(unsigned int idof = 0; idof < n_dofs_cell; idof++)
            {
                const unsigned int istate = fe_values_volume.get_fe().system_to_component_index(idof).first;
                soln_at_q += dg->solution[dof_indices[idof]]*fe_values_volume.shape_value_component(idof, iquad, istate);
            }
            solution_at_quad_pts.push_back(soln_at_q);
        }
            
    }
    std::cout<<"solution = [ "<<solution_at_quad_pts[0];
    for (unsigned int i=1; i<solution_at_quad_pts.size(); i++)
    {
        std::cout<<", "<<solution_at_quad_pts[i];
    }
    std::cout<<"]"<<std::endl;;

    std::cout<<"x_quad_pts = [ "<<x_quad_pts[0];
    for (unsigned int i=1; i<x_quad_pts.size(); i++)
    {
        std::cout<<", "<<x_quad_pts[i];
    }
    std::cout<<"]"<<std::endl;;
*/
    return 0;
}

#if PHILIP_DIM==1
    template class MeshRAdaptation<PHILIP_DIM,PHILIP_DIM>;
#endif
} // namespace Tests
} // namespace PHiLiP

