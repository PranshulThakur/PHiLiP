#include <fstream>
#include <iostream>
#include "Sacado.hpp"
#include <deal.II/lac/full_matrix.h>
#include "dg/dg_factory.hpp"
#include "parameters/parameters.h"
#include "physics/physics_factory.h"
#include "functional/objective_function_for_mesh_adaptation.h"
#include <deal.II/grid/grid_generator.h>

using PDEType   = PHiLiP::Parameters::AllParameters::PartialDifferentialEquation;
#if PHILIP_DIM==1
    using Triangulation = dealii::Triangulation<PHILIP_DIM>;
#else
    using Triangulation = dealii::parallel::distributed::Triangulation<PHILIP_DIM>;
#endif


template <typename real>
real func1(real &x1)
{
    return 2.0*x1*x1;
}

template <typename real>
real func(real &x1, real &x2, real &x3)
{
    real f = func1(x1) - 3*x2*x2 + 4*x1*x2 + pow((x3+2.0),2.0) + 4.0*x1;
    return f;
}


int main (int argc, char* argv[])
{
    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    int mpi_rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    dealii::ConditionalOStream pcout(std::cout, mpi_rank==0);

    using namespace PHiLiP;

                    std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation>(
                    #if PHILIP_DIM!=1
                    MPI_COMM_WORLD,
                    #endif
                    typename dealii::Triangulation<PHILIP_DIM>::MeshSmoothing(
                    dealii::Triangulation<PHILIP_DIM>::smoothing_on_refinement |
                    dealii::Triangulation<PHILIP_DIM>::smoothing_on_coarsening));

    unsigned int grid_refinement_val = 1;
    dealii::GridGenerator::hyper_cube(*grid);
    grid->refine_global(grid_refinement_val);


    dealii::ParameterHandler parameter_handler;
    Parameters::AllParameters::declare_parameters (parameter_handler);
    Parameters::AllParameters all_parameters;
    all_parameters.parse_parameters (parameter_handler);
    const unsigned int poly_degree = 2;

    std::shared_ptr < DGBase<PHILIP_DIM, double> > dg = DGFactory<PHILIP_DIM,double>::create_discontinuous_galerkin(&all_parameters, poly_degree,poly_degree,1, grid);
    dg->allocate_system();
    dealii::LinearAlgebra::distributed::Vector<double> solution_fine(dg->n_dofs()), solution_tilde(dg->n_dofs());
    std::cout<<"Active cells = "<<grid->n_active_cells()<<std::endl;
    std::cout<<"N_dofs = "<<dg->n_dofs()<<std::endl;

    for(unsigned int idof=0; idof < dg->n_dofs(); idof++)
    {
        solution_fine[idof] = idof*idof + 3.0;
        solution_tilde[idof] = idof*idof*idof + 4.0;
    }

    std::cout<<"Solution fine = "<<std::endl;
    solution_fine.print(std::cout, 3, true, false);

    ObjectiveFunctionMeshAdaptation<PHILIP_DIM,PHILIP_DIM,double,Triangulation> objfunc(dg, solution_fine, solution_tilde);
    //objfunc.evaluate_objective_function_hessian();
    double val = objfunc.evaluate_objective_function_and_derivatives();

    std::cout<<"derivative wrt solution fine = "<<std::endl;
    objfunc.derivative_objfunc_wrt_solution_fine.print(std::cout, 3, true, false);

    std::cout<<"derivative wrt solution tilde = "<<std::endl;
    objfunc.derivative_objfunc_wrt_solution_tilde.print(std::cout, 3, true, false);

    std::cout<<"derivative wrt metric nodes = "<<std::endl;
    objfunc.derivative_objfunc_wrt_metric_nodes.print(std::cout, 3, true, false);

    std::cout<<"Objective function value = "<<val<<std::endl;

       
/*
    using FadType = Sacado::Fad::DFad<double>;
    using FadFadType = Sacado::Fad::DFad<FadType>;
    unsigned int n_independent_variables = 3;
    std::vector<FadFadType> x_variables(n_independent_variables);

    // Set up AD variables
    unsigned int i_variable = 0;
    for(int i=0; i<3; i++)
    {
        x_variables[i] = (i+10)*(i+3); // set value
        x_variables[i].diff(i_variable++, n_independent_variables);
    }
   
    i_variable = 0;
    for(int i=0; i<3; i++)
    {
        x_variables[i].val() = (i+10)*(i+3); 
        x_variables[i].val().diff(i_variable++, n_independent_variables);
    }

    // Compute function
    FadFadType function_value = func(x_variables[0], x_variables[1], x_variables[2]);
    double function_val_double = function_value.val().val();
    std::cout<<"Function value = "<<function_val_double<<std::endl;

    // Compute gradient
    std::vector<double> gradient(n_independent_variables);
    i_variable = 0;
    for(unsigned int i=0; i<n_independent_variables; i++)
    {
        gradient[i] = function_value.dx(i_variable++).val();
    }
    std::cout<<"Gradient = "<<std::endl;
    for(unsigned int i=0; i<n_independent_variables; i++)
    {
        std::cout<<gradient[i]<<std::endl;
    }

    // Compute Hessian
    dealii::FullMatrix<double> Hessian(n_independent_variables, n_independent_variables);

    i_variable = 0;
    for(unsigned int i =0; i<n_independent_variables; i++)
    {
        const FadType df_dxi = function_value.dx(i_variable++);

        unsigned int j_derivative = 0;
        for(unsigned int j = 0; j < n_independent_variables; j++)
        {
            double d2f_dxidxj = df_dxi.dx(j_derivative++);
            Hessian(i,j) = d2f_dxidxj;
        }

    }

    std::cout<<std::endl<<std::endl;
    for(unsigned int i = 0; i<n_independent_variables; i++)
    {
        for(unsigned int j=0; j<n_independent_variables; j++)
        {
            std::cout<<Hessian(i,j)<<"  ";
        }
        std::cout<<std::endl;
    }
*/
    return 0;

}
