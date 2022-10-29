#include <deal.II/grid/grid_generator.h>
#include "dg/dg_factory.hpp"
#include "physics/physics_factory.h"
#include <deal.II/numerics/vector_tools.h>
#include "functional/functional.h"
#include "parameters/all_parameters.h"
#include "linear_solver/linear_solver.h"
#include "optimization/design_parameterization/inner_vol_parameterization.hpp"
    
const int nstate = 1;
const int dim = PHILIP_DIM;

using namespace PHiLiP;   
using VectorType = typename dealii::LinearAlgebra::distributed::Vector<double>;
using MatrixType = dealii::TrilinosWrappers::SparseMatrix;
#if PHILIP_DIM == 1
    using MeshType = typename dealii::Triangulation<dim>;
#else
    using MeshType = typename dealii::parallel::distributed::Triangulation<dim>;
#endif


double get_functional_val(std::shared_ptr<Functional<dim,nstate,double,MeshType>> functional)
{
    const VectorType coarse_solution = functional->dg->solution;
//    functional->dg->change_cells_fe_degree_by_deltadegree_and_interpolate_solution(1);
    
    functional->dg->assemble_residual(true);
    VectorType delU(functional->dg->solution);
    solve_linear(functional->dg->system_matrix, functional->dg->right_hand_side, delU, functional->dg->all_parameters->linear_solver_param);
    delU *= -1.0;
    delU.update_ghost_values();

    functional->dg->solution += delU;
    functional->dg->solution.update_ghost_values();

    const double functional_val = functional->evaluate_functional();

//    functional->dg->change_cells_fe_degree_by_deltadegree_and_interpolate_solution(-1);
    functional->dg->solution = coarse_solution;
    functional->dg->solution.update_ghost_values();

    return functional_val;
}

void get_dIdX_analytical(std::shared_ptr<Functional<dim,nstate,double,MeshType>> functional, VectorType &dIdX)
{
    const VectorType coarse_solution = functional->dg->solution;
//    functional->dg->change_cells_fe_degree_by_deltadegree_and_interpolate_solution(1);
    
    functional->dg->assemble_residual(true);
    MatrixType R_u_transpose;
    R_u_transpose.copy_from(functional->dg->system_matrix_transpose);
    R_u_transpose.compress(dealii::VectorOperation::add);
    VectorType delU(functional->dg->solution);
    solve_linear(functional->dg->system_matrix, functional->dg->right_hand_side, delU, functional->dg->all_parameters->linear_solver_param);
    delU *= -1.0;
    delU.update_ghost_values();

    functional->dg->set_dual(delU);
    functional->dg->assemble_residual(false, false, true);
    MatrixType delU_times_R_ux;
    delU_times_R_ux.copy_from(functional->dg->d2RdWdX);
    delU_times_R_ux.compress(dealii::VectorOperation::add);

    functional->dg->assemble_residual(false, true);
    MatrixType R_x;
    R_x.copy_from(functional->dg->dRdXv);
    R_x.compress(dealii::VectorOperation::add);

    functional->dg->solution += delU;
    functional->dg->solution.update_ghost_values();

    functional->evaluate_functional(true, true);

    VectorType adjoint2(delU);
    solve_linear(R_u_transpose, functional->dIdw, adjoint2, functional->dg->all_parameters->linear_solver_param);
    adjoint2 *= -1.0;
    adjoint2.update_ghost_values();

    dIdX = functional->dIdX;

    R_x.Tvmult_add(dIdX, adjoint2);
    dIdX.update_ghost_values();
    delU_times_R_ux.Tvmult_add(dIdX, adjoint2);
    dIdX.update_ghost_values();
    
//    functional->dg->change_cells_fe_degree_by_deltadegree_and_interpolate_solution(-1);
    functional->dg->solution = coarse_solution;
    functional->dg->solution.update_ghost_values();
}

int main (int argc, char * argv[])
{
    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);    
    int mpi_rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    dealii::ConditionalOStream pcout(std::cout, mpi_rank==0);

    // Create grid and dg. 
    std::shared_ptr<MeshType> grid = std::make_shared<MeshType>(
            #if PHILIP_DIM != 1
            MPI_COMM_WORLD
            #endif
            );
    unsigned int grid_refinement_val = 3;
    dealii::GridGenerator::hyper_cube(*grid);
    grid->refine_global(grid_refinement_val);

    dealii::ParameterHandler parameter_handler; // Using default parameters. 
    Parameters::AllParameters::declare_parameters (parameter_handler);
    Parameters::AllParameters all_parameters;
    all_parameters.parse_parameters(parameter_handler);
    all_parameters.linear_solver_param.linear_residual = 1.0e-14;
    all_parameters.manufactured_convergence_study_param.manufactured_solution_param.use_manufactured_source_term = true;
    all_parameters.manufactured_convergence_study_param.manufactured_solution_param.manufactured_solution_type = Parameters::ManufacturedSolutionParam::ManufacturedSolutionType::poly_solution;
    all_parameters.pde_type = Parameters::AllParameters::PartialDifferentialEquation::burgers_viscous;

    const unsigned int poly_degree = 1;
    const unsigned int grid_degree = 1;

    std::shared_ptr < DGBase<dim, double> > dg = DGFactory<dim, double>::create_discontinuous_galerkin(&all_parameters, poly_degree,poly_degree + 1, grid_degree, grid);
    dg->allocate_system();
    pcout<<"Created and allocated DG."<<std::endl;

    std::shared_ptr <Physics::PhysicsBase<dim,nstate,double>> physics_double = Physics::PhysicsFactory<dim, nstate, double>::create_Physics(&all_parameters);
    pcout<<"Created physics double."<<std::endl;
    VectorType solution_no_ghost;
    solution_no_ghost.reinit(dg->locally_owned_dofs, MPI_COMM_WORLD);
    dealii::VectorTools::interpolate(dg->dof_handler, *(physics_double->manufactured_solution_function), solution_no_ghost);
    pcout<<"Interpolated solution."<<std::endl;
    dg->solution = solution_no_ghost;
    dg->solution.add(1.0);
    dg->solution.update_ghost_values();

    std::shared_ptr<Functional<dim,nstate,double,MeshType>> functional = FunctionalFactory<dim,nstate,double,MeshType>::create_Functional(dg->all_parameters->functional_param, dg);

    VectorType dIdX_fd(dg->high_order_grid->volume_nodes);

    double step_length = 1.0e-6;

    double original_val = get_functional_val(functional);
    
    const dealii::IndexSet vol_range = dg->high_order_grid->volume_nodes.get_partitioner()->locally_owned_range();
    
    for(unsigned int i = 0; i<dg->high_order_grid->volume_nodes.size(); ++i)
    {
        if(vol_range.is_element(i))
        {
            dg->high_order_grid->volume_nodes(i) += step_length;
            std::cout<<"Perturbed node."<<std::endl;
        }
        dg->high_order_grid->volume_nodes.update_ghost_values();

        double perturbed_val = get_functional_val(functional);

        if(vol_range.is_element(i))
        {
            dIdX_fd(i) = (perturbed_val - original_val)/step_length;
            dg->high_order_grid->volume_nodes(i) -= step_length; //reset
            std::cout<<"Reset node."<<std::endl;
        }
        dg->high_order_grid->volume_nodes.update_ghost_values();
    }
    dIdX_fd.update_ghost_values();
    
    VectorType dIdX_analytical;
    get_dIdX_analytical(functional, dIdX_analytical);
    pcout<<" dIdX_fd = "<<std::endl;
    dIdX_fd.print(std::cout, 3, true, false);
    pcout<<" dIdX_analytical = "<<std::endl;
    dIdX_analytical.print(std::cout, 3, true, false);

    VectorType diff = dIdX_analytical;
    diff -= dIdX_fd;
    diff.update_ghost_values();
    
    std::unique_ptr<BaseParameterization<dim>> design_parameterization = 
                        std::make_unique<InnerVolParameterization<dim>>(dg->high_order_grid);

    VectorType diff_inner_nodes;
    design_parameterization->initialize_design_variables(diff_inner_nodes); // get inner volume nodes
    pcout<<"Initialized design variables."<<std::endl;
    MatrixType dXv_dXp;
    design_parameterization->compute_dXv_dXp(dXv_dXp);
    pcout<<"Computed dXv_dXp."<<std::endl;

    dXv_dXp.Tvmult(diff_inner_nodes, diff);
    diff_inner_nodes.update_ghost_values();

    pcout<<"Analytical - FD dIdX = "<<diff_inner_nodes.l2_norm()<<std::endl;

    return 0;
}
