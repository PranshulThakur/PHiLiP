#include <stdlib.h>     /* srand, rand */
#include <iostream>

#include <deal.II/base/convergence_table.h>

#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/numerics/vector_tools.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/base/function.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/fe/mapping_manifold.h>

#include "dual_weighted_residual_mesh_adaptation.h"

#include "physics/initial_conditions/initial_condition.h"
#include "physics/manufactured_solution.h"
#include "dg/dg_factory.hpp"

#include "ode_solver/ode_solver_factory.h"
#include "mesh/mesh_adaptation.h"

namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
DualWeightedResidualMeshAdaptation<dim, nstate> :: DualWeightedResidualMeshAdaptation(const Parameters::AllParameters *const parameters_input)
    : TestsBase::TestsBase(parameters_input)
    {}


template <int dim, int nstate>
int DualWeightedResidualMeshAdaptation<dim, nstate> :: run_test () const
{
    const Parameters::AllParameters param = *(TestsBase::all_parameters);
    using ManParam = Parameters::ManufacturedConvergenceStudyParam;
    ManParam manu_grid_conv_param = param.manufactured_convergence_study_param;

    const unsigned int p_start             = manu_grid_conv_param.degree_start;
    const unsigned int p_end               = manu_grid_conv_param.degree_end;
    const unsigned int n_grids       = manu_grid_conv_param.number_of_grids;
    const unsigned int initial_grid_size           = manu_grid_conv_param.initial_grid_size;
    
    for (unsigned int poly_degree = p_start; poly_degree <= p_end; ++poly_degree)
    {
        for (unsigned int igrid=0; igrid<n_grids; ++igrid) 
        {
            // Create grid.
            using Triangulation = dealii::parallel::distributed::Triangulation<dim>;
            std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation>(
                 MPI_COMM_WORLD,
                 typename dealii::Triangulation<dim>::MeshSmoothing(
                 dealii::Triangulation<dim>::smoothing_on_refinement |
                 dealii::Triangulation<dim>::smoothing_on_coarsening));

            // Currently, the domain is [0,1]
            bool colorize = true;
            dealii::GridGenerator::hyper_cube(*grid, 0, 1, colorize);
            const int steps_to_create_grid = initial_grid_size + igrid;
            grid->refine_global(steps_to_create_grid);

            std::shared_ptr< DGBase<dim, double, Triangulation> > dg
                = DGFactory<dim,double,Triangulation>::create_discontinuous_galerkin(
                 &param,
                 poly_degree,
                 poly_degree+1,
                 poly_degree,
                 grid);



            dg->allocate_system();
            ZeroInitialCondition<dim,double> initial_conditions(nstate);
            const auto mapping = *(dg->high_order_grid->mapping_fe_field);
            dealii::VectorTools::interpolate(mapping, dg->dof_handler, initial_conditions, dg->solution);
   
   using VectorType       = typename dealii::LinearAlgebra::distributed::Vector<double>;
     using DoFHandlerType   = typename dealii::DoFHandler<dim>;
     using SolutionTransfer = typename MeshTypeHelper<Triangulation>::template SolutionTransfer<dim,VectorType,DoFHandlerType>;
    SolutionTransfer solution_transfer(dg->dof_handler);
     solution_transfer.prepare_for_coarsening_and_refinement(dg->solution);
 
     dg->high_order_grid->prepare_for_coarsening_and_refinement();
     dg->triangulation->prepare_coarsening_and_refinement();
 
     for (auto cell = dg->dof_handler.begin_active(); cell != dg->dof_handler.end(); ++cell)
         if (cell->is_locally_owned()) 
         {
            dealii::Point<dim> smallest_cell_coord = cell->center();
            if (smallest_cell_coord[0] > 0.5)
             cell->set_future_fe_index(cell->active_fe_index()+1);
        }
 
     dg->triangulation->execute_coarsening_and_refinement();
     dg->high_order_grid->execute_coarsening_and_refinement();
 
     dg->allocate_system();
     dg->solution.zero_out_ghosts();
 
         solution_transfer.interpolate(dg->solution);
     
     
     dg->solution.update_ghost_values();
             
            // generate ODE solver
            std::shared_ptr< ODE::ODESolverBase<dim,double,Triangulation> > ode_solver = ODE::ODESolverFactory<dim,double,Triangulation>::create_ODESolver(dg);

            std::cout<<"In loop"<<std::endl;
            ode_solver->steady_state();
            
            if (param.mesh_adaptation_param.total_refinement_steps > 0)
                 {
                    dealii::Point<dim> smallest_cell_coord = dg->high_order_grid->smallest_cell_coordinates();
                    pcout<<" x = "<<smallest_cell_coord[0]<<" y = "<<smallest_cell_coord[1]<<std::endl;
                    // Check if the mesh is refined near the shock i.e x,y in [0.3,0.7].
                    if ((smallest_cell_coord[0] > 0.3) && (smallest_cell_coord[0] < 0.7) && (smallest_cell_coord[1] > 0.3) && (smallest_cell_coord[1] < 0.7))
                    {
                        pcout<<"Mesh is refined near the shock. Test passed"<<std::endl;
                        return 0; // Mesh adaptation test passed.
                    }
                    else
                    {
                        pcout<<"Mesh Adaptation failed"<<std::endl;
                        return 1; // Mesh adaptation failed.
                    }
                 }
        } // for loop of igrid
    } // loop of poly_degree

    return 0; // Mesh adaptation test passed.
    
}

#if PHILIP_DIM!=1
template class DualWeightedResidualMeshAdaptation <PHILIP_DIM, 1>;
template class DualWeightedResidualMeshAdaptation <PHILIP_DIM, 2>;
template class DualWeightedResidualMeshAdaptation <PHILIP_DIM, 3>;
template class DualWeightedResidualMeshAdaptation <PHILIP_DIM, 4>;
template class DualWeightedResidualMeshAdaptation <PHILIP_DIM, 5>;
#endif

} // namespace Tests
} // namespace PHiLiP 
    
