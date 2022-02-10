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
 #include "physics/physics_factory.h"
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
std::shared_ptr< Physics::PhysicsBase<dim,nstate,double> > physics_double
         = Physics::PhysicsFactory<dim,nstate,double>::create_Physics(&param);
    const unsigned int p_start             = manu_grid_conv_param.degree_start;
    const unsigned int p_end               = manu_grid_conv_param.degree_end;
    const unsigned int n_grids       = manu_grid_conv_param.number_of_grids;
    const unsigned int initial_grid_size           = manu_grid_conv_param.initial_grid_size;
    const unsigned int m_degree = 0;
    double global_point[dim];
    double dx;
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
                 poly_degree+m_degree,
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
             cell->set_future_fe_index(cell->active_fe_index()+m_degree);
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
            
            //compute errors starts 
    //       int overintegrate = 10;
     dealii::hp::MappingCollection<dim> mapping_collection(mapping);
             dealii::hp::FEValues<dim,dim> fe_values_extra_net(mapping_collection, dg->fe_collection, dg->volume_quadrature_collection,
                   dealii::update_values | dealii::update_JxW_values | dealii::update_quadrature_points);
           //  const unsigned int n_quad_pts = fe_values_extra.n_quadrature_points;
             std::array<double,nstate> soln_at_q;

             double linf_norm = 0.0;
             double l2_norm = 0.0;

             dealii::Point<dim> coord_max_error;


             for(auto cell = dg->dof_handler.begin_active(); cell < dg->dof_handler.end(); ++cell){
                 if(!cell->is_locally_owned()) 
                    continue;

            // dealii::QGauss<dim> quad_extra(cell->active_fe_index()+overintegrate);
            // dealii::FEValues<dim,dim> fe_values_extra(*(dg->high_order_grid->mapping_fe_field), dg->fe_collection[cell->active_fe_index()], quad_extra,
             //        dealii::update_values | dealii::update_JxW_values | dealii::update_quadrature_points);
                 fe_values_extra_net.reinit(cell);
                 const dealii::FEValues<dim,dim> &fe_values_extra = fe_values_extra_net.get_present_fe_values();
             const unsigned int n_quad_pts = fe_values_extra.n_quadrature_points;
             std::vector<dealii::types::global_dof_index> dofs_indices(fe_values_extra.dofs_per_cell);

                 cell->get_dof_indices(dofs_indices);

                 double cell_l2error = 0.0;
                 std::array<double,nstate> cell_linf;
                 std::fill(cell_linf.begin(), cell_linf.end(), 0);

                 for(unsigned int iquad = 0; iquad < n_quad_pts; ++iquad){
                     std::fill(soln_at_q.begin(), soln_at_q.end(), 0);
                     for(unsigned int idof = 0; idof < fe_values_extra.dofs_per_cell; ++idof){
                         const unsigned int istate = fe_values_extra.get_fe().system_to_component_index(idof).first;
                         soln_at_q[istate] += dg->solution[dofs_indices[idof]] * fe_values_extra.shape_value_component(idof, iquad, istate);
                     }

                     const dealii::Point<dim> qpoint = (fe_values_extra.quadrature_point(iquad));

                     for(unsigned int istate = 0; istate < nstate; ++ istate){
                         const double uexact = physics_double->manufactured_solution_function->value(qpoint, istate);
                         cell_l2error += pow(soln_at_q[istate] - uexact, 2) * fe_values_extra.JxW(iquad);
                         if(cell_linf[istate] < abs(soln_at_q[istate]-uexact))
                        // cell_linf[istate] = std::max(cell_linf[istate], abs(soln_at_q[istate]-uexact));
                            cell_linf[istate] = abs(soln_at_q[istate]-uexact);
                     }
                 }


                 l2_norm += cell_l2error;
                 const double linf_norm_prev = linf_norm;
                 for(unsigned int istate = 0; istate < nstate; ++ istate){
                    if(linf_norm  < cell_linf[istate])
                     linf_norm = cell_linf[istate];
                 }


                 if(linf_norm_prev != linf_norm)
                 {
                    coord_max_error = cell->center();
                 }


             }
                 dealii::Utilities::MPI::MinMaxAvg minindexstore;
                 minindexstore = dealii::Utilities::MPI::min_max_avg(linf_norm, mpi_communicator);
                 int n_proc_max = minindexstore.max_index;

                  const int iproc = dealii::Utilities::MPI::this_mpi_process(mpi_communicator);
                 if(iproc == n_proc_max)
                 {
                    for (int i=0; i<dim; i++)
                    global_point[i] = coord_max_error[i];

                  }


                 MPI_Bcast(global_point, dim, MPI_DOUBLE, n_proc_max, mpi_communicator); // Update values in all processors
                dx = dealii::GridTools::maximal_cell_diameter(*grid);
             const double l2_norm_mpi = std::sqrt(dealii::Utilities::MPI::sum(l2_norm, mpi_communicator));
             const double linf_norm_mpi = dealii::Utilities::MPI::max(linf_norm, mpi_communicator);
             pcout<<std::endl;
             pcout<<"h = "<<dx<<std::endl;
            pcout<<"p_left = "<<poly_degree<<std::endl;
            pcout<<"p_right = "<<dg->get_max_fe_degree()<<std::endl;
            pcout<<"L2 Norm Error = "<<l2_norm_mpi<<std::endl;
            pcout<<"Linf Norm Error = "<<linf_norm_mpi<<std::endl;
            pcout<<"x_inf_max = "<<global_point[0]<<std::endl;
            pcout<<"y_inf_max = "<<global_point[1]<<std::endl;


             // Compute error ends
        } // for loop of igrid
    } // loop of poly_degree

    return 0; // Mesh adaptation test passed.
            
        /*    if (param.mesh_adaptation_param.total_refinement_steps > 0)
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
    */
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
    
