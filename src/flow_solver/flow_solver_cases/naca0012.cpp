#include "naca0012.h"

#include <deal.II/base/function.h>
#include <deal.II/base/table_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <stdlib.h>
#include <deal.II/grid/grid_out.h>
#include <iostream>

#include "dg/dg_base.hpp"
#include "functional/lift_drag.hpp"
#include "mesh/gmsh_reader.hpp"
#include "mesh/grids/naca_airfoil_grid.hpp"
#include "mesh/grids/cylinder_channel.h"
#include "physics/physics_factory.h"

namespace PHiLiP {
namespace FlowSolver {
//=========================================================
// NACA0012
//=========================================================
template <int dim, int nstate>
NACA0012<dim, nstate>::NACA0012(const PHiLiP::Parameters::AllParameters *const parameters_input)
        : FlowSolverCaseBase<dim, nstate>(parameters_input)
        , unsteady_data_table_filename_with_extension(this->all_param.flow_solver_param.unsteady_data_table_filename+".txt")
{}

template <int dim, int nstate>
void NACA0012<dim,nstate>::display_additional_flow_case_specific_parameters() const
{
}

template <int dim, int nstate>
std::shared_ptr<Triangulation> NACA0012<dim,nstate>::generate_grid() const
{
    std::cout<<"Generating grid"<<std::endl;
    std::shared_ptr <Triangulation> grid = std::make_shared<Triangulation>(
    #if PHILIP_DIM!=1
    this->mpi_communicator,
    #endif
    typename dealii::Triangulation<dim>::MeshSmoothing(
        dealii::Triangulation<dim>::smoothing_on_refinement |
    dealii::Triangulation<dim>::smoothing_on_coarsening));

    dealii::Point<dim> p1;
    dealii::Point<dim> p2;
    for(unsigned int d=0; d<dim; ++d)
    {
        p1[d] = 0.0;
        p2[d] = 128.0;
    }
    dealii::GridGenerator::hyper_rectangle	( *grid, p1, p2,true );
    grid->refine_global(this->all_param.flow_solver_param.number_of_mesh_refinements);
        
    return grid;
}

template <int dim, int nstate>
void NACA0012<dim,nstate>::set_higher_order_grid(std::shared_ptr<DGBase<dim, double>> dg) const
{
    (void) dg;
}

template <int dim, int nstate>
double NACA0012<dim,nstate>::compute_lift(std::shared_ptr<DGBase<dim, double>> dg) const
{
    (void) dg;
    return 0.0;
}

template <int dim, int nstate>
double NACA0012<dim,nstate>::compute_drag(std::shared_ptr<DGBase<dim, double>> dg) const
{
    (void) dg;
    return 0.0;
}

template <int dim, int nstate>
void NACA0012<dim,nstate>::steady_state_postprocessing(std::shared_ptr<DGBase<dim, double>> dg) const
{
    const double lift = this->compute_lift(dg);
    const double drag = this->compute_drag(dg);

    this->pcout << " Resulting lift : " << lift << std::endl;
    this->pcout << " Resulting drag : " << drag << std::endl;
}

template <int dim, int nstate>
void NACA0012<dim, nstate>::compute_unsteady_data_and_write_to_table(
        const unsigned int current_iteration,
        const double current_time,
        const std::shared_ptr <DGBase<dim, double>> dg,
        const std::shared_ptr <dealii::TableHandler> unsteady_data_table)
{
    // Compute aerodynamic values
    const double lift = 0.0; //this->compute_lift(dg);
    const double drag = 0.0;
    (void) dg;

    if(this->mpi_rank==0) {
        // Add values to data table
        this->add_value_to_data_table(current_time,"time",unsteady_data_table);
        this->add_value_to_data_table(lift,"lift",unsteady_data_table);
        this->add_value_to_data_table(drag,"drag",unsteady_data_table);
        // Write to file
        std::ofstream unsteady_data_table_file(this->unsteady_data_table_filename_with_extension);
        unsteady_data_table->write_text(unsteady_data_table_file);
    }
    // Print to console
    this->pcout << "    Iter: " << current_iteration
                << "    Time: " << current_time
                << "    Lift: " << lift
                << "    Drag: " << drag;
    this->pcout << std::endl;

    // Abort if energy is nan
    if(std::isnan(lift) || std::isnan(drag)) {
        this->pcout << " ERROR: Lift or drag at time " << current_time << " is nan." << std::endl;
        this->pcout << "        Consider decreasing the time step / CFL number." << std::endl;
        std::abort();
    }
}

#if PHILIP_DIM==1
    template class NACA0012<PHILIP_DIM,PHILIP_DIM>;
#endif
#if PHILIP_DIM!=1
    template class NACA0012<PHILIP_DIM,PHILIP_DIM+2>;
#endif

} // FlowSolver namespace
} // PHiLiP namespace

