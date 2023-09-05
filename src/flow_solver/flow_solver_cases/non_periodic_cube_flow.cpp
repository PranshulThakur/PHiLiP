#include "non_periodic_cube_flow.h"
#include <deal.II/grid/grid_generator.h>
#include "mesh/gmsh_reader.hpp"

namespace PHiLiP {
namespace FlowSolver {

template <int dim, int nstate>
NonPeriodicCubeFlow<dim, nstate>::NonPeriodicCubeFlow(const PHiLiP::Parameters::AllParameters *const parameters_input)
    : FlowSolverCaseBase<dim, nstate>(parameters_input)
{}

template <int dim, int nstate>
std::shared_ptr<Triangulation> NonPeriodicCubeFlow<dim,nstate>::generate_grid() const
{
    std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation> (this->mpi_communicator); // Mesh smoothing is set to none by default.
    
    const unsigned int number_of_refinements = this->all_param.flow_solver_param.number_of_mesh_refinements;
    const double domain_left = this->all_param.flow_solver_param.grid_left_bound;
    const double domain_right = this->all_param.flow_solver_param.grid_right_bound;
    const bool colorize = true;
    
    dealii::GridGenerator::hyper_cube(*grid, domain_left, domain_right, colorize);
    grid->refine_global(number_of_refinements);

    return grid;
}

template <int dim, int nstate>
void NonPeriodicCubeFlow<dim,nstate>::set_higher_order_grid(std::shared_ptr<DGBase<dim, double>> dg) const
{
    const std::string mesh_filename = this->all_param.flow_solver_param.input_mesh_filename+std::string(".msh");
    const bool use_mesh_smoothing = false;
    std::shared_ptr<HighOrderGrid<dim,double>> mesh_high_order = read_gmsh<dim, dim> (mesh_filename, 0, use_mesh_smoothing);
    dg->set_high_order_grid(mesh_high_order);
}

template <int dim, int nstate>
void NonPeriodicCubeFlow<dim,nstate>::display_additional_flow_case_specific_parameters() const
{
    // Do nothing for now.
}

#if PHILIP_DIM==2
    template class NonPeriodicCubeFlow<PHILIP_DIM, 1>;
    template class NonPeriodicCubeFlow<PHILIP_DIM, 2>;
#endif
} // FlowSolver namespace
} // PHiLiP namespace
