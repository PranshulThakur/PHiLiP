#include "non_periodic_cube_flow.h"
#include <deal.II/grid/grid_generator.h>

namespace PHiLiP {
namespace FlowSolver {

template <int dim, int nstate>
NonPeriodicCubeFlow<dim, nstate>::NonPeriodicCubeFlow(const PHiLiP::Parameters::AllParameters *const parameters_input)
    : FlowSolverCaseBase<dim, nstate>(parameters_input)
{}

template <int dim, int nstate>
std::shared_ptr<MeshType> NonPeriodicCubeFlow<dim,nstate>::generate_grid() const
{
    std::shared_ptr<MeshType> grid = std::make_shared<MeshType> (
#if PHILIP_DIM!=1
    this->mpi_communicator
#endif
    ); // Mesh smoothing is set to none by default.
    
    const unsigned int number_of_refinements = this->all_param.flow_solver_param.number_of_mesh_refinements;
    const double domain_left = this->all_param.flow_solver_param.grid_left_bound;
    const double domain_right = this->all_param.flow_solver_param.grid_right_bound;
    const bool colorize = true;
    
    dealii::GridGenerator::hyper_cube(*grid, domain_left, domain_right, colorize);
    grid->refine_global(number_of_refinements);

    return grid;
}

template <int dim, int nstate>
void NonPeriodicCubeFlow<dim,nstate>::display_additional_flow_case_specific_parameters() const
{
    // Display nothing for now.
}

template class NonPeriodicCubeFlow<PHILIP_DIM, 1>;

} // FlowSolver namespace
} // PHiLiP namespace
