#ifndef __NON_PERIODIC_CUBE_FLOW__
#define __NON_PERIODIC_CUBE_FLOW__

#include "flow_solver_case_base.h"

namespace PHiLiP {
namespace FlowSolver {

#if PHILIP_DIM==1
     using MeshType = dealii::Triangulation<PHILIP_DIM>;
#else
     using MeshType = dealii::parallel::distributed::Triangulation<PHILIP_DIM>;
#endif

template <int dim, int nstate>
class NonPeriodicCubeFlow : public FlowSolverCaseBase<dim, nstate>
{
 public:
     NonPeriodicCubeFlow(const Parameters::AllParameters *const parameters_input);
     
     ~NonPeriodicCubeFlow() {};
 
     std::shared_ptr<MeshType> generate_grid() const override;

     void display_additional_flow_case_specific_parameters() const override;
};

} // FlowSolver namespace
} // PHiLiP namespace

#endif

