#ifndef __GOAL_ORIENTED_MESH_OPTIMIZATION_H__ 
#define __GOAL_ORIENTED_MESH_OPTIMIZATION_H__ 

#include "tests.h"
#include "dg/dg.h"
#include "physics/physics.h"
#include "parameters/all_parameters.h"

namespace PHiLiP {
namespace Tests {

/// Test to check reduced and full space goal oriented mesh optimization.
template <int dim, int nstate>
class GoalOrientedMeshOptimization : public TestsBase
{
public:
    /// Constructor of GoalOrientedMeshOptimization.
    GoalOrientedMeshOptimization(const Parameters::AllParameters *const parameters_input,
                                       const dealii::ParameterHandler &parameter_handler_input);
    
    /// Parameter handler.
    const dealii::ParameterHandler &parameter_handler;

    /// Runs the test of mesh optimization.
    int run_test() const;
}; 

} // Tests namespace
} // PHiLiP namespace

#endif

