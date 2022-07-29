#ifndef __MESH_R_ADAPTATION__
#define __MESH_R_ADAPTATION__

#include "tests.h"
#include "parameters/all_parameters.h"

namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
class MeshRAdaptation: public TestsBase
{
public:
    MeshRAdaptation(const Parameters::AllParameters *const parameters_input);
    ~MeshRAdaptation(){};

    int run_test() const;
};


} // namespace Tests
} // namespace PHiLiP
#endif
