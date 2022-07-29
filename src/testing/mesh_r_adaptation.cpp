#include "mesh_r_adaptation.h"

namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
MeshRAdaptation<dim, nstate>::MeshRAdaptation(const Parameters::AllParameters *const parameters_input)
: TestsBase::TestsBase(parameters_input)
{}

template <int dim, int nstate>
int MeshRAdaptation<dim, nstate>::run_test() const
{
    return 0;
}

#if PHILIP_DIM==1
    template class MeshRAdaptation<PHILIP_DIM,PHILIP_DIM>;
#endif
} // namespace Tests
} // namespace PHiLiP

