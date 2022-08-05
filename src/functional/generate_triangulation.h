#ifndef __GENERATE_TRIANGULATION__
#define __GENERATE_TRIANGULATION__

#include "total_derivatives_of_objective_function.h"

namespace PHiLiP {

#if PHILIP_DIM==1
template <int dim, int nstate, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, int nstate, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class GenerateTriangulation
{
    using VectorType = typename dealii::LinearAlgebra::distributed::Vector<real>;
public:
    GenerateTriangulation(VectorType &metric, unsigned int refinement_val, bool output_vertex_positions = false);

    std::shared_ptr<MeshType> triangulation;
};

} // namespace PHiLiP


#endif
