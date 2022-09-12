#ifndef __RESIDUAL_ERROR_ESTIMATE_H__
#define __RESIDUAL_ERROR_ESTIMATE_H__

#include "mesh_error_estimate_base.h"

namespace PHiLiP {

#if PHILIP_DIM==1
template <int dim, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
/// Class to compute residual based error
class ResidualErrorEstimate : public MeshErrorEstimateBase <dim, real, MeshType>
{

public:
    /// Computes maximum residual error in each cell.
    dealii::Vector<real> compute_cellwise_errors () override;

    /// Constructor
    ResidualErrorEstimate(std::shared_ptr<DGBase<dim,real,MeshType>> dg_input);

    /// Destructor
    ~ResidualErrorEstimate() {};

};

} // namespace PHiLiP

#endif
