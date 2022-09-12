#ifndef __MESH_ERROR_ESTIMATE_BASE_H__
#define __MESH_ERROR_ESTIMATE_BASE_H__

#include "parameters/all_parameters.h"
#include "dg/dg.h"
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria.h>
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>

#include <vector>
#include <iostream>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/distributed/solution_transfer.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include "functional/functional.h"
#include "physics/physics.h"

namespace PHiLiP {

#if PHILIP_DIM==1
template <int dim, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif

/// Abstract class to estimate error for mesh adaptation
class MeshErrorEstimateBase
{

public:

    /// Computes the vector containing errors in each cell.
    virtual dealii::Vector<real> compute_cellwise_errors () = 0;

    /// Constructor
    MeshErrorEstimateBase(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input);

    /// Virtual Destructor
    virtual ~MeshErrorEstimateBase() = 0;

    /// Pointer to DGBase
    std::shared_ptr<DGBase<dim,real,MeshType>> dg;

};
} // Namespace PHiLiP
#endif
