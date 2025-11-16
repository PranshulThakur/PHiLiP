#ifndef __CYLINDER_CHANNEL_H__
#define __CYLINDER_CHANNEL_H__

#include <deal.II/grid/manifold_lib.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/tria.h>

namespace PHiLiP {
namespace Grids {

/// Create a cylindricaly channel with an associated nonlinear manifold.
template<int dim>
void cylindrical_channel(
    dealii::parallel::distributed::Triangulation<dim> &grid,
    const unsigned int left_length,
    const unsigned int right_length,
    const unsigned int height_bottom,
    const unsigned int height_top,
    const double depth,
    unsigned int  depth_division,
    const double   shell_region_radius,
    const unsigned int  n_shells,
    const double skewness,
    const bool  use_transfinite_region,
    const unsigned int n_refinements);

template <int dim>
void dealiiuniform_channel_with_cylinder(
    dealii::Triangulation<dim>                &tria,
    const std::vector<unsigned int> &lengths_and_heights,
    const double                     depth = 1,
    unsigned int                     depth_division = 1,
    const double                     shell_region_radius = 0.75,
    const unsigned int               n_shells = 2,
    const double                     skewness = 2.0,
    const bool                       use_transfinite_region = false,
    const bool                       colorize = true);
    
template <int dim>
double minimal_vertex_distance(const dealii::Triangulation<dim> &triangulation);

} // namespace Grids
} // namespace PHiLiP
#endif
