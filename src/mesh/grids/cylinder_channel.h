#ifndef __CYLINDER_CHANNEL_H__
#define __CYLINDER_CHANNEL_H__

#include <deal.II/grid/manifold_lib.h>
#include <deal.II/distributed/tria.h>

namespace PHiLiP {
namespace Grids {

/// Create a cylindricaly channel with an associated nonlinear manifold.

template<int dim>
void cylindrical_channel(
    dealii::parallel::distributed::Triangulation<dim> &grid,
    const std::vector<unsigned int> n_subdivisions,
    const double channel_length,
    const double channel_heigh,
    const double bump_height);
void uniform_channel_with_cylinder(
    dealii::Triangulation<3>                &tria,
    const std::vector<unsigned int> &lengths_and_heights,
    const double                     depth,
    unsigned int                     depth_division,
    const double                     shell_region_radius,
    const unsigned int               n_shells,
    const double                     skewness,
    const bool                       use_transfinite_region,
    const bool                       colorize);
void uniform_channel_with_cylinder(
    dealii::Triangulation<2>                &tria,
    const std::vector<unsigned int> &lengths_and_heights,
    const double,
    unsigned int,
    const double       shell_region_radius,
    const unsigned int n_shells,
    const double       skewness,
    const bool         use_transfinite_region,
    const bool         colorize);

}
#endif
