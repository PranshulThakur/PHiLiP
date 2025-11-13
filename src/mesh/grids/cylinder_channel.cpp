#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include "cylinder_channel.h"

namespace PHiLiP {
namespace Grids {
  void uniform_channel_with_cylinder(
    dealii::Triangulation<3>                &tria,
    const std::vector<unsigned int> &lengths_and_heights,
    const double                     depth,
    unsigned int                     depth_division,
    const double                     shell_region_radius,
    const unsigned int               n_shells,
    const double                     skewness,
    const bool                       use_transfinite_region,
    const bool                       colorize)
  {
    using namespace dealii;
    dealii::Triangulation<2> tria_2;
    uniform_channel_with_cylinder(tria_2,
                                  lengths_and_heights,
                                  depth,
                                  depth_division,
                                  shell_region_radius,
                                  n_shells,
                                  skewness,
                                  use_transfinite_region,
                                  colorize);


    // extrude to 3d
    extrude_triangulation(tria_2, depth_division, depth, tria, true);


    // set up the new 3d manifolds
    const types::manifold_id      cylindrical_manifold_id = 0;
    const types::manifold_id      tfi_manifold_id         = 1;
    const dealii::PolarManifold<2> *const m_ptr =
      dynamic_cast<const dealii::PolarManifold<2> *>(
        &tria_2.get_manifold(cylindrical_manifold_id));
    Assert(m_ptr != nullptr, dealii::ExcInternalError());
    const dealii::Point<3>     axial_point(m_ptr->get_center()[0],
                               m_ptr->get_center()[1],
                               0.0);
    const dealii::Tensor<1, 3> direction{{0.0, 0.0, 1.0}};


    tria.set_manifold(cylindrical_manifold_id, FlatManifold<3>());
    tria.set_manifold(tfi_manifold_id, FlatManifold<3>());
    const dealii::CylindricalManifold<3> cylindrical_manifold(direction, axial_point);


    tria.set_manifold(cylindrical_manifold_id, cylindrical_manifold);


    if (use_transfinite_region)
      {
        dealii::TransfiniteInterpolationManifold<3> inner_manifold;
        inner_manifold.initialize(tria);
        tria.set_manifold(tfi_manifold_id, inner_manifold);
      }


    // From extrude_triangulation: since the maximum boundary id of tria_2 was
    // 4, the front boundary id is 4 and the back is 5. They remain unchanged.
  }
void uniform_channel_with_cylinder(
    dealii::Triangulation<2>                &tria,
    const std::vector<unsigned int> &lengths_and_heights,
    const double,
    unsigned int,
    const double       shell_region_radius,
    const unsigned int n_shells,
    const double       skewness,
    const bool         use_transfinite_region,
    const bool         colorize)
  {
    using namespace dealii;
    const dealii::types::manifold_id polar_manifold_id = 0;
    const dealii::types::manifold_id tfi_manifold_id   = 1;


    // The radius of the cylinder is 0.5, so the diameter is 1.
    const double radius     = 0.5;
    const double box_radius = 1;


    // We assume that the cylinder is centered at (0,0) and has a diameter of 1.
    // We use the cylinder diameter as the characteristic length of the channel.
    // The number of repetitions is chosen to ensure that the cylinder
    // occupies four cells.


    const unsigned int length_pre   = lengths_and_heights[0];
    const unsigned int length_post  = lengths_and_heights[1];
    const unsigned int height_below = lengths_and_heights[2];
    const unsigned int height_above = lengths_and_heights[3];


    const unsigned int length_repetitions = length_pre + length_post;
    const unsigned int height_repetitions = height_above + height_below;


    // We begin by setting up a grid that is length_repetition by
    // height_repetitions cells. These cells are all square
    dealii::Triangulation<2> bulk_tria;
    dealii::GridGenerator::subdivided_hyper_rectangle(
      bulk_tria,
      {(length_repetitions), height_repetitions},
      Point<2>(-double(length_pre), -double(height_below)),
      Point<2>(double(length_post), double(height_above)));


    // bulk_tria now looks like this:
    //
    //   +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
    //   |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
    //   +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
    //   |  |XX|XX|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
    //   +--+--O--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
    //   |  |XX|XX|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
    //   +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
    //   |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
    //   +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
    //
    // The next step is to remove the cells marked with XXs: we will place
    // the grid around the cylinder there later. The following loop determines
    // which cells need to be removed from the Triangulation
    //    (i.e., find the cells marked with XX in the picture).
    std::set<dealii::Triangulation<2>::active_cell_iterator> cells_to_remove;
    for (const auto &cell : bulk_tria.active_cell_iterators())
      {
        if ((cell->center() - Point<2>(0., 0.)).norm() < 1.1 * box_radius)
          cells_to_remove.insert(cell);
      }


    dealii::Triangulation<2> tria_without_cylinder;
    dealii::GridGenerator::create_triangulation_with_removed_cells(
      bulk_tria, cells_to_remove, tria_without_cylinder);


    // Set up the cylinder triangulation. Note that this function sets the
    // manifold ids of the interior boundary cells to 0
    // (polar_manifold_id).
    dealii::Triangulation<2> cylinder_tria;
    dealii::GridGenerator::hyper_cube_with_cylindrical_hole(cylinder_tria,
                                                    shell_region_radius,
                                                    box_radius);


    // Assign interior manifold ids to be the TFI id.
    for (const auto &cell : cylinder_tria.active_cell_iterators())
      {
        cell->set_manifold_id(tfi_manifold_id);
        for (const unsigned int face_n : GeometryInfo<2>::face_indices())
          if (!cell->face(face_n)->at_boundary())
            cell->face(face_n)->set_manifold_id(tfi_manifold_id);
      }


    // The shell region should have a radius that is larger than the radius of
    // the cylinder
    if (radius < shell_region_radius)
      {
        Assert(0 < n_shells,
               ExcMessage("If the shell region has positive width then "
                          "there must be at least one shell."));
        dealii::Triangulation<2> shell_tria;
        dealii::GridGenerator::concentric_hyper_shells(shell_tria,
                                               Point<2>(),
                                               radius,
                                               shell_region_radius,
                                               n_shells,
                                               skewness,
                                               8);


        // Make the tolerance as large as possible since these cells can
        // be quite close together
        const double vertex_tolerance =
          std::min(dealii::minimial_vertex_distance(shell_tria),
                   dealii::minimial_vertex_distance(cylinder_tria)) *
          0.5;


        shell_tria.set_all_manifold_ids(polar_manifold_id);
        dealii::Triangulation<2> temp;
        dealii::GridGenerator::merge_triangulations(
          shell_tria, cylinder_tria, temp, vertex_tolerance, true);
        cylinder_tria = std::move(temp);
      }


    // Compute the tolerance again, since the shells may be very close to
    // each-other:
    const double vertex_tolerance =
      std::min(dealii::minimial_vertex_distance(tria_without_cylinder),
               dealii::minimial_vertex_distance(cylinder_tria)) /
      10;


    dealii::GridGenerator::merge_triangulations(
      tria_without_cylinder, cylinder_tria, tria, vertex_tolerance, true);




    // Ensure that all manifold ids on a polar cell really are set to the
    // polar manifold id:
    for (const auto &cell : tria.active_cell_iterators())
      if (cell->manifold_id() == polar_manifold_id)
        cell->set_all_manifold_ids(polar_manifold_id);


    // Ensure that all other manifold ids (including the interior faces
    // opposite the cylinder) are set to the flat manifold id:
    for (const auto &cell : tria.active_cell_iterators())
      if (cell->manifold_id() != polar_manifold_id &&
          cell->manifold_id() != tfi_manifold_id)
        cell->set_all_manifold_ids(numbers::flat_manifold_id);


    // attach manifolds
    dealii::PolarManifold<2> polar_manifold(Point<2>(0., 0.));
    tria.set_manifold(polar_manifold_id, polar_manifold);


    if (use_transfinite_region)
      {
        tria.set_manifold(tfi_manifold_id, FlatManifold<2>());
        dealii::TransfiniteInterpolationManifold<2> inner_manifold;
        inner_manifold.initialize(tria);
        tria.set_manifold(tfi_manifold_id, inner_manifold);
      }


    if (colorize)
      for (const auto &face : tria.active_face_iterators())
        if (face->at_boundary())
          {
            const dealii::Point<2> center = face->center();
            // left side
            if (std::abs(center[0] - (-static_cast<double>(length_pre))) <
                1e-10)
              face->set_boundary_id(0);
            // right side
            else if (std::abs(center[0] - static_cast<double>(length_post)) <
                     1e-10)
              face->set_boundary_id(1);
            // cylinder boundary
            else if (face->manifold_id() == polar_manifold_id)
              face->set_boundary_id(2);
            // bottom side
            else if (std::abs(center[1] -
                              (-static_cast<double>(height_below))) < 1e-10)
              face->set_boundary_id(3);
            // top side
            else
              face->set_boundary_id(4);
          }
  }
















































}
}
