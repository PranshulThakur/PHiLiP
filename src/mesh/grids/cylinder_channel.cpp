#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include "cylinder_channel.h"

namespace PHiLiP {
namespace Grids {

template<int dim>
void cylindrical_channel(
    dealii::parallel::distributed::Triangulation<dim> &grid,
    const unsigned int left_length,
    const unsigned int right_length,
    const unsigned int height_bottom,
    const unsigned int height_top,
    const double  depth,
    unsigned int  depth_division, //4
    const double  shell_region_radius,
    const unsigned int   n_shells, //14
    const double         skewness, //2.75
    const bool           use_transfinite_region,
    const unsigned int 	n_cells_per_shell, //60
    const unsigned int n_refinements)
{
/*
//=====================================================================================================================
    // Grid for testing the lid-driven cavity test case
    dealii::Point<dim> p1;
    dealii::Point<dim> p2;
    for(unsigned int d=0; d<dim; ++d)
    {
        p1[d] = -1.0;
        p2[d] = 1.0;
    }
    dealii::GridGenerator::hyper_rectangle	( grid, p1, p2,true );
    for (typename dealii::parallel::distributed::Triangulation<dim>::active_cell_iterator cell = grid.begin_active(); cell != grid.end(); ++cell) {
        for (unsigned int face=0; face<dealii::GeometryInfo<dim>::faces_per_cell; ++face) {
            if (cell->face(face)->at_boundary()) {
                unsigned int current_id = cell->face(face)->boundary_id();
                if (current_id == 3 ) {cell->face(face)->set_boundary_id (1010);} // top
                else {cell->face(face)->set_boundary_id (1001);}
            }
        }
    }
    grid.refine_global(n_refinements);
*/

/*
//=====================================================================================================================
    // Grid for testing the p+1 convergence orders of wall BC
    dealii::Point<dim> p1;
    dealii::Point<dim> p2;
    p1[0] = -2.0; p1[1] = -1.0;
    p2[0] = 2.0; p2[1] = 1.0;
    const double pi = 3.141592653589793238462643383279502884e+00;
    std::vector<unsigned int> repetitions(2);
    repetitions[0] = 2;
    repetitions[1] = 1;
    dealii::GridGenerator::subdivided_hyper_rectangle	( grid, repetitions, p1, p2,true );
    
    for (typename dealii::parallel::distributed::Triangulation<dim>::active_cell_iterator cell = grid.begin_active(); cell != grid.end(); ++cell) {
        for (unsigned int face=0; face<dealii::GeometryInfo<dim>::faces_per_cell; ++face) {
            if (cell->face(face)->at_boundary()) {
                unsigned int current_id = cell->face(face)->boundary_id();
                if (current_id == 2 || current_id == 3 ) cell->face(face)->set_boundary_id (1001); // top and bottom
            }
        }
    }
    std::vector<dealii::GridTools::PeriodicFacePair<typename dealii::Triangulation<dim>::cell_iterator> > matched_pairs;
    dealii::GridTools::collect_periodic_faces(grid,0,1,0,matched_pairs);
    grid.add_periodicity(matched_pairs);
    grid.refine_global(n_refinements);
    std::vector<double> y_expected(pow(2,n_refinements)-1);
    for(unsigned int i=0; i<y_expected.size(); ++i)
    {
        y_expected[i] = -1.0 + 2.0/(y_expected.size()+1)*(i+1.0);
    }
    for (typename dealii::parallel::distributed::Triangulation<dim>::active_cell_iterator cell = grid.begin_active(); cell != grid.end(); ++cell) {
        for(unsigned int ivertex = 0; ivertex < 4; ++ivertex)
        {
            for(unsigned int i=0; i<y_expected.size(); ++i)
            {
                if(abs(cell->vertex(ivertex)[1] - y_expected[i])<1.0e-6)
                {
                    cell->vertex(ivertex)[1] = y_expected[i] + 0.25*sin(pi*y_expected[i])+1.0e-5; 
                }
            }
        }
    }
//=====================================================================================================================
*/

        dealii::Triangulation<2> tria_2;
        dealii::Point<2> center;
        for(unsigned int d=0; d<2; ++d)
        {
            center[d] = 0.0;
        }

        const double inner_radius = 0.5;
        const double outer_radius = 30.0;
        const unsigned int N_shells = n_shells; //14
        const double Skewness = skewness; //2.75
        const bool 	colorize = true;

        dealii::GridGenerator::concentric_hyper_shells(tria_2, center, inner_radius, outer_radius,N_shells, Skewness, n_cells_per_shell, colorize);
    if constexpr (dim==3)
    {
        dealii::Triangulation<3> tria;
        // extrude to 3d
        dealii::GridGenerator::extrude_triangulation(tria_2, depth_division, 2.0, tria, true);
        // set up the new 3d manifolds
        const dealii::types::manifold_id      cylindrical_manifold_id = 0;
        const dealii::types::manifold_id      tfi_manifold_id         = 1;
        const dealii::PolarManifold<2> *const m_ptr =
          dynamic_cast<const dealii::PolarManifold<2> *>(
            &tria_2.get_manifold(cylindrical_manifold_id));
        Assert(m_ptr != nullptr, dealii::ExcInternalError());
        const dealii::Point<3>     axial_point(m_ptr->center[0],
                                   m_ptr->center[1],
                                   0.0);
        const dealii::Tensor<1, 3> direction{{0.0, 0.0, 1.0}};


        tria.set_manifold(cylindrical_manifold_id, dealii::FlatManifold<3>());
        tria.set_manifold(tfi_manifold_id, dealii::FlatManifold<3>());
        const dealii::CylindricalManifold<3> cylindrical_manifold(direction, axial_point);


        tria.set_manifold(cylindrical_manifold_id, cylindrical_manifold);
    
        grid.copy_triangulation(tria);

        // Set boundary ids
        const unsigned int boundary_id_wall = 1001;
        const unsigned int boundary_id_reimann = 1004;
        
        for (typename dealii::parallel::distributed::Triangulation<dim>::active_cell_iterator cell = grid.begin_active(); cell != grid.end(); ++cell) {
            for (unsigned int face=0; face<dealii::GeometryInfo<dim>::faces_per_cell; ++face) {
                if (cell->face(face)->at_boundary()) {
                    unsigned int current_id = cell->face(face)->boundary_id();
                    if (current_id == 1) cell->face(face)->set_boundary_id (boundary_id_reimann); // outer
                    if (current_id == 0) cell->face(face)->set_boundary_id (boundary_id_wall); // Cylindrical wall
                }
            }
        }
        std::vector<dealii::GridTools::PeriodicFacePair<typename dealii::Triangulation<dim>::cell_iterator> > matched_pairs;
        dealii::GridTools::collect_periodic_faces(grid,2,3,2,matched_pairs);
        grid.add_periodicity(matched_pairs);
     }   

/*
//=====================================================================================================================
    // Grid for cylindrical channel
    dealii::Triangulation<dim> grid_serial;

    std::vector<unsigned int> lengths_and_heights(4);
    lengths_and_heights[0] = left_length;
    lengths_and_heights[1] = right_length;
    lengths_and_heights[2] = height_bottom;
    lengths_and_heights[3] = height_top;

    dealiiuniform_channel_with_cylinder(grid_serial, lengths_and_heights, depth, depth_division, shell_region_radius, n_shells, skewness, use_transfinite_region, true);

    grid.copy_triangulation(grid_serial);

    // Set boundary ids
    const unsigned int boundary_id_wall = 1001;
    const unsigned int boundary_id_reimann = 1004;
    
    for (typename dealii::parallel::distributed::Triangulation<dim>::active_cell_iterator cell = grid.begin_active(); cell != grid.end(); ++cell) {
        for (unsigned int face=0; face<dealii::GeometryInfo<dim>::faces_per_cell; ++face) {
            if (cell->face(face)->at_boundary()) {
                unsigned int current_id = cell->face(face)->boundary_id();
                if (current_id == 0 || current_id == 3 || current_id == 4) cell->face(face)->set_boundary_id (boundary_id_reimann); // left, top and bottom
                if (current_id == 2) cell->face(face)->set_boundary_id (boundary_id_wall); // Cylindrical wall
                if (current_id == 1) cell->face(face)->set_boundary_id (boundary_id_reimann); // right
            }
        }
    }
    if constexpr(dim==3)
    {
        std::vector<dealii::GridTools::PeriodicFacePair<typename dealii::Triangulation<dim>::cell_iterator> > matched_pairs;
        dealii::GridTools::collect_periodic_faces(grid,5,6,2,matched_pairs);
        grid.add_periodicity(matched_pairs);
    }
    grid.refine_global(n_refinements);
//=====================================================================================================================
*/
    (void) left_length;
    (void) right_length;
    (void) height_bottom;
    (void) height_top;
    (void)  depth;
    (void)  depth_division;
    (void)  shell_region_radius;
    (void)   n_shells;
    (void)         skewness;
    (void)           use_transfinite_region;
    (void) n_refinements;
}

template <>
void dealiiuniform_channel_with_cylinder<2>(
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
      dealii::Point<2>(-double(length_pre), -double(height_below)),
      dealii::Point<2>(double(length_post), double(height_above)));


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
        if ((cell->center() - dealii::Point<2>(0., 0.)).norm() < 1.1 * box_radius)
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
        for (const unsigned int face_n : dealii::GeometryInfo<2>::face_indices())
          if (!cell->face(face_n)->at_boundary())
            cell->face(face_n)->set_manifold_id(tfi_manifold_id);
      }


    // The shell region should have a radius that is larger than the radius of
    // the cylinder
    if (radius < shell_region_radius)
      {
        Assert(0 < n_shells,
               dealii::ExcMessage("If the shell region has positive width then "
                          "there must be at least one shell."));
        dealii::Triangulation<2> shell_tria;
        dealii::GridGenerator::concentric_hyper_shells(shell_tria,
                                               dealii::Point<2>(),
                                               radius,
                                               shell_region_radius,
                                               n_shells,
                                               skewness,
                                               8);


        // Make the tolerance as large as possible since these cells can
        // be quite close together
        const double vertex_tolerance =
          std::min(minimal_vertex_distance(shell_tria),
                   minimal_vertex_distance(cylinder_tria)) *
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
      std::min(minimal_vertex_distance(tria_without_cylinder),
               minimal_vertex_distance(cylinder_tria)) /
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
        cell->set_all_manifold_ids(dealii::numbers::flat_manifold_id);


    // attach manifolds
    dealii::PolarManifold<2> polar_manifold(dealii::Point<2>(0., 0.));
    tria.set_manifold(polar_manifold_id, polar_manifold);


    if (use_transfinite_region)
      {
        tria.set_manifold(tfi_manifold_id, dealii::FlatManifold<2>());
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

template <>
  void dealiiuniform_channel_with_cylinder<3>(
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
    dealii::Triangulation<2> tria_2;
    dealiiuniform_channel_with_cylinder(tria_2,
                                  lengths_and_heights,
                                  depth,
                                  depth_division,
                                  shell_region_radius,
                                  n_shells,
                                  skewness,
                                  use_transfinite_region,
                                  colorize);


    // extrude to 3d
    dealii::GridGenerator::extrude_triangulation(tria_2, depth_division, depth, tria, true);


    // set up the new 3d manifolds
    const dealii::types::manifold_id      cylindrical_manifold_id = 0;
    const dealii::types::manifold_id      tfi_manifold_id         = 1;
    const dealii::PolarManifold<2> *const m_ptr =
      dynamic_cast<const dealii::PolarManifold<2> *>(
        &tria_2.get_manifold(cylindrical_manifold_id));
    Assert(m_ptr != nullptr, dealii::ExcInternalError());
    const dealii::Point<3>     axial_point(m_ptr->center[0],
                               m_ptr->center[1],
                               0.0);
    const dealii::Tensor<1, 3> direction{{0.0, 0.0, 1.0}};


    tria.set_manifold(cylindrical_manifold_id, dealii::FlatManifold<3>());
    tria.set_manifold(tfi_manifold_id, dealii::FlatManifold<3>());
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

    template <int dim>
    double
    minimal_vertex_distance(const dealii::Triangulation<dim> &triangulation)
    {
      double length = std::numeric_limits<double>::max();
      for (const auto &cell : triangulation.active_cell_iterators())
        for (unsigned int n = 0; n < dealii::GeometryInfo<dim>::lines_per_cell; ++n)
          length = std::min(length, cell->line(n)->diameter());
      return length;
    }


#if PHILIP_DIM != 1
template void cylindrical_channel<PHILIP_DIM>(
    dealii::parallel::distributed::Triangulation<PHILIP_DIM> &grid,
    const unsigned int left_length,
    const unsigned int right_length,
    const unsigned int height_bottom,
    const unsigned int height_top,
    const double depth,
    unsigned int depth_division,
    const double shell_region_radius,
    const unsigned int  n_shells,
    const double skewness,
    const bool  use_transfinite_region,
    const unsigned int 	n_cells_per_shell, //60
    const unsigned int n_refinements);

template void dealiiuniform_channel_with_cylinder<PHILIP_DIM>(
    dealii::Triangulation<PHILIP_DIM>                &tria,
    const std::vector<unsigned int> &lengths_and_heights,
    const double                     depth,
    unsigned int                     depth_division,
    const double                     shell_region_radius,
    const unsigned int               n_shells,
    const double                     skewness,
    const bool                       use_transfinite_region,
    const bool                       colorize);
    
template double  minimal_vertex_distance<PHILIP_DIM>(const dealii::Triangulation<PHILIP_DIM> &triangulation);
#endif

} // namespace Grids
} // namespace PHiLiP
