#include "generate_triangulation.h"
#include <deal.II/grid/grid_generator.h>
namespace PHiLiP {

template <int dim, int nstate, typename real, typename MeshType>
GenerateTriangulation<dim, nstate, real, MeshType>::GenerateTriangulation(dealii::Vector<real> &metric, unsigned int grid_refinement_val, bool output_vertex_positions)
{
    triangulation = std::make_shared<MeshType> ();
    dealii::GridGenerator::hyper_cube(*triangulation);
    triangulation->refine_global(grid_refinement_val);
        if(output_vertex_positions)
            std::cout<<std::endl<<"vertex_positions = [0, ";

    for (const auto &cell : triangulation->active_cell_iterators())
    {
        if(cell->active_cell_index() == metric.size())
            break;


        cell->vertex(1)[0] = metric(cell->active_cell_index());


        
        if(output_vertex_positions)
            std::cout<<cell->vertex(1)[0]<<", ";

    }
        if(output_vertex_positions)
            std::cout<<"1]"<<std::endl;


}

#if PHILIP_DIM == 1
template class GenerateTriangulation<PHILIP_DIM, 1, double, dealii::Triangulation<PHILIP_DIM>>;
template class GenerateTriangulation<PHILIP_DIM, 2, double, dealii::Triangulation<PHILIP_DIM>>;
template class GenerateTriangulation<PHILIP_DIM, 3, double, dealii::Triangulation<PHILIP_DIM>>;
template class GenerateTriangulation<PHILIP_DIM, 4, double, dealii::Triangulation<PHILIP_DIM>>;
template class GenerateTriangulation<PHILIP_DIM, 5, double, dealii::Triangulation<PHILIP_DIM>>;
#endif
/*
#if PHILIP_DIM != 1
template class GenerateTriangulation<PHILIP_DIM, 1, double,  dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class GenerateTriangulation<PHILIP_DIM, 2, double,  dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class GenerateTriangulation<PHILIP_DIM, 3, double,  dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class GenerateTriangulation<PHILIP_DIM, 4, double,  dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class GenerateTriangulation<PHILIP_DIM, 5, double,  dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif
*/
} // namespae PHiLiP
