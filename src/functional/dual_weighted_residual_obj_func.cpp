#include "dual_weighted_residual_obj_func.h"
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>

namespace PHiLiP {

template <int dim, int nstate, typename real>
DualWeightedResidualObjFunc<dim, nstate, real> :: DualWeightedResidualObjFunc( 
    std::shared_ptr<DGBase<dim,real>> dg_input,
    const bool uses_solution_values,
    const bool uses_solution_gradient)
    : Functional<dim, nstate, real> (dg_input, uses_solution_values, uses_solution_gradient)
{
    compute_cell_index_range();
    compute_interpolation_matrix();
}

template<int dim, int nstate, typename real>
void DualWeightedResidualObjFunc<dim, nstate, real> :: compute_cell_index_range()
{
    const unsigned int n_global_cells = this->dg->triangulation->n_global_active_cells();
    cell_index_range.set_size(n_global_cells);
    for(const auto &cell : this->dg->dof_handler.active_cell_iterators())
    {
        if(cell->is_locally_owned())
        {
            cell_index_range.add_index(cell->active_cell_index());
        }
    }
}

template<int dim, int nstate, typename real>
void DualWeightedResidualObjFunc<dim, nstate, real> :: compute_interpolation_matrix()
{ 
    vector_coarse = this->dg->solution; // copies values and parallel layout.
    unsigned int n_dofs_coarse = this->dg->n_dofs();
    this->dg->change_cells_fe_degree_by_deltadegree_and_interpolate_solution(1);
    vector_fine = this->dg->solution;
    unsigned int n_dofs_fine = this->dg->n_dofs();
    const dealii::IndexSet dofs_fine_locally_relevant_range = this->dg->locally_relevant_dofs;
    std::vector<std::vector<dealii::types::global_dof_index>> cellwise_dofs_fine = get_cellwise_dof_indices();
    this->dg->change_cells_fe_degree_by_deltadegree_and_interpolate_solution(-1);
    AssertDimension(vector_coarse.size(), this->dg->solution.size());     

    // Get all possible interpolation matrices for available poly order combinations.
    dealii::Table<2,dealii::FullMatrix<real>> interpolation_hp;
    extract_interpolation_matrices(interpolation_hp);

    // Get locally owned dofs
    const dealii::IndexSet &dofs_range_coarse = vector_coarse.get_partitioner()->locally_owned_range();
    const dealii::IndexSet &dofs_range_fine = vector_fine.get_partitioner()->locally_owned_range();

    dealii::DynamicSparsityPattern dsp(n_dofs_fine, n_dofs_coarse, dofs_range_fine);
    std::vector<dealii::types::global_dof_index> dof_indices;
    
    for(const auto &cell : this->dg->dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned()) continue;

        const dealii::types::global_dof_index cell_index = cell->active_cell_index();
        const unsigned int i_fele = cell->active_fe_index();
        const dealii::FESystem<dim,dim> &fe_ref = this->dg->fe_collection[i_fele];
        const unsigned int n_dofs_cell = fe_ref.n_dofs_per_cell();

        dof_indices.resize(n_dofs_cell);
        cell->get_dof_indices (dof_indices);
        const std::vector<dealii::types::global_dof_index> &dof_indices_fine = cellwise_dofs_fine[cell_index];

        for(unsigned int i=0; i < dof_indices_fine.size(); ++i)
        {
            for(unsigned int j=0; j < n_dofs_cell; ++j)
            {
                dsp.add(dof_indices_fine[i], dof_indices[j]);
            }
        }

    } // cell loop ends

    dealii::SparsityTools::distribute_sparsity_pattern(dsp, dofs_range_fine, MPI_COMM_WORLD, dofs_fine_locally_relevant_range);
    interpolation_matrix.reinit(dofs_range_fine, dofs_range_coarse, dsp, MPI_COMM_WORLD);

    for(const auto &cell : this->dg->dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned()) continue;

        const dealii::types::global_dof_index cell_index = cell->active_cell_index();
        const unsigned int i_fele = cell->active_fe_index();
        const dealii::FESystem<dim,dim> &fe_ref = this->dg->fe_collection[i_fele];
        const unsigned int n_dofs_cell = fe_ref.n_dofs_per_cell();

        dof_indices.resize(n_dofs_cell);
        cell->get_dof_indices (dof_indices);
        const std::vector<dealii::types::global_dof_index> &dof_indices_fine = cellwise_dofs_fine[cell_index];
        
        assert(i_fele + 1 <= this->dg->max_degree);
        const dealii::FullMatrix<real> &interpolation_matrix_local = interpolation_hp(i_fele + 1, i_fele);
        AssertDimension(interpolation_matrix_local.m(), dof_indices_fine.size());
        AssertDimension(interpolation_matrix_local.n(), n_dofs_cell);

        for(unsigned int i=0; i < dof_indices_fine.size(); ++i)
        {
            for(unsigned int j=0; j < n_dofs_cell; ++j)
            {
                interpolation_matrix.set(dof_indices_fine[i], dof_indices[j], interpolation_matrix_local(i,j));
            }
        }

    } // cell loop ends

    interpolation_matrix.compress(dealii::VectorOperation::insert);
}

template<int dim, int nstate, typename real>
void DualWeightedResidualObjFunc<dim, nstate, real> :: extract_interpolation_matrices(
    dealii::Table<2, dealii::FullMatrix<real>> &interpolation_hp)
{
    const dealii::hp::FECollection<dim> &fe = this->dg->dof_handler.get_fe_collection();
    interpolation_hp.reinit(fe.size(), fe.size());

    for(unsigned int i=0; i<fe.size(); ++i)
    {
        for(unsigned int j=0; j<fe.size(); ++j)
        {
            if(i != j)
            {
                interpolation_hp(i, j).reinit(fe[i].n_dofs_per_cell(), fe[j].n_dofs_per_cell());
                try
                {
                    fe[i].get_interpolation_matrix(fe[j], interpolation_hp(i,j));
                } 
                // If interpolation matrix cannot be generated, reset matrix size to 0.
                catch (const typename dealii::FiniteElement<dim>::ExcInterpolationNotImplemented &)
                {
                    interpolation_hp(i,j).reinit(0,0);
                }

            }
        }
    }
}

template<int dim, int nstate, typename real>
std::vector<std::vector<dealii::types::global_dof_index>> DualWeightedResidualObjFunc<dim, nstate, real> :: get_cellwise_dof_indices()
{
    unsigned int n_cells_global = this->dg->triangulation->n_global_active_cells();
    std::vector<std::vector<dealii::types::global_dof_index>> cellwise_dof_indices(n_cells_global);
    std::vector<dealii::types::global_dof_index> dof_indices;
    for(const auto &cell : this->dg->dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned()) continue;

        const unsigned int i_fele = cell->active_fe_index();
        const dealii::FESystem<dim,dim> &fe_ref = this->dg->fe_collection[i_fele];
        const unsigned int n_dofs_cell = fe_ref.n_dofs_per_cell();

        dof_indices.resize(n_dofs_cell);
        cell->get_dof_indices (dof_indices);

        const dealii::types::global_dof_index cell_index = cell->active_cell_index();
        cellwise_dof_indices[cell_index] = dof_indices;
    }
    return cellwise_dof_indices;
}


template class DualWeightedResidualObjFunc <PHILIP_DIM, 1, double>;
template class DualWeightedResidualObjFunc <PHILIP_DIM, 2, double>;
template class DualWeightedResidualObjFunc <PHILIP_DIM, 3, double>;
template class DualWeightedResidualObjFunc <PHILIP_DIM, 4, double>;
template class DualWeightedResidualObjFunc <PHILIP_DIM, 5, double>;
} // namespace PHiLiP

