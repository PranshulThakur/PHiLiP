#include "unit_vector_parameterization.hpp"
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>

namespace PHiLiP {

template<int dim>
UnitVectorParameterization<dim>::UnitVectorParameterization(
    std::shared_ptr<HighOrderGrid<dim, double>> _high_order_grid)
    : BaseParameterization<dim>(_high_order_grid)
    , n_vol_nodes(this->high_order_grid->volume_nodes.size())
    , n_control_variables(n_vol_nodes - 1)
    , left_end(this->high_order_grid->volume_nodes(0))
    , right_end(this->high_order_grid->volume_nodes(n_control_variables))
    , min_mesh_size(1.0e-11)
    , rho(right_end - left_end - min_mesh_size*n_control_variables)
{
    if( (this->n_mpi > 1) || (dim > 1) )
    {
        std::cout<<"Cannot use more than one processor. This class is designed for 1D."<<std::endl;
        std::abort();
    }
}

template<int dim>
void UnitVectorParameterization<dim> :: initialize_design_variables(VectorType &control_var)
{
    control_var.reinit(n_control_variables);
    for(unsigned int i=0; i<n_control_variables; ++i)
    {
        control_var(i) = this->high_order_grid->volume_nodes(i+1) - this->high_order_grid->volume_nodes(i);
    }
    control_var.update_ghost_values();

    current_control_variables = control_var;
    current_control_variables.update_ghost_values();
    control_var_norm_squared = control_var * control_var; 
    scaling_k = rho/control_var_norm_squared;
}

//========================== Compute dXv_dXp and update it. ============================================================
template<int dim>
void UnitVectorParameterization<dim> :: compute_dXv_dXp(MatrixType &dXv_dXp) const
{
    const dealii::IndexSet &volume_range = this->high_order_grid->volume_nodes.get_partitioner()->locally_owned_range();
    
    dealii::DynamicSparsityPattern dsp(n_vol_nodes, n_control_variables, volume_range);
    for(unsigned int i=1; i<n_vol_nodes; ++i)
    {
        for(unsigned int j=0; j<n_control_variables; ++j)
        {
            dsp.add(i,j);
        }
    }

    dealii::IndexSet locally_relevant_dofs;
    dealii::DoFTools::extract_locally_relevant_dofs(this->high_order_grid->dof_handler_grid, locally_relevant_dofs);

    dealii::SparsityTools::distribute_sparsity_pattern(dsp, volume_range, this->mpi_communicator, locally_relevant_dofs);
    
    dealii::IndexSet control_variables_range(n_control_variables);
    control_variables_range.add_range(0, n_control_variables);

    dXv_dXp.reinit(volume_range, control_variables_range, dsp, this->mpi_communicator);
    update_dXv_dXp(dXv_dXp);
}

template<int dim>
void UnitVectorParameterization<dim> :: update_dXv_dXp(MatrixType &dXv_dXp) const
{
    for(unsigned int i=1; i<n_vol_nodes; ++i)
    {
        for(unsigned int p=0; p<n_control_variables; ++p)
        {
            dXv_dXp.set(i,p, dxi_dhp(i,p));
        }
    }

    dXv_dXp.compress(dealii::VectorOperation::insert);
}
//========================= Other functions from base class which are overridden. =============================================

template<int dim>
bool UnitVectorParameterization<dim> :: update_mesh_from_design_variables(
    const MatrixType &/*dXv_dXp*/,
    const VectorType &control_var) 
{
    // check if control variables have changed.
    bool control_variable_has_changed = this->has_design_variable_been_updated(current_control_variables, control_var);
    bool mesh_updated;
    if( !(control_variable_has_changed) )
    {
        mesh_updated = false;
        return mesh_updated;
    }
    current_control_variables = control_var;
    current_control_variables.update_ghost_values();
    
    control_var_norm_squared = control_var * control_var; 
    scaling_k = rho/control_var_norm_squared;

    for(unsigned int i=1; i<n_vol_nodes; ++i)
    {
       this->high_order_grid->volume_nodes(i) = this->high_order_grid->volume_nodes(i-1) + scaling_k*pow(control_var(i-1),2) + min_mesh_size;
    }
    this->high_order_grid->volume_nodes.update_ghost_values();
    mesh_updated = true;
    return mesh_updated;
}

template<int dim>
unsigned int UnitVectorParameterization<dim> :: get_number_of_design_variables() const
{
    return n_control_variables;
}

//========================== Functions related to first order deivatives ==============================================
template<int dim>
double UnitVectorParameterization<dim> :: dxi_dhp(const unsigned int i, const unsigned int p) const
{
    double sum_val_i = 0;
    for(unsigned int j=0; j<i; ++j)
    {
        sum_val_i += pow(current_control_variables(j),2);
    }
    double derivative_val = sum_val_i * dk_dh(p);

    if(  p <= (i-1)   )
    {
        derivative_val += scaling_k*2.0*current_control_variables(p);
    }
    return derivative_val;
}

template<int dim>
double UnitVectorParameterization<dim> :: dk_dh(const unsigned int p) const
{
    double derivative_val = -2.0 * current_control_variables(p) * rho / pow(control_var_norm_squared,2);
    return derivative_val;
}

template class UnitVectorParameterization<PHILIP_DIM>;
} // namespace PHiLiP
