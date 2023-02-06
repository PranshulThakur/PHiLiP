#include "metric_parameterization.hpp"
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>

namespace PHiLiP {

template<int dim>
MetricParameterization<dim> :: MetricParameterization(std::shared_ptr<HighOrderGrid<dim,double>> _high_order_grid)
	: BaseParameterization<dim>(_high_order_grid)
    , n_vol_nodes(this->high_order_grid->volume_nodes.size())
    , n_control_variables(n_vol_nodes - 1)
{
    if( (this->n_mpi > 1) || (dim > 1) )
    {
        std::cout<<"This parameterization is coded for 1D and, at the moment, cannot handle multiple dimensions."<<std::endl;
        std::abort();
    }
}

template<int dim>
void MetricParameterization<dim> :: initialize_design_variables(VectorType &control_var)
{
    control_var.reinit(n_control_variables);
    for(unsigned int i=0; i<n_control_variables; ++i)
    {
		const double cell_size = this->high_order_grid->volume_nodes(i+1) - this->high_order_grid->volume_nodes(i);
        control_var(i) = 1.0/(pow(cell_size,2));
    }
    control_var.update_ghost_values();

    current_control_variables = control_var;
    current_control_variables.update_ghost_values();
}

//========================== Compute dXv_dXp and update it. ============================================================
template<int dim>
void MetricParameterization<dim> :: compute_dXv_dXp(MatrixType &dXv_dXp) const
{
    const dealii::IndexSet &volume_range = this->high_order_grid->volume_nodes.get_partitioner()->locally_owned_range();
    
    dealii::DynamicSparsityPattern dsp(n_vol_nodes, n_control_variables, volume_range);
    for(int i=0; i<(int)(n_vol_nodes); ++i)
    {
        for(int j=i-1; j<=i; ++j)
        {
			if( (j==-1) || (j==((int)(n_vol_nodes)-1)) ) {continue;}
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
void MetricParameterization<dim> :: update_dXv_dXp(MatrixType &dXv_dXp) const
{
    for(int i=0; i<(int)(n_vol_nodes); ++i)
    {
        for(int j=i-1; j<=i; ++j)
        {
			if( (j==-1) || (j==((int)(n_vol_nodes)-1)) ) {continue;}
			double derivative_val = 0.0;
			const double metric_val = current_control_variables(j);
			if(j == i)
			{
				derivative_val = 1.0/4.0 * pow(metric_val, -1.5);
			}
			else if(j == i-1)
			{
				derivative_val = -1.0/4.0 * pow(metric_val, -1.5);
			}
            dXv_dXp.set(i,j,derivative_val);
        }
    } // loop ends

    dXv_dXp.compress(dealii::VectorOperation::insert);
}

//========================= Other functions from base class which are overridden. =============================================
template<int dim>
bool MetricParameterization<dim> :: update_mesh_from_design_variables(
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
    
	double sum_val = 0.0;
	for(unsigned int i=0; i<n_control_variables; ++i)
	{
		const double metric_val = current_control_variables(i);
		sum_val += 1.0/sqrt(metric_val);
	}
	const double scaling = 1.0/sum_val;

	for(unsigned int i=1; i<n_vol_nodes; ++i)
	{
		const double metric_val = current_control_variables(i-1);
		const double mesh_size = scaling/sqrt(metric_val);
		this->high_order_grid->volume_nodes(i) = this->high_order_grid->volume_nodes(i-1) + mesh_size; 
	}
    
	this->high_order_grid->volume_nodes.update_ghost_values();
    mesh_updated = true;
    return mesh_updated;
}

template<int dim>
unsigned int MetricParameterization<dim> :: get_number_of_design_variables() const
{
    return n_control_variables;
}

template<int dim>    
int MetricParameterization<dim> :: is_design_variable_valid(
	const MatrixType &/*dXv_dXp*/, 
	const VectorType &control_var) const
{
    this->pcout<<"Checking if mesh is valid before updating variables..."<<std::endl;
	for(unsigned int i=0; i<n_control_variables; ++i)
	{
		if(control_var(i) < 1.0e-10) {return 1;}
	}

	return 0;
}

template<int dim>
double MetricParameterization<dim> :: control_var_norm() const
{
    return current_control_variables.l2_norm();
}

template class MetricParameterization<PHILIP_DIM>;
} // PHiLiP namespace
