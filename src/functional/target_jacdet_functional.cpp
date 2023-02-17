#include "target_jacdet_functional.h"
#include <deal.II/dofs/dof_tools.h>

namespace PHiLiP {

template<int dim>
Target_Jacdet<dim>::Target_Jacdet(std::shared_ptr<PHiLiP::DGBase<dim,double>> _dg)
	: dg(_dg)
	, d2IdXdX(std::make_shared<MatrixType>())
	, pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
{
	NormalVector target_jacdet_initial(dg->triangulation->n_active_cells());
	target_jacdet_initial *= 0.0;
	set_target_jacdet(target_jacdet_initial);
}

template<int dim>
void Target_Jacdet<dim> :: set_target_jacdet(const NormalVector & _target_jacdet)
{
	target_jacdet = _target_jacdet;
}

template<int dim>
void Target_Jacdet<dim> :: allocate_derivatives()
{
	// allocate dIdX
	dealii::IndexSet locally_owned_dofs = dg->high_order_grid->dof_handler_grid.locally_owned_dofs();
	dealii::IndexSet locally_relevant_dofs, ghost_dofs;
	dealii::DoFTools::extract_locally_relevant_dofs(dg->high_order_grid->dof_handler_grid, locally_relevant_dofs);
	ghost_dofs = locally_relevant_dofs;
	ghost_dofs.subtract_set(locally_owned_dofs);
	dIdX.reinit(locally_owned_dofs, ghost_dofs, MPI_COMM_WORLD);

	// allocate d2IdXdX
	dealii::SparsityPattern sparsity_pattern_d2IdXdX = dg->get_d2RdXdX_sparsity_pattern();
	const dealii::IndexSet &row_parallel_partitioning_d2IdXdX = dg->high_order_grid->locally_owned_dofs_grid;
	const dealii::IndexSet &col_parallel_partitioning_d2IdXdX = dg->high_order_grid->locally_owned_dofs_grid;
	d2IdXdX->reinit(row_parallel_partitioning_d2IdXdX, col_parallel_partitioning_d2IdXdX, sparsity_pattern_d2IdXdX, MPI_COMM_WORLD);
}

template<int dim>
template<typename real2>
real2 Target_Jacdet<dim> :: evaluate_volume_cell_functional(
		const std::vector< real2 > &coords_coeff, 
		const dealii::FESystem<dim> &fe_metric,
		const dealii::Quadrature<dim> &volume_quadrature,
		const double target_cell_jacdet) const
{
	const unsigned int n_vol_quad_pts = volume_quadrature.size();
	const unsigned int n_metric_dofs_cell = coords_coeff.size();
	
	real2 sum_val = 0.0;
	for(unsigned int iquad = 0; iquad < n_vol_quad_pts; ++iquad)
	{
		const dealii::Point<dim,double> &ref_point = volume_quadrature.point(iquad);
		dealii::Point<dim,real2> phys_coord;

		for (int d=0;d<dim;++d) {phys_coord[d] = 0.0;}

		std::array< dealii::Tensor<1,dim,real2>, dim > coord_grad; // Tensor initialize with zeros
		dealii::Tensor<2,dim,real2> metric_jacobian;

		for (unsigned int idof=0; idof<n_metric_dofs_cell; ++idof) {
            const unsigned int axis = fe_metric.system_to_component_index(idof).first;
            phys_coord[axis] += coords_coeff[idof] * fe_metric.shape_value(idof, ref_point);
            coord_grad[axis] += coords_coeff[idof] * fe_metric.shape_grad(idof, ref_point);
        }
        for (int row=0;row<dim;++row) {
            for (int col=0;col<dim;++col) {
                metric_jacobian[row][col] = coord_grad[row][col];
            }
        }
        const real2 jacobian_determinant = dealii::determinant(metric_jacobian);
		std::cout<<"Jacdet val = "<<jacobian_determinant.val().val()<<std::endl;
		real2 jacdiff = jacobian_determinant - target_cell_jacdet;
		sum_val += 1.0/2.0*jacdiff*jacdiff;
	} // quad loop ends

	return sum_val;
}

template<int dim>
double Target_Jacdet<dim> :: evaluate_functional(bool compute_derivatives)
{
	double local_functional = 0.0;

	const dealii::FESystem<dim,dim> &fe_metric = dg->high_order_grid->fe_system;
	const unsigned int n_metric_dofs_cell = fe_metric.dofs_per_cell;
	std::vector<dealii::types::global_dof_index> cell_metric_dofs_indices(n_metric_dofs_cell);

	std::vector<double> local_dIdX(n_metric_dofs_cell);

	const auto mapping = (*(dg->high_order_grid->mapping_fe_field));
	dealii::hp::MappingCollection<dim> mapping_collection(mapping);

	if(compute_derivatives)
	{
		allocate_derivatives();
		pcout<<"Evaluating Target_Jacdet functional with derivatives."<<std::endl;
	}

	auto metric_cell = dg->high_order_grid->dof_handler_grid.begin_active();
	for(auto soln_cell = dg->dof_handler.begin_active(); soln_cell !=  dg->dof_handler.end(); ++soln_cell, ++metric_cell)
	{
		if(!soln_cell->is_locally_owned()) continue;

		const unsigned int active_fe_index = soln_cell->active_fe_index();
		metric_cell->get_dof_indices(cell_metric_dofs_indices);
		std::vector< FadFadType > coords_coeff(n_metric_dofs_cell);

		unsigned int n_total_indep = 0;
		if(compute_derivatives) {n_total_indep = n_metric_dofs_cell;}

		unsigned int i_derivative = 0;
		for(unsigned int idof = 0; idof < n_metric_dofs_cell; ++idof)
		{
			const double val = dg->high_order_grid->volume_nodes[cell_metric_dofs_indices[idof]];
			coords_coeff[idof] = val;
			if(compute_derivatives) {coords_coeff[idof].diff(i_derivative++, n_total_indep);}
		}

		AssertDimension(i_derivative, n_total_indep);

		if(compute_derivatives)
		{
			i_derivative = 0;
			for(unsigned int idof = 0; idof < n_metric_dofs_cell; ++idof)
			{
				const double val = dg->high_order_grid->volume_nodes[cell_metric_dofs_indices[idof]];
				coords_coeff[idof].val() = val;
				coords_coeff[idof].val().diff(i_derivative++, n_total_indep);
			}
		}
		
		AssertDimension(i_derivative, n_total_indep);

        // Get quadrature point on reference cell
        const dealii::Quadrature<dim> &volume_quadrature = dg->volume_quadrature_collection[active_fe_index];

		FadFadType volume_local_sum = evaluate_volume_cell_functional(coords_coeff, fe_metric, volume_quadrature, target_jacdet(soln_cell->active_cell_index()));

		local_functional += volume_local_sum.val().val();

		if(compute_derivatives)
		{
			i_derivative = 0;
			local_dIdX.resize(n_metric_dofs_cell);
			for(unsigned int idof = 0; idof < n_metric_dofs_cell; ++idof)
			{
				local_dIdX[idof] = volume_local_sum.dx(i_derivative++).val();
			}

			dIdX.add(cell_metric_dofs_indices, local_dIdX);

			std::vector<double> dXidX(n_metric_dofs_cell);
			i_derivative = 0;
            for (unsigned int idof=0; idof<n_metric_dofs_cell; ++idof) 
			{
                const FadType dXi = volume_local_sum.dx(i_derivative++);
                unsigned int j_derivative = 0;
                for (unsigned int jdof=0; jdof<n_metric_dofs_cell; ++jdof) 
				{
                    dXidX[jdof] = dXi.dx(j_derivative++);
                }
                d2IdXdX->add(cell_metric_dofs_indices[idof], cell_metric_dofs_indices, dXidX);
            }
		}
	} // cell loop ends

	if(compute_derivatives)
	{
		dIdX.compress(dealii::VectorOperation::add);
		d2IdXdX->compress(dealii::VectorOperation::add);
	}

	const double global_functional_value = dealii::Utilities::MPI::sum(local_functional, MPI_COMM_WORLD);

	return global_functional_value;
}

template class Target_Jacdet<PHILIP_DIM>;
} // PHiLiP namespace 
