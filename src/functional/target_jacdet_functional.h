#ifndef __TARGET_JACDET_FUNCTIONAL__
#define __TARGET_JACDET_FUNCTIONAL__

#include <vector>
#include <iostream>

#include <Sacado.hpp>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/differentiation/ad/sacado_math.h>
#include <deal.II/differentiation/ad/sacado_number_types.h>
#include <deal.II/differentiation/ad/sacado_product_types.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include "dg/dg.h"
#include "physics/physics.h"

namespace PHiLiP {

template <int dim>
class Target_Jacdet
{
	using FadType = Sacado::Fad::DFad<double>; ///< Sacado AD type for first derivatives.
	using FadFadType = Sacado::Fad::DFad<FadType>; ///< Sacado AD type that allows 2nd derivatives.
    using VectorType = dealii::LinearAlgebra::distributed::Vector<double>; ///< Alias for dealii's parallel distributed vector.
    using MatrixType = dealii::TrilinosWrappers::SparseMatrix; ///< Alias for dealii::TrilinosWrappers::SparseMatrix.
    using NormalVector = dealii::Vector<double>; ///< Alias for serial vector.

public:
	Target_Jacdet(std::shared_ptr<PHiLiP::DGBase<dim,double>> _dg);
	~Target_Jacdet(){};

	double evaluate_functional(bool compute_derivatives = false);

	void set_target_jacdet(const NormalVector & _target_jacdet);
	
    VectorType dIdX;
	std::shared_ptr<MatrixType> d2IdXdX;
private:
	void allocate_derivatives();

	template<typename real2>
	real2 evaluate_volume_cell_functional(
		const std::vector< real2 > &coords_coeff, 
		const dealii::FESystem<dim> &fe_metric,
		const dealii::Quadrature<dim> &volume_quadrature,
		const double target_cell_jacdet) const;

	NormalVector target_jacdet;
	std::shared_ptr<DGBase<dim,double>> dg;
	dealii::ConditionalOStream pcout; ///< Parallel std::cout that only outputs on mpi_rank==
	
}; // class Target_Jacdet
} // PHiLiP namespace

#endif