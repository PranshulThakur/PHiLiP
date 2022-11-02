#ifndef __DUAL_WEIGHTED_RESIDUAL_OBJ_FUNC_H__
#define __DUAL_WEIGHTED_RESIDUAL_OBJ_FUNC_H__

#include "functional.h"

namespace PHiLiP {

/// Class to compute the objective function of dual weighted residual used for optimization based mesh adaptation. \f[ \mathcal{F} = \frac{1}{2} \sum_k \eta_k^2 \f].
template <int dim, int nstate, typename real>
class DualWeightedResidualObjFunc : public Functional<dim, nstate, real>
{
    using VectorType = dealii::LinearAlgebra::distributed::Vector<real>; ///< Alias for dealii's parallel distributed vector.
    using MatrixType = dealii::TrilinosWrappers::SparseMatrix; ///< Alias for dealii::TrilinosWrappers::SparseMatrix.
    using NormalVector = dealii::Vector<real>; ///< Alias for serial vector.

public:
    /// Constructor
    DualWeightedResidualObjFunc( 
        std::shared_ptr<DGBase<dim,real>> dg_input,
        const real _functional_exact,
        const bool uses_solution_values = true,
        const bool uses_solution_gradient = false,
        const bool _use_coarse_residual = false);

    /// Destructor
    ~DualWeightedResidualObjFunc(){}


    /// Computes \f[ out_vector = d2IdWdW*in_vector \f]. 
    void d2IdWdW_vmult(VectorType &out_vector, const VectorType &in_vector) const override;
    /// Computes \f[ out_vector = d2IdWdX*in_vector \f]. 
    void d2IdWdX_vmult(VectorType &out_vector, const VectorType &in_vector) const override; 
    /// Computes \f[ out_vector = d2IdWdX^T*in_vector \f]. 
    void d2IdWdX_Tvmult(VectorType &out_vector, const VectorType &in_vector) const override;
    /// Computes \f[ out_vector = d2IdXdX*in_vector \f]. 
    void d2IdXdX_vmult(VectorType &out_vector, const VectorType &in_vector) const override;

    /// Evaluates \f[ \mathcal{F} = \frac{1}{2} \sum_k \eta_k^2 \f] and derivatives, if needed.
    real evaluate_functional(
        const bool compute_dIdW = false,
        const bool compute_dIdX = false,
        const bool compute_d2I = false) override;

private:
    /// Stores true if coarse residual is used in the objective function.
    const bool use_coarse_residual;

    /// Extracts all matrices possible for various combinations of polynomial degrees.
    void extract_interpolation_matrices(dealii::Table<2, dealii::FullMatrix<real>> &interpolation_hp);
    
    /// Returns cellwise dof indices. Used to store cellwise dof indices of higher poly order grid to form interpolation matrix and cto compute matrix-vector products.
    std::vector<std::vector<dealii::types::global_dof_index>> get_cellwise_dof_indices();

    /// Evaluates objective function and stores adjoint and residual.
    real evaluate_objective_function();

    /// Computes common vectors and matrices (R_u, R_u_transpose, adjoint*d2R) required for dIdW, dIdX and d2I.
    void compute_common_vectors_and_matrices();

    /// Computes interpolation matrix.
    /** Assumes the polynomial order remains constant throughout the optimization algorithm.
     *  Also assumes that all cells have the same polynomial degree.
     */
    void compute_interpolation_matrix();

    /// Stores dIdW
    void store_dIdW();

    /// Stores dIdX
    void store_dIdX();
    
    /// Stores \f[ \mathcal{J}_{exact} - \mathcal{J} \f].
    real functional_error;

    /// Stores \f[ \mathcal{J}_{exact} \f].
    const real functional_exact;

    /// Stores vector on coarse space to copy parallel partitioning later.
    VectorType vector_coarse;
    
    /// Stores vector on fine space (p+1) to copy parallel partitioning later.
    VectorType vector_fine;
    
    /// Stores vector of volume nodes to copy parallel partitioning later.
    VectorType vector_vol_nodes;

    /// Stores \f[ \mathcal{J}_x \f].
    VectorType functional_x;
    
    /// Stores \f[ \mathcal{J}_u \f].
    VectorType functional_u;

    /// Stores \f[ \mathcal{J}_{uu} \f].
    MatrixType functional_uu;
    
    /// Stores \f[ \mathcal{J}_{ux} \f].
    MatrixType functional_ux;

    /// Stores \f[ \mathcal{J}_{xx} \f].
    MatrixType functional_xx;

    /// Functional used to create the objective function.
    std::shared_ptr< Functional<dim, nstate, real> > functional;
    
    /// Functional used to evaluate cell weight.
    std::unique_ptr< Functional<dim, nstate, real> > cell_weight_functional;
    
public:
    /// Stores global dof indices of the fine mesh.
    std::vector<std::vector<dealii::types::global_dof_index>> cellwise_dofs_fine;

    /// Stores interpolation matrix \f[ I_h \f] to interpolate onto fine space. Used to compute \f[ U_h^H = I_h u_H \f]. 
    MatrixType interpolation_matrix;
};

} // namespace PHiLiP

#endif
