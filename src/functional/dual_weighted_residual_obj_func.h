#ifndef __DUAL_WEIGHTED_RESIDUAL_OBJ_FUNC_H__
#define __DUAL_WEIGHTED_RESIDUAL_OBJ_FUNC_H__

#include "functional.h"

namespace PHiLiP {

/// Class to compute the objective function of dual weighted residual used for optimization based mesh adaptation. \f[ \mathcal{F} = \eta^T \eta \f].
template <int dim, int nstate, typename real>
class DualWeightedResidualObjFunc : public Functional<dim, nstate, real>
{
    using VectorType = dealii::LinearAlgebra::distributed::Vector<real>; ///< Alias for dealii's parallel distributed vector.
    using MatrixType = dealii::TrilinosWrappers::SparseMatrix; ///< Alias for dealii::TrilinosWrappers::SparseMatrix.

public:
    /// Constructor
    DualWeightedResidualObjFunc( 
        std::shared_ptr<DGBase<dim,real>> dg_input,
        const bool uses_solution_values = true,
        const bool uses_solution_gradient = false);

    /// Destructor
    ~DualWeightedResidualObjFunc(){}

    /// Computes common vectors (adjoint, residual_fine) and matrices (R_u, R_u_transpose, eta_psi and eta_R) required for dIdW, dIdX and d2I.
    void compute_common_vectors_and_matrices();

    /// Computes interpolation matrix.
    /** Assumes the polynomial order remains constant throughout the optimization algorithm.
     *  Also assumes that all cells have the same polynomial degree.
     */
    void compute_interpolation_matrix();

    /// Computes  \f[ out_vector = \eta_x*in_vector \f].
    void eta_x_vmult(VectorType &out_vector, const VectorType &in_vector) const;

    /// Computes  \f[ out_vector = \eta_u*in_vector \f].
    void eta_u_vmult(VectorType &out_vector, const VectorType &in_vector) const;
    
    /// Computes  \f[ out_vector = \eta_x^T*in_vector \f].
    void eta_x_Tvmult(VectorType &out_vector, const VectorType &in_vector) const;

    /// Computes  \f[ out_vector = \eta_u^T*in_vector \f].
    void eta_u_Tvmult(VectorType &out_vector, const VectorType &in_vector) const;

    /// Stores dIdW
    void store_dIdW();

    /// Stores dIdX
    void store_dIdX();
/*
    /// Computes \f[ out_vector = d2IdWdW*in_vector \f]. 
    void d2IdWdW_vmult(VectorType &out_vector, const VectorType &in_vector) const override;
    /// Computes \f[ out_vector = d2IdWdX*in_vector \f]. 
    void d2IdWdX_vmult(VectorType &out_vector, const VectorType &in_vector) const override; 
    /// Computes \f[ out_vector = d2IdWdX^T*in_vector \f]. 
    void d2IdWdX_Tvmult(VectorType &out_vector, const VectorType &in_vector) const override;
    /// Computes \f[ out_vector = d2IdXdX*in_vector \f]. 
    void d2IdXdX_vmult(VectorType &out_vector, const VectorType &in_vector) const override;
*/
    /// Evaluates \f[ \mathcal{F} = \eta^T \eta \f] and derivatives, if needed.
    real evaluate_functional(
        const bool compute_dIdW = false,
        const bool compute_dIdX = false,
        const bool compute_d2I = false) override;
private:
    /// Extracts all matrices possible for various combinations of polynomial degrees.
    void extract_interpolation_matrices(dealii::Table<2, dealii::FullMatrix<real>> &interpolation_hp);
    
    /// Returns cellwise dof indices. Used to store cellwise dof indices of higher poly order grid to form interpolation matrix.
    std::vector<std::vector<dealii::types::global_dof_index>> get_cellwise_dof_indices();

    /// Evaluates objective function and also stores adjoint and residual.
    real evaluate_objective_function();

    /// Stores adjoint weighted residual on each cell.
    VectorType eta;

    /// Stores \f[R_u \f] on fine space. 
    MatrixType R_u;
    
    /// Stores \f[R_u^T \f] on fine space. 
    MatrixType R_u_transpose;

    /// Stores adjoint.
    VectorType adjoint;

    /// Stores residual evaluated on fine (p+1) space.
    VectorType residual_fine;

    /// Stores vector on coarse space to copy parallel partitioning later.
    VectorType vector_coarse;
    
    /// Stores vector on fine space (p+1) to copy parallel partitioning later.
    VectorType vector_fine;
    
public:
    /// Stores global dof indices of the fine mesh.
    std::vector<std::vector<dealii::types::global_dof_index>> cellwise_dofs_fine;

    /// Stores interpolation matrix \f[ I_h \f] to interpolate onto fine space. Used to compute \f[ U_h^H = I_h u_H \f]. 
    MatrixType interpolation_matrix;
};

} // namespace PHiLiP

#endif
