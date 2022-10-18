#ifndef __DUAL_WEIGHTED_RESIDUAL_OBJ_FUNC_H__
#define __DUAL_WEIGHTED_RESIDUAL_OBJ_FUNC_H__

#include "functional.h"

namespace PHiLiP {

/// Class to compute the objective function of dual weighted residual used for optimization based mesh adaptation. \f[ \mathcal{F} = \frac{1}{2}\eta^T\eta \f].
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

    /// Evaluates \f[ \mathcal{F} = \frac{1}{2} \eta^T\eta \f] and derivatives, if needed.
    real evaluate_functional(
        const bool compute_dIdW = false,
        const bool compute_dIdX = false,
        const bool compute_d2I = false) override;

private:
    /// Stores true if coarse residual is used to compute objective function.
    const bool use_coarse_residual;

    /// Extracts all matrices possible for various combinations of polynomial degrees.
    void extract_interpolation_matrices(dealii::Table<2, dealii::FullMatrix<real>> &interpolation_hp);
    
    /// Returns cellwise dof indices. Used to store cellwise dof indices of higher poly order grid to form interpolation matrix.
    std::vector<std::vector<dealii::types::global_dof_index>> get_cellwise_dof_indices();

    /// Evaluates objective function and stores adjoint and residual.
    real evaluate_objective_function();

    /// Computes common vectors (adjoint, residual_fine) and matrices (R_u, R_u_transpose, adjoint*d2R) required for dIdW, dIdX and d2I.
    void compute_common_vectors_and_matrices();

    /// Computes interpolation matrix.
    /** Assumes the polynomial order remains constant throughout the optimization algorithm.
     *  Also assumes that all cells have the same polynomial degree.
     */
    void compute_interpolation_matrix();

    /// Computes  \f[ out_vector = \psi_x in_vector \f].
    void adjoint_x_vmult(VectorType &out_vector, const VectorType &in_vector) const;

    /// Computes  \f[ out_vector = \eta_u in_vector \f].
    void adjoint_u_vmult(VectorType &out_vector, const VectorType &in_vector) const;
    
    /// Computes  \f[ out_vector = \eta_x^T in_vector \f].
    void adjoint_x_Tvmult(VectorType &out_vector, const VectorType &in_vector) const;

    /// Computes  \f[ out_vector = \eta_u^T in_vector \f].
    void adjoint_u_Tvmult(VectorType &out_vector, const VectorType &in_vector) const;
    
    /// Computes \f[out_vector = \eta_{\psi} in_vector \f]. 
    /** @note: \f[ \eta_{\psi}\f] is a diagonal matrix, so Tvmult is the same as vmult.
     */
    void d_goalresidual_d_adjoint_vmult(VectorType &out_vector, const VectorType &in_vector) const;
    
    /// Computes \f[out_vector = \eta_R in_vector \f]. 
    /** @note: \f[ \eta_R \f] is a diagonal matrix, so Tvmult is the same as vmult.
     */
    void d_goalresidual_d_residual_vmult(VectorType &out_vector, const VectorType &in_vector) const;
    
    ///Computes \f[out_vector = \eta_x in_vector \f]
    void goalresidual_x_vmult(VectorType &out_vector, const VectorType &in_vector) const;
    
    ///Computes \f[out_vector = \eta_u in_vector \f]
    void goalresidual_u_vmult(VectorType &out_vector, const VectorType &in_vector) const;
    
    ///Computes \f[out_vector = \eta_x^T in_vector \f]
    void goalresidual_x_Tvmult(VectorType &out_vector, const VectorType &in_vector) const;
    
    ///Computes \f[out_vector = \eta_u^T in_vector \f]
    void goalresidual_u_Tvmult(VectorType &out_vector, const VectorType &in_vector) const; 

    /// Stores dIdW
    void store_dIdW();

    /// Stores dIdX
    void store_dIdX();
    
    /// Stores goal residual i.e. a vector of \f[ \eta = [\psi_1 R_1, \psi_2 R_2, ..., \psi_i R_i,... \psi_n R_n] \f] of size n_dofs fine.
    VectorType goalresidual;

    /// Stores \f[R_u \f] on fine space. 
    MatrixType R_u;
    
    /// Stores \f[R_u^T \f] on fine space. 
    MatrixType R_u_transpose;
    
    /// Stores \f[R_x \f] on fine space. 
    MatrixType R_x;

    /// Stores adjoint.
    VectorType adjoint;

    /// Residual used to evaluate objective function. Can be residual_fine or residual_fine - residual_coarse_interpolated.
    VectorType residual_used;

    /// Stores vector on coarse space to copy parallel partitioning later.
    VectorType vector_coarse;
    
    /// Stores vector on fine space (p+1) to copy parallel partitioning later.
    VectorType vector_fine;

    /// Stores \f[ J_{ux} + \psi^TR_{ux} \f]
    MatrixType matrix_ux;

    /// Stores \f[ J_{uu} + \psi^TR_{uu} \f]
    MatrixType matrix_uu;
    
    /// Stores \f[r_u \f] on coarse space. 
    MatrixType r_u;
    
    /// Stores \f[r_x \f] on coarse space. 
    MatrixType r_x;

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
