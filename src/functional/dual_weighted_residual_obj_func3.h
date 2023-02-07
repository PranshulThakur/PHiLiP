#ifndef __DUAL_WEIGHTED_RESIDUAL_OBJ_FUNC3_H__
#define __DUAL_WEIGHTED_RESIDUAL_OBJ_FUNC3_H__

#include "functional.h"

namespace PHiLiP {

/// Class to compute the objective function of dual weighted residual used for optimization based mesh adaptation.
template <int dim, int nstate, typename real>
class DualWeightedResidualObjFunc3 : public Functional<dim, nstate, real>
{
    using VectorType = dealii::LinearAlgebra::distributed::Vector<real>; ///< Alias for dealii's parallel distributed vector.
    using MatrixType = dealii::TrilinosWrappers::SparseMatrix; ///< Alias for dealii::TrilinosWrappers::SparseMatrix.
    using NormalVector = dealii::Vector<real>; ///< Alias for serial vector.

public:
    /// Constructor
    DualWeightedResidualObjFunc3( 
        std::shared_ptr<DGBase<dim,real>> dg_input,
        const bool uses_solution_values = true,
        const bool uses_solution_gradient = false,
        const bool _use_coarse_residual = false);

    /// Destructor
    ~DualWeightedResidualObjFunc3(){}


    /// Computes \f[ out_vector = d2IdWdW*in_vector \f]. 
    void d2IdWdW_vmult(VectorType &out_vector, const VectorType &in_vector) const override;
    /// Computes \f[ out_vector = d2IdWdX*in_vector \f]. 
    void d2IdWdX_vmult(VectorType &out_vector, const VectorType &in_vector) const override; 
    /// Computes \f[ out_vector = d2IdWdX^T*in_vector \f]. 
    void d2IdWdX_Tvmult(VectorType &out_vector, const VectorType &in_vector) const override;
    /// Computes \f[ out_vector = d2IdXdX*in_vector \f]. 
    void d2IdXdX_vmult(VectorType &out_vector, const VectorType &in_vector) const override;

    /// Evaluates functional and derivatives, if needed.
    real evaluate_functional(
        const bool compute_dIdW = false,
        const bool compute_dIdX = false,
        const bool compute_d2I = false) override;

private:
    /// Stores true if coarse residual is used in the objective function.
    const bool use_coarse_residual;

    /// Extracts all matrices possible for various combinations of polynomial degrees.
    void extract_interpolation_matrices(dealii::Table<2, dealii::FullMatrix<real>> &interpolation_hp);
    
    /// Returns cellwise dof indices. Used to store cellwise dof indices of higher poly order grid to form interpolation matrix and to compute matrix-vector products.
    std::vector<std::vector<dealii::types::global_dof_index>> get_cellwise_dof_indices();

    /// Evaluates objective function and stores adjoint and residual.
    real evaluate_objective_function();

    /// Computes common vectors and matrices.
    void compute_common_vectors_and_matrices();

    /// Computes interpolation matrix.
    /** Assumes the polynomial order remains constant throughout the optimization algorithm.
     *  Also assumes that all cells have the same polynomial degree.
     */
    void compute_interpolation_matrix();
    
    /// Computes projection matrix of size i x j (i.e. projects from fe_j to fe_i).
    void get_projection_matrix(
        const dealii::FESystem<dim,dim> &fe_i, 
        const dealii::FESystem<dim,dim> &fe_j, 
        dealii::FullMatrix<real> &projection_matrix);

    /// Computes  \f[ out_vector = \psi_x in_vector \f].
    void adjoint_x_vmult(VectorType &out_vector, const VectorType &in_vector) const;

    /// Computes  \f[ out_vector = \psi_u in_vector \f].
    void adjoint_u_vmult(VectorType &out_vector, const VectorType &in_vector) const;
    
    /// Computes  \f[ out_vector = \psi_x^T in_vector \f].
    void adjoint_x_Tvmult(VectorType &out_vector, const VectorType &in_vector) const;

    /// Computes  \f[ out_vector = \psi_u^T in_vector \f].
    void adjoint_u_Tvmult(VectorType &out_vector, const VectorType &in_vector) const;

    /// Computes  \f[ out_vector = \eta_{\psi} in_vector \f].
    void dwr_adjoint_vmult(NormalVector &out_vector, const VectorType &in_vector) const;
    
    /// Computes  \f[ out_vector = \eta_{R} in_vector \f].
    void dwr_residual_vmult(NormalVector &out_vector, const VectorType &in_vector) const;
    
    /// Computes  \f[ out_vector = \eta_{\psi}^T in_vector \f].
    void dwr_adjoint_Tvmult(VectorType &out_vector, const NormalVector &in_vector) const;
    
    /// Computes  \f[ out_vector = \eta_{R}^T in_vector \f].
    void dwr_residual_Tvmult(VectorType &out_vector, const NormalVector &in_vector) const;

    /// Computes \f[ out_vector = \eta_x in_vector \f].
    void dwr_x_vmult(NormalVector &out_vector, const VectorType &in_vector) const;
    
    /// Computes \f[ out_vector = \eta_u in_vector \f], with \f[ \eta_u \f] computed wrt the coarse solution
    void dwr_u_vmult(NormalVector &out_vector, const VectorType &in_vector) const;
    
    /// Computes \f[ out_vector = \eta_x^T in_vector \f].
    void dwr_x_Tvmult(VectorType &out_vector, const NormalVector &in_vector) const;
    
    /// Computes \f[ out_vector = \eta_u^T in_vector \f]. with \f[ \eta_u \f] computed wrt coarse solution.
    void dwr_u_Tvmult(VectorType &out_vector, const NormalVector &in_vector) const;

	/// Computes vmult with tensor matrix.
	void tensor_matrix_vmult(NormalVector &out_vector, const NormalVector &in_vector) const;
	
	/// Computes vmult with tensor matrix squared.
	void tensor_matrix_squared_vmult(NormalVector &out_vector, const NormalVector &in_vector) const;
	
	/// Returns dot product of two vectors of size n_cells
	real normalvector_dot_product(const NormalVector &in_vector1, const NormalVector &in_vector2) const;
    

    /// Stores dIdW
    void store_dIdW();

    /// Stores dIdX
    void store_dIdX();
    
    /// Stores \f[ \eta = [\eta_k, k=1,2,..N_k], \eta_k = sqrt(\left(\psi^T R \right)_k^2 + \delta) \f].
    NormalVector dwr_error;
	
	/// Stores signed cel error \f[ \left(\psi^T R \right)_k \f]. 
	NormalVector signed_cell_error;
	
	/// Stores tensor squared times dwr_error.
	NormalVector tensor2_times_dwr;

    /// Stores \f[R_u \f] on fine space. 
    MatrixType R_u;
    
    /// Stores \f[R_u^T \f] on fine space. 
    MatrixType R_u_transpose;
    
    /// Stores \f[R_x \f] on fine space. 
    MatrixType R_x;
 
    /// Stores adjoint evaluated on fine space.
    VectorType adjoint;

    /// Residual used to evaluate objective function. Can be residual_fine or residual_fine - residual_coarse_interpolated.
    VectorType residual_used;

    /// Stores vector on coarse space to copy parallel partitioning later.
    VectorType vector_coarse;
    
    /// Stores vector on fine space (p+1) to copy parallel partitioning later.
    VectorType vector_fine;
    
    /// Stores vector of volume nodes to copy parallel partitioning later.
    VectorType vector_vol_nodes;
    
    /// Stores \f[ - \left(J_{ux} + \psi^TR_{ux} \right) \f]
    MatrixType matrix_ux;

    /// Stores \f[ - \left(J_{uu} + \psi^TR_{uu} \right) \f]
    MatrixType matrix_uu;
 
    /// Stores \f[r_u \f] on coarse space. 
    MatrixType r_u;
    
    /// Stores \f[r_x \f] on coarse space. 
    MatrixType r_x;

    /// Functional used to create the objective function.
    std::shared_ptr< Functional<dim, nstate, real> > functional;

    /// Stores the weight of mesh distortion term added to the objective function.
    const real mesh_weight;

    /// Stores initial volume nodes for implementing mesh weight.
    const VectorType initial_vol_nodes;

    /// Functional used to evaluate cell distortion.
//    std::unique_ptr< Functional<dim, nstate, real> > cell_distortion_functional;
    
public:
    /// Stores global dof indices of the fine mesh.
    std::vector<std::vector<dealii::types::global_dof_index>> cellwise_dofs_fine;

    /// Stores interpolation matrix \f[ I_h \f] to interpolate onto fine space. Used to compute \f[ U_h^H = I_h u_H \f]. 
    MatrixType interpolation_matrix;
};

} // namespace PHiLiP

#endif
