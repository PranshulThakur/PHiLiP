#ifndef __DIRECT_GOAL_ORIENTED_ERROR_H__
#define __DIRECT_GOAL_ORIENTED_ERROR_H__

#include "functional.h"

namespace PHiLiP {
/// Class to evaluate direct goal oriented error.
/**
 * Computes direct error in the functional of interest for mesh adaptation. We use two solutions:
 * 1.  Interpolated solution \f[ \mathbf{U_h^H} \f] : Solution coeffs interpolated on a fine (p-refined) grid. \f[ \mathbf{U_h^H} = \mathbf{I_h^H} \mathbf{u_H} \f] with 
 *     with \f[ \mathbf{u_H} \f] being the coarse solution.
 * 2.  Fine solution \f[ \mathbf{U_h} \f] : Solution coeffs obtained after taylor expanding the residual about \f[ \mathbf{U_h^H} \f]. 
 *     \f[ \mathbf{U_h} = \mathbf{U_h^H} - \mathbf{\frac{\partial R}{\partial U}^{-1}} \mathbf{R(U_h^H)}  \f]
 *   
 * It is assumed that the interpolated and fine solutions are passed to the constructor of this class. The direct error in functional \f[ \mathcal{J} \f] at each cell k is then computed as
 * \f[\eta_k = \left(\mathcal{J}(\mathbf{U_h}) - \mathcal{J}(\mathbf{U_h^H}) \right) \f]
 * And the goal oriented error currently used as the objective function for optimization base goal oriented mesh adaptation is 
 * \f[ \mathcal{F}(\mathbf{U_h}, \mathbf{U_h^H}, \mathbf{x}) = \sum_k \eta_k^2 \f].
 * @note This class is structured similar to the Functional class for evaluating derivatives using AD.
 */
#if PHiLiP_DIM==1
template <int dim, int nstate, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, int nstate, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class DirectGoalOrientedError : public Functional <dim, nstate, real, MeshType>
{
    using FadType = Sacado::Fad::DFad<real>; ///< Sacado AD type for first derivative
    using FadFadType = Sacado::Fad::DFad<FadType>; ///< Sacado AD type for second derivative
    using VectorType = dealii::LinearAlgebra::distributed::Vector<real>; ///< Alias for parallel distributed vector.
    using MatrixType = dealii::TrilinosWrappers::SparseMatrix; ///< Alias for SparseMatrix.

public:
    /// Pointer to functional to evaluate functional values at each cell (required for computing direct error).
    std::shared_ptr<Functional<dim, nstate, real, MeshType>> functional;

    /// Derivative of direct goal oriented error \f[\mathcal{F} \f] w.r.t. the interpolated solution \f[ \mathbf{U_h^H} \f].
    VectorType derivative_functionalerror_wrt_solution_interpolated;
    /// Derivative of direct goal oriented error \f[\mathcal{F} \f] w.r.t. the fine solution \f[\mathbf{U_h} \f]. Refer class description for nomenclature.
    VectorType derivative_functionalerror_wrt_solution_fine;
    /// Derivative of direct goal oriented error \f[\mathcal{F} \f] w.r.t. the volume nodes \f[ \mathbf{X} \f].
    VectorType derivative_functionalerror_wrt_volume_nodes;

    /// Stores \f[ \frac{\partial^2 \mathcal{F}}{\partial \mathbf{U_h} \partial \mathbf{U_h}} \f].
    MatrixType d2F_solfine_solfine;
    /// Stores \f[ \frac{\partial^2 \mathcal{F}}{\partial \mathbf{U_h} \partial \mathbf{U_h^H}} \f].
    MatrixType d2F_solfine_solinterp;
    /// Stores \f[ \frac{\partial^2 \mathcal{F}}{\partial \mathbf{U_h} \partial \mathbf{X}} \f].
    MatrixType d2F_solfine_volnodes;

    /// Stores \f[ \frac{\partial^2 \mathcal{F}}{\partial \mathbf{U_h^H} \partial \mathbf{U_h^H}} \f].
    MatrixType d2F_solinterp_solinterp;
    /// Stores \f[ \frac{\partial^2 \mathcal{F}}{\partial \mathbf{U_h^H} \partial \mathbf{X}} \f].
    MatrixType d2F_solinterp_volnodes;

    /// Stores \f[ \frac{\partial^2 \mathcal{F}}{\partial \mathbf{X} \partial \mathbf{X}} \f].
    MatrixType d2F_volnodes_volnodes;

    VectorType solution_fine; ///< Stores fine solution \f[ \mathbf{U_h} \f].
    VectorType solution_interpolated; ///< Stores interpolated solution \f[ \mathbf{U_h} \f].
    VectorType del_solution; ///< Stores solution_fine - solution_interpolated
    real current_error_value; ///< Stores computed error value.

    real weight_of_mesh_error; ///< Ensures that the error is high when volume nodes are too close to each other.

    /// Constructor of the class.
    DirectGoalOrientedError( std::shared_ptr<DGBase<dim,real,MeshType>> _dg,
                             const bool uses_solution_values = true,
                             const bool uses_solution_gradient = false);
    /// Destructor.
    ~DirectGoalOrientedError(){}
    /// Allocate memory for storing first and second order derivatives.
    void allocate_partial_derivatives(const bool compute_dF_dWfine, const bool compute_dF_dWinterp, const bool compute_dF_dX, const bool compute_d2F);
    /// Evaluate functional error in cell volume \f[ \eta_k^2 \f]. Templated with real2 to facilitate AD.
    template <typename real2>
    real2 evaluate_functional_error_in_cell_volume(const std::vector< real2 > &soln_coeff_fine,
                                                   const std::vector< real2 > &soln_coeff_interpolated, 
                                                   const dealii::FESystem<dim> &fe_solution,
                                                   const std::vector<real2> &coords_coeff,
                                                   const dealii::FESystem<dim> &fe_metric,
                                                   const dealii::Quadrature<dim> &volume_quadrature) const;
    /// Evaluate functional error in cell boundary \f[ \eta_k^2 \f] (evaluated when the cell's face is located at the boundary). Templated with real2 to facilitate AD.
    template <typename real2> 
    real2 evaluate_functional_error_in_cell_boundary(const unsigned int boundary_id,
                                                     const std::vector< real2 > &soln_coeff_fine,
                                                     const std::vector< real2 > &soln_coeff_interpolated,
                                                     const dealii::FESystem<dim> &fe_solution,
                                                     const std::vector< real2 > &coords_coeff,
                                                     const dealii::FESystem<dim> &fe_metric,
                                                     const unsigned int face_number,
                                                     const dealii::Quadrature<dim-1> &face_quadrature) const;
    /// Evaluates the goal oriented error \f[ \mathcal{F} \f] and, if needed, it's first and second derivatives using AD.
    real evaluate_functional(
        const bool compute_dIdW = false,
        const bool compute_dIdX = false,
        const bool compute_d2I = false) override;

private:
    /// Computes solution fine and solution interpolated and stores them in member variables.
    void compute_solution_fine_and_solution_interpolated();

    /// Assigns values to soln and metric coeffs on each cell and sets them up as independent variables for AD.
    void assign_and_setup_independent_variables_for_ad(
        const bool compute_dF_dWfine,
        const bool compute_dF_dWinterp,
        const bool compute_dF_dX,
        const bool compute_d2F,
        std::vector< FadFadType > &soln_coeff_fine,
        std::vector< FadFadType > &soln_coeff_interpolated,
        std::vector< FadFadType > &coords_coeff); // TODO: Use this to simplify evaluate_functional_error_and_derivatives();

    /// Evaluate derivatives using AD.
    void evaluate_derivatives(
        const bool compute_dF_dWfine,
        const bool compute_dF_dWinterp,
        const bool compute_dF_dX,
        const bool compute_d2F,
        const FadFadType local_error_fadfad);  // TODO: Use this to simplify evaluate_functional_error_and_derivatives();
};

} // namespace PHiLiP
#endif
