#ifndef __DIRECT_GOAL_ORIENTED_ERROR_H__
#define __DIRECT_GOAL_ORIENTED_ERROR_H__

#include "mesh_error_estimate_base.h"

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
 */

#if PHiLiP_DIM==1
template <int dim, int nstate, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, int nstate, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif

class DirectGoalOrientedError : public MeshErrorEstimateBase <dim, real, MeshType>
{
    using FadType = Sacado::Fad::DFad<real>; ///< Sacado AD type for first derivative
    using FadFadType = Sacado::Fad::DFad<FadType>; ///< Sacado AD type for second derivative
    using VectorType = dealii::LinearAlgebra::distributed::Vector<real>; ///< Alias for parallel distributed vector.
    using MatrixType = dealii::TrilinosWrappers::SparseMatrix; ///< Alias for SparseMatrix.

public:
    /** Shared pointer to fine dg.
     * @note It is assumed that the dg is already p-enriched (i.e. it's polynomial order has already been increased by 1).
     */
    std::shared_ptr<DGBase<dim, real, MeshType>> dg_fine;
    /// Pointer to functional to evaluate functional values at each cell (required for computing direct error).
    std::shared_ptr<Functional<dim, nstate, real, MeshType>> functional;

    /// Stores dealii's volume update flags.
    const dealii::UpdateFlags volume_update_flags = dealii::update_values | dealii::update_gradients 
                                                    | dealii::update_quadrature_points | dealii::update_JxW_values;
    /// Stores dealii's face update flags.
    const dealii::UpdateFlags face_update_flags = dealii::update_values | dealii::update_gradients | dealii::update_quadrature_points 
                                                  | dealii::update_JxW_values | dealii::update_normal_vectors;
    
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

    dealii::ConditionalOStream pcout; ///< std::cout only by processor #0.

    /** Constructor of the class.
     * @note This class assumes that dg is fine (p-refined) and solution fine and solution interpolated are already computed.
     */
     DirectGoalOrientedError(std::shared_ptr<DGBase<dim,real,MeshType>> _dg_fine,
                             VectorType & _solution_fine,
                             VectorType & _solution_interpolated);
     /// Destructor (does nothing but included for readability).
     ~DirectGoalOrientedError(){};
     /// Allocate memory for storing first and second order derivatives.
     void allocate_derivatives(const bool compute_first_order_derivatives, const bool compute_second_order_derivatives);
     /// Evaluate functional error in cell volume \f[ \eta_k^2 \f]. Templated with real2 to facilitate AD.
     template <typename real2>
     real2 evaluate_functional_error_in_cell_volume(const std::vector< real2 > &soln_coeff_fine,
                                                    const std::vector< real2 > &soln_coeff_interpolated, 
                                                    const dealii::FESystem<dim> &fe_solution,
                                                    const std::vector<real2> &coords_coeff,
                                                    const dealii::FESystem<dim> &fe_metric,
                                                    const dealii::Quadrature<dim> &volume_quadrature) const;
    /// Evaluate functional error in cell boundary \f[ \eta_k^2 \f] (evaluated when the cell's face is located at the boundary). Templated with real2 to facilitate AD.
    template <typename real2> evaluate_functional_error_in_cell_boundary(const unsigned int boundary_id,
                                                                         const std::vector< real2 > &soln_coeff_fine,
                                                                         const std::vector< real2 > &soln_coeff_interpolated,
                                                                         const dealii::FESystem<dim> &fe_solution,
                                                                         const std::vector< real2 > &coords_coeff,
                                                                         const dealii::FESystem<dim> &fe_metric,
                                                                         const unsigned int face_number,
                                                                         const dealii::Quadrature<dim-1> &face_quadrature) const;
    /// Evaluates the goal oriented error \f[ \mathcal{F} \f] and, if needed, it's first and second derivatives using AD.
    real evaluate_functional_error_and_derivatives(const bool evaluate_first_order_derivatives, const bool evaluate_second_order_derivatives);

    /// Sets up variables for AD
    void setup_variables_for_automatic_differentiation(const unsigned int &n_soln_dofs_cell, 
                                                       const unsigned int &n_metric_dofs_cell,
                                                       const std::vector< FadFadType > &soln_coeff_fine,
                                                       const std::vector< FadFadType > &soln_coeff_interpolated,
                                                       const std::vector< FadFadType > &coords_coeff,
                                                       const std::vector<dealii::types::global_dof_index> cell_soln_dofs_indices,
                                                       const std::vector<dealii::types::global_dof_index> cell_metric_dofs_indices,
                                                       const bool compute_second_order_derivatives);
    /// Evaluates and stores derivatives using automatic differentiation.
    void evaluate_derivatives( const unsigned int &n_soln_dofs_cell, 
                               const unsigned int &n_metric_dofs_cell,
                               const FadType cell_functional_error,
                               const std::vector<dealii::types::global_dof_index> cell_soln_dofs_indices,
                               const std::vector<dealii::types::global_dof_index> cell_metric_dofs_indices,
                               const bool compute_second_order_derivatives);
    /// Update solution fine and solution interpolated, if changed. Volume nodes are automatically updated in DG.
    void update_solution_fine_and_solution_interpolated(const VectorType &_solution_fine, const VectorType &_solution_interpolated);
    /** 
     * We compute the derivatives of the error w.r.t. solution fine, solution interpolated and volume nodes all at once and store it. However, Trilinos's ROL calls  
     * these derivatives individually. This function checks if the derivatives have already been computed for the present solution-node configuration. If it has, 
     * we just return the error already computed for the solution-node configuration.  
     */
    bool have_error_and_its_derivatives_already_been_computed(const bool compute_first_order_derivatives, const bool compute_second_order_derivatives);
};


#endif
