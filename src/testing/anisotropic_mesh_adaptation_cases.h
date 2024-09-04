#ifndef __ANISOTROPICMESHADAPTATIONCASES_H__ 
#define __ANISOTROPICMESHADAPTATIONCASES_H__ 

#include "tests.h"
#include "dg/dg.h"
#include "physics/physics.h"
#include "parameters/all_parameters.h"

namespace PHiLiP {
namespace Tests {

/// Test to check anisotropic mesh adaptation.
template <int dim, int nstate>
class AnisotropicMeshAdaptationCases : public TestsBase
{
public:
    /// Constructor
    AnisotropicMeshAdaptationCases(const Parameters::AllParameters *const parameters_input,
                                       const dealii::ParameterHandler &parameter_handler_input);
    
    /// Parameter handler.
    const dealii::ParameterHandler &parameter_handler;

    /// Runs the test related to anisotropic mesh adaptation.
    int run_test() const;

    /// Checks PHiLiP::FEValuesShapeHessian for MappingFEField with dealii's shape hessian for MappingQGeneric.
    void verify_fe_values_shape_hessian(const DGBase<dim, double> &dg) const;
    
    /// Evaluates \f[ J_exact - J(u_h) \f].
    double evaluate_functional_error(std::shared_ptr<DGBase<dim,double>> dg) const;
    
    /// Evaluates \f[ J_exact - J(u_h) \f].
    double evaluate_abs_dwr_error(std::shared_ptr<DGBase<dim,double>> dg) const;
    
    /// Outputs vtk files with primal and adjoint solutions.
    double output_vtk_files(std::shared_ptr<DGBase<dim,double>> dg) const;
    
    /// Evaluates l2 norm of solution error.
    double evaluate_solution_error(std::shared_ptr<DGBase<dim,double>> dg) const;

    void evaluate_regularization_matrix(
        dealii::TrilinosWrappers::SparseMatrix &regularization_matrix,
        std::shared_ptr<DGBase<dim,double>> dg) const;

    void increase_grid_degree_and_interpolate_solution(std::shared_ptr<DGBase<dim,double>> dg) const;
    
    /// Evaluates exact solution.
    std::array<double, nstate> evaluate_soln_exact(const dealii::Point<dim> &point) const;

    /// Move nodes to the curve.
    void move_nodes_to_shock(std::shared_ptr<DGBase<dim,double>> dg) const;

}; 

} // Tests namespace
} // PHiLiP namespace

#endif

