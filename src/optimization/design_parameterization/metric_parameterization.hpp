#ifndef __METRIC_PARAMETERIZATION_H__ 
#define __METRIC_PARAMETERIZATION_H__ 

#include "base_parameterization.hpp"

namespace PHiLiP {

/// Class to parameterize according to metric field. Currently implemented for 1D only.
template <int dim>
class MetricParameterization : public BaseParameterization<dim>
{
    using VectorType = dealii::LinearAlgebra::distributed::Vector<double>; ///< Alias for dealii's parallel distributed vector.
    using MatrixType = dealii::TrilinosWrappers::SparseMatrix; ///< Alias for dealii::TrilinosWrappers::SparseMatrix.

public:
    /// Constructor
    MetricParameterization(
        std::shared_ptr<HighOrderGrid<dim,double>> _high_order_grid);

    /// Destructor
    ~MetricParameterization(){};
   
//=============Functions overridden from base parameterization ===============================================  
    /// Initializes design variables.
    void initialize_design_variables(VectorType &control_var) override;
    
    /// Computes the derivative of volume nodes w.r.t. control variables.
    void compute_dXv_dXp(MatrixType &dXv_dXp) const override;
    
    /// Updates the derivative of volume nodes w.r.t. current control variables.
    void update_dXv_dXp(MatrixType &dXv_dXp) const override;

    /// Checks if the design variables have changed and updates the mesh. 
    bool update_mesh_from_design_variables(
        const MatrixType &/*dXv_dXp*/,
        const VectorType &design_var) override;

    /// Returns the number of design variables.
    unsigned int get_number_of_design_variables() const override;
    
    /// Return the norm of control variables. 
    double control_var_norm() const override;
    
	/// Checks if the updated control variable doesn't distort the mesh (which is possible when backtracking with high initial step length). Returns 0 if everything is good.
    int is_design_variable_valid(const MatrixType &/*dXv_dXp*/, const VectorType &control_var) const override;

private:
   const unsigned int n_vol_nodes;
   const unsigned int n_control_variables;
   VectorType current_control_variables;
}; //class ends

} // PHiLiP namespace

#endif
