#ifndef __UNIT_VECTOR_PARAMETERIZATION_H__ 
#define __UNIT_VECTOR_PARAMETERIZATION_H__ 

#include "base_parameterization.hpp"

namespace PHiLiP {

/// Class to parameterize according to unit vectors, for 1D only.
template <int dim>
class UnitVectorParameterization : public BaseParameterization<dim>
{
    using VectorType = dealii::LinearAlgebra::distributed::Vector<double>; ///< Alias for dealii's parallel distributed vector.
    using MatrixType = dealii::TrilinosWrappers::SparseMatrix; ///< Alias for dealii::TrilinosWrappers::SparseMatrix.

public:
    /// Constructor
    UnitVectorParameterization(
        std::shared_ptr<HighOrderGrid<dim,double>> _high_order_grid);

    /// Destructor
    ~UnitVectorParameterization(){};
   
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

//=============== Functions specific to this class =========================================================
    
    double dk_dh(const unsigned int h_val) const;

    double dxi_dhp(const unsigned int i, const unsigned int p) const;

    double d2k_dhq_dhp(const unsigned int q, const unsigned int p) const;

    double d2xi_dhq_dhp(const unsigned int i, const unsigned int q, const unsigned int p) const;

    double d2_dh2_matrix_at_coordinate(const unsigned int i, const unsigned int j, const VectorType &lambda) const;

    void get_lambda_times_d2X_dh2(dealii::FullMatrix<double>& lambda_times_d2X_dh2, const VectorType& lambda) const;

    void v1_times_d2XdXp2_times_v2(VectorType &out_vector, const VectorType& v1, const VectorType &v2) const override; 

private:
   const unsigned int n_vol_nodes;
   const unsigned int n_control_variables;
   const double left_end; 
   const double right_end; 
   const double min_mesh_size;
   const double rho;
   double control_var_norm_squared;
   double scaling_k;
   VectorType current_control_variables;
}; //class ends

} // PHiLiP namespace

#endif
