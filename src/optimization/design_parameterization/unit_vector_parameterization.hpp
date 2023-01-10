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
    void initialize_design_variables(VectorType &design_var) override;
    
    /// Computes the derivative of volume nodes w.r.t. control variables.
    void compute_dXv_dXp(MatrixType &dXv_dXp) const override;
    
    /// Computes the derivative of volume nodes w.r.t. control variables.
    void update_dXv_dXp(MatrixType &dXv_dXp) const override;

    void set_lambda_d2XdXp2(VectorType &lambda_input) override;

    void lambda_d2Xdp2_vmult(VectorType &out_vector, const VectorType& in_vector) const override;
    
    /// Checks if the design variables have changed and updates the mesh. 
    bool update_mesh_from_design_variables(
        const MatrixType &dXv_dXp,
        const VectorType &design_var) override;

    /// Returns the number of design variables.
    unsigned int get_number_of_design_variables() const override;

//=============== Functions specific to this class =========================================================
    
    double dk_dh(unsigned int h_val);

    double d2k_dh1_dh2(unsigned int hval1, unsigned int hval2);

    double dxi_dhp(unsigned int i, unsigned int p);

    double d2xi_dhq_dhp(unsigned int i, unsigned int q, unsigned int p);

private:
   const double left_end; 
   const double right_end; 
   const unsigned int n_control_variables;
   const double min_mesh_size;
   const double rho;
   VectorType lambda;
   MatrixType lambda_d2Xdp2;
}; //class ends

} // PHiLiP namespace

#endif
