#ifndef __TOTAL_DERIVATIVES_OBJFUNC__
#define __TOTAL_DERIVATIVES_OBJFUNC__

#include "objective_function_for_mesh_adaptation.h"
#include "linear_solver/linear_solver.h"
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
namespace PHiLiP {

#if PHILIP_DIM==1
template <int dim, int nstate, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, int nstate, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif

class TotalDerivativeObjfunc 
{
public:
    TotalDerivativeObjfunc(std::shared_ptr<DGBase<dim,real,MeshType>> _dg);
    void compute_solution_tilde_and_solution_fine();
    void refine_or_coarsen_dg(unsigned int degree);
    void form_interpolation_matrix();
    void compute_adjoints();
    void compute_total_derivative();
    void compute_total_hessian();

    std::shared_ptr<DGBase<dim,real,MeshType>> dg;

    dealii::LinearAlgebra::distributed::Vector<real> dF_dX_total;
    dealii::TrilinosWrappers::SparseMatrix r_x;
    dealii::TrilinosWrappers::SparseMatrix r_u;
    dealii::TrilinosWrappers::SparseMatrix r_u_transpose;
    dealii::TrilinosWrappers::SparseMatrix R_x;
    dealii::TrilinosWrappers::SparseMatrix R_u;
    dealii::TrilinosWrappers::SparseMatrix R_u_transpose;

    dealii::LinearAlgebra::distributed::Vector<real> solution_fine; // U_h
    dealii::LinearAlgebra::distributed::Vector<real> solution_coarse_taylor_expanded; // U_H_tilde
    dealii::LinearAlgebra::distributed::Vector<real> solution_tilde_fine; // U_h^H tilde
    dealii::LinearAlgebra::distributed::Vector<real> solution_coarse_old; // Store old solution
    
    dealii::LinearAlgebra::distributed::Vector<real> adjoint_fine; 
    dealii::LinearAlgebra::distributed::Vector<real> adjoint_tilde;
    
    dealii::SparseMatrix<real> interpolation_matrix;
    dealii::SparsityPattern      sparsity_pattern;
    std::unique_ptr<ObjectiveFunctionMeshAdaptation<dim, nstate, real, MeshType>> objfunc;
    real objective_function_val;
};

} // namespace PHiLiP

#endif
