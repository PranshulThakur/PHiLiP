#ifndef __ADJOINT_MARCH_H__
#define __ADJOINT_MARCH_H__

#include "dg/dg_base.hpp"
#include "physics/physics.h"
#include "functional.h"

namespace PHiLiP {

#if PHILIP_DIM==1
template <int dim, int nstate, int n_subspace_vectors, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, int nstate, int n_subspace_vectors, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif

class AdjointMarch
{
    using VectorType = dealii::LinearAlgebra::distributed::Vector<double>;
    std::shared_ptr<DGBase<dim,double,MeshType>> dg;
    const int restart_index_terminal;
    const double dt;
    const double delT;
    const double T;
    const double T_extra;
    const int K;
    const int nsteps;
    const double perturbation_mach;
    Parameters::AllParameters param_perturbed;
    dealii::ConditionalOStream pcout; ///< Parallel std::cout that only outputs on mpi_rank==0

    static const int n_rk_stages = 3;
    std::array<VectorType,n_rk_stages> Ytilde_rk;
    std::array<VectorType,n_rk_stages-1> mass_inv_residuals_rk;

    std::array<std::array<double,n_rk_stages>,n_rk_stages> a_rk;
    std::array<double,n_rk_stages> b_rk;
    std::array< std::array<VectorType,n_subspace_vectors+1>, n_rk_stages> lambda_rk;
    
    std::shared_ptr<DGBase<dim,double,MeshType>> dg_perturbed;
    std::shared_ptr<Functional<dim,nstate,double,MeshType>> functional;
    std::shared_ptr<Functional<dim,nstate,double,MeshType>> functional_perturbed;
    std::vector<int> unstable_indices;
    std::vector<int> stable_indices;
    std::vector<int> neutral_indices;
    std::vector<std::array<std::array<double,n_subspace_vectors>,n_subspace_vectors>> R_vec;
    std::vector<std::array<double,n_subspace_vectors>> s_vec;
    std::vector<std::array<double,n_subspace_vectors>> b_vec;
    std::vector<std::array<double,n_subspace_vectors>> d_vec;
    std::vector<double> h_vec;
    std::array<double, n_subspace_vectors> lyapunov_exp;
    void compute_s_stable_backward_march();
    void compute_s_unstable_forward_march();
    void compute_unstable_neutral_stable_subspace_indices();
    void compute_Y_terminal(std::array< VectorType, n_subspace_vectors> &Y_terminal);
    void compute_v_terminal(VectorType &v_terminal);
    
    template<int collength>
    void compute_QR_decomposition(const std::array<VectorType,collength> &A,
                                  std::array<VectorType,collength> &Q,
                                  std::array<std::array<double,collength>,collength> &R) const;

    double simpson_integration(const std::vector<double> &integrand, const int n, const double h ) const; // Integration using n+1 points

    void advance_in_time_hom(const std::array<VectorType,n_subspace_vectors> & psi_n, 
                             std::array<VectorType,n_subspace_vectors> &psi_nminus);
    //void advance_in_time_nonhom(const VectorType & psi_n, 
    //                            VectorType &psi_nminus);
    void advance_in_time_hom_and_nonhom(const std::array<VectorType,n_subspace_vectors> & Y_n,
                                        const VectorType &v_n,
                                        std::array<VectorType,n_subspace_vectors> & Y_nminus,
                                        VectorType &v_nminus);
    void advance_in_time(const std::array<VectorType,n_subspace_vectors+1> & psi_nplus, 
                         std::array<VectorType,n_subspace_vectors+1> &psi_n,
                         const bool compute_nonhom_term);

    void apply_f_u_transposed(const std::array<VectorType,n_subspace_vectors+1> &in_vec, std::array<VectorType,n_subspace_vectors+1> &out_vec);

public:
    AdjointMarch(std::shared_ptr<DGBase<dim,double,MeshType>> _dg,
                 const int _restart_index_terminal,
                 const double dt_, const double delT_, const double T_, const double T_extra_, const double _perturbation_mach = 1.0e-5); // Total trajecotry length is T+T_extra
    ~AdjointMarch(){};
    double compute_sensitivity();
    void load_solution_at_time(const double _time);
    double compute_f_dot_adjoint_average() const;
    void compute_df_dc_and_dJ_dc(VectorType &f_c, double &J_c);
    void compute_R_b_d_h_vecs();
    void compute_lyapunov_exponents();

};
} // PHiLiP namespace
#endif

