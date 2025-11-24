#ifndef __ADJOINT_MARCH_H__
#define __ADJOINT_MARCH_H__

#include "dg/dg_base.hpp"
#include "physics/physics.h"
#include"functional.h"

namespace PHiLiP {

#if PHILIP_DIM==1
template <int dim, int nstate, int n_subspace_vectors, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, int nstate, int n_subspace_vectors, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif

class AdjointMarch
{
    using VectorType = dealii::LinearAlgebra::distributed::Vector<double>;
    const double dt;
    const double delT;
    const double T;
    const double T_extra;
    const int K;
    const int n_steps;
    const std::string restart_file_index_terminal,
    /// Smart pointer to DGBase
    std::shared_ptr<DGBase<dim,double,MeshType>> dg;
    std::shared_ptr<Functional<dim,nstate,double,MeshType>> functional;
    std::vector<int> unstable_indices;
    std::vector<int> stable_indices;
    std::vector<int> neutral_indices;
    std::vector<std::array<std::array<double,n_subspace_vectors>,n_subspace_vectors>> R_vec;
    std::vector<std::array<double,n_subspace_vectors>> s_vec;
    std::vector<std::array<double,n_subspace_vectors>> b_vec;
    std::vector<std::array<double,n_subspace_vectors>> d_vec;
    std::vector<double> h_vec;
    std::vector<std::array<double,n_subspace_vectors>> d_f_vec;
    std::vector<double> h_f_vec;
    std::vector<double> weights_simpson_nsteps;

    void compute_s_stable_backward_march();
    void compute_s_unstable_forward_march();
    void compute_unstable_neutral_stable_subspace_indices();
    void compute_Y_terminal(std::array< VectorType, n_subspace_vectors> &Y_terminal) const;
    void compute_v_terminal(VectorType &v_terminal) const;
    
    template<int collength>
    void compute_QR_decomposition(const std::array<VectorType,collength> &A,
                                  std::array<VectorType,collength> &Q,
                                  std::array<std::array<double,collength>,collength> &R) const;

    double simpson_integration(const std::vector<double> &integrand, const int n, const double h ) const; // Integration using n+1 points

    void compute_R_b_d_h_vecs();
    void store_R_b_d_h_in_files(const std::array<std::array<double,n_subspace_vectors>,n_subspace_vectors> &R,
                                const std::array<double,n_subspace_vectors> &b,
                                const std::array<double,n_subspace_vectors> &d,
                                const std::array<double,n_subspace_vectors> &h) const;
    void load_solution_at_time(const double _time);
    void advance_in_time_hom(const std::array<VectorType,n_subspace_vectors> & psi_n, 
                             std::array<VectorType,n_subspace_vectors> &psi_nminus) const;
    void advance_in_time_nonhom(const VectorType & psi_n, 
                                VectorType &psi_nminus) const;

    void compute_df_dc(VectorType &f_c);
    void compute_dJ_dc(double &J_c);

public:
    AdjointMarch(std::shared_ptr<DGBase<dim,double,MeshType>> _dg,
                 std::shared_ptr<Functional<dim,nstate,double,MeshType>> _functional,
                 const std::string _restart_file_index_terminal,
                 const double dt_, const double delT_, const double T_, const double T_extra_); // Total trajecotry length is T+T_extra
    ~AdjointMarch(){};
    double compute_sensitivity();
    double compute_f_dot_adjoint_average() const;

};
} // PHiLiP namespace
#endif

