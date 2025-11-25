#include "adjoint_march.h"

namespace PHiLiP {

template <int dim, int nstate, int n_subspace_vectors, typename MeshType>
AdjointMarch<dim,nstate,n_subspace_vectors,MeshType>::
AdjointMarch(std::shared_ptr<DGBase<dim,double,MeshType>> _dg,
             std::shared_ptr<Functional<dim,nstate,double,MeshType>> _functional,
             const std::string _restart_file_index_terminal,
             const double dt_, const double delT_, const double T_, const double T_extra_) // Total trajecotry length is T+T_extra
    : dg(_dg)
    , functional(_functional)
    , dt(dt_)
    , delT(delT_)
    , T(T_)
    , T_extra(T_extra_)
    , dg(_dg)
    , K(T/delT)
    , n_steps(delT/dt)
    , restart_file_index_terminal(_restart_file_index_terminal)
{
    R_vec.resize(K);
    s_vec.resize(K);
    b_vec.resize(K);
    d_vec.resize(K);
    h_vec.resize(K);
    d_f_vec.resize(K);
    h_f_vec.resize(K);
    weights_simpson_nsteps.resize(n_steps+1);
    for(int i=0; i<=n_steps; ++i)
    {
        double w = 1.0;
        if( (i==0) || (i==n_steps)) {w = 17.0/48.0;}
        else if( (i==1) || (i==(n_steps-1))) {w = 59.0/48.0;}
        else if( (i==2) || (i==(n_steps-2))) {w = 43.0/48.0;}
        else if( (i==3) || (i==(n_steps-3))) {w = 49.0/48.0;}
        weights_simpson_nsteps[i] = w;
    }
}

template <int dim, int nstate, int n_subspace_vectors, typename MeshType>
double AdjointMarch<dim,nstate,n_subspace_vectors,MeshType>::
simpson_integration(const std::vector<double> &integrand, const int n, const double h ) const // Integration using n+1 points
{
    double val=0.0;
    for(int i=0; i<=n; ++i)
    {
        double w = 1.0;
        if( (i==0) || (i==n)) {w = 17.0/48.0;}
        else if( (i==1) || (i==(n-1))) {w = 59.0/48.0;}
        else if( (i==2) || (i==(n-2))) {w = 43.0/48.0;}
        else if( (i==3) || (i==(n-3))) {w = 49.0/48.0;}
        val+= integrand[i]*w*h;
    }
    return val;
}

template <int dim, int nstate, int n_subspace_vectors, typename MeshType>
template<int n_col>
void AdjointMarch<dim,nstate,n_subspace_vectors,MeshType>::
compute_QR_decomposition(const std::array<VectorType,n_col> &A,
                          std::array<VectorType,n_col> &Q,
                          std::array<std::array<double,n_col>,n_col> &R) const
{
    for(unsigned int i=0; i<n_col; ++i)
    {
        Q[i].reinit(dg->solution);
    }

    for(unsigned int i=0; i<n_col; ++i)
    {
        Q[i] = A[i];
        for(unsigned int j=0; j<i; ++j)
        {
            R[j][i] = Q[j]*A[i];
            Q[i] -= R[j][i]*Q[j];
        }

        R[i][i] = Q[i].l2_norm();
        Q[i]/=R[i][i];
        Q[i].update_ghost_values();        
    }
}
    


template <int dim, int nstate, int n_subspace_vectors, typename MeshType>
void AdjointMarch<dim,nstate,n_subspace_vectors,MeshType>::
advance_in_time_hom(const std::array<VectorType,n_subspace_vectors> & psi_n, 
                    std::array<VectorType,n_subspace_vectors> &psi_nminus) const
{
    VectorType temp(dg->right_hand_side);
    // Assumes dRdW is already computed at time n
    for(unsigned int k=0; k<n_subspace_vectors; ++k)
    {
        dg->system_matrix.Tvmult(temp, psi_n[k]);
        temp*=dt;
        dg->apply_inverse_global_mass_matrix(temp,psi_nminus[k]);
        psi_nminus[k] += psi_n[k];
        psi_nminus[k].update_ghost_values();
    }
}

template <int dim, int nstate, int n_subspace_vectors, typename MeshType>
void AdjointMarch<dim,nstate,n_subspace_vectors,MeshType>::
advance_in_time_nonhom(const VectorType & psi_n, 
                        VectorType &psi_nminus) const
{
    VectorType temp(dg->right_hand_side);
    // Assumes dRdW and dIdw is already computed at time n
    dg->system_matrix.Tvmult(temp, psi_n);
    temp+= functional->dIdw;
    temp*=dt;
    dg->apply_inverse_global_mass_matrix(temp,psi_nminus);
    psi_nminus += psi_n;
    psi_nminus.update_ghost_values();
}
    
template <int dim, int nstate, int n_subspace_vectors, typename MeshType>
void AdjointMarch<dim,nstate,n_subspace_vectors,MeshType>::
compute_Y_terminal(std::array< VectorType, n_subspace_vectors> &Y_terminal) const
{
    std::array< VectorType, n_subspace_vectors+1> Y_augmented;
    load_solution_at_time(T+T_extra);
    dg->assemble_residual(true);
    Y_augmented[0] = dg->right_hand_side;
    for(unsigned int i=1; i<n_subspace_vectors+1; ++i)
    {
        Y_augmented[i].reinit(dg->right_hand_side);
        if(Y_augmented[i].get_partitioner().in_local_range(i-1))
        {
            Y_augmented[i][i-1] = 1.0;
        }
    }
    for(unsigned int i=0; i<n_subspace_vectors+1; ++i)
    {
        Y_augmented[i].update_ghost_values();
    }

    std::array<VectorType,n_subspace_vectors+1> Q_augmented;
    std::array<std::array<double,n_subspace_vectors+1>,n_subspace_vectors+1> R_augmented;
    compute_QR_decomposition<n_subspace_vectors+1>(Y_augmented,Q_augmented, R_augmented);
    std::array<VectorType,n_subspace_vectors> Q_n;
    std::array<VectorType,n_subspace_vectors> Q_nminus;
    std::array<std::array<double,n_subspace_vectors>,n_subspace_vectors> R;
    for(unsigned int i=0; i<n_subspace_vectors; ++i)
    {
        Q_n[i] = Q_augmented[i+1];
        Q_n[i].update_ghost_values();
    }


    const int K_extra = T_extra/delT;

    for(int i=K+K_extra; i>K; --i)
    {
        // Integrate from Ti to T_{i-1}
        for(int j=n_steps; j>0; --j) // Move from j to j-1
        {
            const double current_time = (i-1)*delT + j*dt;
            load_solution_at_time(current_time);
            dg->assemble_residual(true);
            advance_in_time_hom(Q_n, Q_nminus);
            for(unsigned int k=0; k<n_subspace_vectors; ++k)
            {
                Q_n[k] = Q_nminus[k];
            }
        }
        // Perform QR decomposition
        compute_QR_decomposition<n_subspace_vectors>(Q_nminus, Q_n, R);
    }

    for(unsigned int i=0; i<n_subspace_vectors; ++i)
    {
        Y_terminal[i] = Q_n[i];
    }
}

template <int dim, int nstate, int n_subspace_vectors, typename MeshType>
void AdjointMarch<dim,nstate,n_subspace_vectors,MeshType>::
compute_v_terminal(VectorType &v_terminal) const
{ 
    const int m_T = T/dt;
    std::vector<double> j_vals(m_T+1);
    for(unsigned int i=0; i<m_T+1; ++i)
    {
        const double current_time = i*dt;
        load_solution_at_time(current_time);
        j_vals[i] = functional.evaluate_functional();
    }
    const double j_bar = 1.0/T * simpson_integration(j_vals,m_T,dt);
    dg->assemble_residual();
    v_terminal = ((j_bar - j_vals[m_T])/(dg->right_hand_side*dg->right_hand_side)) * dg->right_hand_side;
}
    
template <int dim, int nstate, int n_subspace_vectors, typename MeshType>
void AdjointMarch<dim,nstate,n_subspace_vectors,MeshType>::
compute_R_b_d_h_vecs()
{
    std::ofstream cout_R("R_vec.txt"); dealii::ConditionalOStream pcout_R(cout_R, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
    std::ofstream cout_b("b_vec.txt"); dealii::ConditionalOStream pcout_b(cout_b, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
    std::ofstream cout_d("d_vec.txt"); dealii::ConditionalOStream pcout_d(cout_d, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
    std::ofstream cout_h("h_vec.txt"); dealii::ConditionalOStream pcout_h(cout_h, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
    std::ofstream cout_J_c("integral_J_c.txt"); dealii::ConditionalOStream pcout_J_c(cout_J_c, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

    std::array<VectorType,n_subspace_vectors> Y;
    VectorType v;
    compute_Y_terminal(Y);
    compute_v_terminal(v);
    std::array<VectorType,n_subspace_vectors> Y_minus;
    VectorType v_minus;
    std::array<VectorType,n_subspace_vectors> Q;
    std::array<std::array<double,n_subspace_vectors>,n_subspace_vectors> R;
    
    std::array<std::vector<double>,n_subspace_vectors> integrand_d;
    for(unsigned int k=0; k<n_subspace_vectors; ++k) {integrand_d[k].resize(n_steps+1);}
    std::vector<double> integrand_h(n_steps+1);
    std::vector<double> integrand_J_c(n_steps+1);
    VectorType f_c;

    for(int i=K; i>0; --i) // Between Ti and T_{i-1}
    {
        for(int j=n_steps; j>0; --j) // Between j and j-1
        {           
           const double current_time = (i-1)*delT + j*dt;
           load_solution_at_time(current_time);
           dg->assemble_residual(true);
           functional->evaluate_functional(true);
            // Compute integrands to be integrated
            //=========================================
            compute_df_dc(f_c);
            compute_dJ_dc(integrand_J_c[j]);
            for(unsigned int k=0; k<n_subspace_vectors; ++k)
            {
                integrand_d[k][j] = Y[k]*f_c;
            }
            integrand_h[j] = v*f_c;
            //========================================
            advance_in_time_hom(Y,Y_minus);
            advance_in_time_nonhom(v,v_minus);
            for(unsigned int k=0; k<n_subspace_vectors; ++k)
            {
                Y[k] = Y_minus[k];
            }
            v = v_minus;
            if(j==1)
            {
                const double current_time_minus = current_time - dt;
                load_solution_at_time(current_time_minus);
                compute_df_dc(f_c);
                compute_dJ_dc(integrand_J_c[j-1]);
                for(unsigned int k=0; k<n_subspace_vectors; ++k)
                {
                    integrand_d[k][j-1] = Y_minus[k]*f_c;
                }
                integrand_h[j-1] = v_minus*f_c;
            }
        } // nsteps ends
        // Compute integrals
        const double integral_jc = simpson_integration(integrand_J_c, nsteps, dt);
        const double integral_h = simpson_integration(integrand_h, nsteps, dt);
        std::array<double,n_subspace_vectors> integrals_d;
        for(unsigned int k=0; k<n_subspace_vectors; ++k)
        {
            integrals_d[k] = simpson_integration(integrand_d[k], nsteps, dt);
        }

        // Compute QR and variables for the next iteration
        compute_QR_decomposition(Y_minus, Q, R);
        std::array<double, n_subspace_vectors> b;
        for(unsigned int k=0; k<n_subspace_vectors; ++k)
        {
            Y[k] = Q[k];
            b[k] = -Q[k]*v_minus;
            v += Q[k]*b[k];
        }
        v.update_ghost_values();

        // Write R, b, integral_jc, integral_h and integrals_d to file.
        for(unsigned int k1=0; k1<n_subspace_vectors; ++k1)
        {
            for(unsigned int k2 = 0; k2<n_subspace_vectors; ++k2)
            {
                pcout_R<<R[k1][k2]<<"\n";
            }
            pcout_b<<b[k1]<<"\n";
            pcout_d<<integrals_d[k1]<<"\n";
        }
        pcout_J_c<<integral_jc<<"\n";
        pcout_h<<integral_h<<"\n";
    } // K loop
}
    
template <int dim, int nstate, int n_subspace_vectors, typename MeshType>
void AdjointMarch<dim,nstate,n_subspace_vectors,MeshType>::
load_solution_at_time(const double _time)
{
    const int steps = _time/dt;
    const int restart_index = restart_index_end - ((int) (T+T_extra)/dt) + steps;

    // Computes restart index string
    std::string restart_index_string = std::to_string(restart_index);
    const unsigned int length_of_index_with_padding = 5;
    const unsigned int number_of_zeros = length_of_index_with_padding - restart_index_string.length();
    restart_index_string.insert(0, number_of_zeros, '0');
    const std::string prefix = "restart-";
    const std::string restart_filename_without_extension = prefix+restart_index_string;
#if PHILIP_DIM>1
    dg->triangulation->load(flow_solver_param.restart_files_directory_name + std::string("/") + restart_filename_without_extension);
    
    // Note: Future development with hp-capabilities, see section "Note on usage with DoFHandler with hp-capabilities"
    // ----- Ref: https://www.dealii.org/current/doxygen/deal.II/classparallel_1_1distributed_1_1SolutionTransfer.html
    dealii::LinearAlgebra::distributed::Vector<double> solution_no_ghost;
    solution_no_ghost.reinit(dg->locally_owned_dofs, this->mpi_communicator);
    dealii::parallel::distributed::SolutionTransfer<dim, dealii::LinearAlgebra::distributed::Vector<double>, dealii::DoFHandler<dim>> solution_transfer(dg->dof_handler);
    solution_transfer.deserialize(solution_no_ghost);
    dg->solution = solution_no_ghost; //< assignment
    dg->solution.update_ghost_values();
#endif
}

/*
template<int n_int_grid_points,int n_subspace_vectors>
int AdjointMarch<n_int_grid_points,n_subspace_vectors>::
compute_unstable_subspace_dimension() const
{
    std::array<double,n_subspace_vectors> lyapunov_exponents;
    for(int i=0; i<n_subspace_vectors; ++i)
    {
        lyapunov_exponents[i] = 0.0;
        for(int j=0; j<K; ++j)
        {
            lyapunov_exponents[i] += log(R_vec[j][i][i]);
        }
        lyapunov_exponents[i]/=T;
    }

    int dimension_unstable = 0;
    for(int i=0; i<n_subspace_vectors; ++i)
    {
        if(lyapunov_exponents[i]>0.0) {++dimension_unstable;}
        else {break;}
    }
/a*
    for(int i=0;i<n_subspace_vectors; ++i)
    {
        for(int j=0; j<K; ++j)
        {
            std::cout<<R_vec[j][i][i];
            if(j<(K-1)) {std::cout<<" ";}
            else {std::cout<<" Lyapunov exponent = "<<lyapunov_exponents[i]<<"\n"<<"\n";}
        }
    }

    std::cout<<"Unstable subspace dimension = "<<dimension_unstable<<std::endl;
*a/
    return dimension_unstable;
}
    
template<int n_int_grid_points,int n_subspace_vectors>
void AdjointMarch<n_int_grid_points,n_subspace_vectors>::
compute_s_stable_backward_march()
{
    for(int i=n_unstable; i<n_subspace_vectors; ++i)
    {
        s_vec[K-1][i] = 0.0;
    }

    for(int i=(K-1); i>=1; --i)
    {
        for(int j=n_unstable; j<n_subspace_vectors; ++j)
        {
            s_vec[i-1][j] = -b_vec[i][j];
            for(int k=j; k<n_subspace_vectors; ++k)
            {
                s_vec[i-1][j] += R_vec[i][j][k]*s_vec[i][k];
            }
        }
    }
}

template<int n_int_grid_points,int n_subspace_vectors>
void AdjointMarch<n_int_grid_points,n_subspace_vectors>::
compute_s_unstable_forward_march()
{
    for(int i=0; i<K; ++i)
    {
        const bool i_is_positive = (i>0);
        for(int j=(n_unstable-1); j>=0; --j)
        {
            // Compute rj
            double rj=b_vec[i][j];
            for(int k=n_unstable; k<n_subspace_vectors; ++k)
            {
                rj -= R_vec[i][j][k]*s_vec[i][k];
            }
            if(i_is_positive) {rj+= s_vec[i-1][j];}

            // Compute sj
            s_vec[i][j] = rj;
            for(int k=j+1; k<n_unstable; ++k)
            {
                s_vec[i][j] -= R_vec[i][j][k]*s_vec[i][k];
            }
            s_vec[i][j]/=R_vec[i][j][j];
        }
    }
}

template<int n_int_grid_points,int n_subspace_vectors>
double AdjointMarch<n_int_grid_points,n_subspace_vectors>::
compute_sensitivity()
{
    compute_R_b_d_h_vecs();
    n_unstable = compute_unstable_subspace_dimension();
    compute_s_stable_backward_march();
    compute_s_unstable_forward_march();

    // J_c = 0, hence ignoring that term.
    double sensitivity = 0.0;
    for(int i=0; i<K; ++i)
    {
        for(int j=0; j<n_subspace_vectors; ++j)
        {
            sensitivity += s_vec[i][j]*d_vec[i][j];
        }
        sensitivity += h_vec[i];
    }
    sensitivity/=T;
    return sensitivity;
}

template<int n_int_grid_points,int n_subspace_vectors>
double AdjointMarch<n_int_grid_points,n_subspace_vectors>::
compute_f_dot_adjoint_average() const
{
    double f_dot_adj_avg = 0.0;
    for(int i=0; i<K; ++i)
    {
        for(int j=0; j<n_subspace_vectors; ++j)
        {
            f_dot_adj_avg += s_vec[i][j]*d_f_vec[i][j];
        }
        f_dot_adj_avg += h_f_vec[i];
    }
    f_dot_adj_avg/=T;
    return abs(f_dot_adj_avg);
}
*/
template class AdjointMarch<127,20>;
template class AdjointMarch<255,20>;
template class AdjointMarch<511,20>;

} // PHiLiP namespace
