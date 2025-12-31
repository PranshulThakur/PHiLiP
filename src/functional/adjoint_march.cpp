#include "adjoint_march.h"
#include "dg/dg_factory.hpp"

namespace PHiLiP {

template <int dim, int nstate, int n_subspace_vectors, typename MeshType>
AdjointMarch<dim,nstate,n_subspace_vectors,MeshType>::
AdjointMarch(std::shared_ptr<DGBase<dim,double,MeshType>> _dg,
             const int _restart_index_terminal,
             const double dt_, const double delT_, const double T_, const double T_extra_, const double _perturbation_val) // Total trajecotry length is T+T_extra
    : dg(_dg)
    , restart_index_terminal(_restart_index_terminal)
    , dt(dt_)
    , delT(delT_)
    , T(T_)
    , T_extra(T_extra_)
    , K(T/delT)
    , nsteps(delT/dt)
    , perturbation_val(_perturbation_val)
    , param_perturbed(*(dg->all_parameters))
    , pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
{
    // Initialize and allocate perturbed variables
    dg_perturbed = DGFactory<dim,double>::create_discontinuous_galerkin(&param_perturbed, param_perturbed.flow_solver_param.poly_degree, param_perturbed.flow_solver_param.max_poly_degree_for_adaptation, param_perturbed.flow_solver_param.grid_degree, dg->triangulation);
    dg_perturbed->c_ks = dg->c_ks + perturbation_val;
    dg_perturbed->allocate_system(false,false,false);

    functional = FunctionalFactory<dim,nstate,double,MeshType>::create_Functional(dg->all_parameters, dg);
    functional_perturbed = FunctionalFactory<dim,nstate,double,MeshType>::create_Functional(&param_perturbed, dg_perturbed);
    // Initialize rk variables
    for(int istage = 0; istage<n_rk_stages; ++istage)
    {
        Ytilde_rk[istage].reinit(dg->solution);
        if(istage<(n_rk_stages-1))
        {
            mass_inv_residuals_rk[istage].reinit(dg->solution);
        }

        for(int jstage = 0; jstage<n_rk_stages; ++jstage)
        {
            a_rk[istage][jstage] = 0.0;
        }

        b_rk[istage] = 0.0;
    }

    a_rk[1][0] = 1.0; a_rk[2][0] = 0.25; a_rk[2][1] = 0.25;
    b_rk[0] = 1.0/6.0; b_rk[1] = 1.0/6.0; b_rk[2] = 2.0/3.0;
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
        for(unsigned int j = 0; j<i; ++j)
        {
            R[i][j] = 0.0;
        }
    }

    for(unsigned int i=0; i<n_col; ++i)
    {
        Q[i] = A[i];
        for(unsigned int j=0; j<i; ++j)
        {
            R[j][i] = Q[j]*A[i];
            VectorType temp = Q[j];
            temp *= R[j][i];
            Q[i] -= temp;
        }

        R[i][i] = Q[i].l2_norm();
        Q[i]/=R[i][i];
        Q[i].update_ghost_values();        
    }

    // Check for linear independence
    for(int i=0; i<n_col; ++i)
    {
        if(R[i][i]<1.0e-5) 
        {
            std::cout<<"Linearly dependent"<<std::endl;
            std::abort();
        }
    }
}

template <int dim, int nstate, int n_subspace_vectors, typename MeshType>
void AdjointMarch<dim,nstate,n_subspace_vectors,MeshType>::
advance_in_time(const std::array<VectorType,n_subspace_vectors+1> & psi_nplus, 
                std::array<VectorType,n_subspace_vectors+1> &psi_n,
                const bool compute_nonhom_term)
{
    // Assumes the solution is already loaded in at time n
     
    // Compute and store Ytilde
    for(int istage = 0; istage<n_rk_stages; ++istage)
    {
        Ytilde_rk[istage] = 0;

        for( int jstage=0; jstage<istage; ++jstage)
        {
            Ytilde_rk[istage].add(a_rk[istage][jstage],mass_inv_residuals_rk[jstage]);
        }
        Ytilde_rk[istage] *= dt;

        Ytilde_rk[istage] += dg->solution;

        if(istage <(n_rk_stages-1))
        {
            dg->solution = Ytilde_rk[istage];
            dg->solution.update_ghost_values();
            dg->assemble_residual();
            dg->apply_inverse_global_mass_matrix(dg->right_hand_side,mass_inv_residuals_rk[istage]);
            dg->solution = Ytilde_rk[0];
        }
    }

    for(int kstage=n_rk_stages-1; kstage>=0; --kstage)
    {
        dg->solution = Ytilde_rk[kstage];
        dg->solution.update_ghost_values();
        std::array<VectorType,n_subspace_vectors+1> temp;
        for(unsigned int s=0;s<n_subspace_vectors+1; ++s)
        {
            temp[s] = psi_nplus[s];
            temp[s] *= b_rk[kstage];
            for(unsigned int jstage=kstage+1; jstage<n_rk_stages; ++jstage)
            {
                temp[s].add(a_rk[jstage][kstage],lambda_rk[jstage][s]);
            }
            temp[s] *= dt;
        }
        
        apply_f_u_transposed(temp,lambda_rk[kstage]);
        if(compute_nonhom_term)
        {
            functional->evaluate_functional(true);
            lambda_rk[kstage][n_subspace_vectors].add(dt*b_rk[kstage],functional->dIdw);
        }
    }

    // Compute psi_n
    for(unsigned int s=0; s<n_subspace_vectors+1; ++s)
    {
        psi_n[s] = psi_nplus[s];
        for(int kstage=0; kstage<n_rk_stages; ++kstage)
        {
            psi_n[s] += lambda_rk[kstage][s];
        }
    }

    // residual has been assembled at time minus.
    // functional has been assembled at time minus if compute_nonhom_term == true.
}
template <int dim, int nstate, int n_subspace_vectors, typename MeshType>
void AdjointMarch<dim,nstate,n_subspace_vectors,MeshType>::
advance_in_time_hom(const std::array<VectorType,n_subspace_vectors> & psi_n, 
                    std::array<VectorType,n_subspace_vectors> &psi_nminus)
{
    // Assumes DG solution is loaded in at time nminus
    std::array<VectorType,n_subspace_vectors+1> psi_n_aug;
    std::array<VectorType,n_subspace_vectors+1> psi_nminus_aug;
    for(unsigned int k=0; k<n_subspace_vectors; ++k)
    {
        psi_n_aug[k] = psi_n[k];
        psi_nminus_aug[k].reinit(dg->right_hand_side);
    }
    psi_n_aug[n_subspace_vectors].reinit(dg->right_hand_side); 
    psi_n_aug[n_subspace_vectors]=0;
    psi_nminus_aug[n_subspace_vectors].reinit(dg->right_hand_side);

    const bool compute_nonhom_term = false;
    advance_in_time(psi_n_aug,psi_nminus_aug,compute_nonhom_term);

    for(unsigned int k=0; k<n_subspace_vectors; ++k)
    {
        psi_nminus[k] = psi_nminus_aug[k];
    }
    // residual has been assembled at time minus.
}

template <int dim, int nstate, int n_subspace_vectors, typename MeshType>
void AdjointMarch<dim,nstate,n_subspace_vectors,MeshType>::
advance_in_time_hom_and_nonhom(const std::array<VectorType,n_subspace_vectors> & Y_n,
                                const VectorType &v_n,
                                std::array<VectorType,n_subspace_vectors> & Y_nminus,
                                VectorType &v_nminus)
{
    // Assumes DG solution is loaded in at time nminus
    std::array<VectorType,n_subspace_vectors+1> psi_n_aug;
    std::array<VectorType,n_subspace_vectors+1> psi_nminus_aug;
    for(unsigned int k=0; k<n_subspace_vectors; ++k)
    {
        psi_n_aug[k] = Y_n[k];
        psi_nminus_aug[k].reinit(dg->right_hand_side);
    }
    psi_n_aug[n_subspace_vectors]= v_n;
    psi_nminus_aug[n_subspace_vectors].reinit(dg->right_hand_side);

    const bool compute_nonhom_term = true;
    advance_in_time(psi_n_aug,psi_nminus_aug,compute_nonhom_term);

    for(unsigned int k=0; k<n_subspace_vectors; ++k)
    {
        Y_nminus[k] = psi_nminus_aug[k];
    }
    v_nminus = psi_nminus_aug[n_subspace_vectors];
    // residual has been assembled at time minus.
    // functional has been assembled at time minus.
}
    
template <int dim, int nstate, int n_subspace_vectors, typename MeshType>
void AdjointMarch<dim,nstate,n_subspace_vectors,MeshType>::
compute_Y_terminal(std::array< VectorType, n_subspace_vectors> &Y_terminal) 
{
    std::array< VectorType, n_subspace_vectors+1> Y_augmented;
    load_solution_at_time(T+T_extra);
    dg->assemble_residual();
    Y_augmented[0] = dg->right_hand_side;
    dg->apply_inverse_global_mass_matrix(dg->right_hand_side,Y_augmented[0]);
    for(unsigned int i=1; i<n_subspace_vectors+1; ++i)
    {
        Y_augmented[i].reinit(dg->right_hand_side);
        Y_augmented[i] = 0;
        if(Y_augmented[i].get_partitioner()->in_local_range(i-1))
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
        Q_nminus[i] = Q_n[i];
    }


    const int K_extra = T_extra/delT;

    pcout<<" Initial runs homogeneous:"<<std::endl;
    for(int i=K+K_extra; i>K; --i)
    {
        // Integrate from Ti to T_{i-1}
        for(int j=nsteps; j>0; --j) // Move from j to j-1
        {
            const double current_time = (i-1)*delT + j*dt;
            pcout<<"Time: "<<current_time<<std::endl;
            load_solution_at_time(current_time-dt);
            advance_in_time_hom(Q_n, Q_nminus);
            for(unsigned int k=0; k<n_subspace_vectors; ++k)
            {
                Q_n[k] = Q_nminus[k];
            }
        }
        pcout<<"================================================================"<<std::endl;
        // Perform QR decomposition
        //std::cout<<"Computing QR Decomposition"<<std::endl;
        compute_QR_decomposition<n_subspace_vectors>(Q_nminus, Q_n, R);
        //std::cout<<"Done computing QR Decomposition"<<std::endl;
    }

    for(unsigned int i=0; i<n_subspace_vectors; ++i)
    {
        Y_terminal[i] = Q_n[i];
    }
}

template <int dim, int nstate, int n_subspace_vectors, typename MeshType>
void AdjointMarch<dim,nstate,n_subspace_vectors,MeshType>::
compute_v_terminal(VectorType &v_terminal)
{ 
    const unsigned int m_T = T/dt;
    std::vector<double> j_vals(m_T+1);
    for(unsigned int i=0; i<m_T+1; ++i)
    {
        const double current_time = i*dt;
        load_solution_at_time(current_time);
        j_vals[i] = functional->evaluate_functional();
    }
    const double j_bar = 1.0/T * simpson_integration(j_vals,m_T,dt);
    dg->assemble_residual();
    VectorType f_val = dg->right_hand_side;
    dg->apply_inverse_global_mass_matrix(dg->right_hand_side,f_val);
    v_terminal = f_val;
    v_terminal *= ((j_bar - j_vals[m_T])/(f_val*f_val));
    v_terminal.update_ghost_values();
    // Residual and the functional have been evaluated at time T.
}
    
template <int dim, int nstate, int n_subspace_vectors, typename MeshType>
void AdjointMarch<dim,nstate,n_subspace_vectors,MeshType>::
compute_R_b_d_h_vecs()
{
    //std::cout<<"Here 1"<<std::endl;
    std::ofstream cout_R("R_vec.txt"); dealii::ConditionalOStream pcout_R(cout_R, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
    std::ofstream cout_b("b_vec.txt"); dealii::ConditionalOStream pcout_b(cout_b, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
    std::ofstream cout_d("d_vec.txt"); dealii::ConditionalOStream pcout_d(cout_d, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
    std::ofstream cout_h("h_vec.txt"); dealii::ConditionalOStream pcout_h(cout_h, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
    std::ofstream cout_J_c("integral_J_c.txt"); dealii::ConditionalOStream pcout_J_c(cout_J_c, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

    std::array<VectorType,n_subspace_vectors> Y;
    VectorType v;
    //std::cout<<"Here 2"<<std::endl;
    compute_Y_terminal(Y);
    //std::cout<<"Here 3"<<std::endl;
    compute_v_terminal(v);
    //std::cout<<"Here 4"<<std::endl;
    std::array<VectorType,n_subspace_vectors> Y_minus;
    VectorType v_minus;
    // Initialize minus vectors
    v_minus = v;
    for(unsigned int k=0; k<n_subspace_vectors; ++k)
    {
        Y_minus[k] = Y[k];
    }
    std::array<VectorType,n_subspace_vectors> Q;
    std::array<std::array<double,n_subspace_vectors>,n_subspace_vectors> R;
    
    std::array<std::vector<double>,n_subspace_vectors> integrand_d;
    for(unsigned int k=0; k<n_subspace_vectors; ++k) 
    {
        integrand_d[k].resize(nsteps+1);
        lyapunov_exp[k] = 0.0;
    }
    std::vector<double> integrand_h(nsteps+1);
    std::vector<double> integrand_J_c(nsteps+1);
    VectorType f_c;

    pcout<<"Runs nonhomogeneous:"<<std::endl;
    for(int i=K; i>0; --i) // Between Ti and T_{i-1}
    {
        for(int j=nsteps; j>0; --j) // Between j and j-1
        {           
            const double current_time = (i-1)*delT + j*dt;
            if(j==nsteps)
            {
                compute_df_dc_and_dJ_dc(f_c,integrand_J_c[j]);
                for(unsigned int k=0; k<n_subspace_vectors; ++k)
                {
                    integrand_d[k][j] = Y[k]*f_c;
                }
                integrand_h[j] = v*f_c;
            }
            pcout<<"Time: "<<current_time<<std::endl;
            load_solution_at_time(current_time-dt);
            advance_in_time_hom_and_nonhom(Y,v,Y_minus,v_minus);
            for(unsigned int k=0; k<n_subspace_vectors; ++k)
            {
                Y[k] = Y_minus[k];
            }
            v = v_minus;
            // Compute integrands to be integrated
            //=========================================
            compute_df_dc_and_dJ_dc(f_c,integrand_J_c[j-1]);
            for(unsigned int k=0; k<n_subspace_vectors; ++k)
            {
                integrand_d[k][j-1] = Y[k]*f_c;
            }
            integrand_h[j-1] = v*f_c;
            //========================================
        } // nsteps ends
        pcout<<"================================================================"<<std::endl;
        // Compute integrals
        const double integral_jc = simpson_integration(integrand_J_c, nsteps, dt);
        const double integral_h = simpson_integration(integrand_h, nsteps, dt);
        std::array<double,n_subspace_vectors> integrals_d;
        for(unsigned int k=0; k<n_subspace_vectors; ++k)
        {
            integrals_d[k] = simpson_integration(integrand_d[k], nsteps, dt);
        }

        // Compute QR and variables for the next iteration
        compute_QR_decomposition<n_subspace_vectors>(Y_minus, Q, R);
        std::array<double, n_subspace_vectors> b;
        for(unsigned int k=0; k<n_subspace_vectors; ++k)
        {
            Y[k] = Q[k];
            b[k] = -(Q[k]*v_minus);
            v.add(b[k],Q[k]);
        }
        v.update_ghost_values();

        // compute lyapunov exp
        for(unsigned int k=0; k<n_subspace_vectors; ++k)
        {
            lyapunov_exp[k] += log(R[k][k]);
        }

        // Write R, b, integral_jc, integral_h and integrals_d to file.
        for(unsigned int k1=0; k1<n_subspace_vectors; ++k1)
        {
            for(unsigned int k2 = 0; k2<n_subspace_vectors; ++k2)
            {
                pcout_R<<std::setprecision(16)<<R[k1][k2]<<"\n";
            }
            pcout_b<<std::setprecision(16)<<b[k1]<<"\n";
            pcout_d<<std::setprecision(16)<<integrals_d[k1]<<"\n";
        }
        pcout_J_c<<std::setprecision(16)<<integral_jc<<"\n";
        pcout_h<<std::setprecision(16)<<integral_h<<"\n";
    } // K loop
    
    pcout<<"Lyapunov exponents: "; 
    for(unsigned int k=0; k<n_subspace_vectors; ++k)
    {
        lyapunov_exp[k] /= T;
        pcout<<lyapunov_exp[k]<<", ";
    }
    pcout<<std::endl;
}

template <int dim, int nstate, int n_subspace_vectors, typename MeshType>
void AdjointMarch<dim,nstate,n_subspace_vectors,MeshType>::
compute_lyapunov_exponents()
{
    std::array<VectorType,n_subspace_vectors> Y;
    compute_Y_terminal(Y);
    std::array<VectorType,n_subspace_vectors> Y_minus;
    for(unsigned int k=0; k<n_subspace_vectors; ++k)
    {
        Y_minus[k] = Y[k];
    }
    std::array<VectorType,n_subspace_vectors> Q;
    std::array<std::array<double,n_subspace_vectors>,n_subspace_vectors> R;

    const int m_T = T/dt;
    for(unsigned int s=0; s<n_subspace_vectors; ++s)
    {
        lyapunov_exp[s] = 0.0;
    }

    for(int i=m_T; i>0; --i)
    {
        const double current_time = i*dt;
        pcout<<"Current time = "<<current_time<<std::endl;
        load_solution_at_time(current_time-dt);
        advance_in_time_hom(Y,Y_minus);
        compute_QR_decomposition<n_subspace_vectors>(Y_minus,Q,R);
        for(unsigned int s=0; s<n_subspace_vectors; ++s)
        {
            Y[s] = Q[s];
            lyapunov_exp[s] += log(R[s][s]); 
        }

        pcout<<"Current lyapunov exponents: ";
        const double elapsed_time = T - (current_time-dt);
        for(unsigned int k=0; k<n_subspace_vectors; ++k)
        {
            pcout<<lyapunov_exp[k]/elapsed_time<<", ";
        }
        pcout<<std::endl;
    }
    
    pcout<<"Lyapunov exponents: "; 
    for(unsigned int k=0; k<n_subspace_vectors; ++k)
    {
        lyapunov_exp[k] /= T;
        pcout<<lyapunov_exp[k]<<", ";
    }
    pcout<<std::endl;

}
    
template <int dim, int nstate, int n_subspace_vectors, typename MeshType>
void AdjointMarch<dim,nstate,n_subspace_vectors,MeshType>::
load_solution_at_time(const double _time)
{
    const int steps = _time/dt;
    const int restart_index = restart_index_terminal - (int)((T+T_extra)/dt) + steps;

    // Computes restart index string
    std::string restart_index_string = std::to_string(restart_index);
    const unsigned int length_of_index_with_padding = 5;
    const unsigned int number_of_zeros = length_of_index_with_padding - restart_index_string.length();
    restart_index_string.insert(0, number_of_zeros, '0');
    const std::string prefix = "restart-";
    const std::string restart_filename_without_extension = prefix+restart_index_string;
    //std::cout<<restart_filename_without_extension<<std::endl;
    load_solution(dg->all_parameters->flow_solver_param.restart_files_directory_name + std::string("/") + restart_filename_without_extension);
}

template <int dim, int nstate, int n_subspace_vectors, typename MeshType>
void AdjointMarch<dim,nstate,n_subspace_vectors,MeshType>::
load_solution(const std::string filename)
{
    std::ifstream sol_file(filename);
    if (!sol_file.is_open()) { // 2. Check if the file opened successfully
        std::cerr << "Error opening file: " << filename << std::endl;
        std::abort(); // Exit if file cannot be opened
    }
    for(unsigned int i=0; i<dg->dof_handler.n_dofs(); ++i)
    {
        sol_file>>dg->solution[i];
    }
    pcout << "done." << std::endl;
}
    
template <int dim, int nstate, int n_subspace_vectors, typename MeshType>
void AdjointMarch<dim,nstate,n_subspace_vectors,MeshType>::
compute_df_dc_and_dJ_dc(VectorType &f_c, double &J_c)
{
    // Assumes dg->assemble_residual() and functional->evaluate_functional() have already been called earlier.
    dg_perturbed->solution = dg->solution;
    dg_perturbed->solution.update_ghost_values();
    dg_perturbed->assemble_residual();

    VectorType R_c = dg_perturbed->right_hand_side;
    R_c -= dg->right_hand_side;
    R_c /= perturbation_val;
    f_c = R_c;
    dg->apply_inverse_global_mass_matrix(R_c,f_c);
    f_c.update_ghost_values();

    J_c = functional_perturbed->evaluate_functional();
    J_c -= functional->current_functional_value;
    J_c/= perturbation_val;
}
    
template <int dim, int nstate, int n_subspace_vectors, typename MeshType>
void AdjointMarch<dim,nstate,n_subspace_vectors,MeshType>::
apply_f_u_transposed(const std::array<VectorType,n_subspace_vectors+1> &in_vec, std::array<VectorType,n_subspace_vectors+1> &out_vec)
{
    for(unsigned int k=0; k<n_subspace_vectors+1; ++k)
    {
        dg->apply_inverse_global_mass_matrix(in_vec[k],dg->duals[k]);
        dg->duals[k].update_ghost_values();
    }
    dg->assemble_residual(true);
    for(unsigned int k=0; k<n_subspace_vectors+1; ++k)
    {
        out_vec[k] = dg->duals_transpose_dRdW[k];
    }
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

template <int dim, int nstate, int n_subspace_vectors, typename MeshType>
void AdjointMarch<dim,nstate,n_subspace_vectors,MeshType>::
load_R_b_d_h_vecs()
{
    R_vec.resize(K);
    s_vec.resize(K);
    b_vec.resize(K);
    d_vec.resize(K);
    h_vec.resize(K);
    integralJc_vec.resize(K);
    
    std::ifstream in_R("R_vec.txt");
    std::ifstream in_b("b_vec.txt");
    std::ifstream in_d("d_vec.txt");
    std::ifstream in_h("h_vec.txt");
    std::ifstream in_Jc("integral_J_c.txt");
    for(unsigned int i=K-1; i>=0; --i)
    {
        
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
#if PHILIP_DIM==1
template class AdjointMarch<PHILIP_DIM, PHILIP_DIM, 20, dealii::Triangulation<PHILIP_DIM>>;
template class AdjointMarch<PHILIP_DIM, PHILIP_DIM, 30, dealii::Triangulation<PHILIP_DIM>>;
#endif
#if PHILIP_DIM != 1
template class AdjointMarch<PHILIP_DIM, PHILIP_DIM+2, 30, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class AdjointMarch<PHILIP_DIM, PHILIP_DIM+2, 20, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class AdjointMarch<PHILIP_DIM, PHILIP_DIM+2, 12, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif

} // PHiLiP namespace
