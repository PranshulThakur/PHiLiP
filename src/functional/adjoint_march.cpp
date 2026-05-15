#include "adjoint_march.h"
#include "dg/dg_factory.hpp"
#include <deal.II/lac/qr.h>
#include "linear_solver/linear_solver.h"
#include "ode_solver/runge_kutta_methods/runge_kutta_methods.h"

namespace PHiLiP {

template <int dim, int nstate, int n_subspace_vectors, typename MeshType>
AdjointMarch<dim,nstate,n_subspace_vectors,MeshType>::
AdjointMarch(std::shared_ptr<DGBase<dim,double,MeshType>> _dg,
             const int _restart_index_terminal,
             const double dt_, const double delT_, const double T_, const double T_extra_, const double j_bar_, const bool _use_adjoint_restart_files, const double _adjoint_restart_time, const double _perturbation_val) // Total trajecotry length is T+T_extra
    : dg(_dg)
    , restart_index_terminal(_restart_index_terminal)
    , dt(dt_)
    , delT(delT_)
    , T(T_)
    , T_extra(T_extra_)
    , K(T/delT)
    , nsteps(delT/dt)
    , j_bar(j_bar_)
    , use_adjoint_restart_files(_use_adjoint_restart_files)
    , adjoint_restart_time(_adjoint_restart_time)
    , perturbation_val(_perturbation_val)
    , param_perturbed(*(dg->all_parameters))
    , pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
{
    // Initialize and allocate perturbed variables
    //param_perturbed.euler_param.mach_inf = dg->all_parameters->euler_param.mach_inf + perturbation_val;
    param_perturbed.euler_param.angle_of_attack = dg->all_parameters->euler_param.angle_of_attack + perturbation_val;
    dg_perturbed = DGFactory<dim,double>::create_discontinuous_galerkin(&param_perturbed, param_perturbed.flow_solver_param.poly_degree, param_perturbed.flow_solver_param.max_poly_degree_for_adaptation, param_perturbed.flow_solver_param.grid_degree, dg->triangulation);
    dg_perturbed->allocate_system(false,false,false);

    functional = FunctionalFactory<dim,nstate,double,MeshType>::create_Functional(dg->all_parameters, dg);
    functional_perturbed = FunctionalFactory<dim,nstate,double,MeshType>::create_Functional(&param_perturbed, dg_perturbed);
    // Initialize rk variables
    for(int istage = 0; istage<n_rk_stages; ++istage)
    {
        for(unsigned int s=0; s<n_subspace_vectors+1; ++s)
        {
            lambda_rk[istage][s].reinit(dg->solution);
            lambda_tilde_rk[istage][s].reinit(dg->solution);
        }

        for(int jstage = 0; jstage<n_rk_stages; ++jstage)
        {
            a_rk[istage][jstage] = 0.0;
        }

        b_rk[istage] = 0.0;
    }

    // For rk4
    //a_rk[1][0] = 0.5; a_rk[2][1] = 0.5; a_rk[3][2] = 1.0;
    //b_rk[0] = 1.0/6.0; b_rk[1] = 1.0/3.0; b_rk[2] = 1.0/3.0; b_rk[3] = 1.0/6.0;
    
    // For 3rd order dirk
    std::shared_ptr<PHiLiP::ODE::RKTableauButcherBase<dim,double,MeshType>> rk_tableau_butcher = std::make_shared<PHiLiP::ODE::DIRK3Implicit<dim, double, MeshType>>  (3, "3nd order diagonally-implicit (implicit)");
    std::shared_ptr<PHiLiP::ODE::EmptyRRKBase<dim,double,MeshType>> RRK_object = std::make_shared<PHiLiP::ODE::EmptyRRKBase<dim,double,MeshType>> (rk_tableau_butcher);
    rk_solver  = std::make_shared<PHiLiP::ODE::RungeKuttaODESolver<dim,double,3,MeshType>>(dg,rk_tableau_butcher,RRK_object);
    rk_solver->allocate_ode_system();
    
    for(int istage = 0; istage<3; ++istage)
    {
        for(int jstage = 0; jstage<3; ++jstage)
        {
            a_rk[istage][jstage] = rk_solver->butcher_tableau->get_a(istage,jstage);
        }
        b_rk[istage] = rk_solver->butcher_tableau->get_b(istage);
    }

    if(! use_adjoint_restart_files)
    {
        reconstruct_solution(T+T_extra - n_soln_steps_stored*dt);
    }
    else
    {
        // Find the interval in which adjoint_restart_time exists
        double reconstruct_time = T + T_extra;
        while(adjoint_restart_time <= reconstruct_time)
        {
            reconstruct_time -= n_soln_steps_stored*dt;
        }
        reconstruct_solution(reconstruct_time);
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
    dealii::QR<VectorType> qr_factorization;
    for(unsigned int i=0; i<n_col; ++i)
    {
        qr_factorization.append_column(A[i]);
        Q[i].reinit(dg->solution);
    }
    // Compute Q
    for(unsigned int i=0; i<n_col; ++i)
    {
        dealii::Vector<double> x(n_col);
        x[i] = 1.0;
        qr_factorization.multiply_with_Q(Q[i],x);
    }

    dealii::LAPACKFullMatrix<double> R_temp = qr_factorization.get_R();
    for(unsigned int i=0; i<n_col; ++i)
    {
        for(unsigned int j=0; j<n_col; ++j)
        {
            R[i][j] = R_temp(i,j);
        }
    }


/*
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
*/
    // Check for linear independence
    for(int i=0; i<n_col; ++i)
    {
        if(abs(R[i][i])<1.0e-5) 
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
    std::array<VectorType,n_subspace_vectors+1> psi_nplus_tilde;
    for(unsigned int s=0; s<n_subspace_vectors+1; ++s)
    {
        psi_nplus_tilde[s] = psi_nplus[s];
        dg->global_inverse_mass_matrix.vmult(psi_nplus_tilde[s],psi_nplus[s]);
    }

    rk_solver->step_in_time(dt,false);

    for(int kstage=n_rk_stages-1; kstage>=0; --kstage)
    {
        dg->solution = rk_solver->soln_stored[kstage];
        dg->solution.update_ghost_values();
        dg->assemble_residual(true);
        std::array<VectorType,n_subspace_vectors+1> temp;
        std::array<VectorType,n_subspace_vectors+1> rhs;
        for(unsigned int s=0;s<n_subspace_vectors+1; ++s)
        {
            temp[s] = psi_nplus_tilde[s];
            rhs[s] = temp[s];
            temp[s] *= b_rk[kstage];
            for(unsigned int jstage=kstage+1; jstage<n_rk_stages; ++jstage)
            {
                temp[s].add(a_rk[jstage][kstage],lambda_tilde_rk[jstage][s]);
            }
            temp[s] *= dt;
            dg->system_matrix_transpose.vmult(rhs[s],temp[s]);
        }
        
        if(compute_nonhom_term)
        {
            functional->evaluate_functional(true);
            rhs[n_subspace_vectors].add(dt*b_rk[kstage],functional->dIdw);
        }

        dg->system_matrix_transpose *= -dt*a_rk[kstage][kstage];
        dg->system_matrix_transpose.add(1.0,dg->global_mass_matrix);
        for(unsigned int s=0;s<n_subspace_vectors+1; ++s)
        {
            solve_linear(dg->system_matrix_transpose,rhs[s],lambda_tilde_rk[kstage][s], dg->all_parameters->linear_solver_param);
            dg->global_mass_matrix.vmult(lambda_rk[kstage][s],lambda_tilde_rk[kstage][s]);
        }
    }

    // Compute psi_n
    for(unsigned int s=0; s<n_subspace_vectors+1; ++s)
    {
        psi_n[s].reinit(dg->solution);
        psi_n[s] = psi_nplus[s];
        for(int kstage=0; kstage<n_rk_stages; ++kstage)
        {
            psi_n[s] += lambda_rk[kstage][s];
        }
    }

    // residual has been assembled at time minus.
    // functional has been assembled at time minus if compute_nonhom_term == true.
    // dg->solution is at time minus
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
        psi_nminus_aug[k].reinit(dg->solution);
    }
    psi_n_aug[n_subspace_vectors].reinit(dg->solution); 
    psi_n_aug[n_subspace_vectors]*=0;
    psi_nminus_aug[n_subspace_vectors].reinit(dg->solution);

    const bool compute_nonhom_term = false;
    advance_in_time(psi_n_aug,psi_nminus_aug,compute_nonhom_term);

    for(unsigned int k=0; k<n_subspace_vectors; ++k)
    {
        psi_nminus[k] = psi_nminus_aug[k];
    }
    // residual has been assembled at time minus.
    // dg->solution is at time minus
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
        psi_nminus_aug[k].reinit(dg->solution);
    }
    psi_n_aug[n_subspace_vectors]= v_n;
    psi_nminus_aug[n_subspace_vectors].reinit(dg->solution);

    const bool compute_nonhom_term = true;
    advance_in_time(psi_n_aug,psi_nminus_aug,compute_nonhom_term);

    for(unsigned int k=0; k<n_subspace_vectors; ++k)
    {
        Y_nminus[k] = psi_nminus_aug[k];
    }
    v_nminus = psi_nminus_aug[n_subspace_vectors];
    // residual has been assembled at time minus.
    // functional has been assembled at time minus.
    // dg->solution is at time minus
}
    
template <int dim, int nstate, int n_subspace_vectors, typename MeshType>
void AdjointMarch<dim,nstate,n_subspace_vectors,MeshType>::
compute_Y_terminal(std::array< VectorType, n_subspace_vectors> &Y_terminal) 
{
    std::array< VectorType, n_subspace_vectors+1> Y_augmented;
    get_solution_at_time(T+T_extra);
    dg->assemble_residual();
    Y_augmented[0] = dg->right_hand_side;
    dg->apply_inverse_global_mass_matrix(dg->right_hand_side,Y_augmented[0]);
    for(unsigned int i=1; i<n_subspace_vectors+1; ++i)
    {
        Y_augmented[i].reinit(dg->solution);
        Y_augmented[i]*= 0;

        if(Y_augmented[i].locally_owned_elements().is_element(i-1))
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
            get_solution_at_time(current_time-dt);
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
    /*
    const unsigned int m_T = T/dt;
    std::vector<double> j_vals(m_T+1);
    for(unsigned int i=0; i<m_T+1; ++i)
    {
        const double current_time = i*dt;
        get_solution_at_time(current_time);
        j_vals[i] = functional->evaluate_functional();
    }
    const double j_bar = 1.0/T * simpson_integration(j_vals,m_T,dt);
    */
    get_solution_at_time(T);
    const double j_val_T = functional->evaluate_functional(); 
    dg->assemble_residual();
    VectorType f_val = dg->right_hand_side;
    dg->apply_inverse_global_mass_matrix(dg->right_hand_side,f_val);
    v_terminal = f_val;
    v_terminal *= ((j_bar - j_val_T)/(f_val*f_val));
    v_terminal.update_ghost_values();
    // Residual and the functional have been evaluated at time T.
}
    
template <int dim, int nstate, int n_subspace_vectors, typename MeshType>
void AdjointMarch<dim,nstate,n_subspace_vectors,MeshType>::
compute_R_b_d_h_Jc_vecs()
{
    std::array<VectorType,n_subspace_vectors> Y;
    VectorType v;
    for(unsigned int k=0; k<n_subspace_vectors; ++k)
    {
        Y[k].reinit(dg->solution);
    }
    v.reinit(dg->solution);
    int index_start = K;
    R_vec.resize(K);
    d_vec.resize(K);
    b_vec.resize(K);
    h_vec.resize(K);
    integral_jc_vec.resize(K);
    if(! use_adjoint_restart_files)
    {
        //std::cout<<"Here 2"<<std::endl;
        compute_Y_terminal(Y);
        //std::cout<<"Here 3"<<std::endl;
        compute_v_terminal(v);
        //std::cout<<"Here 4"<<std::endl;
        output_adjoint_restarts(Y,v,K*delT);
    }
    else
    {
        read_adjoint_restarts(Y,v,adjoint_restart_time);
        index_start = std::round(adjoint_restart_time/delT);
    }
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
    }
    std::vector<double> integrand_h(nsteps+1);
    std::vector<double> integrand_J_c(nsteps+1);
    VectorType f_c;

    pcout<<"Runs nonhomogeneous:"<<std::endl;
    for(int i=index_start; i>0; --i) // Between Ti and T_{i-1}
    {
        for(int j=nsteps; j>0; --j) // Between j and j-1
        {           
            const double current_time = (i-1)*delT + j*dt;
            if(j==nsteps)
            {
                get_solution_at_time(current_time);
                compute_df_dc_and_dJ_dc(f_c,integrand_J_c[j]);
                for(unsigned int k=0; k<n_subspace_vectors; ++k)
                {
                    integrand_d[k][j] = Y[k]*f_c;
                }
                integrand_h[j] = v*f_c;
            }
            pcout<<"Time: "<<current_time<<std::endl;
            get_solution_at_time(current_time-dt);
            advance_in_time_hom_and_nonhom(Y,v,Y_minus,v_minus);
            for(unsigned int k=0; k<n_subspace_vectors; ++k)
            {
                Y[k] = Y_minus[k];
            }
            v = v_minus;
            // Compute integrands to be integrated
            //=========================================
            get_solution_at_time(current_time-dt);
            compute_df_dc_and_dJ_dc(f_c,integrand_J_c[j-1]);
            for(unsigned int k=0; k<n_subspace_vectors; ++k)
            {
                integrand_d[k][j-1] = Y_minus[k]*f_c;
            }
            integrand_h[j-1] = v_minus*f_c;
            //========================================
        } // nsteps ends
        pcout<<"================================================================"<<std::endl;
        // Compute integrals
        const double integral_jc = simpson_integration(integrand_J_c, nsteps, dt);
        integral_jc_vec[i-1] = integral_jc;
        const double integral_h = simpson_integration(integrand_h, nsteps, dt);
        h_vec[i-1] = integral_h;
        std::array<double,n_subspace_vectors> integrals_d;
        for(unsigned int k=0; k<n_subspace_vectors; ++k)
        {
            integrals_d[k] = simpson_integration(integrand_d[k], nsteps, dt);
        }
        d_vec[i-1] = integrals_d;

        // Compute QR and variables for the next iteration
        compute_QR_decomposition<n_subspace_vectors>(Y_minus, Q, R);
        R_vec[i-1] = R;
        std::array<double, n_subspace_vectors> b;
        for(unsigned int k=0; k<n_subspace_vectors; ++k)
        {
            Y[k] = Q[k];
            b[k] = -(Q[k]*v_minus);
            v.add(b[k],Q[k]);
        }
        b_vec[i-1] = b;

        if( ( ( (int)((i-1)*delT) ) % 5 ) == 0 )
        {
            output_adjoint_restarts(Q,v,(i-1)*delT);
        }
    } // K loop
    
    pcout<<"Lyapunov exponents: "; 
    // compute lyapunov exp
    for(unsigned int k=0; k<n_subspace_vectors; ++k)
    {
        lyapunov_exp[k] = 0;
        for(int i=K; i>0; --i)
        {
            lyapunov_exp[k] += log(abs(R_vec[i-1][k][k]));
        }
        lyapunov_exp[k] /= T;
    }

    for(unsigned int k=0; k<n_subspace_vectors; ++k)
    {
        pcout<<lyapunov_exp[k]<<", ";
    }
    pcout<<std::endl;

}
template <int dim, int nstate, int n_subspace_vectors, typename MeshType>
void AdjointMarch<dim,nstate,n_subspace_vectors,MeshType>::
output_adjoint_restarts(const std::array<VectorType,n_subspace_vectors> & Q, 
                        const VectorType &v, 
                        const double current_time) const
{
    std::ofstream cout_R("restart_files/R_vec_T" + std::to_string(current_time)  + ".txt"); dealii::ConditionalOStream pcout_R(cout_R, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
    std::ofstream cout_b("restart_files/b_vec_T" + std::to_string(current_time)  + ".txt"); dealii::ConditionalOStream pcout_b(cout_b, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
    std::ofstream cout_d("restart_files/d_vec_T" + std::to_string(current_time)  + ".txt"); dealii::ConditionalOStream pcout_d(cout_d, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
    std::ofstream cout_h("restart_files/h_vec_T" + std::to_string(current_time)  + ".txt"); dealii::ConditionalOStream pcout_h(cout_h, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
    std::ofstream cout_J_c("restart_files/integral_J_c_T" + std::to_string(current_time)  + ".txt"); dealii::ConditionalOStream pcout_J_c(cout_J_c, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

    int index_end = std::round(current_time/delT);
    for(int i=K; i>index_end; --i)
    {
        // Write R, b, integral_jc, integral_h and integrals_d to file.
        for(unsigned int k1=0; k1<n_subspace_vectors; ++k1)
        {
            for(unsigned int k2 = 0; k2<n_subspace_vectors; ++k2)
            {
                pcout_R<<std::setprecision(16)<<R_vec[i-1][k1][k2]<<std::endl;
            }
            pcout_b<<std::setprecision(16)<<b_vec[i-1][k1]<<std::endl;
            pcout_d<<std::setprecision(16)<<d_vec[i-1][k1]<<std::endl;
        }
        pcout_J_c<<std::setprecision(16)<<integral_jc_vec[i-1]<<std::endl;
        pcout_h<<std::setprecision(16)<<h_vec[i-1]<<std::endl;        
    }
    
    cout_R.close(); 
    cout_b.close(); 
    cout_d.close(); 
    cout_h.close(); 
    cout_J_c.close(); 
    
    #if PHILIP_DIM > 1
    for(unsigned int k=0; k<n_subspace_vectors;++k)
    {
        std::string filenameQ = "restart_files/Q_T" + std::to_string(current_time) + "_subspacevec_" + std::to_string(k);
        save_vector(Q[k],filenameQ);
    }
    std::string filenamev = "restart_files/v_T" + std::to_string(current_time);
    save_vector(v,filenamev);
    #endif
}

template <int dim, int nstate, int n_subspace_vectors, typename MeshType>
void AdjointMarch<dim,nstate,n_subspace_vectors,MeshType>::
read_adjoint_restarts(std::array<VectorType,n_subspace_vectors> & Q, 
                      VectorType &v, 
                      const double current_time)
{
    std::ifstream cin_R("restart_files/R_vec_T" + std::to_string(current_time)  + ".txt"); 
    std::ifstream cin_b("restart_files/b_vec_T" + std::to_string(current_time)  + ".txt"); 
    std::ifstream cin_d("restart_files/d_vec_T" + std::to_string(current_time)  + ".txt"); 
    std::ifstream cin_h("restart_files/h_vec_T" + std::to_string(current_time)  + ".txt"); 
    std::ifstream cin_J_c("restart_files/integral_J_c_T" + std::to_string(current_time)  + ".txt"); 

    int index_end = std::round(current_time/delT);
    for(int i=K; i>index_end; --i)
    {
        // Write R, b, integral_jc, integral_h and integrals_d to file.
        for(unsigned int k1=0; k1<n_subspace_vectors; ++k1)
        {
            for(unsigned int k2 = 0; k2<n_subspace_vectors; ++k2)
            {
                cin_R>>R_vec[i-1][k1][k2];
            }
            cin_b>>b_vec[i-1][k1];
            cin_d>>d_vec[i-1][k1];
        }
        cin_J_c>>integral_jc_vec[i-1];
        cin_h>>h_vec[i-1];        
    }
    
    cin_R.close(); 
    cin_b.close(); 
    cin_d.close(); 
    cin_h.close(); 
    cin_J_c.close(); 
    
    #if PHILIP_DIM > 1
    for(unsigned int k=0; k<n_subspace_vectors;++k)
    {
        std::string filenameQ = "restart_files/Q_T" + std::to_string(current_time) + "_subspacevec_" + std::to_string(k);
        load_vector(Q[k],filenameQ);
    }
    std::string filenamev = "restart_files/v_T" + std::to_string(current_time);
    load_vector(v,filenamev);
    #endif
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
        get_solution_at_time(current_time-dt);
        advance_in_time_hom(Y,Y_minus);
        compute_QR_decomposition<n_subspace_vectors>(Y_minus,Q,R);
        for(unsigned int s=0; s<n_subspace_vectors; ++s)
        {
            Y[s] = Q[s];
            lyapunov_exp[s] += log(abs(R[s][s])); 
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
reconstruct_solution(const double initial_time)
{
    T_solnstored_start = initial_time;
    T_solnstored_end = initial_time + dt*n_soln_steps_stored;
    pcout<<"Reconstructing solution from T = "<<T_solnstored_start<<" to T = "<<T_solnstored_end<<std::endl; 
    load_solution_at_time(initial_time);
    soln_stored[0] = dg->solution;

    for(int i=0; i<n_soln_steps_stored; ++i)
    {
        rk_solver->step_in_time(dt,false);
        soln_stored[i+1] = dg->solution;
    } 
}

template <int dim, int nstate, int n_subspace_vectors, typename MeshType>
void AdjointMarch<dim,nstate,n_subspace_vectors,MeshType>::
get_solution_at_time(const double _time)
{
    if( (T_solnstored_start-1.0e-14<= _time) && (_time<= T_solnstored_end+1.0e-14))
    {
        const int vector_index = std::round((_time - T_solnstored_start)/dt);
        dg->solution = soln_stored[vector_index];
        dg->solution.update_ghost_values();
    }
    else if( ( (T_solnstored_start - n_soln_steps_stored*dt)<= _time) && (_time<= (T_solnstored_end - dt*n_soln_steps_stored)))
    {
        reconstruct_solution((T_solnstored_start - n_soln_steps_stored*dt));
        get_solution_at_time(_time);
    }
    else
    {
        pcout<<"Shouldn't have reached here in AdjointMarch::get_solution_at_time(). Aborting.."<<std::endl;
        std::abort();
    }
}
    
template <int dim, int nstate, int n_subspace_vectors, typename MeshType>
void AdjointMarch<dim,nstate,n_subspace_vectors,MeshType>::
load_solution_at_time(const double _time)
{
    const int steps_k = std::round((T + T_extra - _time)/(dt*n_soln_steps_stored));
    const int restart_index = restart_index_terminal - steps_k + 1;

    // Computes restart index string
    std::string restart_index_string = std::to_string(restart_index);
    const unsigned int length_of_index_with_padding = 6;
    const unsigned int number_of_zeros = length_of_index_with_padding - restart_index_string.length();
    restart_index_string.insert(0, number_of_zeros, '0');
    const std::string prefix = "restart-";
    const std::string restart_filename_without_extension = prefix+restart_index_string;
    pcout<<dg->all_parameters->flow_solver_param.restart_files_directory_name + std::string("/") +restart_filename_without_extension<<std::endl;
#if PHILIP_DIM>1
    dg->triangulation->load(dg->all_parameters->flow_solver_param.restart_files_directory_name + std::string("/") + restart_filename_without_extension);
    
    // Note: Future development with hp-capabilities, see section "Note on usage with DoFHandler with hp-capabilities"
    // ----- Ref: https://www.dealii.org/current/doxygen/deal.II/classparallel_1_1distributed_1_1SolutionTransfer.html
    dealii::LinearAlgebra::distributed::Vector<double> solution_no_ghost;
    solution_no_ghost.reinit(dg->locally_owned_dofs, MPI_COMM_WORLD);
    dealii::parallel::distributed::SolutionTransfer<dim, dealii::LinearAlgebra::distributed::Vector<double>, dealii::DoFHandler<dim>> solution_transfer(dg->dof_handler);
    solution_transfer.deserialize(solution_no_ghost);
    dg->solution = solution_no_ghost; //< assignment
    dg->solution.update_ghost_values();
#endif
    //std::cout<<"Done loading solution"<<std::endl;
}

#if PHILIP_DIM>1
template <int dim, int nstate, int n_subspace_vectors, typename MeshType>
void AdjointMarch<dim,nstate,n_subspace_vectors,MeshType>::
save_vector(const VectorType &v, const std::string filename) const
{
    dealii::parallel::distributed::SolutionTransfer<dim, dealii::LinearAlgebra::distributed::Vector<double>, dealii::DoFHandler<dim>> solution_transfer(dg->dof_handler);
    // Note: Future development with hp-capabilities, see section "Note on usage with DoFHandler with hp-capabilities"
    // ----- Ref: https://www.dealii.org/current/doxygen/deal.II/classparallel_1_1distributed_1_1SolutionTransfer.html
    solution_transfer.prepare_for_serialization(v);
    dg->triangulation->save(filename);
}

template <int dim, int nstate, int n_subspace_vectors, typename MeshType>
void AdjointMarch<dim,nstate,n_subspace_vectors,MeshType>::
load_vector(VectorType &v, const std::string filename)
{
    dg->triangulation->load(filename);
    
    // Note: Future development with hp-capabilities, see section "Note on usage with DoFHandler with hp-capabilities"
    // ----- Ref: https://www.dealii.org/current/doxygen/deal.II/classparallel_1_1distributed_1_1SolutionTransfer.html
    dealii::LinearAlgebra::distributed::Vector<double> v_no_ghost;
    v_no_ghost.reinit(dg->locally_owned_dofs, MPI_COMM_WORLD);
    dealii::parallel::distributed::SolutionTransfer<dim, dealii::LinearAlgebra::distributed::Vector<double>, dealii::DoFHandler<dim>> solution_transfer(dg->dof_handler);
    solution_transfer.deserialize(v_no_ghost);
    v = v_no_ghost; //< assignment
    v.update_ghost_values();
}
#endif
    
template <int dim, int nstate, int n_subspace_vectors, typename MeshType>
void AdjointMarch<dim,nstate,n_subspace_vectors,MeshType>::
compute_df_dc_and_dJ_dc(VectorType &f_c, double &J_c)
{
    // Assumes that the solution is already loaded in at the required time.
    dg->solution.update_ghost_values();
    dg->assemble_residual();
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
    J_c -= functional->evaluate_functional();
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

#if PHILIP_DIM != 1
//template class AdjointMarch<PHILIP_DIM, PHILIP_DIM+2, 20, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class AdjointMarch<PHILIP_DIM, PHILIP_DIM+2, 15, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif

} // PHiLiP namespace
