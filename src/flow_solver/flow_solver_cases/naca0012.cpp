#include "naca0012.h"

#include <deal.II/base/function.h>
#include <deal.II/base/table_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <stdlib.h>
#include <deal.II/grid/grid_out.h>
#include <iostream>

#include "dg/dg_base.hpp"
#include "functional/lift_drag.hpp"
#include "mesh/gmsh_reader.hpp"
#include "mesh/grids/naca_airfoil_grid.hpp"
#include "mesh/grids/cylinder_channel.h"
#include "physics/physics_factory.h"

namespace PHiLiP {
namespace FlowSolver {
//=========================================================
// NACA0012
//=========================================================
template <int dim, int nstate>
NACA0012<dim, nstate>::NACA0012(const PHiLiP::Parameters::AllParameters *const parameters_input)
        : FlowSolverCaseBase<dim, nstate>(parameters_input)
        , unsteady_data_table_filename_with_extension(this->all_param.flow_solver_param.unsteady_data_table_filename+".txt")
{}

template <int dim, int nstate>
void NACA0012<dim,nstate>::display_additional_flow_case_specific_parameters() const
{
    using PDE_enum = Parameters::AllParameters::PartialDifferentialEquation;
    const PDE_enum pde_type = this->all_param.pde_type;
    if (pde_type == PDE_enum::navier_stokes){
        this->pcout << "- - Freestream Reynolds number: " << this->all_param.navier_stokes_param.reynolds_number_inf << std::endl;
    }
    this->pcout << "- - Courant-Friedrichs-Lewy number: " << this->all_param.flow_solver_param.courant_friedrichs_lewy_number << std::endl;
    this->pcout << "- - Freestream Mach number: " << this->all_param.euler_param.mach_inf << std::endl;
    const double pi = atan(1.0) * 4.0;
    this->pcout << "- - Angle of attack [deg]: " << this->all_param.euler_param.angle_of_attack*180/pi << std::endl;
    this->pcout << "- - Side-slip angle [deg]: " << this->all_param.euler_param.side_slip_angle*180/pi << std::endl;
    this->pcout << "- - Farfield conditions: " << std::endl;
    const dealii::Point<dim> dummy_point;
    for (int s=0;s<nstate;s++) {
        this->pcout << "- - - State " << s << "; Value: " << this->initial_condition_function->value(dummy_point, s) << std::endl;
    }
}

template <int dim, int nstate>
std::shared_ptr<Triangulation> NACA0012<dim,nstate>::generate_grid() const
{
/*
    //Dummy triangulation
    if constexpr(dim==2) {
        std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation>(
    #if PHILIP_DIM!=1
                this->mpi_communicator
    #endif
        );
        dealii::GridGenerator::Airfoil::AdditionalData airfoil_data;
        dealii::GridGenerator::Airfoil::create_triangulation(*grid, airfoil_data);
        grid->refine_global();
        return grid;
    } 
    else if constexpr(dim==3) {
        const std::string mesh_filename = this->all_param.flow_solver_param.input_mesh_filename+std::string(".msh");
        const bool use_mesh_smoothing = false;
        std::shared_ptr<HighOrderGrid<dim,double>> naca0012_mesh = read_gmsh<dim, dim> (mesh_filename, this->all_param.do_renumber_dofs, 0, use_mesh_smoothing);
        return naca0012_mesh->triangulation;
    }
*/         
/*        const std::string mesh_filename = this->all_param.flow_solver_param.input_mesh_filename+std::string(".msh");
        const bool use_mesh_smoothing = false;
        std::shared_ptr<HighOrderGrid<dim,double>> naca0012_mesh = read_gmsh<dim, dim> (mesh_filename, this->all_param.do_renumber_dofs, 0, use_mesh_smoothing);
        return naca0012_mesh->triangulation;
*/
        std::cout<<"Generating grid"<<std::endl;
        std::shared_ptr <Triangulation> grid = std::make_shared<Triangulation>(
        this->mpi_communicator,
        typename dealii::Triangulation<dim>::MeshSmoothing(
            dealii::Triangulation<dim>::smoothing_on_refinement |
            dealii::Triangulation<dim>::smoothing_on_coarsening));
        const unsigned int number_of_refinements = this->all_param.flow_solver_param.number_of_mesh_refinements;
        const unsigned int length_left = this->all_param.flow_solver_param.length_left_cyl;
        const unsigned int length_right = this->all_param.flow_solver_param.length_right_cyl;
        const unsigned int height_bottom = this->all_param.flow_solver_param.height_bottom_cyl;
        const unsigned int height_top = this->all_param.flow_solver_param.height_top_cyl;
        const unsigned int depth  = this->all_param.flow_solver_param.depth_cyl;
        unsigned int                     depth_division = this->all_param.flow_solver_param.depth_division_cyl;
        const double                     shell_region_radius = this->all_param.flow_solver_param.shell_region_radius_cyl;
        const unsigned int               n_shells = this->all_param.flow_solver_param.n_shells_cyl;
        const double                     skewness = this->all_param.flow_solver_param.skewness_cyl;
        const bool                       use_transfinite_region = this->all_param.flow_solver_param.use_transfinite_region_cyl;

    Grids::cylindrical_channel<dim>(
        *grid,
        length_left,
        length_right,
        height_bottom,
        height_top,
        depth,
        depth_division,
        shell_region_radius,
        n_shells,
        skewness,
        use_transfinite_region,
        number_of_refinements);
        std::cout<<"Done generating grid"<<std::endl;
        return grid;
}

template <int dim, int nstate>
void NACA0012<dim,nstate>::set_higher_order_grid(std::shared_ptr<DGBase<dim, double>> dg) const
{ (void) dg;
/*
    const std::string mesh_filename = this->all_param.flow_solver_param.input_mesh_filename+std::string(".msh");
    const bool use_mesh_smoothing = false;
    std::shared_ptr<HighOrderGrid<dim,double>> naca0012_mesh = read_gmsh<dim, dim> (mesh_filename, this->all_param.do_renumber_dofs, 0, use_mesh_smoothing);
    dg->set_high_order_grid(naca0012_mesh);
    for (int i=0; i<this->all_param.flow_solver_param.number_of_mesh_refinements; ++i) {
        dg->high_order_grid->refine_global();
    }
*/
}

template <int dim, int nstate>
double NACA0012<dim,nstate>::compute_lift(std::shared_ptr<DGBase<dim, double>> dg) const
{
    LiftDragFunctional<dim,dim+2,double,Triangulation> lift_functional(dg, LiftDragFunctional<dim,dim+2,double,Triangulation>::Functional_types::lift);
    const double lift = lift_functional.evaluate_functional();
    return lift;
}

template <int dim, int nstate>
double NACA0012<dim,nstate>::compute_drag(std::shared_ptr<DGBase<dim, double>> dg) const
{
    LiftDragFunctional<dim,dim+2,double,Triangulation> drag_functional(dg, LiftDragFunctional<dim,dim+2,double,Triangulation>::Functional_types::drag);
    const double drag = drag_functional.evaluate_functional();
    return drag;
}

template <int dim, int nstate>
void NACA0012<dim,nstate>::steady_state_postprocessing(std::shared_ptr<DGBase<dim, double>> dg) const
{
    const double lift = this->compute_lift(dg);
    const double drag = this->compute_drag(dg);

    this->pcout << " Resulting lift : " << lift << std::endl;
    this->pcout << " Resulting drag : " << drag << std::endl;
}

template <int dim, int nstate>
void NACA0012<dim, nstate>::compute_unsteady_data_and_write_to_table(
        const unsigned int current_iteration,
        const double current_time,
        const std::shared_ptr <DGBase<dim, double>> dg,
        const std::shared_ptr <dealii::TableHandler> /*unsteady_data_table*/)
{
    // Compute aerodynamic values
    const double lift = 0.0; //this->compute_lift(dg);
    const double drag = this->compute_drag(dg);
    if(current_time>150.0)
    {
        functional_sum += drag;
        countval_functional++;
        functional_avg = functional_sum/countval_functional;
    }
    (void) dg;
/*
    if(this->mpi_rank==0) {
        // Add values to data table
        this->add_value_to_data_table(current_time,"time",unsteady_data_table);
        this->add_value_to_data_table(lift,"lift",unsteady_data_table);
        this->add_value_to_data_table(drag,"drag",unsteady_data_table);
        // Write to file
        //std::ofstream unsteady_data_table_file(this->unsteady_data_table_filename_with_extension);
        //unsteady_data_table->write_text(unsteady_data_table_file);
    }
*/  
    // Print to console
    this->pcout << "    Iter: " << current_iteration
                << "    Time: " << current_time
                << "    Lift: " << lift
                << "    Drag: " << drag
                << "    Avg functional: " << functional_avg;
    this->pcout << std::endl;

    // Abort if energy is nan
    if(std::isnan(lift) || std::isnan(drag)) {
        this->pcout << " ERROR: Lift or drag at time " << current_time << " is nan." << std::endl;
        this->pcout << "        Consider decreasing the time step / CFL number." << std::endl;
        std::abort();
    }
/* 
// Compute max wave speed
    // Initialize the maximum local wave speed to zero
    double maximum_local_wave_speed = 0.0;

    // Overintegrate the error to make sure there is not integration error in the error estimate
    int overintegrate = 10;
    dealii::QGauss<dim> quad_extra(dg->max_degree+1+overintegrate);
    dealii::FEValues<dim,dim> fe_values_extra(*(dg->high_order_grid->mapping_fe_field), dg->fe_collection[dg->max_degree], quad_extra,
                                              dealii::update_values | dealii::update_gradients | dealii::update_JxW_values | dealii::update_quadrature_points);

    const unsigned int n_quad_pts = fe_values_extra.n_quadrature_points;
    std::array<double,nstate> soln_at_q;

    std::vector<dealii::types::global_dof_index> dofs_indices (fe_values_extra.dofs_per_cell);
    for (auto cell = dg->dof_handler.begin_active(); cell!=dg->dof_handler.end(); ++cell) {
        if (!cell->is_locally_owned()) continue;
        fe_values_extra.reinit (cell);
        cell->get_dof_indices (dofs_indices);

        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

            std::fill(soln_at_q.begin(), soln_at_q.end(), 0.0);
            for (unsigned int idof=0; idof<fe_values_extra.dofs_per_cell; ++idof) {
                const unsigned int istate = fe_values_extra.get_fe().system_to_component_index(idof).first;
                soln_at_q[istate] += dg->solution[dofs_indices[idof]] * fe_values_extra.shape_value_component(idof, iquad, istate);
            }

            // Update the maximum local wave speed (i.e. convective eigenvalue)
            const double vel_norm = sqrt(pow(soln_at_q[1]/soln_at_q[0],2)+pow(soln_at_q[2]/soln_at_q[0],2)+pow(soln_at_q[3]/soln_at_q[0],2));
            const double pressure = 0.4*(soln_at_q[nstate-1] - 0.5*soln_at_q[0]*vel_norm*vel_norm);
            const double c = sqrt(1.4*pressure/soln_at_q[0]);
            const double local_wave_speed = vel_norm + c;

            if(local_wave_speed > maximum_local_wave_speed) maximum_local_wave_speed = local_wave_speed;
        }
    }
    const double maximum_wave_speed = dealii::Utilities::MPI::max(maximum_local_wave_speed, this->mpi_communicator);
    this->pcout<<"Max wave speed = "<<maximum_wave_speed<<std::endl;
*/
}

#if PHILIP_DIM!=1
    template class NACA0012<PHILIP_DIM,PHILIP_DIM+2>;
#endif

} // FlowSolver namespace
} // PHiLiP namespace

