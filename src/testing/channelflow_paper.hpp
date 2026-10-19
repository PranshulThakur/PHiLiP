#ifndef __CHANNEL_FLOW_PAPER_H__
#define __CHANNEL_FLOW_PAPER_H__ 

#include <deal.II/grid/manifold_lib.h>

#include "dg/dg_base.hpp"
#include "parameters/all_parameters.h"
#include "physics/physics.h"
#include "physics/initial_conditions/set_initial_condition.h"
#include "tests.h"
#include <deal.II/base/function.h>

namespace PHiLiP {
namespace Tests {

/// Tests entropy stability for lid driven cavity with moving and static wall BCs.
template <int dim, int nspecies, int nstate>
class ChannelFlowPaper: public TestsBase
{
#if PHILIP_DIM==1
    using Triangulation = dealii::Triangulation<PHILIP_DIM>;
#else
    using Triangulation = dealii::parallel::distributed::Triangulation<PHILIP_DIM>;
#endif
public:
    /// Constructor. Deleted the default constructor since it should not be used
    ChannelFlowPaper () = delete;
    /// Constructor.
    /** Simply calls the TestsBase constructor to set its parameters = parameters_input
     */
    ChannelFlowPaper(const Parameters::AllParameters *const parameters_input,
                    const dealii::ParameterHandler &parameter_handler_input);


    /// Parameter handler for storing the .prm file being ran
    const dealii::ParameterHandler &parameter_handler;

    /// Tests entropy stability
    int run_test () const;

};

/// Function used to evaluate initial conservative solution. Using the same initial condition and the test case as in Chan, Jesse, Yimin Lin, and Tim Warburton. "Entropy stable modal discontinuous Galerkin schemes and wall boundary conditions for the compressible Navier-Stokes equations." Journal of Computational Physics 448 (2022).
template <int dim, int nspecies, int nstate>
class InitialConditionChannelFlowPaper: public InitialConditionFunction<dim,nspecies,nstate,double>
{
protected:
    using dealii::Function<dim,double>::value; ///< dealii::Function we are templating on

public:
    /// Farfield conservative solution
    std::array<double,nstate> farfield_conservative;

    /// Constructor.
    /** Evaluates the primary farfield solution and converts it into the store farfield_conservative solution
     */
    explicit InitialConditionChannelFlowPaper (const Physics::Euler<dim,nspecies,nstate,double> euler_physics)
            : InitialConditionFunction<dim,nspecies,nstate,double>()
    {
        const double density_bc = euler_physics.density_inf;
        const double pressure_bc = 1.0/(euler_physics.gam*euler_physics.mach_inf_sqr);
        std::array<double,nstate> primitive_boundary_values;
        primitive_boundary_values[0] = density_bc;
        for (int d=0;d<dim;d++) { primitive_boundary_values[1+d] = euler_physics.velocities_inf[d]; }
        primitive_boundary_values[nstate-1] = pressure_bc;
        farfield_conservative = euler_physics.convert_primitive_to_conservative(primitive_boundary_values);
    }

    /// Returns the value
    double value (const dealii::Point<dim> &point, const unsigned int istate) const override
    {
//==========================================================================================
        // IC for testing p+1 convergence order sof wall BC
        const double pi = 3.141592653589793238462643383279502884e+00; 
        if(istate==0)
        {
            return 1.0;
        }
        else if(istate==1)
        {
            return (1.0/10.0* sin(pi*point[0]/2.0)*cos(pi*point[dim-1]/2.0));
        }
        else if(istate==2)
        {
            return (1.0/10.0* cos(pi*point[0]/2.0)*sin(pi*point[dim-1]));
        }
        else
        {
            const double pressure = (farfield_conservative[3] - 0.5*(pow(farfield_conservative[1],2) + pow(farfield_conservative[2],2))/farfield_conservative[0])*0.4;
            return (pressure/0.4 + 0.5*(pow((1.0/10.0* sin(pi*point[0]/2.0)*cos(pi*point[1]/2.0)),2) + pow((1.0/10.0* cos(pi*point[0]/2.0)*sin(pi*point[1])),2)));
        }
//==========================================================================================
    }
};
} // Tests namespace
} // PHiLiP namespace
#endif


