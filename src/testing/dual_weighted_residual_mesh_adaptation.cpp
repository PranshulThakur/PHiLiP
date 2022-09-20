#include <stdlib.h>     /* srand, rand */
#include <iostream>
#include "dual_weighted_residual_mesh_adaptation.h"
#include "flow_solver/flow_solver_factory.h"

namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
DualWeightedResidualMeshAdaptation<dim, nstate> :: DualWeightedResidualMeshAdaptation(
    const Parameters::AllParameters *const parameters_input,
    const dealii::ParameterHandler &parameter_handler_input)
    : TestsBase::TestsBase(parameters_input)
    , parameter_handler(parameter_handler_input)
{}

template <int dim, int nstate>
int DualWeightedResidualMeshAdaptation<dim, nstate> :: run_test () const
{
    const Parameters::AllParameters param = *(TestsBase::all_parameters);
    bool use_mesh_adaptation = param.mesh_adaptation_param.total_mesh_adaptation_cycles > 0;
    using ManParam = Parameters::ManufacturedConvergenceStudyParam;
    ManParam manu_grid_conv_param = param.manufactured_convergence_study_param;
   /*
    bool check_for_p_refined_cell = false;
    
    using MeshAdaptationTypeEnum = Parameters::MeshAdaptationParam::MeshAdaptationType;
    MeshAdaptationTypeEnum mesh_adaptation_type = param.mesh_adaptation_param.mesh_adaptation_type;
    if(mesh_adaptation_type == MeshAdaptationTypeEnum::p_adaptation)
    {
        check_for_p_refined_cell = true;
    }
    */
    if(!use_mesh_adaptation)
    {
        pcout<<"This test case checks mesh adaptation. However, total mesh adaptation cycles have been set to 0 in the parameters file. Aborting..."<<std::endl; 
        std::abort();
    }

    std::unique_ptr<FlowSolver::FlowSolver<dim,nstate>> flow_solver = FlowSolver::FlowSolverFactory<dim,nstate>::select_flow_case(&param, parameter_handler);
    using VectorType = dealii::LinearAlgebra::distributed::Vector<double>; ///< Alias for parallel distributed vector.
    using MatrixType = dealii::TrilinosWrappers::SparseMatrix; ///< Alias for SparseMatrix.
    VectorType is_a_surface_node(flow_solver->dg->high_order_grid->volume_nodes); // Copy parallel partitioning of volume_nodes.
    is_a_surface_node.update_ghost_values();
    is_a_surface_node *= 0.0;
    const dealii::IndexSet &volume_range = flow_solver->dg->high_order_grid->volume_nodes.get_partitioner()->locally_owned_range();
    const dealii::IndexSet &surface_range = flow_solver->dg->high_order_grid->surface_nodes.get_partitioner()->locally_owned_range();
    const unsigned int n_surf_nodes = flow_solver->dg->high_order_grid->surface_nodes.size(); 
    const unsigned int n_vol_nodes = flow_solver->dg->high_order_grid->volume_nodes.size(); 
    const unsigned int n_inner_nodes = n_vol_nodes - n_surf_nodes;
    for(unsigned int i=0; i<n_surf_nodes; ++i)
    {
        if(surface_range.is_element(i))
        {
            const unsigned int vol_index = flow_solver->dg->high_order_grid->surface_to_volume_indices[i];
            Assert(volume_range.is_element(vol_index), dealii::ExcMessage("Vol index is not in range."));
            is_a_surface_node(vol_index) = 1.0;
        }
    }
    pcout<<"Done setting flag for surface nodes"<<std::endl;
    unsigned int n_elements_locally_owned = volume_range.n_elements() - surface_range.n_elements();
    int mpi_rank, n_mpi;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_mpi);
    std::vector<unsigned int> n_elements_per_mpi(n_mpi);
    MPI_Allgather(&n_elements_locally_owned, 1, MPI_UNSIGNED, &(n_elements_per_mpi[0]), 1, MPI_UNSIGNED, MPI_COMM_WORLD);
    unsigned int lower_index = 0, higher_index = 0;
    for(int i_mpi = 0; i_mpi<mpi_rank; ++i_mpi)
    {
        lower_index += n_elements_per_mpi[i_mpi];
    }
    higher_index = lower_index + n_elements_locally_owned;
    dealii::IndexSet inner_vol_range;
    inner_vol_range.set_size(n_inner_nodes);
    inner_vol_range.add_range(lower_index, higher_index);

    VectorType inner_vol_index_to_vol_index(inner_vol_range, MPI_COMM_WORLD); // no need of ghost indices.
    /*
    dealii::IndexSet inner_vol_ghost;
    inner_vol_ghost.set_size(n_inner_nodes);
    inner_vol_ghost.add_range(0, n_inner_nodes);
    VectorType inner_vol_index_to_vol_index(inner_vol_range, inner_vol_ghost, MPI_COMM_WORLD);*/

    unsigned int count1 = lower_index;
    for(unsigned int i_vol=0; i_vol<n_vol_nodes; ++i_vol)
    {
        if(!volume_range.is_element(i_vol)) continue;

        if(is_a_surface_node(i_vol)) continue;

        inner_vol_index_to_vol_index[count1++] = i_vol;
    }
    AssertDimension(count1, higher_index);

    // Create dXv_dXvinner

    dealii::DynamicSparsityPattern dsp(n_vol_nodes, n_inner_nodes, volume_range);
    for(unsigned int i=0; i<n_inner_nodes; ++i)
    {
        if(!inner_vol_range.is_element(i)) continue;
        dsp.add(inner_vol_index_to_vol_index[i],i);
    }


    dealii::IndexSet locally_relevant_dofs;
    dealii::DoFTools::extract_locally_relevant_dofs(flow_solver->dg->high_order_grid->dof_handler_grid, locally_relevant_dofs);

    dealii::SparsityTools::distribute_sparsity_pattern(dsp, volume_range, MPI_COMM_WORLD, locally_relevant_dofs);

    MatrixType dXv_dXvinner;
    dXv_dXvinner.reinit(volume_range, inner_vol_range, dsp, MPI_COMM_WORLD);
    pcout<<"Computing dXv_dXvinner.."<<std::endl;
    
    for(unsigned int i=0; i<n_inner_nodes; ++i)
    {
        if(!inner_vol_range.is_element(i)) continue;
        dXv_dXvinner.set(inner_vol_index_to_vol_index[i],i, 1.0);
    }

    dXv_dXvinner.compress(dealii::VectorOperation::insert);
    pcout<<"Done computing dXv_dXvinner.."<<std::endl;

    const double delta_x = 3.0;
    VectorType delta_x_vector(inner_vol_range, MPI_COMM_WORLD); // without ghosts
    //VectorType delta_x_vector(inner_vol_range,inner_vol_ghost, MPI_COMM_WORLD);
    VectorType volume_nodes_copy(flow_solver->dg->high_order_grid->volume_nodes);
    VectorType volume_nodes_original(flow_solver->dg->high_order_grid->volume_nodes);
    pcout<<"Reached here 1"<<std::endl;
    
    for(unsigned int i=0; i<n_inner_nodes; ++i)
    {
        if(inner_vol_range.is_element(i))
        {
            delta_x_vector[i] += delta_x;
        }
    }
    pcout<<"Reached here 2"<<std::endl;
    dXv_dXvinner.vmult_add(volume_nodes_copy, delta_x_vector);
    pcout<<"Reached here 3"<<std::endl;

    for(unsigned int i=0; i<n_vol_nodes; ++i)
    {
        if(volume_range.is_element(i))
        {
            if(is_a_surface_node(i))
            {
                if(volume_nodes_copy(i) - volume_nodes_original(i) != 0.0) return 1;
            }
            else
            {
                if(volume_nodes_copy(i) - volume_nodes_original(i) != delta_x) return 1;
            }
        }
    }
    AssertDimension(delta_x_vector.size(), n_inner_nodes);

    pcout<<"Test passed. mult with dXv_dXvinner works."<<std::endl;























    
   /*
   flow_solver->run();

    // Check location of the most refined cell
    dealii::Point<dim> refined_cell_coord = flow_solver->dg->coordinates_of_highest_refined_cell(check_for_p_refined_cell);
    pcout<<" Coordinates of the most refined cell (x,y) = ("<<refined_cell_coord[0]<<", "<<refined_cell_coord[1]<<")"<<std::endl;
    // Check if the mesh is refined near the shock i.e x,y in [0.3,0.6].
    if ((refined_cell_coord[0] > 0.3) && (refined_cell_coord[0] < 0.6) && (refined_cell_coord[1] > 0.3) && (refined_cell_coord[1] < 0.6))
    {
        pcout<<"Mesh is refined near the shock. Test passed!"<<std::endl;
        return 0; // Mesh adaptation test passed.
    }
    else
    {
        pcout<<"Mesh Adaptation has failed."<<std::endl;
        return 1; // Mesh adaptation failed.
    }
    */
    
    return 0;
}

#if PHILIP_DIM==2
template class DualWeightedResidualMeshAdaptation <PHILIP_DIM, 1>;
#endif

} // namespace Tests
} // namespace PHiLiP 
    
