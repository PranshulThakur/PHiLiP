#include "direct_goal_oriented_error.h"
#include <deal.II/dofs/dof_tools.h>
#include "linear_solver/linear_solver.h"

namespace PHiLiP {

template<int dim, int nstate, typename real, typename MeshType>
DirectGoalOrientedError<dim,nstate,real,MeshType>::DirectGoalOrientedError(
    std::shared_ptr<DGBase<dim,real,MeshType>> _dg,
    const bool _uses_solution_values,
    const bool _uses_solution_gradient)
    : Functional<dim,nstate,real,MeshType>::Functional(_dg, _uses_solution_values, _uses_solution_gradient)
{
    functional = FunctionalFactory<dim,nstate,real,MeshType>::create_Functional(this->dg->all_parameters->functional_param, this->dg);
    weight_of_mesh_error = 1.0e-15; // Hard coded for now.
}


template<int dim, int nstate, typename real, typename MeshType>
void DirectGoalOrientedError<dim,nstate,real,MeshType> :: allocate_partial_derivatives(
    const bool compute_dF_dWfine, 
    const bool compute_dF_dWinterp, 
    const bool compute_dF_dX, 
    const bool compute_d2F)
{
    const dealii::IndexSet &locally_owned_solution_dofs = this->dg->dof_handler.locally_owned_dofs();
    const dealii::IndexSet &locally_owned_grid_dofs = this->dg->high_order_grid->dof_handler_grid.locally_owned_dofs();
    
    if(compute_dF_dWfine) {derivative_functionalerror_wrt_solution_fine.reinit(locally_owned_solution_dofs, MPI_COMM_WORLD);}
    if(compute_dF_dWinterp) {derivative_functionalerror_wrt_solution_interpolated.reinit(locally_owned_solution_dofs, MPI_COMM_WORLD);}

    if(compute_dF_dX)
    {
        dealii::IndexSet locally_relevant_grid_dofs, ghost_grid_dofs;
        dealii::DoFTools::extract_locally_relevant_dofs(this->dg->high_order_grid->dof_handler_grid, locally_relevant_grid_dofs);
        ghost_grid_dofs = locally_relevant_grid_dofs;
        ghost_grid_dofs.subtract_set(locally_owned_grid_dofs);
        derivative_functionalerror_wrt_volume_nodes.reinit(locally_owned_grid_dofs, ghost_grid_dofs, MPI_COMM_WORLD);
    }

    if(compute_d2F)
    {
        // Allocate second derivatives
        dealii::SparsityPattern d2F_dWdW_sparsity_pattern = this->dg->get_d2RdWdW_sparsity_pattern();
        dealii::SparsityPattern d2F_dWdX_sparsity_pattern = this->dg->get_d2RdWdX_sparsity_pattern();
        dealii::SparsityPattern d2F_dXdX_sparsity_pattern = this->dg->get_d2RdXdX_sparsity_pattern();

        d2F_solfine_solfine.reinit(locally_owned_solution_dofs, locally_owned_solution_dofs, d2F_dWdW_sparsity_pattern, MPI_COMM_WORLD);
        d2F_solfine_solinterp.reinit(locally_owned_solution_dofs, locally_owned_solution_dofs, d2F_dWdW_sparsity_pattern, MPI_COMM_WORLD);
        d2F_solfine_volnodes.reinit(locally_owned_solution_dofs, locally_owned_grid_dofs, d2F_dWdX_sparsity_pattern, MPI_COMM_WORLD);


        d2F_solinterp_solinterp.reinit(locally_owned_solution_dofs, locally_owned_solution_dofs, d2F_dWdW_sparsity_pattern, MPI_COMM_WORLD);
        d2F_solinterp_volnodes.reinit(locally_owned_solution_dofs, locally_owned_grid_dofs, d2F_dWdX_sparsity_pattern, MPI_COMM_WORLD);


        d2F_volnodes_volnodes.reinit(locally_owned_grid_dofs, locally_owned_grid_dofs, d2F_dXdX_sparsity_pattern, MPI_COMM_WORLD);
    }
}

template<int dim, int nstate, typename real, typename MeshType>
void DirectGoalOrientedError<dim,nstate,real,MeshType> :: compute_solution_fine_and_solution_interpolated()
{
    solution_interpolated = this->dg->solution;
    this->dg->assemble_residual(true); // evaluate dR_dW
    solve_linear(this->dg->system_matrix, this->dg->right_hand_side, del_solution, this->dg->all_parameters->linear_solver_param);
    del_solution *= -1.0;
    solution_fine = solution_interpolated;
    solution_fine += del_solution;
}

template<int dim, int nstate, typename real, typename MeshType>
template<typename real2>
real2 DirectGoalOrientedError<dim,nstate,real,MeshType> :: evaluate_functional_error_in_cell_volume(
    const std::vector< real2 > &soln_coeff_fine,
    const std::vector< real2 > &soln_coeff_interpolated, 
    const dealii::FESystem<dim> &fe_solution,
    const std::vector<real2> &coords_coeff,
    const dealii::FESystem<dim> &fe_metric,
    const dealii::Quadrature<dim> &volume_quadrature) const
{
    real2 cell_functional_value_fine = functional->evaluate_volume_cell_functional(*(this->physics_fad_fad), 
                                                                                   soln_coeff_fine,
                                                                                   fe_solution,
                                                                                   coords_coeff,
                                                                                   fe_metric,
                                                                                   volume_quadrature);
    real2 cell_functional_value_interpolated = functional->evaluate_volume_cell_functional(*(this->physics_fad_fad), 
                                                                                           soln_coeff_interpolated,
                                                                                           fe_solution,
                                                                                           coords_coeff,
                                                                                           fe_metric,
                                                                                           volume_quadrature);
    real2 eta_cell = cell_functional_value_fine - cell_functional_value_interpolated;

    real2 cell_functional_error = std::pow(eta_cell,2);
    real2 sum_mesh_weight = 0.0;
    
    unsigned int n_vertex_points = coords_coeff.size()/dim;
    Assert(n_vertex_points == dealii::GeometryInfo<dim>::vertices_per_cell, dealii::ExcMessage( "The objective function is currently implemented for grid degree = 1"));
    // Store vertex points
    std::vector<dealii::Point<dim, real2>> vertex_points(n_vertex_points);
    unsigned int count1 = 0;

    for(unsigned int ipoint=0; ipoint<n_vertex_points; ipoint++)
    {
        for(unsigned int idim = 0; idim < dim; idim++)
        {
            vertex_points[ipoint][idim] = coords_coeff[count1++];
        }
    }

    for(unsigned int ipoint=0; ipoint<n_vertex_points; ipoint++)
    {
        for(unsigned int jpoint=ipoint+1; jpoint<n_vertex_points; jpoint++)
        {
            sum_mesh_weight += 1.0/vertex_points[ipoint].distance_square(vertex_points[jpoint]);
        }
    }

    cell_functional_error += weight_of_mesh_error*sum_mesh_weight;
    return cell_functional_error;
}

template<int dim, int nstate, typename real, typename MeshType>
template<typename real2>
real2 DirectGoalOrientedError<dim,nstate,real,MeshType> :: evaluate_functional_error_in_cell_boundary(
    const unsigned int boundary_id,
    const std::vector< real2 > &soln_coeff_fine,
    const std::vector< real2 > &soln_coeff_interpolated,
    const dealii::FESystem<dim> &fe_solution,
    const std::vector< real2 > &coords_coeff,
    const dealii::FESystem<dim> &fe_metric,
    const unsigned int face_number,
    const dealii::Quadrature<dim-1> &face_quadrature) const
{
    real2 cell_boundary_functional_value_fine = functional->evaluate_boundary_cell_functional(*(this->physics_fad_fad), 
                                                                                     boundary_id, 
                                                                                     soln_coeff_fine, 
                                                                                     fe_solution, 
                                                                                     coords_coeff, 
                                                                                     fe_metric, 
                                                                                     face_number, 
                                                                                     face_quadrature);
    real2 cell_boundary_functional_value_interpolated = functional->evaluate_boundary_cell_functional(*(this->physics_fad_fad), 
                                                                                             boundary_id, 
                                                                                             soln_coeff_interpolated, 
                                                                                             fe_solution, 
                                                                                             coords_coeff, 
                                                                                             fe_metric, 
                                                                                             face_number, 
                                                                                             face_quadrature);
    real2 eta_cell = cell_boundary_functional_value_fine - cell_boundary_functional_value_interpolated;
    
    real2 cell_boundary_functional_error = std::pow(eta_cell, 2);
    return cell_boundary_functional_error;
}

template<int dim, int nstate, typename real, typename MeshType>
real DirectGoalOrientedError<dim,nstate,real,MeshType> :: evaluate_functional(
    const bool compute_dIdW,
    const bool compute_dIdX,
    const bool compute_d2I)
{
    bool actually_compute_value = true;
    bool actually_compute_dIdW = compute_dIdW;
    bool actually_compute_dIdX = compute_dIdX;
    bool actually_compute_d2I  = compute_d2I;

    this->pcout << "Evaluating functional... "<<std::endl;
    this->need_compute(actually_compute_value, actually_compute_dIdW, actually_compute_dIdX, actually_compute_d2I);
    bool need_to_compute_something = (actually_compute_value || actually_compute_dIdW || actually_compute_dIdX || actually_compute_d2I);
    if(!need_to_compute_something) {return current_error_value;}
    
    bool compute_dF_dWfine=false, compute_dF_dWinterp=false, compute_dF_dX=false, compute_d2F = false;
    if(actually_compute_d2I) {compute_d2F = true; compute_dF_dWfine = true; compute_dF_dWinterp = true; compute_dF_dX = true;}
    if(actually_compute_dIdW) {compute_dF_dWfine = true; compute_dF_dWinterp = true;}
    if(actually_compute_dIdX) {compute_dF_dX = true; compute_dF_dWfine = true;}
    
    // Increase poly order of DG
    if (this->dg->get_max_fe_degree() >= this->dg->max_degree)
    {
        this->pcout<<"Polynomial degree of DG will exceed the maximum allowable after refinement. Update max_degree in dg"<<std::endl;
        std::abort();
    }
    this->dg->change_cells_fe_degree_by_deltadegree_and_interpolate_solution(1);
    
    compute_solution_fine_and_solution_interpolated();
    AssertDimension(this->dg->solution.size(), solution_fine.size());
    AssertDimension(this->dg->solution.size(), solution_interpolated.size());

    real error_value_on_this_processor = 0.0;
    
    const dealii::FESystem<dim,dim> &fe_metric = this->dg->high_order_grid->fe_system;
    const unsigned int n_metric_dofs_cell = fe_metric.dofs_per_cell;
    std::vector<dealii::types::global_dof_index> cell_metric_dofs_indices(n_metric_dofs_cell);
    
    const unsigned int max_dofs_per_cell = this->dg->dof_handler.get_fe_collection().max_dofs_per_cell();
    std::vector<dealii::types::global_dof_index> cell_soln_dofs_indices(max_dofs_per_cell);

    std::vector<FadFadType> soln_coeff_interpolated(max_dofs_per_cell); // Solution interpolated to p+1.
    std::vector<FadFadType> soln_coeff_fine(max_dofs_per_cell);         // Solution interpolated to p+1, then taylor expanded.
    std::vector< FadFadType > coords_coeff(n_metric_dofs_cell);         // Coords coeff. 
    
    const auto mapping = (*(this->dg->high_order_grid->mapping_fe_field));

    dealii::hp::MappingCollection<dim> mapping_collection(mapping);
    dealii::hp::FEFaceValues<dim,dim> fe_values_collection_face(mapping_collection, this->dg->fe_collection, this->dg->face_quadrature_collection, this->face_update_flags);

    allocate_partial_derivatives(
     compute_dF_dWfine, 
     compute_dF_dWinterp, 
     compute_dF_dX, 
     compute_d2F);

    auto metric_cell = this->dg->high_order_grid->dof_handler_grid.begin_active();
    auto soln_cell = this->dg->dof_handler.begin_active();

    for( ; soln_cell != this->dg->dof_handler.end(); ++soln_cell, ++metric_cell)
    {
        if(!soln_cell->is_locally_owned()) continue;

        const unsigned int cell_fe_index = soln_cell->active_fe_index();

        // Resize solution coefficients & get soln & metric dof indices. Grid degree and size of coords_coeff is expected to be the same for all cells.
        const dealii::FESystem<dim,dim> &fe_solution = this->dg->fe_collection[cell_fe_index];
        const unsigned int n_soln_dofs_cell = fe_solution.n_dofs_per_cell();
        cell_soln_dofs_indices.resize(n_soln_dofs_cell);
        soln_cell->get_dof_indices(cell_soln_dofs_indices);
        soln_coeff_fine.resize(n_soln_dofs_cell);
        soln_coeff_interpolated.resize(n_soln_dofs_cell);
        metric_cell->get_dof_indices(cell_metric_dofs_indices);
         
        // Total independent variables for AD
        unsigned int n_total_independent_variables = 0;
        if(compute_dF_dWfine || compute_d2F) n_total_independent_variables += n_soln_dofs_cell;
        if(compute_dF_dWinterp || compute_d2F) n_total_independent_variables += n_soln_dofs_cell;
        if(compute_dF_dX || compute_d2F) n_total_independent_variables += n_metric_dofs_cell;

        unsigned int n_current_independent_variable = 0;
//=====================================================================================================================================================================
        // Setup independent variables for AD (to be moved to a separate function).

        // Setup independent variables for first derivative.
        for(unsigned int idof = 0; idof < n_soln_dofs_cell; idof++)
        {
            const real value_fine = solution_fine[cell_soln_dofs_indices[idof]];
            soln_coeff_fine[idof] = value_fine;
            if(compute_dF_dWfine || compute_d2F)
            {
                soln_coeff_fine[idof].diff(n_current_independent_variable++, n_total_independent_variables);
            }
        }
        
        for(unsigned int idof = 0; idof < n_soln_dofs_cell; idof++)
        {
            const real value_interpolated = solution_interpolated[cell_soln_dofs_indices[idof]];
            soln_coeff_interpolated[idof] = value_interpolated;
            if(compute_dF_dWinterp || compute_d2F)
            {
                soln_coeff_interpolated[idof].diff(n_current_independent_variable++, n_total_independent_variables);
            }
        }
        
        for (unsigned int idof = 0; idof < n_metric_dofs_cell; ++idof)
        {
            const real value_coord_coeff = this->dg->high_order_grid->volume_nodes[cell_metric_dofs_indices[idof]];
            coords_coeff[idof] = value_coord_coeff;
            if(compute_dF_dX || compute_d2F)
            {
                coords_coeff[idof].diff(n_current_independent_variable++, n_total_independent_variables);
            }
        }
        AssertDimension(n_current_independent_variable, n_total_independent_variables);
        
        // Setup AD variables for second derivatives
        if(compute_d2F)
        {
            n_current_independent_variable = 0;

            for(unsigned int idof = 0; idof < n_soln_dofs_cell; idof++)
            {
                const real value_fine = solution_fine[cell_soln_dofs_indices[idof]];
                soln_coeff_fine[idof].val() = value_fine;
                soln_coeff_fine[idof].val().diff(n_current_independent_variable++, n_total_independent_variables);
            }

            for(unsigned int idof = 0; idof < n_soln_dofs_cell; idof++)
            {
                const real value_interpolated = solution_interpolated[cell_soln_dofs_indices[idof]];
                soln_coeff_interpolated[idof].val() = value_interpolated;
                soln_coeff_interpolated[idof].val().diff(n_current_independent_variable++, n_total_independent_variables);
            }

            for (unsigned int idof = 0; idof < n_metric_dofs_cell; ++idof)
            {
                const real value_coord_coeff = this->dg->high_order_grid->volume_nodes[cell_metric_dofs_indices[idof]];
                coords_coeff[idof].val() = value_coord_coeff;
                coords_coeff[idof].val().diff(n_current_independent_variable++, n_total_independent_variables);
            }
            AssertDimension(n_current_independent_variable, n_total_independent_variables);
        }
//=====================================================================================================================================================================
        // Evaluate objective function on the cell.
        const dealii::Quadrature<dim> &volume_quadratures_cell = this->dg->volume_quadrature_collection[cell_fe_index];
        const dealii::Quadrature<dim-1> &face_quadratures = this->dg->face_quadrature_collection[cell_fe_index];
        FadFadType local_functional_error_fadfad = this->evaluate_functional_error_in_cell_volume(soln_coeff_fine, 
                                                                                                  soln_coeff_interpolated, 
                                                                                                  fe_solution,
                                                                                                  coords_coeff, 
                                                                                                  fe_metric, 
                                                                                                  volume_quadratures_cell);
        for(unsigned int iface = 0; iface < dealii::GeometryInfo<dim>::faces_per_cell; ++iface)
        {
            auto face = soln_cell->face(iface);
            if(face->at_boundary())
            {
                const unsigned int boundary_id = face->boundary_id();
                local_functional_error_fadfad += this->evaluate_functional_error_in_cell_boundary(boundary_id, 
                                                                                                  soln_coeff_fine, 
                                                                                                  soln_coeff_interpolated, 
                                                                                                  fe_solution,
                                                                                                  coords_coeff, 
                                                                                                  fe_metric, 
                                                                                                  iface, 
                                                                                                  face_quadratures);
            }
        }

        error_value_on_this_processor += local_functional_error_fadfad.val().val();

//====================================================================================================================================================================
        // Evaluate derivatives (to be moved to a separate function).

        // Evaluate first derivatives.
        unsigned int i_variable = 0;

        if(compute_dF_dWfine)
        {
            // First derivative wrt solution fine
            for(unsigned int idof = 0; idof < n_soln_dofs_cell; idof++) // i_variable is solution fine
            {
                const real dF_dWfinei = local_functional_error_fadfad.dx(i_variable++).val();
                derivative_functionalerror_wrt_solution_fine(cell_soln_dofs_indices[idof]) += dF_dWfinei;
            }
        }

        if(compute_dF_dWinterp)
        {
            // First derivative wrt solution interpolated.
            for(unsigned int idof = 0; idof < n_soln_dofs_cell; idof++) // i_variable is solution interpolated
            {
                const real dF_dWinterpi = local_functional_error_fadfad.dx(i_variable++).val();
                derivative_functionalerror_wrt_solution_interpolated(cell_soln_dofs_indices[idof]) += dF_dWinterpi;
            }
        }

        if(compute_dF_dX)
        {
            // First derivative wrt metric nodes
            for(unsigned int idof = 0; idof < n_metric_dofs_cell; idof++) // i_variable is X (metric nodes)
            {
                const real dF_dXi = local_functional_error_fadfad.dx(i_variable++).val();
                derivative_functionalerror_wrt_volume_nodes(cell_metric_dofs_indices[idof]) += dF_dXi; // += adds contribution from adjacent cells.
            }
        }

        AssertDimension(i_variable, n_total_independent_variables);

        // Evaluate second derivatives
        if(compute_d2F)
        {
            // Note: 
            // variables 0 to n_soln_dofs_cell refer to solution_fine. 
            // variables n_soln_dofs_cell to 2*n_soln_dofs_cell refer to solution_interpolated. 
            // variables 2*n_soln_dofs_cell to n_total_independent_variables refer to volume_nodes. 
            
//============================================================================================================================================
            // i variable is solution fine
            i_variable = 0;
            for(unsigned int idof = 0; idof < n_soln_dofs_cell; idof++)
            {
                const FadType dF_dWfinei = local_functional_error_fadfad.dx(i_variable++);
                unsigned int j_variable = 0;

                for(unsigned int jdof = 0; jdof < n_soln_dofs_cell; jdof++) // j_variable is solution_fine.
                {
                    d2F_solfine_solfine.add(cell_soln_dofs_indices[idof], cell_soln_dofs_indices[jdof], dF_dWfinei.dx(j_variable++));
                }

                for(unsigned int jdof = 0; jdof < n_soln_dofs_cell; jdof++) // j_variable is solution_interpolated.
                {
                   d2F_solfine_solinterp.add(cell_soln_dofs_indices[idof], cell_soln_dofs_indices[jdof], dF_dWfinei.dx(j_variable++));
                }

                for(unsigned int jdof = 0; jdof < n_metric_dofs_cell; jdof++) // j_variable is X (volume_nodes).
                {
                   d2F_solfine_volnodes.add(cell_soln_dofs_indices[idof], cell_metric_dofs_indices[jdof], dF_dWfinei.dx(j_variable++));
                }
                AssertDimension(j_variable, n_total_independent_variables);
            }
//============================================================================================================================================
            // i variable is solution interpolated.
            AssertDimension(i_variable, n_soln_dofs_cell);
            for(unsigned int idof = 0; idof < n_soln_dofs_cell; idof++)
            {
                const FadType dF_dWinterpi = local_functional_error_fadfad.dx(i_variable++);
                unsigned int j_variable = n_soln_dofs_cell; // j_variable starts from solution_interpolated

                for(unsigned int jdof = 0; jdof < n_soln_dofs_cell; jdof++) // j_variable is solution_interpolated
                {
                    d2F_solinterp_solinterp.add(cell_soln_dofs_indices[idof], cell_soln_dofs_indices[jdof], dF_dWinterpi.dx(j_variable++));
                }

                for(unsigned int jdof = 0; jdof < n_metric_dofs_cell; jdof++) // j_variable is X (volume_nodes)
                {
                    d2F_solinterp_volnodes.add(cell_soln_dofs_indices[idof], cell_metric_dofs_indices[jdof], dF_dWinterpi.dx(j_variable++));
                }
                AssertDimension(j_variable, n_total_independent_variables);
            }
//============================================================================================================================================
            // i variable is volume_nodes.
            AssertDimension(i_variable, 2*n_soln_dofs_cell);
            for(unsigned int idof = 0; idof < n_metric_dofs_cell; idof++) // i_variable is X
            {
                const FadType dF_dXi = local_functional_error_fadfad.dx(i_variable++);
                unsigned int j_variable = 2*n_soln_dofs_cell; // j_variable starts from volume_nodes

                for(unsigned int jdof = 0; jdof < n_metric_dofs_cell; jdof++) // j_variable is X (volume_nodes)
                {
                    d2F_volnodes_volnodes.add(cell_metric_dofs_indices[idof], cell_metric_dofs_indices[jdof], dF_dXi.dx(j_variable++));
                }
                AssertDimension(j_variable, n_total_independent_variables);
            }
            AssertDimension(i_variable, n_total_independent_variables);
        } // compute_d2F ends

    } // cell loop ends

    current_error_value = dealii::Utilities::MPI::sum(error_value_on_this_processor, MPI_COMM_WORLD);

    //Compress all vectors and matrices
    if(compute_dF_dWfine) {
        derivative_functionalerror_wrt_solution_fine.compress(dealii::VectorOperation::add);
    }
    if(compute_dF_dWinterp) {
        derivative_functionalerror_wrt_solution_interpolated.compress(dealii::VectorOperation::add);
    }
    if(compute_dF_dX) {
        derivative_functionalerror_wrt_volume_nodes.compress(dealii::VectorOperation::add);
    }

    if(compute_d2F)
    {
        d2F_solfine_solfine.compress(dealii::VectorOperation::add);
        d2F_solfine_solinterp.compress(dealii::VectorOperation::add);
        d2F_solfine_volnodes.compress(dealii::VectorOperation::add);

        d2F_solinterp_solinterp.compress(dealii::VectorOperation::add);
        d2F_solinterp_volnodes.compress(dealii::VectorOperation::add);

        d2F_volnodes_volnodes.compress(dealii::VectorOperation::add);
    }
    // Decrease poly order of DG
    this->dg->change_cells_fe_degree_by_deltadegree_and_interpolate_solution(-1);
    return current_error_value;
}
template class DirectGoalOrientedError<PHILIP_DIM, 1, double, dealii::Triangulation<PHILIP_DIM>>;
template class DirectGoalOrientedError<PHILIP_DIM, 2, double, dealii::Triangulation<PHILIP_DIM>>;
template class DirectGoalOrientedError<PHILIP_DIM, 3, double, dealii::Triangulation<PHILIP_DIM>>;
template class DirectGoalOrientedError<PHILIP_DIM, 4, double, dealii::Triangulation<PHILIP_DIM>>;
template class DirectGoalOrientedError<PHILIP_DIM, 5, double, dealii::Triangulation<PHILIP_DIM>>;
#if PHILIP_DIM != 1
template class DirectGoalOrientedError<PHILIP_DIM, 1, double,  dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class DirectGoalOrientedError<PHILIP_DIM, 2, double,  dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class DirectGoalOrientedError<PHILIP_DIM, 3, double,  dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class DirectGoalOrientedError<PHILIP_DIM, 4, double,  dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class DirectGoalOrientedError<PHILIP_DIM, 5, double,  dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif
} // PHiLiP namespace
