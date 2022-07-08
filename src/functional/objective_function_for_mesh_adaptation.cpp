#include "objective_function_for_mesh_adaptation.h"
#include <deal.II/dofs/dof_tools.h>

namespace PHiLiP {

template <int dim, int nstate, typename real, typename MeshType>
ObjectiveFunctionMeshAdaptation<dim,nstate,real,MeshType>::ObjectiveFunctionMeshAdaptation(std::shared_ptr<DGBase<dim,real,MeshType>> _dg)
    : dg(_dg)
    , pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
{}



template <int dim, int nstate, typename real, typename MeshType>
real ObjectiveFunctionMeshAdaptation<dim,nstate,real,MeshType>::evaluate_objective_function_and_derivatives()
{
    using FadType = Sacado::Fad::DFad<real>;
    using FadFadType = Sacado::Fad::DFad<FadType>;

    real objective_function_of_current_processor = 0.0;

    const dealii::FESystem<dim,dim> &fe_metric = dg->high_order_grid->fe_system;
    const unsigned int n_metric_dofs_cell = fe_metric.dofs_per_cell;
    std::vector<dealii::types::global_dof_index> cell_metric_dofs_indices(n_metric_dofs_cell);

    const unsigned int max_dofs_per_cell = dg->dof_handler.get_fe_collection().max_dofs_per_cell();
    std::vector<dealii::types::global_dof_index> cell_soln_dofs_indices(max_dofs_per_cell); 

    std::vector<FadFadType> soln_coeff_fine(max_dofs_per_cell);     // Solution interpolated to p+1, then taylor expanded.
    std::vector<FadFadType> soln_coeff_tilde(max_dofs_per_cell);     // Solution taylor expanded, then interpolated to p+1.

    std::vector<real>   local_derivative_objfunc_wrt_solution_fine(max_dofs_per_cell);
    std::vector<real>   local_derivative_objfunc_wrt_solution_tilde(max_dofs_per_cell);

    std::vector<real>   local_derivative_objfunc_wrt_metric_nodes(n_metric_dofs_cell);

    const auto mapping = (*(dg->high_order_grid->mapping_fe_field));
    
    dealii::hp::MappingCollection<dim> mapping_collection(mapping);
    dealii::hp::FEFaceValues<dim,dim> fe_values_collection_face(mapping_collection, dg->fe_collection, dg->face_quadrature_collection, face_update_flags);

    // Include a function to allocate derivatives.

    dg->solution.update_ghost_values();

    auto metric_cell = dg->high_order_grid->dof_handler_grid.begin_active();
    auto soln_cell = dg->dof_handler.begin_active();

    for( ; soln_cell != dg->dof_handler.end(); ++soln_cell, ++metric_cell)
    {
        if(!(soln_cell->is_locally_owned()))
            continue;

        const unsigned int current_fe_index = soln_cell->active_fe_index();
        const unsigned int current_quad_index = current_fe_index;


        // Resize solution coefficients
        const dealii::FESystem<dim,dim> &fe_solution = dg->fe_collection[current_fe_index];
        const unsigned int n_soln_dofs_cell = fe_solution.n_dofs_per_cell();
        cell_soln_dofs_indices.resize(n_soln_dofs_cell);
        soln_cell->get_dof_indices(cell_soln_dofs_indices);
        soln_coeff_fine.resize(n_soln_dofs_cell);
        soln_coeff_tilde.resize(n_soln_dofs_cell);

        // Get metric coefficients
        metric_cell->get_dof_indices (cell_metric_dofs_indices);
        std::vector< FadFadType > coords_coeff(n_metric_dofs_cell);
        for (unsigned int idof = 0; idof < n_metric_dofs_cell; ++idof) 
        {
            coords_coeff[idof] = dg->high_order_grid->volume_nodes[cell_metric_dofs_indices[idof]];
        }

        // Setup independent variables of gradient for automatic differentiation
        unsigned int n_total_independent_variables = 2.0*n_soln_dofs_cell + n_metric_dofs_cell;

        unsigned int n_current_independent_variable = 0;
        
        for(unsigned int idof = 0; idof < n_soln_dofs_cell; idof++)
        {
            const real value_fine = solution_fine[cell_soln_dofs_indices[idof]];
            soln_coeff_fine[idof] = value_fine;
            soln_coeff_fine[idof].diff(n_current_independent_variable++, n_total_independent_variables);
        }

        for(unsigned int idof = 0; idof < n_soln_dofs_cell; idof++)
        {
            const real value_tilde = solution_tilde[cell_soln_dofs_indices[idof]];
            soln_coeff_tilde[idof] = value_tilde;
            soln_coeff_tilde[idof].diff(n_current_independent_variable++, n_total_independent_variables);
        }

        for (unsigned int idof = 0; idof < n_metric_dofs_cell; ++idof) 
        {
            const real value_coord_coeff = dg->high_order_grid->volume_nodes[cell_metric_dofs_indices[idof]];
            coords_coeff[idof] = value_coord_coeff;
            coors_coeff[idof].diff(n_current_independent_variable++, n_total_independent_variables);
        }

//============================================================================================================================================================
        // Setup automatic differentiation variables for the second derivatives
        n_current_independent_variable = 0;

        for(unsigned int idof = 0; idof < n_soln_dofs_cell; idof++)
        {
            const real value_fine = solution_fine[cell_soln_dofs_indices[idof]];
            soln_coeff_fine[idof].val() = value_fine;
            soln_coeff_fine[idof].val().diff(n_current_independent_variable++, n_total_independent_variables);
        }

        for(unsigned int idof = 0; idof < n_soln_dofs_cell; idof++)
        {
            const real value_tilde = solution_tilde[cell_soln_dofs_indices[idof]];
            soln_coeff_tilde[idof].val() = value_tilde;
            soln_coeff_tilde[idof].val().diff(n_current_independent_variable++, n_total_independent_variables);
        }

        for (unsigned int idof = 0; idof < n_metric_dofs_cell; ++idof) 
        {
            const real value_coord_coeff = dg->high_order_grid->volume_nodes[cell_metric_dofs_indices[idof]];
            coords_coeff[idof].val() = value_coord_coeff;
            coors_coeff[idof].val().diff(n_current_independent_variable++, n_total_independent_variables);
        }
//============================================================================================================================================================
        // Evaluate objective function on the cell.
        const dealii::Quadrature<dim> &volume_quadratures_cell = dg->volume_quadrature_collection[current_quad_index];
        const dealii::Quadrature<dim-1> &face_quadratures = dg->face_quadrature_collection[current_quad_index];
        FadFadType local_objective_function_fadfad = this->evaluate_volume_cell_objective_function(*physics_fad_fad, soln_coeff_fine, soln_coeff_tilde, fe_solution, 
                                                                                              coords_coeff, fe_metric, volume_quadrature_cell);
        
        for(unsignd int iface = 0; iface < dealii::GeometryInfo<dim>::faces_per_cell; ++iface)
        {
            auto face = soln_cell->face(iface);

            if(face->at_boundary())
            {
                const unsigned int boundary_id = face->boundary_id();
                local_objective_function_fadfad += this->evaluate_boundary_cell_objective_function(*physics_fad_fad, boundary_id, soln_coeff_fine, soln_coeff_tilde, fe_solution, 
                                                                                              coords_coeff, fe_metric, iface, face_quadratures);
            }
        }

        
        objective_function_of_current_processor += local_objective_function_fadfad.val().val();

//=========================================================================================================================================================================
        // Evaluate first derivatives
        unsigned int i_variable = 0;

        // First derivative wrt solution fine
        for(unsigned int idof = 0; idof < n_soln_dofs_cell; idof++) // i_variable is Wfine (solution fine)
        {
            const real dF_dWfinei = local_objective_function_fadfad.dx(i_variable++).val();
            derivative_objfunc_wrt_solution_fine.add(cell_soln_dofs_indices[idof], dF_dWfinei);
        }


        // First derivative wrt solution tilde
        for(unsigned int idof = 0; idof < n_soln_dofs_cell; idof++) // i_variable is Wtilde (solution tilde)
        {
            const real dF_dWtildei = local_objective_function_fadfad.dx(i_variable++).val();
            derivative_objfunc_wrt_solution_tilde.add(cell_soln_dofs_indices[idof], dF_dWtildei);
        }

        // First derivative wrt metric nodes
        for(unsigned int idof = 0; idof < n_metric_dofs_cell; idof++) // i_variable is X (metric nodes)
        {
            const real dF_dXi = local_objective_function_fadfad.dx(i_variable++).val();
            derivative_objfunc_wrt_metric_nodes.add(cell_metric_dofs_indices[idof], dF_dXi);
        }

//=========================================================================================================================================================================
        // Evaluate second derivatives
        i_variable = 0;
        for(unsigned int idof = 0; idof < n_soln_dofs_cell; idof++) // i_variable is Wfine
        {
            const FadType dF_dWfinei = local_objective_function_fadfad.dx(i_variable++); 
            unsigned int j_variable = 0;

            for(unsigned int jdof = 0; jdof < n_soln_dofs_cell; jdof++) // j_variable is Wfine
            {
                d2F_dWfine_dWfine.add(cell_soln_dofs_indices[idof], cell_soln_dofs_indices[jdof], dF_dWfinei.dx(j_variable++));
            }

            for(unsigned int jdof = 0; jdof < n_soln_dofs_cell; jdof++) // j_variable is Wtilde
            {
               d2F_dWfine_dWtilde.add(cell_soln_dofs_indices[idof], cell_soln_dofs_indices[jdof], dF_dWfinei.dx(j_variable++));
            }

            for(unsigned int jdof = 0; jdof < n_metric_dofs_cell; jdof++) // j_variable is X
            {
               d2F_dWfine_dX.add(cell_soln_dofs_indices[idof], cell_metric_dofs_indices[jdof], dF_dWfinei.dx(j_variable++));
            }
        }

        for(unsigned int idof = 0; idof < n_soln_dofs_cell; idof++) // i_variable is Wtilde
        {
            const FadType dF_dWtildei = local_objective_function_fadfad.dx(i_variable++);
            unsigned int j_variable = n_soln_dofs_cell;

            for(unsigned int jdof = 0; jdof < n_soln_dofs_cell; jdof++) // j_variable is Wtilde
            {
                d2F_dWtilde_dWtilde.add(cell_soln_dofs_indices[idof], cell_soln_dofs_indices[jdof], dF_dWtildei.dx(j_variable++));
            }
            
            for(unsigned int jdof = 0; jdof < n_metric_dofs_cell; jdof++) // j_variable is X
            {
                d2F_dWtilde_dX.add(cell_soln_dofs_indices[idof], cell_metric_dofs_indices[jdof], dF_dWtildei.dx(j_variable++));
            }

        }

        for(unsigned int i_dof = 0; i_dof < n_metric_dofs_cell; i_dof++) // i_variable is X
        {
            const FadType dF_dXi = local_objective_function_fadfad.dx(i_variable++);
            unsigned int j_variable = 2*n_soln_dofs_cell;

            for(unsigned int jdof = 0; jdof < n_metric_dofs_cell; jdof++) // j_variable is X
            {
                d2F_dX_dX.add(cell_metric_dofs_indices[idof], cell_metric_dofs_indices[jdof], dF_dXi.dx(j_variable++));
            }
        }
    
    } // cell loop ends

    // Compress dealii vectors and matrices
    derivative_objfunc_wrt_solution_fine.compress(dealii::VectorOperation::add);
    derivative_objfunc_wrt_solution_tilde.compress(dealii::VectorOperation::add);
    derivative_objfunc_wrt_metric_nodes.compress(dealii::VectorOperation::add);

    d2F_dWfine_dWfine.compress(dealii::VectorOperation::add);
    d2F_dWfine_dWtilde.compress(dealii::VectorOperation::add);
    d2F_dWfine_dX.compress(dealii::VectorOperation::add);
    d2F_dWtilde_dWtilde.compress(dealii::VectorOperation::add);
    d2F_dWtilde_dX.compress(dealii::VectorOperation::add);
    d2F_dX_dX.compress(dealii::VectorOperation::add);

    real global_objective_function_value = dealii::Utilities::MPI::sum(objective_function_of_current_processor, MPI_COMM_WORLD);
    return global_objective_function_value; 
}


template <int dim, int nstate, typename real, typename MeshType>
void ObjectiveFunctionMeshAdaptation<dim,nstate,real,MeshType>::evaluate_objective_function_hessian()
{
    dealii::IndexSet locally_owned_dofs = dg->high_order_grid->dof_handler_grid.locally_owned_dofs();
    dealii::IndexSet locally_relevant_dofs, ghost_dofs;
    dealii::DoFTools::extract_locally_relevant_dofs(dg->high_order_grid->dof_handler_grid, locally_relevant_dofs);
    ghost_dofs = locally_relevant_dofs;
    ghost_dofs.subtract_set(locally_owned_dofs);
    dFdX.reinit(locally_owned_dofs, ghost_dofs, MPI_COMM_WORLD);
    std::cout<<"Size of dFdX = "<<dFdX.size()<<std::endl;
}

template class ObjectiveFunctionMeshAdaptation<PHILIP_DIM, PHILIP_DIM, double, dealii::Triangulation<PHILIP_DIM>>;
#if PHILIP_DIM != 1
template class ObjectiveFunctionMeshAdaptation<PHILIP_DIM, PHILIP_DIM, double,  dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif

} // namespace PHiLiP
