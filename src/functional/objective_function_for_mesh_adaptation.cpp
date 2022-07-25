#include "objective_function_for_mesh_adaptation.h"
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/sparse_matrix.h>

namespace PHiLiP {

template <int dim, int nstate, typename real, typename MeshType>
 ObjectiveFunctionMeshAdaptation<dim,nstate,real,MeshType>::ObjectiveFunctionMeshAdaptation(
    std::shared_ptr<DGBase<dim,real,MeshType>> _dg, 
    dealii::LinearAlgebra::distributed::Vector<real> & _solution_fine,  
    dealii::LinearAlgebra::distributed::Vector<real> & _solution_tilde)
    : dg(_dg)
    , solution_fine(_solution_fine)
    , solution_tilde(_solution_tilde)
    , pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
{
    functional = FunctionalFactory<dim,nstate,real,MeshType>::create_Functional(dg->all_parameters->functional_param, dg);

    std::shared_ptr<Physics::ModelBase<dim,nstate,FadFadType>> model_fad_fad = Physics::ModelFactory<dim,nstate,FadFadType>::create_Model(dg->all_parameters);
    physics_fad_fad = Physics::PhysicsFactory<dim,nstate,FadFadType>::create_Physics(dg->all_parameters,model_fad_fad);
}


template <int dim, int nstate, typename real, typename MeshType>
void ObjectiveFunctionMeshAdaptation<dim,nstate,real,MeshType>::allocate_derivatives()
{
    // Allocate first derivatives
    const dealii::IndexSet &locally_owned_solution_dofs = dg->dof_handler.locally_owned_dofs();
    derivative_objfunc_wrt_solution_fine.reinit(locally_owned_solution_dofs, MPI_COMM_WORLD);
    derivative_objfunc_wrt_solution_tilde.reinit(locally_owned_solution_dofs, MPI_COMM_WORLD);

    const dealii::IndexSet &locally_owned_grid_dofs = dg->high_order_grid->dof_handler_grid.locally_owned_dofs();
    dealii::IndexSet locally_relevant_grid_dofs, ghost_grid_dofs;
    dealii::DoFTools::extract_locally_relevant_dofs(dg->high_order_grid->dof_handler_grid, locally_relevant_grid_dofs);
    ghost_grid_dofs = locally_relevant_grid_dofs;
    ghost_grid_dofs.subtract_set(locally_owned_grid_dofs);
    derivative_objfunc_wrt_metric_nodes.reinit(locally_owned_grid_dofs, ghost_grid_dofs, MPI_COMM_WORLD);

    // Allocate second derivatives
    dealii::SparsityPattern d2F_dWdW_sparsity_pattern = dg->get_d2RdWdW_sparsity_pattern();
    dealii::SparsityPattern d2F_dWdX_sparsity_pattern = dg->get_d2RdWdX_sparsity_pattern();
    dealii::SparsityPattern d2F_dXdX_sparsity_pattern = dg->get_d2RdXdX_sparsity_pattern();

    d2F_dWfine_dWfine.reinit(locally_owned_solution_dofs, locally_owned_solution_dofs, d2F_dWdW_sparsity_pattern, MPI_COMM_WORLD);
    d2F_dWfine_dWtilde.reinit(locally_owned_solution_dofs, locally_owned_solution_dofs, d2F_dWdW_sparsity_pattern, MPI_COMM_WORLD);
    d2F_dWfine_dX.reinit(locally_owned_solution_dofs, locally_owned_grid_dofs, d2F_dWdX_sparsity_pattern, MPI_COMM_WORLD);

    d2F_dWtilde_dWtilde.reinit(locally_owned_solution_dofs, locally_owned_solution_dofs, d2F_dWdW_sparsity_pattern, MPI_COMM_WORLD);
    d2F_dWtilde_dX.reinit(locally_owned_solution_dofs, locally_owned_grid_dofs, d2F_dWdX_sparsity_pattern, MPI_COMM_WORLD);

    d2F_dX_dX.reinit(locally_owned_grid_dofs, locally_owned_grid_dofs, d2F_dXdX_sparsity_pattern, MPI_COMM_WORLD);
}

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

    const auto mapping = (*(dg->high_order_grid->mapping_fe_field));
    
    dealii::hp::MappingCollection<dim> mapping_collection(mapping);
    dealii::hp::FEFaceValues<dim,dim> fe_values_collection_face(mapping_collection, dg->fe_collection, dg->face_quadrature_collection, face_update_flags);

    allocate_derivatives();

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
            coords_coeff[idof].diff(n_current_independent_variable++, n_total_independent_variables);
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
            coords_coeff[idof].val().diff(n_current_independent_variable++, n_total_independent_variables);
        }
//============================================================================================================================================================
        // Evaluate objective function on the cell.
        const dealii::Quadrature<dim> &volume_quadratures_cell = dg->volume_quadrature_collection[current_quad_index];
        const dealii::Quadrature<dim-1> &face_quadratures = dg->face_quadrature_collection[current_quad_index];
        FadFadType local_objective_function_fadfad = this->evaluate_volume_cell_objective_function(*physics_fad_fad, soln_coeff_fine, soln_coeff_tilde, fe_solution, 
                                                                                              coords_coeff, fe_metric, volume_quadratures_cell);
        
        for(unsigned int iface = 0; iface < dealii::GeometryInfo<dim>::faces_per_cell; ++iface)
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
            derivative_objfunc_wrt_solution_fine(cell_soln_dofs_indices[idof]) += dF_dWfinei;
        }


        // First derivative wrt solution tilde
        for(unsigned int idof = 0; idof < n_soln_dofs_cell; idof++) // i_variable is Wtilde (solution tilde)
        {
            const real dF_dWtildei = local_objective_function_fadfad.dx(i_variable++).val();
            derivative_objfunc_wrt_solution_tilde(cell_soln_dofs_indices[idof]) += dF_dWtildei;
        }

        // First derivative wrt metric nodes
        for(unsigned int idof = 0; idof < n_metric_dofs_cell; idof++) // i_variable is X (metric nodes)
        {
            const real dF_dXi = local_objective_function_fadfad.dx(i_variable++).val();
            derivative_objfunc_wrt_metric_nodes(cell_metric_dofs_indices[idof]) += dF_dXi; // += adds contribution from adjacent cells.
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

        for(unsigned int idof = 0; idof < n_metric_dofs_cell; idof++) // i_variable is X
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
template <typename real2>
real2 ObjectiveFunctionMeshAdaptation<dim,nstate,real,MeshType> :: evaluate_volume_cell_objective_function(
    const Physics::PhysicsBase<dim,nstate,real2> &physics,
    const std::vector< real2 > &soln_coeff_fine,
    const std::vector< real2 > &soln_coeff_tilde,
    const dealii::FESystem<dim> &fe_solution,
    const std::vector< real2 > &coords_coeff,
    const dealii::FESystem<dim> &fe_metric,
    const dealii::Quadrature<dim> &volume_quadrature) const
{
    real2 cell_functional_value_fine = functional->evaluate_volume_cell_functional(physics, soln_coeff_fine, fe_solution, coords_coeff, fe_metric, volume_quadrature);
    real2 cell_functional_value_tilde = functional->evaluate_volume_cell_functional(physics, soln_coeff_tilde, fe_solution, coords_coeff, fe_metric, volume_quadrature);
    real2 eta_cell = cell_functional_value_fine - cell_functional_value_tilde;
    
    real2 cell_objecive_function_value = std::pow(eta_cell, 2);
    return cell_objecive_function_value;
}

template <int dim, int nstate, typename real, typename MeshType>
template <typename real2>
real2 ObjectiveFunctionMeshAdaptation<dim,nstate,real,MeshType> :: evaluate_boundary_cell_objective_function(
    const Physics::PhysicsBase<dim,nstate,real2> &physics,
    const unsigned int boundary_id,
    const std::vector< real2 > &soln_coeff_fine,
    const std::vector< real2 > &soln_coeff_tilde,
    const dealii::FESystem<dim> &fe_solution,
    const std::vector< real2 > &coords_coeff,
    const dealii::FESystem<dim> &fe_metric,
    const unsigned int face_number,
    const dealii::Quadrature<dim-1> &face_quadrature) const
{
    real2 cell_functional_value_fine = functional->evaluate_boundary_cell_functional(physics, boundary_id, soln_coeff_fine, fe_solution, coords_coeff, fe_metric, face_number, face_quadrature);
    real2 cell_functional_value_tilde = functional->evaluate_boundary_cell_functional(physics, boundary_id, soln_coeff_tilde, fe_solution, coords_coeff, fe_metric, face_number, face_quadrature);
    real2 eta_cell = cell_functional_value_fine - cell_functional_value_tilde;
    
    real2 cell_objecive_function_value = std::pow(eta_cell, 2);
    return cell_objecive_function_value;
}

template <int dim, int nstate, typename real, typename MeshType>
void ObjectiveFunctionMeshAdaptation<dim,nstate,real,MeshType>::evaluate_objective_function_hessian()
{
    dealii::IndexSet locally_owned_dofs = dg->high_order_grid->dof_handler_grid.locally_owned_dofs();
    dealii::IndexSet locally_relevant_dofs, ghost_dofs;
    dealii::DoFTools::extract_locally_relevant_dofs(dg->high_order_grid->dof_handler_grid, locally_relevant_dofs);
    ghost_dofs = locally_relevant_dofs;
    ghost_dofs.subtract_set(locally_owned_dofs);
    derivative_objfunc_wrt_metric_nodes.reinit(locally_owned_dofs, ghost_dofs, MPI_COMM_WORLD);
    std::cout<<"Size of dFdX = "<<derivative_objfunc_wrt_metric_nodes.size()<<std::endl;
}

template <int dim, int nstate, typename real, typename MeshType>
void ObjectiveFunctionMeshAdaptation<dim,nstate,real,MeshType>::truncate_first_derivative(dealii::LinearAlgebra::distributed::Vector<real> &vector_in)
{
   unsigned int vector_size = vector_in.size(); // NOTE: Change for parallel processing
   dealii::LinearAlgebra::distributed::Vector<real> vector_out (vector_size - 2);

   for(unsigned int i=1; i<vector_size-1; i++)
   {
       vector_out(i-1) = vector_in(i);
   }

    vector_in = vector_out;
}

template <int dim, int nstate, typename real, typename MeshType>
void ObjectiveFunctionMeshAdaptation<dim,nstate,real,MeshType>::truncate_second_derivative(dealii::TrilinosWrappers::SparseMatrix &d2F, bool is_dX_dX)
{
    unsigned int n_rows = d2F.m();
    unsigned int n_columns = d2F.n();
    std::vector<unsigned int> row_indices;
    std::vector<unsigned int> column_indices;

   // Remove first and last column. 
    for(unsigned int i=0; i<n_columns-2; i++)
    {
        column_indices.push_back(i+1);
    }

    if(is_dX_dX)
    {
        // Remove first and last row.
        for(unsigned int i=0; i<n_rows-2; i++)
        {
            row_indices.push_back(i+1);
        }        

    }
    else
    {
        for(unsigned int i=0; i<n_rows; i++)
        {
            row_indices.push_back(i);
        }
    }

    // Copy full to sparse matrix
    dealii::FullMatrix<real> d2F_full_modified(row_indices.size(), column_indices.size());
    d2F_full_modified.extract_submatrix_from(d2F, row_indices, column_indices);

    dealii::SparsityPattern sparsity_pattern;
    sparsity_pattern.copy_from(d2F_full_modified);
    dealii::TrilinosWrappers::SparseMatrix d2F_sparse_modified;
    d2F_sparse_modified.reinit(sparsity_pattern);

    for(unsigned int i=0; i<row_indices.size(); i++)
    {
        for(unsigned int j=0; j<column_indices.size(); j++)
        {
            if(d2F_full_modified(i,j)==0) continue;

            d2F_sparse_modified.add(i,j,d2F_full_modified(i,j));
        }
    }
    // Update input sparse matrix
    d2F.copy_from(d2F_sparse_modified);
}

template class ObjectiveFunctionMeshAdaptation<PHILIP_DIM, 1, double, dealii::Triangulation<PHILIP_DIM>>;
template class ObjectiveFunctionMeshAdaptation<PHILIP_DIM, 2, double, dealii::Triangulation<PHILIP_DIM>>;
template class ObjectiveFunctionMeshAdaptation<PHILIP_DIM, 3, double, dealii::Triangulation<PHILIP_DIM>>;
template class ObjectiveFunctionMeshAdaptation<PHILIP_DIM, 4, double, dealii::Triangulation<PHILIP_DIM>>;
template class ObjectiveFunctionMeshAdaptation<PHILIP_DIM, 5, double, dealii::Triangulation<PHILIP_DIM>>;
#if PHILIP_DIM != 1
template class ObjectiveFunctionMeshAdaptation<PHILIP_DIM, 1, double,  dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class ObjectiveFunctionMeshAdaptation<PHILIP_DIM, 2, double,  dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class ObjectiveFunctionMeshAdaptation<PHILIP_DIM, 3, double,  dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class ObjectiveFunctionMeshAdaptation<PHILIP_DIM, 4, double,  dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class ObjectiveFunctionMeshAdaptation<PHILIP_DIM, 5, double,  dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif

} // namespace PHiLiP
