#include <fstream>
#include <iostream>
#include "Sacado.hpp"
#include <deal.II/lac/full_matrix.h>

template <typename real>
real func1(real &x1)
{
    return 2.0*x1*x1;
}

template <typename real>
real func(real &x1, real &x2, real &x3)
{
    real f = func1(x1) - 3*x2*x2 + 4*x1*x2 + pow((x3+2.0),2.0) + 4.0*x1;
    return f;
}


int main (int /*argc*/, char** /**argv[]*/)
{
    using FadType = Sacado::Fad::DFad<double>;
    using FadFadType = Sacado::Fad::DFad<FadType>;
    unsigned int n_independent_variables = 3;
    std::vector<FadFadType> x_variables(n_independent_variables);

    // Set up AD variables
    unsigned int i_variable = 0;
    for(int i=0; i<3; i++)
    {
        x_variables[i] = (i+10)*(i+3); // set value
        x_variables[i].diff(i_variable++, n_independent_variables);
    }
   
    i_variable = 0;
    for(int i=0; i<3; i++)
    {
        x_variables[i].val() = (i+10)*(i+3); 
        x_variables[i].val().diff(i_variable++, n_independent_variables);
    }

    FadFadType function_value = func(x_variables[0], x_variables[1], x_variables[2]);
    double function_val_double = function_value.val().val();
    std::cout<<"Function value = "<<function_val_double<<std::endl;
    std::vector<double> gradient(n_independent_variables);

    // Compute gradient
    i_variable = 0;
    for(unsigned int i=0; i<n_independent_variables; i++)
    {
        gradient[i] = function_value.dx(i_variable++).val();
    }
    std::cout<<"Gradient = "<<std::endl;
    for(unsigned int i=0; i<n_independent_variables; i++)
    {
        std::cout<<gradient[i]<<std::endl;
    }

    // Compute Hessian
    dealii::FullMatrix<double> Hessian(n_independent_variables, n_independent_variables);

    i_variable = 0;
    for(unsigned int i =0; i<n_independent_variables; i++)
    {
        const FadType df_dxi = function_value.dx(i_variable++);

        unsigned int j_derivative = 0;
        for(unsigned int j = 0; j < n_independent_variables; j++)
        {
            double d2f_dxidxj = df_dxi.dx(j_derivative++);
            Hessian(i,j) = d2f_dxidxj;
        }

    }

    std::cout<<std::endl<<std::endl;
    for(unsigned int i = 0; i<n_independent_variables; i++)
    {
        for(unsigned int j=0; j<n_independent_variables; j++)
        {
            std::cout<<Hessian(i,j)<<"  ";
        }
        std::cout<<std::endl;
    }

    return 0;

}
