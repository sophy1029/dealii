/*
 * WGDarcyEquation.cc

 *  Created on: Feb 15, 2018
 *      Author: ubuntu
 */

// @sect3{Include files}
// This program is based on step-7, step-20 and step-51,
// we add these include files.
#include <deal.II/base/quadrature.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_face.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_out_faces.h>
#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/base/point.h>
#include <deal.II/fe/component_mask.h>

#include <fstream>
#include <iostream>

// This is the same as previous programs.
using namespace dealii;

// @sec3{The WGDarcyEquation class template}

// We will solve the numerical pressure in interiors and on faces and calculate $L_2$ error of pressure.
// In the post-processing step,
// we will calculate $L_2$-errors of velocity and flux.
template <int dim>
class WGDarcyEquation
{
public:
  WGDarcyEquation ();
  void run ();

private:
  void make_grid ();
  void setup_system();
  void assemble_system ();
  void solve ();
  void postprocess();
  void process_solution ();
  void output_results () const;

  Triangulation<dim>   triangulation;

  ConstraintMatrix     constraints;

  FE_RaviartThomas<dim>  fe_rt;
  DoFHandler<dim>      dof_handler_rt;

  // The finite element system is used for interior and face solutions.
  FESystem<dim>          fe;
  DoFHandler<dim>      dof_handler;

  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;

  Vector<double>       solution;
  Vector<double>       system_rhs;
};

// @sec3{Right hand side, boundary values, and exact solution}

// Next, we define the coefficient matrix $\mathbf{K}$,
// Dirichelet boundary condition, right hand side,
// exact solution of pressure and velocity for calculating errors.
//
// Coefficient matrix $\mathbf{K}$ is the identity matrix as a test example.
template <int dim>
class Coefficient : public TensorFunction<2,dim>
{
   public:
	 Coefficient () : TensorFunction<2,dim>() {}
     virtual void value_list (const std::vector<Point<dim> > &points,
                              std::vector<Tensor<2,dim> >    &values) const;
};

template <int dim>
void
Coefficient<dim>::value_list (const std::vector<Point<dim> > &points,
                              std::vector<Tensor<2,dim> >    &values) const
{
  Assert (points.size() == values.size(),
          ExcDimensionMismatch (points.size(), values.size()));
  for (unsigned int p=0; p<points.size(); ++p)
    {
      values[p].clear ();
      for (unsigned int d=0; d<dim; ++d)
    	  values[p][d][d] = 1;
    }
}

template <int dim>
class BoundaryValues : public Function<dim>
{
public:
  BoundaryValues (const unsigned int n_components = 2 ) : Function<dim>(2) {}
  BoundaryValues () : Function<dim>(2) {}
  virtual double value (const Point<dim>   &p,
                        const unsigned int  component = 2 ) const;
};

template <int dim>
double BoundaryValues<dim>::value (const Point<dim> &p,
		                           const unsigned int) const
{
	return 0;
}

template <int dim>
class RightHandSide : public Function<dim>
{
public:
  RightHandSide () : Function<dim>() {}
  virtual double value (const Point<dim>   &p,
                        const unsigned int  component = 0) const;
};

template <int dim>
double RightHandSide<dim>::value (const Point<dim> &p,
		                          const unsigned int /*component*/) const
{
  double return_value = 0.0;
  return_value = 2*M_PI*M_PI*sin(M_PI*p[0])*sin(M_PI*p[1]);
  return return_value;
}

template <int dim>
class Solution : public Function<dim>
{
public:
  Solution (const unsigned int n_components) : Function<dim>(2) {}
  virtual double value (const Point<dim>   &p,
                        const unsigned int    ) const;
};

template <int dim>
double Solution<dim>::value (const Point<dim>   &p,
                             const unsigned int) const
{
  double return_value = 0;
  return_value = sin(M_PI*p[0])*sin(M_PI*p[1]);
  return return_value;
}

template <int dim>
class Velocity : public TensorFunction<1,dim>
{
public:
  Velocity () : TensorFunction<1,dim>() {}
  virtual Tensor<1,dim> value (const Point<dim>   &p, const unsigned int component = 0) const;
};

template <int dim>
Tensor<1,dim> Velocity<dim>::value (const Point<dim>   &p, const unsigned int) const
{
  Tensor<1,dim> return_value;
  return_value[0] = -M_PI*cos(M_PI*p[0])*sin(M_PI*p[1]);
  return_value[1] = -M_PI*sin(M_PI*p[0])*cos(M_PI*p[1]);
  return return_value;

}

// @sect3{WGDarcyEquation class implementation}

// @sect4{WGDarcyEquation::WGDarcyEquation}

// In this constructor, we create a finite element space for vector valued functions, <code>FE_RaviartThomas</code>.
// We will need shape functions in this space to approximate discrete weak gradients.
// So discrete weak gradients of pressure, discrete weak gradients of shape functions of polynomial spaces are
// approximated by basis functions of the Raviart-Thomas space.

// <code>FESystem</code> defines finite element spaces in interior and on edges of elements,
// each part only has one copy of the polynomial space.
// Others are the same as previous tutorial programs.
 template <int dim>
 WGDarcyEquation<dim>::WGDarcyEquation ()
   :
   fe_rt (0),
   dof_handler_rt (triangulation),

   fe (FE_DGQ<dim>(0), 1,
       FE_FaceQ<dim>(0), 1),
   dof_handler (triangulation)

 {}

// @sect4{WGDarcyEquation::make_grid}

// We generate a mesh on the unit square domain and refine it.

 template <int dim>
 void WGDarcyEquation<dim>::make_grid ()
 {
   GridGenerator::hyper_cube (triangulation, 0, 1);
   triangulation.refine_global (1);

   std::cout << "   Number of active cells: "
             << triangulation.n_active_cells()
             << std::endl
             << "   Total number of cells: "
             << triangulation.n_cells()
             << std::endl;

 }

 // @sect4{WGDarcyEquation::setup_system}

 // After we create the mesh, we distribute dofs of finite elements defined on <code>FE_RaviartThomas</code>,
 // and dofs of finite elements defined on <code>FESystem</code> by <code>DoFHandler</code>.
 // The @p solution is the global solution.

 template <int dim>
 void WGDarcyEquation<dim>::setup_system ()
 {
   dof_handler_rt.distribute_dofs (fe_rt);
   dof_handler.distribute_dofs(fe);

   std::cout << "   Number of rt degrees of freedom: "
             << dof_handler_rt.n_dofs()
             << std::endl;

   std::cout << "   Number of dof: "
                   << dof_handler.n_dofs()
                   << std::endl;

   DynamicSparsityPattern dsp(dof_handler.n_dofs());
   DoFTools::make_sparsity_pattern (dof_handler, dsp,constraints);
   sparsity_pattern.copy_from(dsp);

   system_matrix.reinit (sparsity_pattern);

   solution.reinit (dof_handler.n_dofs());
   system_rhs.reinit (dof_handler.n_dofs());

 }

 // @sect4{WGDarcyEquation::assemble_system}

 // First, we allocate quadrature points and <code>FEValues</code> for cells and faces.
 // Then we allocate space for all cell matrices and right hand side.
 // The following definitions have been explained in previous tutorials.
 template <int dim>
 void WGDarcyEquation<dim>::assemble_system ()
 {
   QGauss<dim>  quadrature_formula(fe_rt.degree+1);
   QGauss<dim-1>  face_quadrature_formula(fe_rt.degree+1);
   const RightHandSide<dim> right_hand_side;

   // We define objects to evaluate values, gradients of shape functions at the quadrature points.
   // Since we need shape functions and normal vectors on faces, we need <code>FEFaceValues</code>.
   FEValues<dim> fe_values_rt (fe_rt, quadrature_formula,
                               update_values   | update_gradients |
                               update_quadrature_points | update_JxW_values);

   FEValues<dim> fe_values (fe, quadrature_formula,
                              update_values   |
                              update_quadrature_points | update_JxW_values);

   FEFaceValues<dim> fe_face_values (fe, face_quadrature_formula,
                                     update_values   | update_normal_vectors |
                                     update_quadrature_points | update_JxW_values);

   FEFaceValues<dim> fe_face_values_rt (fe_rt, face_quadrature_formula,
                                        update_values   | update_normal_vectors |
                                        update_quadrature_points | update_JxW_values);


   const unsigned int   dofs_per_cell_rt = fe_rt.dofs_per_cell;
   const unsigned int   dofs_per_cell = fe.dofs_per_cell;

   const unsigned int   n_q_points    = fe_values.get_quadrature().size();
   const unsigned int   n_q_points_rt = fe_values_rt.get_quadrature().size();
   const unsigned int   n_face_q_points = fe_face_values.get_quadrature().size();
   const unsigned int   n_face_q_points_rt = fe_face_values_rt.get_quadrature().size();

   std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

   // We will construct these cell matrices to solve for the pressure.
   FullMatrix<double>   cell_matrix_rt (dofs_per_cell_rt,dofs_per_cell_rt);
   FullMatrix<double>   cell_matrix_F (dofs_per_cell,dofs_per_cell_rt);
   FullMatrix<double>   cell_matrix_C (dofs_per_cell,dofs_per_cell_rt);
   FullMatrix<double>   local_matrix (dofs_per_cell,dofs_per_cell);
   FullMatrix<double>   cell_matrix_D (dofs_per_cell_rt,dofs_per_cell_rt);
   FullMatrix<double>   cell_matrix_E (dofs_per_cell_rt,dofs_per_cell_rt);
   Vector<double>       cell_rhs (dofs_per_cell);
   Vector<double>       cell_solution (dofs_per_cell);

   const Coefficient<dim> coefficient;
   std::vector<Tensor<2,dim>> coefficient_values (n_q_points_rt);

   // We need extractors to extract components of shape functions used for
   // calculating pressures, @p interior will be used to approximate integrals
   // defined in the interior, and @p face will be used to approximate integrals defined on faces.
   const FEValuesExtractors::Vector velocities (0);
   const FEValuesExtractors::Scalar interior (0);
   const FEValuesExtractors::Scalar face (1);

   typename DoFHandler<dim>::active_cell_iterator
   cell = dof_handler.begin_active(),
   endc = dof_handler.end();
   typename DoFHandler<dim>::active_cell_iterator
   cell_rt = dof_handler_rt.begin_active();

   // Here, we will calculate cell matrices used to construct the local matrix on each cell.
   // We need shape functions on the Raviart-Thomas space, so the loop is also defined on this space.
   for (; cell!=endc; ++cell,++cell_rt)
     {
	   // On each cell, cell matrices are different, so in every loop, they need to be re-computed.
       fe_values_rt.reinit (cell_rt);
       fe_values.reinit (cell);
       cell_matrix_rt = 0;
       cell_matrix_F = 0;
       cell_matrix_C = 0;
       local_matrix = 0;
       cell_rhs = 0;
       coefficient.value_list (fe_values_rt.get_quadrature_points(),
                               coefficient_values);

       // This cell matrix is the integral of all basis functions of <code>FE_RaviartThomas</code>.
       // The loop is over all quadrature points defined on <code>FE_RaviartThomas</code>.
       // Next we take the inverse of this matrix by using <code>gauss_jordan()</code>.
       for (unsigned int q=0; q<n_q_points_rt; ++q)
       {

    	   for (unsigned int i=0; i<dofs_per_cell_rt; ++i)
           {
    		 const Tensor<1,dim> phi_i_u = fe_values_rt[velocities].value (i, q);
    		 const double phi_i_u_div = fe_values_rt[velocities].divergence(i,q);
             for (unsigned int j=0; j<dofs_per_cell_rt; ++j)
              {
               const Tensor<1,dim> phi_j_u = fe_values_rt[velocities].value (j, q);
               cell_matrix_rt(i,j) += (
            		                   phi_i_u * phi_j_u *
            		                   fe_values_rt.JxW(q));
              }
           }
        }
       cell_matrix_rt.gauss_jordan();

       // From the introduction, we know that the right hand side
       // is an integral on faces minus an integral in the interior.
       // Here, we approximate the negative of integral in interior.
       // Each component of this matrix is the integral of product of the basis function of the polynomial space
       // and divergence of the basis function of the Raviart-Thomas space.
       // These basis functions are defined in interior.
   	   for (unsigned int q=0; q<n_q_points; ++q)
   	       {
   	         for (unsigned int i=0; i<dofs_per_cell; ++i)
   	             {
   	         	   for (unsigned int k=0; k<dofs_per_cell_rt; ++k)
   	         	       {
   	         		     const double phi_k_u_div = fe_values_rt[velocities].divergence(k,q);
   	                     cell_matrix_F(i,k) -= (fe_values[interior].value(i,q) *
   	                                            phi_k_u_div *
   	                                            fe_values.JxW (q));
   	         	       }
   	              }
   	        }

   	 // Now, we approximate the integral on faces.
   	 // Each component is the integral of product of basis function of the polynomial space
   	 // and the dot product of the basis function of the Raviart-Thomas space and the normal vector.
   	 // So this loop is over all the faces of the element.
   	 // And we get the normal vector of each face from <code>FEFaceValues</code>.
       for (unsigned int face_n=0; face_n<GeometryInfo<dim>::faces_per_cell; ++face_n)
           {
           	   fe_face_values.reinit (cell,face_n);
           	   fe_face_values_rt.reinit (cell_rt,face_n);
           	   for (unsigned int q=0; q<n_face_q_points; ++q)
           	       {
       	         	  const Tensor<1,dim> normal = fe_face_values.normal_vector(q);
           	          for (unsigned int i=0; i<dofs_per_cell; ++i)
           	              {
           	         	       for (unsigned int k=0; k<dofs_per_cell_rt; ++k)
           	                       {
           	                          const Tensor<1,dim> phi_k_u = fe_face_values_rt[velocities].value (k, q);
           	                          cell_matrix_F(i,k) += (fe_face_values[face].value(i,q) *
           	                                                (phi_k_u * normal)*
           	                                                 fe_face_values.JxW (q));
           	                        }
           	              }
           	        }
           	 }

     // We calculate @p cell_matrix_C by doing matrix multiplication via <code>SparseMatrix::mmult</code>.
       cell_matrix_F.mmult(cell_matrix_C,cell_matrix_rt);

     // In the local matrix of this element, component is
     // $\int_{E} \mathbf{K} c_{ik} c_{jl} \mathbf{w}_k \cdot \mathbf{w}_l.$
     // We have calculated coefficients $c$ from the previous step.
     // And these basis functions $\mathbf{w}$ are defined in the interior.
       for (unsigned int q=0; q<n_q_points_rt; ++q)
           {
	         for(unsigned int i = 0; i<dofs_per_cell; ++i)
	            {
		          for(unsigned int j = 0; j<dofs_per_cell; ++j)
		             {
			           for(unsigned int k = 0; k<dofs_per_cell_rt; ++k)
			              {
				            const Tensor<1,dim> phi_k_u = fe_values_rt[velocities].value (k, q);
				            for(unsigned int l = 0; l<dofs_per_cell_rt; ++l)
				               {
					             const Tensor<1,dim> phi_l_u = fe_values_rt[velocities].value (l, q);
					             local_matrix(i,j) += coefficient_values[q]*cell_matrix_C[i][k]*cell_matrix_C[j][l]*
							                          phi_k_u*phi_l_u*
							                          fe_values_rt.JxW(q);
				                }
			               }
		              }
	             }
             }

      // Next, we calculate the right hand side, $\int_{E} f q$.
        for (unsigned int q=0; q<n_q_points; ++q)
            {
      	       for (unsigned int i=0; i<dofs_per_cell; ++i)
      	           {
      	    	      cell_rhs(i) += (fe_values[interior].value(i,q) *
      	    			              right_hand_side.value (fe_values.quadrature_point (q)) *
      	    			              fe_values.JxW (q));
      	            }
             }

   // In this part, we distribute components of this local matrix into the system matrix
   // and transfer components of cell right hand side into the system right hand side.
       cell->get_dof_indices (local_dof_indices);
       constraints.distribute_local_to_global (local_matrix, local_dof_indices,
                                               system_matrix);


       for (unsigned int i=0; i<dofs_per_cell; ++i)
           {
             system_rhs(local_dof_indices[i]) += cell_rhs(i);
           }
    }

  // We assign Dirichelet boundary conditions to the solution.
  // Here, we only need dofs of finite elements defined on faces.
  // We use <code>ComponentMask</code> to generate a mask which only denotes
  // the face pressure components on the element.
  ComponentMask face_pressure_mask = fe.component_mask (face);
  std::map<types::global_dof_index,double> boundary_values;
  VectorTools::interpolate_boundary_values (dof_handler,
                                             0,
                                             BoundaryValues<dim>(2),
                                             boundary_values,
                                             face_pressure_mask);

  MatrixTools::apply_boundary_values (boundary_values,
                                      system_matrix,
                                      solution,
                                      system_rhs);
 }

 // @sect4{WGDarcyEquation<dim>::solve}

 // Solving the system of the Darcy equation. Now, we have pressures in interiors and on faces.
 template <int dim>
 void WGDarcyEquation<dim>::solve ()
 {
   SolverControl           solver_control (1000, 1e-8 * system_rhs.l2_norm());
   SolverCG<>              solver (solver_control);
   solver.solve (system_matrix, solution, system_rhs,
                 PreconditionIdentity());
 }

 // @sect4{WGDarcyEquation<dim>::process_solution}

 // This part is to calculate the $L_2$ error of pressure. To calculate the $L_2$ error of interior pressure,
 // we will extract interior solutions from the global solution.
 template <int dim>
 void WGDarcyEquation<dim>::process_solution ()
 {
	// Since we have two different spaces for finite elements in interior and on faces,
	// if we want to calculate $L_2$ errors in interior, we need dofs only defined in cells.
	// In <code>FESystem</code>, we have two components, the first one is for interior,
	// the second one is for skeletons.
	// <code>fe.base_element(0)</code> shows we only need dofs defined in cells.
	DoFHandler<dim> interior_dof_handler (triangulation);
	interior_dof_handler.distribute_dofs (fe.base_element(0));
	// We define a vector to extract pressures in cells.
	// The size of the vector is the number of all dofs in interiors of all elements.
	Vector<double> interior_solution (interior_dof_handler.n_dofs());
	{
		// <code>types::global_dof_index</code> is used to know the global indices of degrees of freedom.
		// So here, we get the global indices of local dofs and the global indices of interior dofs.
		std::vector<types::global_dof_index> local_dof_indices (fe.dofs_per_cell);
		std::vector<types::global_dof_index> interior_local_dof_indices (fe.base_element(0).dofs_per_cell);
		typename DoFHandler<dim>::active_cell_iterator
		cell = dof_handler.begin_active(),
		endc = dof_handler.end(),
		interior_cell = interior_dof_handler.begin_active();

        // In the loop of all cells and interior of the cell,
		// we extract interior solutions from the global solution.
		for (; cell!=endc; ++cell, ++interior_cell)
		{
			cell->get_dof_indices (local_dof_indices);
			interior_cell->get_dof_indices (interior_local_dof_indices);

			for (unsigned int i=0; i<fe.base_element(0).dofs_per_cell; ++i)
	  		  interior_solution (interior_local_dof_indices[i])
	  		  =  solution(local_dof_indices[fe.component_to_system_index(0,i)]);
		}
	}

	// We define a vector that holds the norm of the error on each cell.
	// Next, we use <code>VectorTool::integrate_difference</code>
	// to compute the error in the $L_2$ norm on each cell.
	// Finally, we get the global $L_2$ norm.
	    Vector<float> difference_per_cell (triangulation.n_active_cells());
	    VectorTools::integrate_difference (interior_dof_handler,
	                                       interior_solution,
	                                       Solution<dim>(1),
	                                       difference_per_cell,
	                                       QGauss<dim>(fe.degree+2),
	                                       VectorTools::L2_norm);
	    const double L2_error = VectorTools::compute_global_error(triangulation,
	                                                              difference_per_cell,
	                                                              VectorTools::L2_norm);

//	    const double L2_error = difference_per_cell.l2_norm();
	    std::cout<<"L2_error_pressure "<<L2_error<<std::endl;

 }

 // @sect4{WGDarcyEquation<dim>::postprocess}

 // After we calculate the numerical pressure, we evaluate $L_2$ errors of velocity of a cell and flux on faces.

 // We are going to evaluate velocities on each cell and calculate the difference between numerical
 // and exact velocities. To calculate velocities, we need interior and face pressure values of each element,
 // and some other cell matrices.

 // For calculating the error of flux, we need areas of elements, length of edges and velocities on faces.

 template <int dim>
 void WGDarcyEquation<dim>::postprocess ()
 {
   QGauss<dim>  quadrature_formula(fe_rt.degree+1);
   QGauss<dim-1>  face_quadrature_formula(fe_rt.degree+1);

   FEValues<dim> fe_values_rt (fe_rt, quadrature_formula,
                               update_values   | update_gradients |
                               update_quadrature_points | update_JxW_values);

   FEValues<dim> fe_values (fe, quadrature_formula,
                            update_values   |
                            update_quadrature_points | update_JxW_values);

   FEFaceValues<dim> fe_face_values (fe, face_quadrature_formula,
                                     update_values   | update_normal_vectors |
                                     update_quadrature_points | update_JxW_values);

   FEFaceValues<dim> fe_face_values_rt (fe_rt, face_quadrature_formula,
                                        update_values   | update_normal_vectors |
                                        update_quadrature_points | update_JxW_values);

   const unsigned int   dofs_per_cell_rt = fe_rt.dofs_per_cell;
   const unsigned int   dofs_per_cell_rt_velocity = fe_rt.dofs_per_cell;
   const unsigned int   dofs_per_cell = fe.dofs_per_cell;

   const unsigned int   n_q_points_rt = fe_values_rt.get_quadrature().size();
   const unsigned int   n_q_points    = fe_values.get_quadrature().size();
   const unsigned int   n_face_q_points = fe_face_values.get_quadrature().size();
   const unsigned int   n_face_q_points_rt = fe_face_values_rt.get_quadrature().size();


   std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
   FullMatrix<double>   cell_matrix_rt (dofs_per_cell_rt,dofs_per_cell_rt);
   FullMatrix<double>   cell_matrix_F (dofs_per_cell,dofs_per_cell_rt);
   FullMatrix<double>   cell_matrix_C (dofs_per_cell,dofs_per_cell_rt);
   FullMatrix<double>   local_matrix (dofs_per_cell,dofs_per_cell);
   FullMatrix<double>   cell_matrix_D (dofs_per_cell_rt,dofs_per_cell_rt);
   FullMatrix<double>   cell_matrix_E (dofs_per_cell_rt,dofs_per_cell_rt);
   Vector<double>       cell_rhs (dofs_per_cell);
   Vector<double>       cell_solution (dofs_per_cell);
   Tensor<1,dim>        velocity_cell;
   Tensor<1,dim>        velocity_face;
   Tensor<1,dim>        exact_velocity_face;
   Tensor<0,dim>        difference_velocity_cell_sqr;
   Tensor<0,dim>        difference_velocity_face_sqr;
   Tensor<0,dim>        L2_err_velocity_cell_sqr_local;
   Tensor<0,dim>        L2_err_velocity_cell_sqr_global;
   Tensor<0,dim>        flux_face;
   Tensor<0,dim>        difference_flux_face_sqr;
   Tensor<0,dim>        L2_err_flux_sqr;
   Tensor<0,dim>        L2_err_flux_face_sqr_local;
   Tensor<0,dim>        err_flux_each_face;
   Tensor<0,dim>        err_flux_face;

   typename DoFHandler<dim>::active_cell_iterator
   cell = dof_handler.begin_active(),
   endc = dof_handler.end();

   typename DoFHandler<dim>::active_cell_iterator
   cell_rt = dof_handler_rt.begin_active();

   const Coefficient<dim> coefficient;
   std::vector<Tensor<2,dim>> coefficient_values (n_q_points_rt);
   const FEValuesExtractors::Vector velocities (0);
   const FEValuesExtractors::Scalar pressure (dim);
   const FEValuesExtractors::Scalar interior (0);
   const FEValuesExtractors::Scalar face (1);

   Velocity<dim> exact_velocity;

   // In the loop over all cells, we will calculate $L_2$ errors of velocity and flux.

   // First, we calculate the $L_2$ error of velocity. In weak Galerkin, velocity is defined on the vector space, Raviart-Thomas space.
   // In the introduction, we explained how to calculate the numerical velocity on the cell. We need pressures of the cell,
   // coefficients of the Gram matrix and coefficients of the $L_2$ projection. We have already calculated the global solution,
   // so we will extract cell solution from the global solution. The coefficients of the Gram matrix have been calculated
   // when we evaluate pressures. We will do the same way here. For the coefficients of the projection, we do matrix multiplication, i.e.,
   // the inverse of the Gram matrix times the matrix with $\mathbf{K} \mathbf{w} \cdot \mathbf{w}$ as components.
   // Then we multiply all these coefficients and name it to be @p beta. The numerical velocity is the product of coefficient
   // @p beta and basis functions of the Raviart-Thomas space.
   for (; cell!=endc; ++cell,++cell_rt)
       {
        fe_values_rt.reinit (cell_rt);
        fe_values.reinit (cell);
        cell_matrix_rt = 0;
        cell_matrix_E = 0;
        cell_matrix_D = 0;
        cell_matrix_C = 0;
        cell_matrix_F = 0;
        velocity_cell = 0;
        coefficient.value_list (fe_values_rt.get_quadrature_points(),
                                coefficient_values);

        // The component of this cell matrix is the integral of $\mathbf{K} \mathbf{w} \cdot \mathbf{w}$.
        for (unsigned int q=0; q<n_q_points_rt; ++q)
            {
               for (unsigned int i=0; i<dofs_per_cell_rt; ++i)
                   {
            		 const Tensor<1,dim> phi_i_u = fe_values_rt[velocities].value (i, q);

                     for (unsigned int j=0; j<dofs_per_cell_rt; ++j)
                      {
                       const Tensor<1,dim> phi_j_u = fe_values_rt[velocities].value (j, q);

                       cell_matrix_E(i,j) += (coefficient_values[q] *
                    		                   phi_j_u * phi_i_u *
                    		                   fe_values_rt.JxW(q));
                      }
                   }
                }

        // This is the Gram matrix on the cell.
         for (unsigned int q=0; q<n_q_points_rt; ++q)
              {
                for (unsigned int i=0; i<dofs_per_cell_rt; ++i)
                    {
                      const Tensor<1,dim> phi_i_u = fe_values_rt[velocities].value (i, q);
                      for (unsigned int j=0; j<dofs_per_cell_rt; ++j)
                          {
                            const Tensor<1,dim> phi_j_u = fe_values_rt[velocities].value (j, q);
                            cell_matrix_rt(i,j) += (
                           		                   phi_i_u * phi_j_u *
                           		                   fe_values_rt.JxW(q));
                           }
                     }
                }

         // We take the inverse of the Gram matrix, take matrix multiplication and get the matrix with coefficients of projection.
          cell_matrix_rt.gauss_jordan();
          cell_matrix_rt.mmult(cell_matrix_D,cell_matrix_E);

         // This is to extract pressure values of the element.
          cell->get_dof_indices (local_dof_indices);
          cell_solution = 0;
          for (unsigned int i=0; i<dofs_per_cell; ++i)
              {
             	cell_solution(i) = solution(local_dof_indices[i]);
              }

           // This cell matrix will be used to calculate the coefficients of the Gram matrix.
           // This part is the same as the part in evaluating pressure.
           for (unsigned int q=0; q<n_q_points; ++q)
               {
                 for (unsigned int i=0; i<dofs_per_cell; ++i)
                	 {
                       for (unsigned int k=0; k<dofs_per_cell_rt; ++k)
                           {
                 	          const double phi_k_u_div = fe_values_rt[velocities].divergence(k,q);
                 	          cell_matrix_F(i,k) -= (fe_values[interior].value(i,q) *
                 	                                 phi_k_u_div *
                 	                                 fe_values.JxW (q));
                 	        }
                 	  }
                 }

            for (unsigned int face_n=0; face_n<GeometryInfo<dim>::faces_per_cell; ++face_n)
                {
                   fe_face_values.reinit (cell,face_n);
                   fe_face_values_rt.reinit (cell_rt,face_n);
                   for (unsigned int q=0; q<n_face_q_points; ++q)
                       {
                     	 const Tensor<1,dim> normal = fe_face_values.normal_vector(q);
                         for (unsigned int i=0; i<dofs_per_cell; ++i)
                         	  {
                         	    for (unsigned int k=0; k<dofs_per_cell_rt; ++k)
                         	        {
                         	          const Tensor<1,dim> phi_k_u = fe_face_values_rt[velocities].value (k, q);
                         	          cell_matrix_F(i,k) += (fe_face_values[face].value(i,q) *
                         	                                (phi_k_u * normal)*
                         	                                 fe_face_values.JxW (q));
                         	         }
                         	   }
                         }
                  }

             cell_matrix_F.mmult(cell_matrix_C,cell_matrix_rt);

             // From previous calculations, we have all the coefficients. Then we get the coefficient @p beta.
             Vector<double> beta (dofs_per_cell_rt);
             beta = 0;
             for(unsigned int k = 0; k<dofs_per_cell_rt;++k)
                 {
                   for (unsigned int j = 0; j<dofs_per_cell_rt; ++j)
                        {
                          for (unsigned int i = 0; i<dofs_per_cell;++i)
                              {
                                 beta(k) += -(cell_solution(i)*
                                    		  cell_matrix_C(i,j) *
                                    	      cell_matrix_D(k,j));
                               }
                         }
                  }

              // We calculate numerical velocity on each quadrature point,
              // find the squares of difference between numerical and exact velocity on the quadrature point.
              // Then we approximate the error over all quadrature points and add it to the global $L_2$ error.
              L2_err_velocity_cell_sqr_local = 0;
              for (unsigned int q=0; q<n_q_points_rt; ++q)
                  {
                	 difference_velocity_cell_sqr = 0;
                	 velocity_cell = 0;
                     for(unsigned int k = 0; k<dofs_per_cell_rt;++k)
                        {
                          const Tensor<1,dim> phi_k_u = fe_values_rt[velocities].value (k, q);
                          velocity_cell += beta(k)*phi_k_u;
                        }
                     difference_velocity_cell_sqr = (velocity_cell - exact_velocity.value (fe_values_rt.quadrature_point (q)))*
                        		                    (velocity_cell - exact_velocity.value (fe_values_rt.quadrature_point (q)));
                     L2_err_velocity_cell_sqr_local += difference_velocity_cell_sqr*fe_values_rt.JxW(q);
                   }

               L2_err_velocity_cell_sqr_global += L2_err_velocity_cell_sqr_local;

              // For flux, we need areas of the cell and lengths of faces. Since fluxes are calculated on faces,
              // we have the loop over all four faces of each cell. To calculate face velocity, we use
              // the coefficient @p beta we have calculated previously. We evaluate the squares of difference between
              // numerical velocity dot product with unit normal vector and exact face velocity dot product with unit normal vector.
              // Finally, we calculate $L_2$ error of flux on the cell and add it to the global error.
               L2_err_flux_face_sqr_local = 0;
               difference_flux_face_sqr = 0;
               err_flux_face = 0;
               const double cell_area = cell->measure();
               for (unsigned int face_n=0; face_n<GeometryInfo<dim>::faces_per_cell; ++face_n)
                   {
                	   const double face_length = cell->face(face_n)->measure();
                       fe_face_values.reinit (cell,face_n);
                       fe_face_values_rt.reinit (cell_rt,face_n);
                       L2_err_flux_face_sqr_local = 0;
                       err_flux_each_face = 0;
                       for (unsigned int q=0; q<n_face_q_points_rt; ++q)
                           {
                             difference_velocity_face_sqr = 0;
                             velocity_face = 0;
                             const Tensor<1,dim> normal = fe_face_values.normal_vector(q);
                             for(unsigned int k = 0; k<dofs_per_cell_rt;++k)
                                {
                                   const Tensor<1,dim> phi_k_u = fe_face_values_rt[velocities].value (k, q);
                                   velocity_face += beta(k)*phi_k_u;
                                 }
                             exact_velocity_face = exact_velocity.value (fe_face_values_rt.quadrature_point (q));
                             difference_velocity_face_sqr = (velocity_face*normal - exact_velocity_face*normal)*
                        		                            (velocity_face*normal - exact_velocity_face*normal);
                              L2_err_flux_face_sqr_local += difference_velocity_face_sqr*fe_face_values_rt.JxW(q);
                             }
                      err_flux_each_face = L2_err_flux_face_sqr_local/(face_length)*(cell_area);
                      err_flux_face += err_flux_each_face;
                     }
                  L2_err_flux_sqr +=  err_flux_face;
      }

  // After adding up errors over all cells, we take square root and get the $L_2$ errors of velocity and flux.
  const double L2_err_velocity_cell = sqrt(L2_err_velocity_cell_sqr_global);
  std::cout<< "L2_error_vel "<<L2_err_velocity_cell <<std::endl;
  const double L2_err_flux_face = sqrt(L2_err_flux_sqr);
  std::cout<< "L2_error_flux "<<L2_err_flux_face <<std::endl;

}


 // @sect4{WGDarcyEquation::output_results}

 // We have 2 sets of results to output:  the interior solution
 // and the skeleton solution. We use <code>DataOut</code> to graph interior results.
 // The graphical output for the skeleton results is done by using
 // the <code>DataOutFaces</code> class.
 template <int dim>
 void WGDarcyEquation<dim>::output_results () const
 {
  DataOut<dim> data_out;
  data_out.attach_dof_handler (dof_handler);
  data_out.add_data_vector (solution, "Pressure_Interior");
  data_out.build_patches (fe.degree);
  std::ofstream output ("Pressure_Interior.vtk");
  data_out.write_vtk (output);

  DataOutFaces<dim> data_out_face(false);
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
  face_component_type(2, DataComponentInterpretation::component_is_scalar);
  data_out_face.add_data_vector (dof_handler,
                                 solution,
                                 "Pressure_Edge",
                                 face_component_type);
  data_out_face.build_patches (fe.degree);
  std::ofstream face_output ("Pressure_Edge.vtk");
  data_out_face.write_vtk (face_output);
 }


 // @sect4{WGDarcyEquation::run}

 // This is the final function of the main class. It calls the other functions of our class.
 template <int dim>
 void WGDarcyEquation<dim>::run ()
 {
   std::cout << "Solving problem in " << dim << " space dimensions." << std::endl;
   make_grid();
   setup_system ();
   assemble_system ();
   solve ();
   process_solution ();
   postprocess();
   output_results ();
 }

 // @sect3{The <code>main</code> function}

 // This is the main function. We can change the dimension here to run in 3d.
 int main ()
 {
   deallog.depth_console (2);
   WGDarcyEquation<2> WGDarcyEquationTest;
   WGDarcyEquationTest.run ();

   return 0;
 }

