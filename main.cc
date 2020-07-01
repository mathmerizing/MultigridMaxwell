/*  
 *  Written by Julian Roth
 *  Supervised by Sebastian Kinnewig, Thomas Wick
 *  July 2020
 */

// === Deal.II Includes ==

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/parameter_handler.h>

#include <deal.II/lac/generic_linear_algebra.h>

// Deal Matrix & Constraint matrix
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/distributed/grid_refinement.h>

// Solver
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/relaxation_block.h>

// Multigrid
#include <deal.II/multigrid/mg_constrained_dofs.h>
#include <deal.II/multigrid/multigrid.h>
#include <deal.II/multigrid/mg_transfer.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_base.h>

// Grid generator
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_in.h>

// dof handler
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

// FE - Elements:
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_nedelec.h>

// Matrix and Vector tools
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_postprocessor.h>
#include <deal.II/numerics/data_out.h>


// === C++ Includes ===
#include <iostream>
#include <fstream>

namespace EddyCurrent
{
using namespace dealii;

// === The Parameter Reader class (derived from ParameterHandler) ===

class ParameterReader : public Subscriptor
{
public:
  ParameterReader(ParameterHandler &);
  void read_parameters(const std::string &);

private:
  void declare_parameters();
  ParameterHandler &prm;
};

ParameterReader::ParameterReader(ParameterHandler &paramhandler) : prm(paramhandler)
{
}

//declare the parameters:
void ParameterReader::declare_parameters()
{
  //The Mesh
  prm.enter_subsection("Mesh & geometry parameters");
  {
    prm.declare_entry("Dimension",
                      "2",
                      Patterns::Integer(0),
                      "The dimension"
                      "of the used Nedelec elements");

    prm.declare_entry("Number of refinements",
                      "5",
                      Patterns::Integer(0),
                      "Number of globar mesh refinement steps"
                      "applied to inital coarse grid");

    prm.declare_entry("Polynomial degree",
                      "3",
                      Patterns::Integer(0),
                      "Polynomial degree"
                      "of the used Nedelec elements");
    
    prm.declare_entry("Show parameters",
                      "false",
                      Patterns::Bool(),
                      "");
  }
  prm.leave_subsection();

  // Preconditioner
  prm.enter_subsection("Solver & Preconditioner");
  {
    prm.declare_entry("Right Preconditioning",
                      "true",
                      Patterns::Bool(),
                      "Flag whether right preconditioning"
                      "is used for GMRES");

    prm.declare_entry("Smoother Relaxation", "1.0", Patterns::Double(0), "relaxation parameter");

    // Domain decomposition parameters
    prm.declare_entry("Vertex Patches", "true", Patterns::Bool(), "use vertex or cell patches");
    prm.declare_entry("interior_dofs_only", "true", Patterns::Bool(), "");
    prm.declare_entry("boundary_patches", "false", Patterns::Bool(), "");
    prm.declare_entry("level_boundary_patches", "false", Patterns::Bool(), "");
    prm.declare_entry("single_cell_patches", "false", Patterns::Bool(), "");
    prm.declare_entry("invert_vertex_mapping", "true", Patterns::Bool(), "");
  }

  prm.leave_subsection();

  // Multigrid
  prm.enter_subsection("Multigrid");
  {
    prm.declare_entry("Smoothing Steps",
                      "5",
                      Patterns::Integer(0),
                      "Number of smoothing steps in GMG");
  }
  prm.leave_subsection();

  // filename and format
  prm.enter_subsection("Output parameters");
  {
    prm.declare_entry("Output file",
                      "solution",
                      Patterns::Anything(),
                      "Name of the output file (without extension)");
    //declare parameters handels all the parameters needed for a certain return type
    DataOutInterface<1>::declare_parameters(prm);
  }
  prm.leave_subsection();
}

// read parameters:
void ParameterReader::read_parameters(const std::string &parameter_file)
{
  declare_parameters();
  prm.parse_input(parameter_file);
}

// === post processing ===
template <int dim>
class ComputeIntensity : public DataPostprocessorScalar<dim>
{
public:
  ComputeIntensity();

  virtual void evaluate_vector_field(
      const DataPostprocessorInputs::Vector<dim> &inputs,
      std::vector<Vector<double>> &computed_quantities) const override;
};

template <int dim>
ComputeIntensity<dim>::ComputeIntensity() : DataPostprocessorScalar<dim>("Intensity", update_values)
{
}

template <int dim>
void ComputeIntensity<dim>::evaluate_vector_field(
    const DataPostprocessorInputs::Vector<dim> &inputs,
    std::vector<Vector<double>> &computed_quantities) const
{
  Assert(
      computed_quantities.size() == inputs.solution_values.size(),
      ExcDimensionMismatch(computed_quantities.size(),
                           inputs.solution_values.size()));
  for (unsigned int i = 0; i < computed_quantities.size(); i++)
  {
    Assert(
        computed_quantities[i].size() == 1,
        ExcDimensionMismatch(computed_quantities[i].size(),
                             1));
    Assert(
        inputs.solution_values[i].size() == 2 * dim,
        ExcDimensionMismatch(inputs.solution_values[i].size(),
                             2 * dim));

    double return_value = 0;
    for (int component = 0; component < 2 * dim; component++)
    {
      return_value += std::pow(inputs.solution_values[i](component), 2);
    }

    computed_quantities[i](0) = std::sqrt(return_value);
  } // rof
}

// === used templates ===
template class ComputeIntensity<2>;
template class ComputeIntensity<3>;

// === The Time Harmonic Maxwell Problem Class ===
template <int dim>
class MaxwellProblem
{
public:
  // constructor
  MaxwellProblem(ParameterHandler &, const unsigned int poly_degree);

  // execute
  void run();

  // matrix and vector type:
  using MatrixType = SparseMatrix<double>;
  using VectorType = Vector<double>;

private:
  void make_grid(int n_refinements);
  void setup_system();
  void assemble_system(); // assemble system matrix
  void assemble_multigrid(); // assemble matrices that correspond to discrete operators on intermediate levels
  void solve_multigrid(); // one mutligrid cycle
  void save_constraints(); // save boundary constraints on current triangulation in level_constraints
  void refine_global_and_save_constraints();
  void output_results() const;

  // Parameterhandler
  ParameterHandler &prm;

  Triangulation<dim> triangulation;

  DoFHandler<dim> dof_handler;
  FESystem<dim> fe;

  SparsityPattern sparsity_pattern;
  MatrixType system_matrix;
  VectorType solution, system_rhs;

  // Constraints
  AffineConstraints<double> constraints;
  // Constraints on the MG levels
  std::vector<AffineConstraints<>> level_constraints;
  
  // Multigrid
  MGLevelObject<SparsityPattern> mg_sparsity_patterns;
  MGLevelObject<SparsityPattern> mg_interface_sparsity_patterns;
  MGLevelObject<MatrixType> mg_matrices;
  MGLevelObject<MatrixType> mg_interface_matrices;
  MGConstrainedDoFs mg_constrained_dofs;
};

template <int dim>
MaxwellProblem<dim>::MaxwellProblem(
    ParameterHandler &param,
    const unsigned int poly_degree) : prm(param),
                                      triangulation(Triangulation<dim>::limit_level_difference_at_vertices), // guarantee that the mesh is v-one-irregular, i.e. levels of all active cells sharing a vertex or a face differ by a maximum of one
                                      dof_handler(triangulation),
                                      fe(FE_Nedelec<dim>(poly_degree), 2) // real and imaginary part
{
}

template <int dim>
void MaxwellProblem<dim>::make_grid(int n_refinements)
{
  std::cout << "Generating grid ..." << std::endl;

  // domain := L-shape
  GridGenerator::hyper_L(triangulation);
  save_constraints();
  for (int i = 0; i < n_refinements; ++i)
  {
    refine_global_and_save_constraints();
  }

  std::cout << "\tNumber of active cells:  "
        << triangulation.n_global_active_cells()
        << " (local: " << triangulation.n_active_cells() << ")"
        << std::endl;
  std::cout << "done!" << std::endl << std::endl;
}

template <int dim>
void MaxwellProblem<dim>::setup_system()
{
  std::cout << "Setting up system ..." << std::endl;
  dof_handler.clear();

  // distributre dofs
  dof_handler.distribute_dofs(fe);
  dof_handler.distribute_mg_dofs(); // distribute level degrees of freedom on each level for multigrid
  
  // return how many level dofs we are using
  std::cout << "\tNumber of degrees of freedom: " << dof_handler.n_dofs()
        << " (by level: ";
  for (unsigned int level = 0; level < triangulation.n_levels(); ++level)
    std::cout << dof_handler.n_dofs(level)
          << (level == triangulation.n_levels() - 1 ? ")" : ", ");
  std::cout << std::endl;

  // return how many level constraints there
  std::cout << "\tNumber of constraints: " << level_constraints.back().n_constraints()
        << " (by level: ";
  for (unsigned int level = 0; level < triangulation.n_levels(); ++level)
    std::cout << level_constraints[level].n_constraints()
          << (level == triangulation.n_levels() - 1 ? ")" : ", ");
  std::cout << std::endl;

  { // handle the boundary conditions and hanging nodes:
    constraints.clear();

    DoFTools::make_hanging_node_constraints(
        dof_handler, constraints);

    // FE_Nedelec boundary condition.
    VectorTools::project_boundary_values_curl_conforming_l2(
        dof_handler,
        0 /* vector component*/,
        Functions::ZeroFunction<dim>(2 * dim),
        0 /* boundary id*/,
        constraints);

    // FE_Nedelec boundary condition.
    VectorTools::project_boundary_values_curl_conforming_l2(
        dof_handler,
        2 /* vector component*/,
        Functions::ZeroFunction<dim>(2 * dim),
        0 /* boundary id*/,
        constraints);

    constraints.close();
  }

  { // assemble the sparsity pattern,

    system_matrix.clear();

    DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints);
    sparsity_pattern.copy_from(dsp);

    system_matrix.reinit(sparsity_pattern);
  }

  //intialize the solution and the right hand side
  solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());

  // initialize multigrid constraints
  mg_constrained_dofs.clear();
  mg_constrained_dofs.initialize(dof_handler);

  // resize multigrid data structures
  const unsigned int n_levels = triangulation.n_levels(); // or: .n_global_levels();

  mg_interface_matrices.resize(0, n_levels - 1);
  mg_interface_matrices.clear_elements(); // from step 50

  mg_matrices.resize(0, n_levels - 1);
  mg_matrices.clear_elements(); // from step 50

  mg_sparsity_patterns.resize(0, n_levels - 1);
  mg_interface_sparsity_patterns.resize(0, n_levels - 1);

  // generate sparsity patterns for matrices and interface matrices on each level
  for (unsigned int level = 0; level < n_levels; ++level)
  {
    // add level constraints to MGConstrainedDoFs object
    mg_constrained_dofs.add_user_constraints(level, level_constraints[level]);

    // generate dsp for mg_matrices
    {
      DynamicSparsityPattern dsp(dof_handler.n_dofs(level),dof_handler.n_dofs(level));
      MGTools::make_sparsity_pattern(dof_handler, dsp, level);
      mg_sparsity_patterns[level].copy_from(dsp);
      mg_matrices[level].reinit(mg_sparsity_patterns[level]);
    }
    // generate dsp for mg_interface_matrices
    {
      DynamicSparsityPattern dsp(dof_handler.n_dofs(level),dof_handler.n_dofs(level));
      MGTools::make_interface_sparsity_pattern(dof_handler,
                                               mg_constrained_dofs,
                                               dsp,
                                               level);
      mg_interface_sparsity_patterns[level].copy_from(dsp);
      mg_interface_matrices[level].reinit(mg_interface_sparsity_patterns[level]);
    }
  }

  std::cout << "done!"
        << std::endl;
  std::cout << "" << std::endl;
}

template <int dim>
void MaxwellProblem<dim>::assemble_system()
{
  std::cout << "Assemble system matix ..." << std::endl;

  const unsigned int curl_dim = (dim == 2) ? 1 : 3;

  // choose the quadrature formulas
  QGauss<dim> quadrature_formula(fe.degree + 2);
  QGauss<dim - 1> face_quadrature_formula(fe.degree + 1);

  // get the number of quadrature points and dofs
  const unsigned int n_q_points = quadrature_formula.size();
  const unsigned int dofs_per_cell = fe.dofs_per_cell;

  // set update flags
  FEValues<dim> fe_values(
      fe, quadrature_formula,
      update_values | update_gradients | update_quadrature_points | update_JxW_values);
  FEFaceValues<dim> fe_face_values(
      fe, face_quadrature_formula,
      update_values | update_quadrature_points | update_normal_vectors | update_JxW_values);

  // Extractors to real and imaginary parts
  const FEValuesExtractors::Vector E_re(0);
  const FEValuesExtractors::Vector E_im(dim);
  std::vector<FEValuesExtractors::Vector> vec(2);
  vec[0] = E_re;
  vec[1] = E_im;

  // create the local left hand side and right hand side
  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double> cell_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  // phi_i:
  std::vector<Tensor<1, dim>> phi_i(2 * n_q_points);            // content: (phi_re_i[q_point = 0], phi_im_i[q_point = 0], ... )
  std::vector<Tensor<1, curl_dim>> curl_phi_i(2 * n_q_points);  // content: (curl_phi_re_i[q_point = 0], curl_phi_im_i[q_point = 0], ... )

  // phi_j:
  Tensor<1, dim> phi_j;
  Tensor<1, curl_dim> curl_phi_j;

  Tensor<1, dim> rhs_func;
  rhs_func[0] = 1.0;
  rhs_func[1] = 1.0;
  if (dim == 3)
    rhs_func[2] = 1.0;

  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    if (cell->is_locally_owned())
    {
      cell_matrix = 0;
      cell_rhs = 0;

      fe_values.reinit(cell);

      // Integrate over the domain
      for (unsigned int i = 0; i < dofs_per_cell; i++)
      {
        const unsigned int block_index_i = fe.system_to_block_index(i).first;

        // we only want to compute this once
        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          phi_i[(2 * q_point) + 0]      = fe_values[E_re].value(i, q_point);
          phi_i[(2 * q_point) + 1]      = fe_values[E_im].value(i, q_point);
          curl_phi_i[(2 * q_point) + 0] = fe_values[E_re].curl(i, q_point);
          curl_phi_i[(2 * q_point) + 1] = fe_values[E_im].curl(i, q_point);
        }

        // handles the real and the imaginary part:
        for (unsigned int j = 0; j < dofs_per_cell; j++)
        {
          const unsigned int block_index_j = fe.system_to_block_index(j).first;

          double mass_part = 0;
          double curl_part = 0;

          if (block_index_i == block_index_j)
          {
            for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
            {
              phi_j      = fe_values[vec[block_index_i]].value(j, q_point);
              curl_phi_j = fe_values[vec[block_index_i]].curl(j, q_point);

              curl_part +=
                  curl_phi_i[(2 * q_point) + block_index_i] * curl_phi_j * fe_values.JxW(q_point);

              mass_part +=
                  phi_i[(2 * q_point) + block_index_i] * phi_j * fe_values.JxW(q_point);

            } // rof: q_point

            cell_matrix(i, j) = (curl_part - mass_part);
            cell_matrix(j, i) = cell_matrix(i, j); // j and i belong to the same block and thus have the same sign

          } // fi: base_index_i == base_index_j
        }   // rof:  dof j

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          // RHS = 1.0
          cell_rhs(i) += phi_i[(2 * q_point) + block_index_i] * rhs_func * fe_values.JxW(q_point);
        }

      }     // rof: dof i

      cell->get_dof_indices(local_dof_indices);

      constraints.distribute_local_to_global(
          cell_matrix,
          cell_rhs,
          local_dof_indices,
          system_matrix,
          system_rhs);

    } // fi: cell locally owned
  }   // rof: active cell

  std::cout << "done!" << std::endl;
  std::cout << "" << std::endl;
}

template <int dim>
void MaxwellProblem<dim>::assemble_multigrid()
{
  std::cout << "Assemble multigrid level matrices ..." << std::endl;

  const unsigned int n_levels = triangulation.n_levels();

  const unsigned int curl_dim = (dim == 2) ? 1 : 3;

  // choose the quadrature formulas
  QGauss<dim> quadrature_formula(fe.degree + 2);
  QGauss<dim - 1> face_quadrature_formula(fe.degree + 1);

  // get the number of quadrature points and dofs
  const unsigned int n_q_points = quadrature_formula.size();
  const unsigned int dofs_per_cell = fe.dofs_per_cell;

  // set update flags
  FEValues<dim> fe_values(
      fe, quadrature_formula,
      update_values | update_gradients | update_quadrature_points | update_JxW_values);
  FEFaceValues<dim> fe_face_values(
      fe, face_quadrature_formula,
      update_values | update_quadrature_points | update_normal_vectors | update_JxW_values);

  // Extractors to real and imaginary parts
  const FEValuesExtractors::Vector E_re(0);
  const FEValuesExtractors::Vector E_im(dim);
  std::vector<FEValuesExtractors::Vector> vec(2);
  vec[0] = E_re;
  vec[1] = E_im;

  // create the local left hand side and right hand side
  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double> cell_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  // phi_i:
  std::vector<Tensor<1, dim>> phi_i(2 * n_q_points);            // content: (phi_re_i[q_point = 0], phi_im_i[q_point = 0], ... )
  std::vector<Tensor<1, curl_dim>> curl_phi_i(2 * n_q_points);  // content: (curl_phi_re_i[q_point = 0], curl_phi_im_i[q_point = 0], ... )

  // phi_j:
  Tensor<1, dim> phi_j;
  Tensor<1, curl_dim> curl_phi_j;

  // create AffineConstraints for each level which has boundary and interface DoFs as constrained entries
  std::vector<AffineConstraints<>> boundary_constraints(n_levels);
  for (unsigned int level = 0; level < n_levels; ++level)
  {
    IndexSet dofset;
    DoFTools::extract_locally_relevant_level_dofs(dof_handler, level, dofset);
    boundary_constraints[level].reinit(dofset);
    boundary_constraints[level].add_lines(
        mg_constrained_dofs.get_refinement_edge_indices(level));

   boundary_constraints[level].merge(
      mg_constrained_dofs.get_user_constraint_matrix(level),
      AffineConstraints<double>::MergeConflictBehavior::right_object_wins);
    boundary_constraints[level].close();
  } // rof: levels
  
  // MAIN LOGIC FOR MG MATRIX ASSEMBLY
  for (const auto &cell : dof_handler.cell_iterators())
  {
    cell_matrix = 0;
    fe_values.reinit(cell);

    // Integrate over the domain
    for (unsigned int i = 0; i < dofs_per_cell; i++)
    {
      const unsigned int block_index_i = fe.system_to_block_index(i).first;

      // we only want to compute this once
      for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        phi_i[(2 * q_point) + 0]      = fe_values[E_re].value(i, q_point);
        phi_i[(2 * q_point) + 1]      = fe_values[E_im].value(i, q_point);
        curl_phi_i[(2 * q_point) + 0] = fe_values[E_re].curl(i, q_point);
        curl_phi_i[(2 * q_point) + 1] = fe_values[E_im].curl(i, q_point);
      }

      // handles the real and the imaginary part:
      for (unsigned int j = 0; j < dofs_per_cell; j++)
      {
        const unsigned int block_index_j = fe.system_to_block_index(j).first;

        double mass_part = 0;
        double curl_part = 0;

        if (block_index_i == block_index_j)
        {
          for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            phi_j      = fe_values[vec[block_index_i]].value(j, q_point);
            curl_phi_j = fe_values[vec[block_index_i]].curl(j, q_point);

            curl_part +=
                curl_phi_i[(2 * q_point) + block_index_i] * curl_phi_j * fe_values.JxW(q_point);

            mass_part +=
                phi_i[(2 * q_point) + block_index_i] * phi_j * fe_values.JxW(q_point);

          } // rof: q_point

          cell_matrix(i, j) = (curl_part - mass_part);
          cell_matrix(j, i) = cell_matrix(i, j); // i and j belong to the same block and thus have the same sign

        } // fi: base_index_i == base_index_j
      }   // rof:  dof j
    }     // rof: dof i

    // copy local contributions into the level objects
    cell->get_mg_dof_indices(local_dof_indices);
    boundary_constraints[cell->level()].distribute_local_to_global(
        cell_matrix, local_dof_indices, mg_matrices[cell->level()]);

    // we need the part of the interface between cells at the current level and cells one level coarser
    for (unsigned int i = 0; i < dofs_per_cell; ++i)
    {
      for (unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        if (mg_constrained_dofs.is_interface_matrix_entry(
                cell->level(), local_dof_indices[i], local_dof_indices[j]))
        {
          // system matrix is symmetric !
          mg_interface_matrices[cell->level()].add(local_dof_indices[i],
                                                   local_dof_indices[j],
                                                   cell_matrix(i, j));
        }
      }
    }
  }
  std::cout << "done!" << std::endl;
  std::cout << "" << std::endl;
}

template <int dim>
void MaxwellProblem<dim>::solve_multigrid()
{
  Timer timer;
  std::cout << "Solving linear system using geometric MG and GMRES..." << std::endl;

  // read paremeters for Multigrid
  prm.enter_subsection("Multigrid");
  int smoothing_steps = prm.get_integer("Smoothing Steps");
  prm.leave_subsection();

  // create the object that deals with the transfer between the different levels
  MGTransferPrebuilt<VectorType> mg_transfer(mg_constrained_dofs);
  mg_transfer.build(dof_handler);

  // 1. Coarse Grid Solver
  FullMatrix<double> coarse_matrix;
  coarse_matrix.copy_from(mg_matrices[0]);
  MGCoarseGridHouseholder<> coarse_grid_solver;
  coarse_grid_solver.initialize(coarse_matrix);

  // 2. Smoother

  // read solver parameters
  prm.enter_subsection("Solver & Preconditioner");
  
  bool right_precondition    = prm.get_bool("Right Preconditioning");
  double smoother_relaxation = prm.get_double("Smoother Relaxation");

  // get domain decomposition parameters
  bool vertex_patches         = prm.get_bool("Vertex Patches");
  bool interior_dofs_only     = prm.get_bool("interior_dofs_only");
  bool boundary_patches       = prm.get_bool("boundary_patches");
  bool level_boundary_patches = prm.get_bool("level_boundary_patches");
  bool single_cell_patches    = prm.get_bool("single_cell_patches");
  bool invert_vertex_mapping  = prm.get_bool("invert_vertex_mapping");

  prm.leave_subsection();

  using RELAXATION = RelaxationBlockSOR<MatrixType>;    // MULTIPLICATIVE SCHWARZ
  
  MGLevelObject<typename RELAXATION::AdditionalData> smoother_data;
  mg::SmootherRelaxation<RELAXATION, VectorType> mg_smoother;
  smoother_data.resize(0, triangulation.n_levels() - 1);
  
  const unsigned int n_comp = this->dof_handler.get_fe().n_components();
  BlockMask exclude_boundary_dofs(n_comp, true);
  if (interior_dofs_only == false)
  {
    std::vector<bool> exclude_block(n_comp, false);
    exclude_boundary_dofs = dealii::BlockMask(exclude_block);
  }

  for (unsigned int l = smoother_data.min_level() + 1; l <= smoother_data.max_level(); ++l)
  {
    // CREATE PATCHES
    if (vertex_patches)
    {
      std::vector<unsigned int> vertex_mapping = DoFTools::make_vertex_patches(smoother_data[l].block_list,
                                                     this->dof_handler,
                                                     l,
                                                     exclude_boundary_dofs,
                                                     boundary_patches,
                                                     level_boundary_patches,
                                                     single_cell_patches,
                                                     invert_vertex_mapping);
    } else
    {
      smoother_data[l].block_list.reinit(
        triangulation.n_cells(l), dof_handler.n_dofs(l), fe.dofs_per_cell);
      DoFTools::make_cell_patches(smoother_data[l].block_list, dof_handler, l);
    }
    
    smoother_data[l].block_list.compress();
    smoother_data[l].relaxation = smoother_relaxation;
    smoother_data[l].inversion = PreconditionBlockBase<double>::svd;
    smoother_data[l].threshold = 1.e-12;
  }
  mg_smoother.initialize(mg_matrices, smoother_data);
  mg_smoother.set_steps(smoothing_steps);

  // 3. Multigrid

  // wrap level and interface matrixes in an object having required multiplication functions
  mg::Matrix<VectorType> mg_matrix(mg_matrices);
  mg::Matrix<VectorType> mg_interface_up(mg_interface_matrices);
  mg::Matrix<VectorType> mg_interface_down(mg_interface_matrices);

  // set up V-cycle operator
  Multigrid<VectorType> mg(
                          mg_matrix,                            /* matrix */
                          coarse_grid_solver,                   /* coarse */
                          mg_transfer,                          /* transfer */
                          mg_smoother,                          /* pre_smooth */
                          mg_smoother,                          /* post_smooth */
                          0,                                    /* minlevel */
                          numbers::invalid_unsigned_int,   /* maxlevel */
                          Multigrid<VectorType>::Cycle::v_cycle /* cycle */         
                        );

  mg.set_edge_matrices(mg_interface_down, mg_interface_up);

  //Solve the linear system using GMRES:
  PreconditionMG<dim, VectorType, MGTransferPrebuilt<VectorType>> preconditioner_mg(dof_handler, mg, mg_transfer);

  // 4. GMG preconditioned GMRES
  int MAX_ITER = 5500;
  GrowingVectorMemory<VectorType> vector_memory;
  SolverGMRES<VectorType>::AdditionalData gmres_data;
  gmres_data.max_n_tmp_vectors = MAX_ITER; // 400;
  gmres_data.force_re_orthogonalization = true;
  gmres_data.right_preconditioning = right_precondition;

  // solve linear system in the usual way
  SolverControl solver_control(MAX_ITER, 1e-6 * system_rhs.l2_norm(), false);
  SolverGMRES<VectorType> gmres(solver_control, vector_memory, gmres_data);

  gmres.solve(system_matrix, solution, system_rhs, preconditioner_mg);

  std::cout << "steps: " << solver_control.last_step() << std::endl;

  constraints.distribute(solution);


  timer.stop();
  std::cout << "done! ( in: " << timer.cpu_time() << "s )" << std::endl;
  std::cout << "" << std::endl;
}

template <int dim>
void MaxwellProblem<dim>::save_constraints()
{
  dof_handler.clear();
  dof_handler.distribute_dofs(fe);

  IndexSet dofset;
  DoFTools::extract_locally_relevant_dofs(dof_handler, dofset);

  AffineConstraints<double> new_constraints;
  new_constraints.clear();
  new_constraints.reinit(dofset);

  // FE_Nedelec boundary condition.
  VectorTools::project_boundary_values_curl_conforming_l2(
      dof_handler,
      0 /* vector component*/,
      Functions::ZeroFunction<dim>(2 * dim),
      0 /* boundary id*/,
      new_constraints);

  // FE_Nedelec boundary condition.
  VectorTools::project_boundary_values_curl_conforming_l2(
      dof_handler,
      2 /* vector component*/,
      Functions::ZeroFunction<dim>(2 * dim),
      0 /* boundary id*/,
      new_constraints);

  new_constraints.close();

  level_constraints.push_back(new_constraints);
}

template <int dim>
void MaxwellProblem<dim>::refine_global_and_save_constraints()
{
  triangulation.refine_global(1);
  save_constraints();
}

template <int dim>
void MaxwellProblem<dim>::output_results() const
{
  std::cout << "Generating output...\t";

  //Define objects of our ComputeIntensity class
  ComputeIntensity<dim> intensities;
  DataOut<dim> data_out;

  // and a DataOut object:
  data_out.attach_dof_handler(dof_handler);

  prm.enter_subsection("Output parameters");
  const std::string output_filename = prm.get("Output file");
  data_out.parse_parameters(prm);
  prm.leave_subsection();

  const std::string filename = output_filename + data_out.default_suffix();

  std::vector<std::string> solution_names;
  solution_names.emplace_back("Re_E1");
  solution_names.emplace_back("Re_E2");
  if (dim == 3)
  {
    solution_names.emplace_back("Re_E3");
  }
  solution_names.emplace_back("Im_E1");
  solution_names.emplace_back("Im_E2");
  if (dim == 3)
  {
    solution_names.emplace_back("Im_E3");
  }

  data_out.add_data_vector(solution, solution_names);
  data_out.add_data_vector(solution, intensities);

  data_out.build_patches();
  std::ofstream vtu_out(filename.c_str());
  data_out.write_vtu(vtu_out);

  std::cout << "done" << std::endl;
}

template <int dim>
void MaxwellProblem<dim>::run()
{
  // get the parameters:
  prm.enter_subsection("Mesh & geometry parameters");
  const unsigned int n_refinements = prm.get_integer("Number of refinements");
  bool show_parameters = prm.get_bool("Show parameters");
  prm.leave_subsection();

  make_grid(2); // L-shape

  for (unsigned int i = 2; i <= n_refinements; i++)
  {
    TimerOutput timer(
        std::cout,
        TimerOutput::summary,
        TimerOutput::cpu_and_wall_times_grouped);

    std::cout << std::endl;
    std::cout << "+-------------------------------------------------------------------------------------------------+" << std::endl;
    std::cout << "|                                        Eddy Current Solver                                      |" << std::endl;
    std::cout << "+-------------------------------------------------------------------------------------------------+" << std::endl;

    if (show_parameters)
    {
      std::cout << "" << std::endl;
      std::cout << " --------------------- PARAMETERS --------------------- " << std::endl;
      prm.print_parameters(std::cout, ParameterHandler::ShortText);
      std::cout << " ------------------------------------------------------ " << std::endl;
      std::cout << "" << std::endl;
    }

    std::cout << "-- INFO: Global refinements: " << i << std::endl;

    timer.enter_subsection("Setup system");
    setup_system();
    timer.leave_subsection();

    timer.enter_subsection("Assemble system");
    assemble_system();
    timer.leave_subsection();

    timer.enter_subsection("Assemble multigrid");
    assemble_multigrid();
    timer.leave_subsection();

    timer.enter_subsection("Solve GMG");
    solve_multigrid();
    timer.leave_subsection();

    timer.enter_subsection("Write results");
    output_results();
    timer.leave_subsection();

    timer.enter_subsection("Refine grid");
    if (i < n_refinements)
    {
      refine_global_and_save_constraints();
    }
    timer.leave_subsection();

    std::cout << "+-------------------------------------------------------------------------------------------------+" << std::endl;
    std::cout << "|                                             DONE                                                |" << std::endl;
    std::cout << "+-------------------------------------------------------------------------------------------------+" << std::endl;
  }
}

} // namespace EddyCurrent

int main()
{
  try
  {
    using namespace dealii;
    using namespace EddyCurrent;
    //deallog.depth_console(5);

    //read the Parameters
    ParameterHandler prm;
    ParameterReader param(prm);
    param.read_parameters("options.prm");

    prm.enter_subsection("Mesh & geometry parameters");
    const unsigned int dim = prm.get_integer("Dimension");
    const unsigned int poly_degree = prm.get_integer("Polynomial degree");
    prm.leave_subsection();

    if (dim == 2)
    {
      MaxwellProblem<2> edc(prm, poly_degree); // edc = Eddy Current
      edc.run();
    }
    else if (dim == 3)
    {
      MaxwellProblem<3> edc(prm, poly_degree); // edc = Eddy Current
      edc.run();
    }
  }
  catch (std::exception &exc)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }
  catch (...)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }
  return 0;
}
