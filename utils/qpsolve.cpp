#include "qpsolve.h"

void qpsolve(
    const Eigen::VectorXd & x,
	const double & k,
    const double & rho,
    Eigen::MatrixXd & B,
    Eigen::VectorXd & z)
{
    using namespace Eigen;
    using namespace OsqpEigen;

    const int m = B.rows();

    // min_x 0.5 * x'Px + q'x
    // s.t.  l <= Ax <= u
    // P
    MatrixXd objective_matrix(3+m, 3+m);
    objective_matrix <<
        rho * MatrixXd::Identity(3,3), MatrixXd::Zero(3,m),
        MatrixXd::Zero(m,3), MatrixXd::Zero(m,m);

    // q
    VectorXd objective_vector(3+m);
    objective_vector <<
        -rho * x, k * VectorXd::Ones(m);

    // A
    MatrixXd constraint_matrix(2*m, 3+m);
    constraint_matrix <<
        B, -MatrixXd::Identity(m,m),
        -B, -MatrixXd::Identity(m,m);

    // u & l
    VectorXd upper_bounds = VectorXd::Zero(2*m);
    VectorXd lower_bounds = VectorXd::Constant(2*m, -std::numeric_limits<double>::infinity());
 
    // main
    Solver solver;
    // settings
    solver.settings()->setVerbosity(false);
    solver.settings()->setWarmStart(true);

    // set the initial data of the QP solver
    solver.data()->setNumberOfVariables(3+m);
    solver.data()->setNumberOfConstraints(2*m);
    solver.data()->setHessianMatrix((SparseMatrix<double>)objective_matrix.sparseView());
    solver.data()->setGradient(objective_vector);
    solver.data()->setLinearConstraintsMatrix((SparseMatrix<double>)constraint_matrix.sparseView());
    solver.data()->setLowerBound(lower_bounds);
    solver.data()->setUpperBound(upper_bounds);

    // instantiate the solver
    solver.initSolver();

    Eigen::VectorXd QPSolution;
    // solve the QP problem
    solver.solveProblem();
    QPSolution = solver.getSolution();
    z = QPSolution.head(3);
}