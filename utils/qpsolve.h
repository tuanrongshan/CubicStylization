#ifndef QPSOLVE_H
#define QPSOLVE_H

#include <Eigen/Core>
#include <vector>
#include "OsqpEigen/OsqpEigen.h"

void qpsolve(
    const Eigen::VectorXd & x,
	const double & k,
    const double & rho,
    Eigen::MatrixXd & B,
    Eigen::VectorXd & z);

#endif