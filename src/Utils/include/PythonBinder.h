//
// Created by Adrien Blanchet on 05/12/2023.
//

#ifndef GUNDAM_PYTHONBINDER_H
#define GUNDAM_PYTHONBINDER_H

#include <Eigen/Dense>
#include <cmath>

using Eigen::Matrix, Eigen::Dynamic;
typedef Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> myMatrix;

class MyClass {

  int N;
  double a;
  double b;

public:

  Eigen::VectorXd v_data;
  Eigen::VectorXd v_gamma;

  MyClass(){}
  MyClass( double a_in, double b_in, int N_in)
  {
    N = N_in;
    a = a_in;
    b = b_in;
  }

  void run()
  {
    v_data = Eigen::VectorXd::LinSpaced(N, a, b);

    auto gammafunc = [](double it) { return std::tgamma(it); };
    v_gamma = v_data.unaryExpr(gammafunc);
  }
};

#endif //GUNDAM_PYTHONBINDER_H
