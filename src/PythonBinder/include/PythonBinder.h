//
// Created by Adrien Blanchet on 05/12/2023.
//

#ifndef GUNDAM_PYTHONBINDER_H
#define GUNDAM_PYTHONBINDER_H


#include <pybind11/stl.h> // support for vectors

#include <cmath>
#include <vector>



class PythonBinder{

  int N{};
  double a{};
  double b{};

public:

  std::vector<double> v_data{};
  std::vector<double> v_gamma{};

  PythonBinder() = default;
  PythonBinder( double a_in, double b_in, int N_in) {
    N = N_in;
    a = a_in;
    b = b_in;
  }

  void run() {
    v_data = {a, b};
  }

};


#endif //GUNDAM_PYTHONBINDER_H
