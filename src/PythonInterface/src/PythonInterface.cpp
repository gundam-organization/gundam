//
// Created by Adrien Blanchet on 05/12/2023.
//

#include "PythonInterface.h"

#include "Logger.h"

#include <pybind11/pybind11.h>

#include <string>


namespace py = pybind11;
constexpr auto byref = py::return_value_policy::reference_internal;

PYBIND11_MODULE(PythonBinder, m) {
  m.doc() = "optional module docstring";

  py::class_<PythonBinder>(m, "PythonBinder")
  .def(py::init<std::string>())
  .def("run", &PythonBinder::run, py::call_guard<py::gil_scoped_release>())
  .def_readonly("v_data", &PythonBinder::v_data, byref)
  .def_readonly("v_gamma", &PythonBinder::v_gamma, byref)
  ;
}
