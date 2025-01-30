//
// Created by Adrien Blanchet on 05/12/2023.
//

#include "PythonInterface.h"

#include "Logger.h"

#include <pybind11/pybind11.h>

#include <string>


namespace py = pybind11;
constexpr auto byref = py::return_value_policy::reference_internal;

PYBIND11_MODULE(PyGundam, m) {
  m.doc() = "GUNDAM engine interface for python";

  py::class_<PyGundam>(m, "PyGundam")
  .def(py::init())
  .def("setConfig", &PyGundam::setConfig, py::call_guard<py::gil_scoped_release>())
  .def("addConfigOverride", &PyGundam::addConfigOverride, py::call_guard<py::gil_scoped_release>())
  .def("load", &PyGundam::load, py::call_guard<py::gil_scoped_release>())
  .def("fit", &PyGundam::fit, py::call_guard<py::gil_scoped_release>())
  ;
}
