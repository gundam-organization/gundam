//
// Created by Adrien Blanchet on 05/12/2023.
//

#include "PythonInterface.h"

#include "Logger.h"

#include <pybind11/pybind11.h>

#include <string>


// constexpr auto byref = pybind11::return_value_policy::reference_internal;

PYBIND11_MODULE(PyGundam, module) {
  module.doc() = "GUNDAM engine interface for python";

  pybind11::class_<FitterEngine>(module, "FitterEngine")
  .def(pybind11::init())
  .def("setConfig", &FitterEngine::setConfig, pybind11::call_guard<pybind11::gil_scoped_release>())
  ;

  pybind11::class_<PyGundam>(module, "PyGundam")
  .def(pybind11::init())
  .def("setConfig", &PyGundam::setConfig, pybind11::call_guard<pybind11::gil_scoped_release>())
  .def("addConfigOverride", &PyGundam::addConfigOverride, pybind11::call_guard<pybind11::gil_scoped_release>())
  .def("load", &PyGundam::load, pybind11::call_guard<pybind11::gil_scoped_release>())
  .def("minimize", &PyGundam::minimize, pybind11::call_guard<pybind11::gil_scoped_release>())
  .def("setNbThreads", &PyGundam::setNbThreads, pybind11::call_guard<pybind11::gil_scoped_release>())
  .def("getFitterEngine", &PyGundam::getFitterEngine, pybind11::return_value_policy::reference)
  ;
}

void PyGundam::load(){
  _fitter_.setSaveDir( app.getOutfilePtr() );

  _fitter_.setConfig( GenericToolbox::Json::fetchValue<JsonType>(_configHandler_.getConfig(), "fitterEngineConfig") );
  _fitter_.configure();

  _fitter_.getLikelihoodInterface().setForceAsimovData( true );

  _fitter_.initialize();
}
