//
// Created by Adrien Blanchet on 05/12/2023.
//

#include "PythonInterface.h"

#include "Logger.h"

#include <pybind11/pybind11.h>

#include <string>


PYBIND11_MODULE(PyGundam, module) {
  module.doc() = "GUNDAM engine interface for python";

  auto& likelihoodInterfaceClass = pybind11::class_<LikelihoodInterface>(module, "LikelihoodInterface").def(pybind11::init());
  likelihoodInterfaceClass.def("evalLikelihood", &LikelihoodInterface::evalLikelihood, pybind11::call_guard<pybind11::gil_scoped_release>());

  auto& fitterEngineClass = pybind11::class_<FitterEngine>(module, "FitterEngine").def(pybind11::init());
  fitterEngineClass.def("getLikelihoodInterface", static_cast<LikelihoodInterface& (FitterEngine::*)()>(&FitterEngine::getLikelihoodInterface), pybind11::return_value_policy::reference);

  auto& pyGundamClass = pybind11::class_<PyGundam>(module, "PyGundam").def(pybind11::init());
  pyGundamClass.def("setConfig", &PyGundam::setConfig, pybind11::call_guard<pybind11::gil_scoped_release>());
  pyGundamClass.def("addConfigOverride", &PyGundam::addConfigOverride, pybind11::call_guard<pybind11::gil_scoped_release>());
  pyGundamClass.def("load", &PyGundam::load, pybind11::call_guard<pybind11::gil_scoped_release>());
  pyGundamClass.def("minimize", &PyGundam::minimize, pybind11::call_guard<pybind11::gil_scoped_release>());
  pyGundamClass.def("setNbThreads", &PyGundam::setNbThreads, pybind11::call_guard<pybind11::gil_scoped_release>());
  pyGundamClass.def("getFitterEngine", &PyGundam::getFitterEngine, pybind11::return_value_policy::reference);
}

void PyGundam::load(){
  _fitter_.setSaveDir( app.getOutfilePtr() );

  _fitter_.setConfig( GenericToolbox::Json::fetchValue<JsonType>(_configHandler_.getConfig(), "fitterEngineConfig") );
  _fitter_.configure();

  _fitter_.getLikelihoodInterface().setForceAsimovData( true );

  _fitter_.initialize();
}
