//
// Created by Adrien Blanchet on 05/12/2023.
//

#include "PythonInterface.h"
#include "FitterEngine.h"
#include "ConfigUtils.h"
#include "GundamApp.h"

#include "Logger.h"

#include <pybind11/pybind11.h>

#include <string>


PYBIND11_MODULE(PyGundam, module) {
  module.doc() = "GUNDAM engine interface for python";

  // GundamGlobals namespace
  module.def("setNumberOfThreads", &GundamGlobals::setNumberOfThreads, "Set the number of threads for Gundam");
  module.def("setLightOutputMode", &GundamGlobals::setLightOutputMode, "Reduce the amount of outputs in the root files");
  module.def("setIsDebug", &GundamGlobals::setIsDebug, "Enables debug printouts");

  // JsonType for the return type
  pybind11::class_<JsonType>(module, "JsonType")
  .def(pybind11::init())
  ;

  // basic function to get sub-parts of Json
  auto gtModule = module.def_submodule("GenericToolbox");
  gtModule.def_submodule("Json")
  .def("cd", &GenericToolbox::Json::cd)
  ;

  // ConfigUtils namespace
  auto configUtilsModule = module.def_submodule("ConfigUtils");
  pybind11::class_<ConfigUtils::ConfigHandler>(configUtilsModule, "ConfigHandler")
  .def(pybind11::init())
  .def(pybind11::init<const std::string&>())
  .def(pybind11::init<const JsonType&>())
  .def("setConfig", pybind11::overload_cast<const std::string&>(&ConfigUtils::ConfigHandler::setConfig))
  .def("setConfig", pybind11::overload_cast<const JsonType&>(&ConfigUtils::ConfigHandler::setConfig))
  .def("setConfig", pybind11::overload_cast<const JsonType&>(&ConfigUtils::ConfigHandler::setConfig))
  .def("getConfig", pybind11::overload_cast<>(&ConfigUtils::ConfigHandler::getConfig, pybind11::const_))
  .def("flatOverride", pybind11::overload_cast<const std::string&>(&ConfigUtils::ConfigHandler::flatOverride))
  .def("flatOverride", pybind11::overload_cast<const std::vector<std::string>&>(&ConfigUtils::ConfigHandler::flatOverride))
  .def("override", pybind11::overload_cast<const std::string&>(&ConfigUtils::ConfigHandler::override))
  .def("override", pybind11::overload_cast<const std::vector<std::string>&>(&ConfigUtils::ConfigHandler::override))
  .def("override", pybind11::overload_cast<const JsonType&>(&ConfigUtils::ConfigHandler::override))
  .def("toString", &ConfigUtils::ConfigHandler::toString)
  .def("exportToJsonFile", &ConfigUtils::ConfigHandler::exportToJsonFile)
  ;

  auto& parametersManagerClass = pybind11::class_<ParametersManager>(module, "ParametersManager").def(pybind11::init());
  parametersManagerClass.def("throwParameters", &ParametersManager::throwParameters);

  auto& propagatorClass = pybind11::class_<Propagator>(module, "Propagator").def(pybind11::init());
  propagatorClass.def("getParametersManager", static_cast<ParametersManager& (Propagator::*)()>(&Propagator::getParametersManager), pybind11::return_value_policy::reference);

  auto& likelihoodInterfaceClass = pybind11::class_<LikelihoodInterface>(module, "LikelihoodInterface").def(pybind11::init());
  likelihoodInterfaceClass.def("evalLikelihood", &LikelihoodInterface::evalLikelihood, pybind11::call_guard<pybind11::gil_scoped_release>());
  likelihoodInterfaceClass.def("setForceAsimovData", &LikelihoodInterface::setForceAsimovData, pybind11::call_guard<pybind11::gil_scoped_release>());
  likelihoodInterfaceClass.def("throwToyParameters", &LikelihoodInterface::throwToyParameters, pybind11::call_guard<pybind11::gil_scoped_release>());
  likelihoodInterfaceClass.def("throwStatErrors", &LikelihoodInterface::throwStatErrors, pybind11::call_guard<pybind11::gil_scoped_release>());
  likelihoodInterfaceClass.def("getModelPropagator", static_cast<Propagator& (LikelihoodInterface::*)()>(&LikelihoodInterface::getModelPropagator), pybind11::return_value_policy::reference);
  likelihoodInterfaceClass.def("getDataPropagator", static_cast<Propagator& (LikelihoodInterface::*)()>(&LikelihoodInterface::getDataPropagator), pybind11::return_value_policy::reference);

  pybind11::class_<MinimizerBase>(module, "MinimizerBase")
  .def("minimize", &MinimizerBase::minimize)
  ;

  pybind11::class_<FitterEngine>(module, "FitterEngine")
  .def(pybind11::init())
  .def("setSaveDir", pybind11::overload_cast<TDirectory*>(&FitterEngine::setSaveDir))
  .def("setConfig", pybind11::overload_cast<const JsonType&>(&FitterEngine::setConfig))
  .def("configure", pybind11::overload_cast<const JsonType&>(&FitterEngine::configure))
  .def("configure", pybind11::overload_cast<>(&FitterEngine::configure))
  .def("initialize", &FitterEngine::initialize)
  .def("getMinimizer", pybind11::overload_cast<>(&FitterEngine::getMinimizer), pybind11::return_value_policy::reference)
  .def("getLikelihoodInterface", pybind11::overload_cast<>(&FitterEngine::getLikelihoodInterface))
  ;
}

