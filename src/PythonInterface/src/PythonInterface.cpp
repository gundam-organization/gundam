//
// Created by Adrien Blanchet on 05/12/2023.
//

#include "PythonInterface.h"
#include "FitterEngine.h"
#include "ConfigUtils.h"
#include "GundamApp.h"

#include "Logger.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>

#include <string>


PYBIND11_MODULE(GUNDAM, module) {
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
  pybind11::class_<ConfigUtils::ConfigBuilder>(configUtilsModule, "ConfigBuilder")
  .def(pybind11::init())
  .def(pybind11::init<const std::string&>())
  .def(pybind11::init<const JsonType&>())
  .def("setConfig", pybind11::overload_cast<const std::string&>(&ConfigUtils::ConfigBuilder::setConfig))
  .def("setConfig", pybind11::overload_cast<const JsonType&>(&ConfigUtils::ConfigBuilder::setConfig))
  .def("setConfig", pybind11::overload_cast<const JsonType&>(&ConfigUtils::ConfigBuilder::setConfig))
  .def("getConfig", pybind11::overload_cast<>(&ConfigUtils::ConfigBuilder::getConfig, pybind11::const_), pybind11::return_value_policy::reference)
  .def("flatOverride", pybind11::overload_cast<const std::string&>(&ConfigUtils::ConfigBuilder::flatOverride))
  .def("flatOverride", pybind11::overload_cast<const std::vector<std::string>&>(&ConfigUtils::ConfigBuilder::flatOverride))
  .def("override", pybind11::overload_cast<const std::string&>(&ConfigUtils::ConfigBuilder::override))
  .def("override", pybind11::overload_cast<const std::vector<std::string>&>(&ConfigUtils::ConfigBuilder::override))
  .def("override", pybind11::overload_cast<const JsonType&>(&ConfigUtils::ConfigBuilder::override))
  .def("toString", &ConfigUtils::ConfigBuilder::toString)
  .def("exportToJsonFile", &ConfigUtils::ConfigBuilder::exportToJsonFile)
  ;

  auto configReaderClass = pybind11::class_<ConfigUtils::ConfigReader>(configUtilsModule, "ConfigReader")
  .def(pybind11::init())
  .def(pybind11::init<const JsonType&>())
  .def("defineField", &ConfigUtils::ConfigReader::defineField)
  .def("fetchValueConfigReader", static_cast<ConfigUtils::ConfigReader (ConfigUtils::ConfigReader::*)(const std::string&) const>(
    &ConfigUtils::ConfigReader::fetchValue<ConfigUtils::ConfigReader>))
  ;

  pybind11::class_<ConfigUtils::ConfigReader::FieldDefinition>(configReaderClass, "FieldDefinition")
  .def(pybind11::init())
  .def(pybind11::init<std::string, std::vector<std::string>, std::string>(),
       pybind11::arg("name"),
       pybind11::arg("path") = std::vector<std::string>{},
       pybind11::arg("defaultValue") = "")
  ;

  pybind11::class_<GundamApp>(module, "GundamApp")
  .def(pybind11::init<std::string>())
  .def("openOutputFile", &GundamApp::openOutputFile)
  .def("writeAppInfo", &GundamApp::writeAppInfo)
  // .def("getOutfilePtr", &GundamApp::getOutfilePtr) // CAN'T EXPOSE ROOT PTRs
  ;

  pybind11::class_<ParametersManager>(module, "ParametersManager")
  .def(pybind11::init())
  .def("throwParameters", &ParametersManager::throwParameters)
  .def("exportParameterInjectorConfig", &ParametersManager::exportParameterInjectorConfig)
  .def("injectParameterValues", &ParametersManager::injectParameterValues)
  ;

  pybind11::class_<Propagator>(module, "Propagator")
  .def(pybind11::init())
  .def("getParametersManager", pybind11::overload_cast<>(&Propagator::getParametersManager), pybind11::return_value_policy::reference)
  .def("copyHistBinContentFrom", pybind11::overload_cast<const Propagator&>(&Propagator::copyHistBinContentFrom), pybind11::return_value_policy::reference)
  ;

  pybind11::class_<LikelihoodInterface>(module, "LikelihoodInterface")
  .def(pybind11::init())
  .def("getSummary", &LikelihoodInterface::getSummary, pybind11::call_guard<pybind11::gil_scoped_release>())
  .def("propagateAndEvalLikelihood", &LikelihoodInterface::propagateAndEvalLikelihood, pybind11::call_guard<pybind11::gil_scoped_release>())
  .def("evalLikelihood", &LikelihoodInterface::evalLikelihood, pybind11::call_guard<pybind11::gil_scoped_release>())
  .def("setForceAsimovData", &LikelihoodInterface::setForceAsimovData, pybind11::call_guard<pybind11::gil_scoped_release>())
  .def("throwToyParameters", &LikelihoodInterface::throwToyParameters, pybind11::call_guard<pybind11::gil_scoped_release>())
  .def("throwStatErrors", &LikelihoodInterface::throwStatErrors, pybind11::call_guard<pybind11::gil_scoped_release>())
  .def("setCurrentParameterValuesAsPrior", &LikelihoodInterface::setCurrentParameterValuesAsPrior, pybind11::call_guard<pybind11::gil_scoped_release>())
  .def("getModelPropagator", pybind11::overload_cast<>(&LikelihoodInterface::getModelPropagator), pybind11::return_value_policy::reference)
  .def("getDataPropagator", pybind11::overload_cast<>(&LikelihoodInterface::getDataPropagator), pybind11::return_value_policy::reference)
  ;

  // no CTOR here
  pybind11::class_<MinimizerBase>(module, "MinimizerBase")
  .def("minimize", &MinimizerBase::minimize)
  ;

  pybind11::class_<FitterEngine>(module, "FitterEngine")
  .def(pybind11::init())
  // .def("setSaveDir", pybind11::overload_cast<TDirectory*>(&FitterEngine::setSaveDir)) // CAN'T EXPOSE ROOT PTRs
  .def("setSaveDir", pybind11::overload_cast<GundamApp&, const std::string&>(&FitterEngine::setSaveDir))
  .def("setConfig", pybind11::overload_cast<const ConfigReader&>(&FitterEngine::setConfig))
  .def("setDoAllParamVariations", &FitterEngine::setDoAllParamVariations)
  .def("configure", pybind11::overload_cast<const ConfigReader&>(&FitterEngine::configure))
  .def("configure", pybind11::overload_cast<>(&FitterEngine::configure))
  .def("initialize", &FitterEngine::initialize)
  .def("setRandomSeed", &FitterEngine::setRandomSeed)
  .def("fit", &FitterEngine::fit)
  .def("getMinimizer", pybind11::overload_cast<>(&FitterEngine::getMinimizer), pybind11::return_value_policy::reference)
  .def("getLikelihoodInterface", pybind11::overload_cast<>(&FitterEngine::getLikelihoodInterface), pybind11::return_value_policy::reference)
  ;
}

