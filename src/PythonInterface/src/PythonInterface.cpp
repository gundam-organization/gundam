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

  // ROOT
  // I need to expose the ROOT classes that are used as arguments or return types in the C++ code.
  // Normal pyROOT is not compatible with pybind11, because it uses cppyy.
  pybind11::class_<TMatrixD, std::shared_ptr<TMatrixD>>(module, "TMatrixD")
  .def(pybind11::init<int, int>())
  .def(pybind11::init<const TMatrixD&>())
  .def("operator()", [](TMatrixD& m, int i, int j) -> double& { return m[i][j]; })
  .def("GetNrows", &TMatrixD::GetNrows, "Get the number of rows in the matrix")
  .def("GetNcols", &TMatrixD::GetNcols, "Get the number of columns in the matrix")
  .def("__call__", [](TMatrixD& self, int i, int j) -> double& {
  return self(i, j);
  })
  .def("__setitem__", [](TMatrixD& self, std::pair<int, int> ij, double val) {
  self(ij.first, ij.second) = val;
  })
  .def("__getitem__", [](TMatrixD& self, std::pair<int, int> ij) -> double {
  return self(ij.first, ij.second);
  })
  ;



  // basic function to get sub-parts of Json
  auto gtModule = module.def_submodule("GenericToolbox");
  gtModule.def_submodule("Json")
  .def("cd", &GenericToolbox::Json::cd)
  .def("readConfigJsonStr", &GenericToolbox::Json::readConfigJsonStr, "Read a JSON string and return a JsonType object")
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

  pybind11::class_<GenericToolbox::Range>(module, "Range")
  .def(pybind11::init<>())
  .def(pybind11::init<double, double>())
  .def_readwrite("min", &GenericToolbox::Range::min)
  .def_readwrite("max", &GenericToolbox::Range::max)
  .def("fillMostConstrainingBounds", &GenericToolbox::Range::fillMostConstrainingBounds)
  .def("hasLowerBound", &GenericToolbox::Range::hasLowerBound)
  .def("hasUpperBound", &GenericToolbox::Range::hasUpperBound)
  .def("hasBound", &GenericToolbox::Range::hasBound)
  .def("hasBothBounds", &GenericToolbox::Range::hasBothBounds)
  .def("isUnbounded", &GenericToolbox::Range::isUnbounded)
  .def("isBellowMin", &GenericToolbox::Range::isBellowMin)
  .def("isAboveMax", &GenericToolbox::Range::isAboveMax)
  .def("isInBounds", &GenericToolbox::Range::isInBounds)
  ;

  pybind11::class_<Parameter>(module, "Parameter")
  // Queries
  .def("isEnabled", &Parameter::isEnabled)
  .def("isValueWithinBounds", &Parameter::isValueWithinBounds)
  .def("isInDomain", &Parameter::isInDomain)
  // Getters
  .def("getName", &Parameter::getName)
  .def("getSummary", &Parameter::getSummary)
  .def("getFullTitle", &Parameter::getFullTitle)
  .def("getParameterValue", &Parameter::getParameterValue)
  .def("getParameterLimits", &Parameter::getParameterLimits, pybind11::return_value_policy::reference)
  .def("getPriorValue", &Parameter::getPriorValue)
  .def("getStdDevValue", &Parameter::getStdDevValue)
  .def("getPhysicalLimits", &Parameter::getPhysicalLimits, pybind11::return_value_policy::reference)
  // Setters
  .def("setPriorValue", &Parameter::setPriorValue)
  .def("setParameterValue", &Parameter::setParameterValue)
;

  pybind11::class_<ParameterSet>(module, "ParameterSet")
  .def("getName", &ParameterSet::getName)
  .def("isEnabled", &ParameterSet::isEnabled)
  .def("getParameterList", static_cast<std::vector<Parameter>& (ParameterSet::*)()>(&ParameterSet::getParameterList),pybind11::return_value_policy::reference)
  .def("isEnableEigenDecomp", &ParameterSet::isEnableEigenDecomp)
  .def("propagateOriginalToEigen", &ParameterSet::propagateOriginalToEigen)
  ;

  pybind11::class_<ParametersManager>(module, "ParametersManager")
  .def(pybind11::init())
  .def("throwParameters", &ParametersManager::throwParameters)
  .def("exportParameterInjectorConfig", &ParametersManager::exportParameterInjectorConfig)
  .def("injectParameterValues", &ParametersManager::injectParameterValues, pybind11::arg("config_"), pybind11::arg("quietVerbose_") = false)
  .def("getParameterSetsList", static_cast<std::vector<ParameterSet>& (ParametersManager::*)()>(&ParametersManager::getParameterSetsList), pybind11::return_value_policy::reference)
//  .def("throwParametersFromGlobalCovariance", pybind11::overload_cast<std::vector<double>&>(&ParametersManager::throwParametersFromGlobalCovariance), pybind11::call_guard<pybind11::gil_scoped_release>())
  .def("throwParametersFromGlobalCovariance", [](ParametersManager &self) {
  std::vector<double> weights;
  self.throwParametersFromGlobalCovariance(weights);
  return weights;
  })
  .def("setThrowerAsCustom", &ParametersManager::setThrowerAsCustom)
  .def("setThrowerAsDefault", &ParametersManager::setThrowerAsDefault)
  .def("setGlobalCovarianceMatrix", static_cast<void (ParametersManager::*)(const std::shared_ptr<TMatrixD>&)>(&ParametersManager::setGlobalCovarianceMatrix), pybind11::call_guard<pybind11::gil_scoped_release>())
  .def("getGlobalCovarianceMatrix", [](ParametersManager& self) {return self.getGlobalCovarianceMatrix().get();}, pybind11::return_value_policy::reference) // Need to unwrap the shared pointer
  ;

  pybind11::class_<Sample>(module, "Sample")
  .def(pybind11::init())
  .def("isEnabled", &Sample::isEnabled)
  .def("getBinningFilePath", &Sample::getBinningFilePath) // returns a ConfigReader object
  .def("getName", &Sample::getName, pybind11::return_value_policy::reference)
  .def("getHistogram", static_cast<Histogram& (Sample::*)()>(&Sample::getHistogram), pybind11::return_value_policy::reference)
  .def("getSummary", &Sample::getSummary)
  ;
  pybind11::class_<Histogram>(module, "Histogram")
  .def(pybind11::init())
  .def("getNbBins", &Histogram::getNbBins)
  .def("getBinContentList", static_cast<std::vector<Histogram::BinContent>& (Histogram::*)()>(&Histogram::getBinContentList), pybind11::return_value_policy::reference)
  ;
  pybind11::class_<Histogram::BinContent>(module, "BinContent")
  .def(pybind11::init())
  .def_readwrite("sumWeights", &Histogram::BinContent::sumWeights)
  .def_readwrite("sqrtSumSqWeights", &Histogram::BinContent::sqrtSumSqWeights)
  ;
  pybind11::class_<SampleSet>(module, "SampleSet")
  .def(pybind11::init())
  .def("getSampleList", static_cast<std::vector<Sample>& (SampleSet::*)()>(&SampleSet::getSampleList), pybind11::return_value_policy::reference)
  ;
  pybind11::class_<Propagator>(module, "Propagator")
  .def(pybind11::init())
  .def("getParametersManager", pybind11::overload_cast<>(&Propagator::getParametersManager), pybind11::return_value_policy::reference)
  .def("copyHistBinContentFrom", pybind11::overload_cast<const Propagator&>(&Propagator::copyHistBinContentFrom), pybind11::return_value_policy::reference)
  .def("propagateParameters", &Propagator::propagateParameters, pybind11::call_guard<pybind11::gil_scoped_release>())
  .def("getSampleSet", static_cast<SampleSet& (Propagator::*)()>(&Propagator::getSampleSet), pybind11::return_value_policy::reference)
  ;

  pybind11::class_<LikelihoodInterface::Buffer>(module, "Buffer")
  .def(pybind11::init())
  .def_readwrite("totalLikelihood", &LikelihoodInterface::Buffer::totalLikelihood)
  .def_readwrite("statLikelihood", &LikelihoodInterface::Buffer::statLikelihood)
  .def_readwrite("penaltyLikelihood", &LikelihoodInterface::Buffer::penaltyLikelihood)
  .def("updateTotal", &LikelihoodInterface::Buffer::updateTotal)
  .def("isValid", &LikelihoodInterface::Buffer::isValid)
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
  .def("getSamplePairList", static_cast<std::vector<SamplePair>& (LikelihoodInterface::*)()>(&LikelihoodInterface::getSamplePairList), pybind11::return_value_policy::reference)
  .def("getSampleBreakdownTable", &LikelihoodInterface::getSampleBreakdownTable)
  .def("getBuffer", pybind11::overload_cast<>(&LikelihoodInterface::getBuffer), pybind11::return_value_policy::reference)
  ;

  pybind11::class_<SamplePair>(module, "SamplePair")
  .def(pybind11::init())
  .def_readwrite("data", &SamplePair::data)
  .def_readwrite("model", &SamplePair::model)
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
  .def("configure", pybind11::overload_cast<const ConfigReader&>(&FitterEngine::configure))
  .def("configure", pybind11::overload_cast<>(&FitterEngine::configure))
  .def("initialize", &FitterEngine::initialize)
  .def("setRandomSeed", &FitterEngine::setRandomSeed)
  .def("fit", &FitterEngine::fit)
  .def("getMinimizer", pybind11::overload_cast<>(&FitterEngine::getMinimizer), pybind11::return_value_policy::reference)
  .def("getLikelihoodInterface", pybind11::overload_cast<>(&FitterEngine::getLikelihoodInterface), pybind11::return_value_policy::reference)
  ;
}

