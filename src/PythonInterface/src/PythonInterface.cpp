//
// Created by Adrien Blanchet on 05/12/2023.
//

#include "PythonInterface.h"
#include "FitterEngine.h"
#include "DatasetDefinition.h"
#include "ConfigUtils.h"
#include "GundamApp.h"

#include "GenericToolbox.Fs.h"
#include "GenericToolbox.Os.h"

#include "Logger.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>

#include <cerrno>
#include <cstring>
#include <string>
#include <unistd.h>


namespace {

void setRuntimeWorkingDirectory(const std::string& dirPath_, bool createIfMissing_){
  auto expandedDirPath = GenericToolbox::expandEnvironmentVariables(dirPath_);

  if( expandedDirPath.empty() ){
    throw pybind11::value_error("Runtime working directory cannot be empty.");
  }

  if( not GenericToolbox::isDir(expandedDirPath) ){
    if( createIfMissing_ ){
      GenericToolbox::mkdir(expandedDirPath);
    }
    if( not GenericToolbox::isDir(expandedDirPath) ){
      throw pybind11::value_error("Runtime working directory does not exist: " + expandedDirPath);
    }
  }

  if( chdir(expandedDirPath.c_str()) != 0 ){
    throw std::runtime_error("Could not set runtime working directory to \"" + expandedDirPath + "\": " + std::strerror(errno));
  }
}

}


PYBIND11_MODULE(GUNDAM, module) {
  module.doc() = "GUNDAM engine interface for python";

  // GundamGlobals namespace
  module.def("setNumberOfThreads", &GundamGlobals::setNumberOfThreads, "Set the number of threads for Gundam");
  module.def("setLightOutputMode", &GundamGlobals::setLightOutputMode, "Reduce the amount of outputs in the root files");
  module.def("setIsDebug", &GundamGlobals::setIsDebug, "Enables debug printouts");
  module.def("setRuntimeWorkingDirectory", &setRuntimeWorkingDirectory,
             pybind11::arg("dirPath"),
             pybind11::arg("createIfMissing") = false,
             "Set the process working directory used by GUNDAM to resolve relative runtime paths.");
  module.def("getRuntimeWorkingDirectory", &GenericToolbox::getCurrentWorkingDirectory,
             "Get the process working directory used by GUNDAM to resolve relative runtime paths.");

  // JsonType for the return type
  pybind11::class_<JsonType>(module, "JsonType")
  .def(pybind11::init())
  ;


  // basic function to get sub-parts of Json
  auto gtModule = module.def_submodule("GenericToolbox");
  gtModule.def_submodule("Json")
  .def("cd", &GenericToolbox::Json::cd)
  ;

  pybind11::class_<GenericToolbox::TFilePath>(gtModule, "TFilePath")
  .def("getSubDir", &GenericToolbox::TFilePath::getSubDir)
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

  pybind11::class_<Parameter>(module, "Parameter")
  .def("getName", &Parameter::getName)
  .def("getFullTitle", &Parameter::getFullTitle)
  .def("isEnabled", &Parameter::isEnabled)
  .def("getStepSize", &Parameter::getStepSize)
  .def("getPriorValue", &Parameter::getPriorValue)
  .def("getStdDevValue", &Parameter::getStdDevValue)
  .def("getThrowValue", &Parameter::getThrowValue)
  .def("getParameterValue", &Parameter::getParameterValue)
  .def("setParameterValue", &Parameter::setParameterValue)
  ;

  pybind11::class_<ParameterSet>(module, "ParameterSet")
  .def(pybind11::init())
  .def("getParameterList", pybind11::overload_cast<>(&ParameterSet::getParameterList), pybind11::return_value_policy::reference)
  ;

  pybind11::class_<ParametersManager>(module, "ParametersManager")
  .def(pybind11::init())
  .def("throwParameters", &ParametersManager::throwParameters)
  .def("exportParameterInjectorConfig", &ParametersManager::exportParameterInjectorConfig)
  .def("injectParameterValues", &ParametersManager::injectParameterValues)
  .def("getParameterSetsList", pybind11::overload_cast<>(&ParametersManager::getParameterSetsList), pybind11::return_value_policy::reference)
  ;

  pybind11::class_<Propagator>(module, "Propagator")
  .def(pybind11::init())
  .def("initialize", pybind11::overload_cast<>(&Propagator::initialize), pybind11::return_value_policy::reference)
  .def("getParametersManager", pybind11::overload_cast<>(&Propagator::getParametersManager), pybind11::return_value_policy::reference)
  .def("copyHistBinContentFrom", pybind11::overload_cast<const Propagator&>(&Propagator::copyHistBinContentFrom), pybind11::return_value_policy::reference)
  .def("writeParameterStateTree", &Propagator::writeParameterStateTree)
  ;

  pybind11::class_<DatasetDefinition>(module, "DatasetDefinition")
  .def("initialize", pybind11::overload_cast<>(&DatasetDefinition::initialize), pybind11::return_value_policy::reference)
  ;

  auto likelihoodInterfaceClass = pybind11::class_<LikelihoodInterface>(module, "LikelihoodInterface");

  auto dataTypeClass = pybind11::class_<LikelihoodInterface::DataType>(likelihoodInterfaceClass, "DataType")
  .def(pybind11::init())
  .def(pybind11::init([](const std::string& name_){
    auto dataType{LikelihoodInterface::DataType::toEnum(name_, true)};
    if( dataType.value == static_cast<LikelihoodInterface::DataType::EnumTypeName>(LikelihoodInterface::DataType::overflowValue) ){
      throw pybind11::value_error("Unknown DataType: " + name_);
    }
    return dataType;
  }), pybind11::arg("name"))
  .def_static("toEnum", [](const std::string& name_, bool ignoreCase_){
    auto dataType{LikelihoodInterface::DataType::toEnum(name_, ignoreCase_)};
    if( dataType.value == static_cast<LikelihoodInterface::DataType::EnumTypeName>(LikelihoodInterface::DataType::overflowValue) ){
      throw pybind11::value_error("Unknown DataType: " + name_);
    }
    return dataType;
  }, pybind11::arg("name"), pybind11::arg("ignoreCase") = true)
  .def_static("generateVectorStr", &LikelihoodInterface::DataType::generateVectorStr)
  .def("toString", [](const LikelihoodInterface::DataType& this_){ return this_.toString(); })
  .def("__str__", [](const LikelihoodInterface::DataType& this_){ return this_.toString(); })
  .def("__repr__", [](const LikelihoodInterface::DataType& this_){ return "DataType." + this_.toString(); })
  .def("__eq__", [](const LikelihoodInterface::DataType& lhs_, const LikelihoodInterface::DataType& rhs_){ return lhs_ == rhs_; })
  ;
  dataTypeClass.attr("Asimov") = pybind11::cast(LikelihoodInterface::DataType(LikelihoodInterface::DataType::Asimov));
  dataTypeClass.attr("Toy") = pybind11::cast(LikelihoodInterface::DataType(LikelihoodInterface::DataType::Toy));
  dataTypeClass.attr("RealData") = pybind11::cast(LikelihoodInterface::DataType(LikelihoodInterface::DataType::RealData));

  likelihoodInterfaceClass
  .def(pybind11::init())
  .def("getSummary", &LikelihoodInterface::getSummary, pybind11::call_guard<pybind11::gil_scoped_release>())
  .def("getLastLikelihood", &LikelihoodInterface::getLastLikelihood, pybind11::call_guard<pybind11::gil_scoped_release>())
  .def("propagateAndEvalLikelihood", &LikelihoodInterface::propagateAndEvalLikelihood, pybind11::call_guard<pybind11::gil_scoped_release>())
  .def("evalLikelihood", &LikelihoodInterface::evalLikelihood, pybind11::call_guard<pybind11::gil_scoped_release>())
  .def("setForceAsimovData", &LikelihoodInterface::setForceAsimovData, pybind11::call_guard<pybind11::gil_scoped_release>())
  .def("throwToyParameters", &LikelihoodInterface::throwToyParameters, pybind11::call_guard<pybind11::gil_scoped_release>())
  .def("throwStatErrors", &LikelihoodInterface::throwStatErrors, pybind11::call_guard<pybind11::gil_scoped_release>())
  .def("setCurrentParameterValuesAsPrior", &LikelihoodInterface::setCurrentParameterValuesAsPrior, pybind11::call_guard<pybind11::gil_scoped_release>())
  .def("setDataType", &LikelihoodInterface::setDataType, pybind11::call_guard<pybind11::gil_scoped_release>())
  .def("getModelPropagator", pybind11::overload_cast<>(&LikelihoodInterface::getModelPropagator), pybind11::return_value_policy::reference)
  .def("getDataPropagator", pybind11::overload_cast<>(&LikelihoodInterface::getDataPropagator), pybind11::return_value_policy::reference)
  .def("getDatasetList", pybind11::overload_cast<>(&LikelihoodInterface::getDatasetList), pybind11::return_value_policy::reference)
  .def("initialize", pybind11::overload_cast<>(&LikelihoodInterface::initialize), pybind11::return_value_policy::reference)
  ;

  pybind11::class_<ParameterScanner>(module, "ParameterScanner")
  .def(pybind11::init())
  .def("initialize", pybind11::overload_cast<>(&ParameterScanner::initialize), pybind11::return_value_policy::reference)
  ;

  // no CTOR here
  pybind11::class_<MinimizerBase>(module, "MinimizerBase")
  .def("minimize", &MinimizerBase::minimize)
  .def("fetchNbDegreeOfFreedom", &MinimizerBase::fetchNbDegreeOfFreedom)
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
  .def("getParameterScanner", pybind11::overload_cast<>(&FitterEngine::getParameterScanner), pybind11::return_value_policy::reference)
  .def("getTFilePath", &FitterEngine::getTFilePath)
  ;
}
