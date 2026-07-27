#include "ExternalWeightDialFactory.h"

#include "ExternalWeight.h"
#include "Event.h"
#include "GenericToolbox.Os.h"
#include "GenericToolbox.Utils.h"
#include "Logger.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <unistd.h>

namespace {
  std::string trimString(const std::string& str_){
    auto begin = std::find_if_not(str_.begin(), str_.end(), [](unsigned char c){ return std::isspace(c); });
    auto end = std::find_if_not(str_.rbegin(), str_.rend(), [](unsigned char c){ return std::isspace(c); }).base();
    if( begin >= end ){ return {}; }
    return {begin, end};
  }

  std::string shellQuote(const std::string& str_){
    std::string out{"'"};
    for( const char c : str_ ){
      if( c == '\'' ){ out += "'\\''"; }
      else{ out += c; }
    }
    out += "'";
    return out;
  }
}

ExternalWeightDialFactory::ExternalWeightDialFactory(const ConfigReader& config_) {
  ConfigReader config(config_);

  if( config.hasField("externalWeight") ){
    config = config.fetchValue<ConfigReader>("externalWeight");
  }

  config.defineFields({
      {"pythonExecutable"},
      {"pythonVenv"},
      {"initScript"},
      {"evalScript"},
      {"inputList"},
    });
  config.checkConfiguration();

  config.fillValue(_inputNameList_, "inputList");
  LogThrowIf(_inputNameList_.empty(), "ExternalWeight requires a non-empty inputList.");

  for( auto& inputName : _inputNameList_ ){
    inputName = normalizeInputName(inputName);
    LogThrowIf(inputName.empty(), "ExternalWeight inputList contains an empty input name.");
  }
  _inputValueList_.resize(_inputNameList_.size());

  PythonWorkerConfig workerConfig{};
  config.fillValue(workerConfig.pythonExecutable, "pythonExecutable");
  config.fillValue(workerConfig.pythonVenv, "pythonVenv");
  config.fillValue(workerConfig.initScript, "initScript");
  config.fillValue(workerConfig.evalScript, "evalScript");

  if( workerConfig.pythonExecutable.empty() and not workerConfig.pythonVenv.empty() ){
    workerConfig.pythonExecutable = workerConfig.pythonVenv + "/bin/python";
  }

  _worker_.configure(workerConfig);
  _worker_.initialize();
}

DialBase* ExternalWeightDialFactory::makeDial(const Event& event_) {
  std::lock_guard<std::mutex> lock(_eventRegistrationMutex_);

  const std::size_t eventIndex = _weightList_.size();

  for( std::size_t iInput = 0 ; iInput < _inputNameList_.size() ; ++iInput ){
    _inputValueList_[iInput].emplace_back(event_.getVariables().fetchVariable(_inputNameList_[iInput]).getVarAsDouble());
  }

  _weightList_.emplace_back(1.);
  return new ExternalWeight(&_weightList_, eventIndex);
}

void ExternalWeightDialFactory::updateWeights(DialInputBuffer& inputBuffer_) {
  if( not _eventsLoadedInWorker_ ){
    this->finalizeEventLoading();
  }

  for( const auto& values : _inputValueList_ ){
    LogThrowIf(values.size() != _weightList_.size(),
               "ExternalWeight internal size mismatch: "
               << _weightList_.size() << " weights for "
               << values.size() << " registered input values.");
  }
  _worker_.evaluate(inputBuffer_, _weightList_);
}

void ExternalWeightDialFactory::finalizeEventLoading() {
  if( _eventsLoadedInWorker_ ){ return; }

  for( auto& values : _inputValueList_ ){
    LogThrowIf(values.size() != _weightList_.size(),
               "ExternalWeight input size mismatch: "
               << values.size() << " values for " << _weightList_.size() << " registered events.");
    values.shrink_to_fit();
  }
  _weightList_.shrink_to_fit();
  _worker_.loadEvents(_inputNameList_, _inputValueList_, _weightList_.size());
  _eventsLoadedInWorker_ = true;
}

std::string ExternalWeightDialFactory::normalizeInputName(const std::string& inputName_) {
  auto out = trimString(inputName_);
  if( out.size() >= 2 and out.front() == '[' and out.back() == ']' ){
    out = out.substr(1, out.size() - 2);
    out = trimString(out);
  }
  return out;
}

void ExternalWeightDialFactory::PythonWorker::configure(const PythonWorkerConfig& config_) {
  _config_ = config_;
}

void ExternalWeightDialFactory::PythonWorker::initialize() {
  if( _isInitialized_ ){ return; }

  LogThrowIf(_config_.pythonExecutable.empty(),
             "ExternalWeight pythonExecutable is not configured. Provide pythonExecutable or pythonVenv.");
  LogThrowIf(_config_.evalScript.empty(),
             "ExternalWeight evalScript is not configured.");

  if( not _config_.initScript.empty() ){
    auto output = GenericToolbox::getOutputOfShellCommand(
        shellQuote(_config_.pythonExecutable) + " " + shellQuote(_config_.initScript)
    );
    for( const auto& line : output ){
      LogInfo << "ExternalWeight initScript: " << line << std::endl;
    }
  }

  _isInitialized_ = true;
}

void ExternalWeightDialFactory::PythonWorker::loadEvents(
    const std::vector<std::string>& inputNameList_,
    const std::vector<std::vector<double>>& inputValueList_,
    std::size_t eventCount_) {
  LogThrowIf(not _isInitialized_, "ExternalWeight PythonWorker is not initialized.");
  LogThrowIf(inputNameList_.size() != inputValueList_.size(),
             "ExternalWeight worker received " << inputNameList_.size()
             << " input names but " << inputValueList_.size() << " input arrays.");
  _loadedInputNameList_ = inputNameList_;
  _loadedInputValueList_ = inputValueList_;
  LogInfo << "ExternalWeight worker registered " << eventCount_
          << " events with inputs " << GenericToolbox::toString(inputNameList_) << "." << std::endl;
}

void ExternalWeightDialFactory::PythonWorker::evaluate(
    const DialInputBuffer& inputBuffer_,
    std::vector<double>& weightList_) {
  LogThrowIf(not _isInitialized_, "ExternalWeight PythonWorker is not initialized.");

  JsonType payload;
  payload["inputs"] = JsonType::object();
  payload["parameters"] = JsonType::array();

  for( std::size_t iInput = 0 ; iInput < _loadedInputNameList_.size() ; ++iInput ){
    payload["inputs"][_loadedInputNameList_[iInput]] = _loadedInputValueList_[iInput];
  }

  for( int iPar = 0 ; iPar < inputBuffer_.getInputSize() ; ++iPar ){
    JsonType parEntry;
    parEntry["name"] = inputBuffer_.getParameter(iPar).getName();
    parEntry["title"] = inputBuffer_.getParameter(iPar).getFullTitle();
    parEntry["value"] = inputBuffer_.getInputBuffer().at(iPar);
    payload["parameters"].emplace_back(std::move(parEntry));
  }

  std::stringstream payloadPathSs;
  payloadPathSs << GenericToolbox::getCurrentWorkingDirectory()
                << "/gundam_external_weight_payload_"
                << getpid() << "_" << reinterpret_cast<std::uintptr_t>(this)
                << ".json";
  const std::string payloadPath = payloadPathSs.str();

  {
    std::ofstream payloadFile(payloadPath);
    LogThrowIf(not payloadFile.is_open(), "Could not open ExternalWeight payload file: " << payloadPath);
    payloadFile << payload.dump();
  }

  auto output = GenericToolbox::getOutputOfShellCommand(
      shellQuote(_config_.pythonExecutable) + " "
      + shellQuote(_config_.evalScript) + " "
      + shellQuote(payloadPath)
  );
  std::remove(payloadPath.c_str());

  LogThrowIf(output.empty(), "ExternalWeight evalScript produced no output.");

  JsonType result;
  bool gotResult{false};
  for( const auto& line : output ){
    try{
      result = JsonType::parse(line);
      gotResult = true;
      break;
    }
    catch(...){}
  }
  LogThrowIf(not gotResult, "ExternalWeight evalScript did not produce a valid JSON line.");
  LogThrowIf(not result.contains("weights"), "ExternalWeight evalScript output does not contain a \"weights\" entry.");

  auto weights = result.at("weights").get<std::vector<double>>();
  LogThrowIf(weights.size() != weightList_.size(),
             "ExternalWeight evalScript returned " << weights.size()
             << " weights for " << weightList_.size() << " events.");
  weightList_ = std::move(weights);
}
