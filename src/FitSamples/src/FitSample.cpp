//
// Created by Nadrino on 22/07/2021.
//

#include "FitSample.h"
#include "GlobalVariables.h"
#include "JsonUtils.h"

#include "GenericToolbox.h"
#include "Logger.h"

#include "vector"
#include "string"
#include <memory>


LoggerInit([]{
  Logger::setUserHeaderStr("[FitSample]");
});


FitSample::FitSample() { this->reset(); }
FitSample::~FitSample() { this->reset(); }

void FitSample::reset() {
  // YAML
  _config_.clear();
  _isEnabled_ = false;
  _name_ = "";
  _selectionCuts_ = "";
  _enabledDatasetList_.clear();
  _mcNorm_ = 1;
  _dataNorm_ = 1;

  // Internals
  _binning_.reset();
  _mcContainer_ = SampleElement();
  _dataContainer_ = SampleElement();
}

void FitSample::setConfig(const nlohmann::json &config_) {
  _config_ = config_;
}

void FitSample::initialize() {

  LogAssert(not _config_.empty(), GET_VAR_NAME_VALUE(_config_.empty()))

  _name_ = JsonUtils::fetchValue<std::string>(_config_, "name");
  LogThrowIf(
      GenericToolbox::doesStringContainsSubstring(_name_, "/"),
      "Invalid sample name: \"" << _name_ << "\": should not have '/'.")

  _isEnabled_ = JsonUtils::fetchValue(_config_, "isEnabled", true);
  if( not _isEnabled_ ){
    LogWarning << "\"" << _name_ << "\" is disabled." << std::endl;
    return;
  }

  LogInfo << "Initializing FitSample: " << _name_ << std::endl;

  _selectionCuts_ = JsonUtils::fetchValue(_config_, "selectionCuts", _selectionCuts_);
  _enabledDatasetList_ = JsonUtils::fetchValue(_config_, std::vector<std::string>{"datasets", "dataSets"}, _enabledDatasetList_);
  _mcNorm_ = JsonUtils::fetchValue(_config_, "mcNorm", _mcNorm_);
  _dataNorm_ = JsonUtils::fetchValue(_config_, "dataNorm", _dataNorm_);
  _binning_.readBinningDefinition( JsonUtils::fetchValue<std::string>(_config_, "binning") );

  TH1::SetDefaultSumw2(true);

  _mcContainer_.name = "MC_" + _name_;
  _mcContainer_.binning = _binning_;
  _mcContainer_.histScale = _dataNorm_/_mcNorm_;
  _mcContainer_.perBinEventPtrList.resize(_binning_.getBinsList().size());
  _mcContainer_.histogram = std::make_shared<TH1D>(
      Form("%s_MC_bins", _name_.c_str()), Form("%s_MC_bins", _name_.c_str()),
      int(_binning_.getBinsList().size()), 0, int(_binning_.getBinsList().size())
  );
  _mcContainer_.histogram->SetDirectory(nullptr);

  _dataContainer_.name = "Data_" + _name_;
  _dataContainer_.binning = _binning_;
  _dataContainer_.perBinEventPtrList.resize(_binning_.getBinsList().size());
  _dataContainer_.histogram = std::make_shared<TH1D>(
      Form("%s_Data_bins", _name_.c_str()), Form("%s_Data_bins", _name_.c_str()),
      int(_binning_.getBinsList().size()), 0, int(_binning_.getBinsList().size())
  );
  _dataContainer_.histogram->SetDirectory(nullptr);
}

bool FitSample::isEnabled() const {
  return _isEnabled_;
}
const std::string &FitSample::getName() const {
  return _name_;
}
const std::string &FitSample::getSelectionCutsStr() const {
  return _selectionCuts_;
}
const DataBinSet &FitSample::getBinning() const {
  return _binning_;
}
const SampleElement &FitSample::getMcContainer() const {
  return _mcContainer_;
}
const SampleElement &FitSample::getDataContainer() const {
  return _dataContainer_;
}
SampleElement &FitSample::getMcContainer() {
  return _mcContainer_;
}
SampleElement &FitSample::getDataContainer() {
  return _dataContainer_;
}

bool FitSample::isDatasetValid(const std::string& datasetName_){
  if( _enabledDatasetList_.empty() ) return true;
  for( auto& dataSetName : _enabledDatasetList_){
    if( dataSetName == "*" or dataSetName == datasetName_ ){
      return true;
    }
  }
  return false;
}

