//
// Created by Nadrino on 22/07/2021.
//

#include "GenericToolbox.h"
#include "Logger.h"

#include "JsonUtils.h"
#include "FitSample.h"

LoggerInit([](){
  Logger::setUserHeaderStr("[FitSample]");
})


FitSample::FitSample() { this->reset(); }
FitSample::~FitSample() { this->reset(); }

void FitSample::reset() {
  _config_.clear();

  _isEnabled_ = false;
  _name_ = "";

  _binning_.reset();

  _selectionCuts_ = "";
  _dataSetsSelections_.clear();

  _mcNorm_ = 1;
  _mcEventList_.clear();
  _mcHistogram_ = nullptr;

  _dataNorm_ = 1;
  _dataEventList_.clear();
  _dataHistogram_ = nullptr;
}

void FitSample::setConfig(const nlohmann::json &config_) {
  _config_ = config_;
}

void FitSample::initialize() {
  LogWarning << __METHOD_NAME__ << std::endl;

  LogAssert(not _config_.empty(), GET_VAR_NAME_VALUE(_config_.empty()));

  _name_ = JsonUtils::fetchValue<std::string>(_config_, "name");
  _isEnabled_ = JsonUtils::fetchValue(_config_, "isEnabled", true);
  if( not _isEnabled_ ){
    LogWarning << "\"" << _name_ << "\" is disabled." << std::endl;
    return;
  }

  _binning_.readBinningDefinition( JsonUtils::fetchValue<std::string>(_config_, "binning") );
//  for( const auto& bin : _binning_.getBinsList() ){
//    LogTrace << bin.getSummary() << std::endl;
//  }

  _mcHistogram_ = std::shared_ptr<TH1D>(
    new TH1D(
      Form("%s_MC_bins", _name_.c_str()), Form("%s_MC_bins", _name_.c_str()),
      int(_binning_.getBinsList().size()), 0, int(_binning_.getBinsList().size())
    )
  );
  _dataHistogram_ = std::shared_ptr<TH1D>(
    new TH1D(
      Form("%s_Data_bins", _name_.c_str()), Form("%s_Data_bins", _name_.c_str()),
      int(_binning_.getBinsList().size()), 0, int(_binning_.getBinsList().size())
    )
  );

  _selectionCuts_ = JsonUtils::fetchValue(_config_, "selectionCuts", _selectionCuts_);
  _dataSetsSelections_ = JsonUtils::fetchValue(_config_, "dataSets", _dataSetsSelections_);

  _mcNorm_ = JsonUtils::fetchValue(_config_, "mcNorm", _mcNorm_);
  _dataNorm_ = JsonUtils::fetchValue(_config_, "dataNorm", _dataNorm_);
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
std::vector<PhysicsEvent> &FitSample::getMcEventList() {
  return _mcEventList_;
}
std::vector<PhysicsEvent> &FitSample::getDataEventList() {
  return _dataEventList_;
}
const DataBinSet &FitSample::getBinning() const {
  return _binning_;
}
const std::vector<size_t> & FitSample::getDataSetIndexList() const {
  return _dataSetIndexList_;
}
const std::vector<size_t> &FitSample::getMcEventOffSetList() const {
  return _mcEventOffSetList_;
}
const std::vector<size_t> &FitSample::getMcEventNbList() const {
  return _mcEventNbList_;
}
const std::vector<size_t> &FitSample::getDataEventOffSetList() const {
  return _dataEventOffSetList_;
}
const std::vector<size_t> &FitSample::getDataEventNbList() const {
  return _dataEventNbList_;
}


bool FitSample::isDataSetValid(const std::string& dataSetName_){
  if( _dataSetsSelections_.empty() ) return true;
  for( auto& dataSetName : _dataSetsSelections_){
    if( dataSetName == "*" or dataSetName == dataSetName_ ){
      return true;
    }
  }
  return false;
}
void FitSample::reserveMemoryForMcEvents(size_t nbEvents_, size_t dataSetIndex_, const PhysicsEvent& eventBuffer_) {
  if( GenericToolbox::doesElementIsInVector(dataSetIndex_, _dataSetIndexList_) ){
    LogThrowIf(_dataSetIndexList_.back() != dataSetIndex_,"Dataset has already been filled");
  }
  else{
    _dataSetIndexList_.emplace_back(dataSetIndex_);
  }
  _mcEventOffSetList_.emplace_back(_mcEventList_.size());
  _mcEventNbList_.emplace_back(nbEvents_);
  _mcEventList_.resize(_mcEventOffSetList_.back() + _mcEventNbList_.back(), eventBuffer_);
}
void FitSample::reserveMemoryForDataEvents(size_t nbEvents_, size_t dataSetIndex_, const PhysicsEvent& eventBuffer_){
  if( GenericToolbox::doesElementIsInVector(dataSetIndex_, _dataSetIndexList_) ){
    LogThrowIf(_dataSetIndexList_.back() != dataSetIndex_,"Dataset has already been filled");
  }
  else{
    _dataSetIndexList_.emplace_back(dataSetIndex_);
  }
  _dataEventOffSetList_.emplace_back(_dataEventList_.size());
  _dataEventNbList_.emplace_back(nbEvents_);
  _dataEventList_.resize(_dataEventOffSetList_.back() + _dataEventNbList_.back(), eventBuffer_);
}


