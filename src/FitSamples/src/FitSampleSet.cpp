//
// Created by Nadrino on 22/07/2021.
//



#include "JsonUtils.h"
#include "GlobalVariables.h"
#include "FitSampleSet.h"

#include "Logger.h"
#include "GenericToolbox.h"

#include "nlohmann/json.hpp"
#include <TTreeFormulaManager.h>

#include <memory>


LoggerInit([]{ Logger::setUserHeaderStr("[FitSampleSet]"); });

FitSampleSet::FitSampleSet() { this->reset(); }
FitSampleSet::~FitSampleSet() { this->reset(); }

void FitSampleSet::reset() {
  _isInitialized_ = false;
  _config_.clear();

  _fitSampleList_.clear();

  _eventByEventDialLeafList_.clear();
}

void FitSampleSet::setConfig(const nlohmann::json &config) {
  _config_ = config;
  while( _config_.is_string() ){
    LogWarning << "Forwarding " << __CLASS_NAME__ << " config: \"" << _config_.get<std::string>() << "\"" << std::endl;
    _config_ = JsonUtils::readConfigFile(_config_.get<std::string>());
  }
}

void FitSampleSet::initialize() {
  LogWarning << __METHOD_NAME__ << std::endl;

  LogAssert(not _config_.empty(), "_config_ is not set." << std::endl);

//  _dataEventType_ = DataEventTypeEnumNamespace::toEnum(
//    JsonUtils::fetchValue<std::string>(_config_, "dataEventType"), true
//    );
//  LogInfo << "Data events type is set to: " << DataEventTypeEnumNamespace::toString(_dataEventType_) << std::endl;

  LogInfo << "Reading samples definition..." << std::endl;
  auto fitSampleListConfig = JsonUtils::fetchValue(_config_, "fitSampleList", nlohmann::json());
  bool _isEnabled_{false};
  
  for( const auto& fitSampleConfig: fitSampleListConfig ){
    _isEnabled_ = JsonUtils::fetchValue(fitSampleConfig, "isEnabled", true);
    if( not _isEnabled_ ) continue;
    _fitSampleList_.emplace_back();
    _fitSampleList_.back().setConfig(fitSampleConfig);
    _fitSampleList_.back().initialize();
  }

  LogInfo << "Creating parallelisable jobs" << std::endl;

  // Fill the bin index inside of each event
  std::function<void(int)> updateSampleEventBinIndexesFct = [this](int iThread){
    for( auto& sample : _fitSampleList_ ){
      sample.getMcContainer().updateEventBinIndexes(iThread);
      sample.getDataContainer().updateEventBinIndexes(iThread);
    }
  };
  GlobalVariables::getParallelWorker().addJob("FitSampleSet::updateSampleEventBinIndexes", updateSampleEventBinIndexesFct);

  // Fill bin event caches
  std::function<void(int)> updateSampleBinEventListFct = [this](int iThread){
    for( auto& sample : _fitSampleList_ ){
      sample.getMcContainer().updateBinEventList(iThread);
      sample.getDataContainer().updateBinEventList(iThread);
    }
  };
  GlobalVariables::getParallelWorker().addJob("FitSampleSet::updateSampleBinEventList", updateSampleBinEventListFct);


  // Histogram fills
  std::function<void(int)> refillMcHistogramsFct = [this](int iThread){
    for( auto& sample : _fitSampleList_ ){
      sample.getMcContainer().refillHistogram(iThread);
      sample.getDataContainer().refillHistogram(iThread);
    }
  };
  std::function<void()> rescaleMcHistogramsFct = [this](){
    for( auto& sample : _fitSampleList_ ){
      sample.getMcContainer().rescaleHistogram();
      sample.getDataContainer().rescaleHistogram();
    }
  };
  GlobalVariables::getParallelWorker().addJob("FitSampleSet::updateSampleHistograms", refillMcHistogramsFct);
  GlobalVariables::getParallelWorker().setPostParallelJob("FitSampleSet::updateSampleHistograms", rescaleMcHistogramsFct);

  std::string llhMethod = JsonUtils::fetchValue(_config_, "llhStatFunction", "PoissonLLH");
  LogInfo << "Using \"" << llhMethod << "\" LLH function." << std::endl;
  if     ( llhMethod == "PoissonLLH" ){  _jointProbabilityPtr_ = std::make_shared<JointProbability::PoissonLLH>(); }
  else if( llhMethod == "BarlowLLH" ) {  _jointProbabilityPtr_ = std::make_shared<JointProbability::BarlowLLH>(); }
  else if( llhMethod == "BANFF_OA2020" or llhMethod == "BarlowLLH_OA2020_Bad" ) {  _jointProbabilityPtr_ = std::make_shared<JointProbability::BarlowLLH_BANFF_OA2020>(); }
  else if( llhMethod == "BANFF_OA2021" or llhMethod == "BarlowLLH_BANFF_OA2021" ) {  _jointProbabilityPtr_ = std::make_shared<JointProbability::BarlowLLH_BANFF_OA2021>(); }
  else{
    LogThrow("Unknown LLH Method: " << llhMethod)
  }

  _isInitialized_ = true;
}

const std::vector<FitSample> &FitSampleSet::getFitSampleList() const {
  return _fitSampleList_;
}
std::vector<FitSample> &FitSampleSet::getFitSampleList() {
  return _fitSampleList_;
}
const nlohmann::json &FitSampleSet::getConfig() const {
  return _config_;
}
const std::shared_ptr<JointProbability::JointProbability> &FitSampleSet::getJointProbabilityFct() const{
  return _jointProbabilityPtr_;
}

bool FitSampleSet::empty() const {
  return _fitSampleList_.empty();
}
double FitSampleSet::evalLikelihood() const{
  double llh = 0.;
  for( auto& sample : _fitSampleList_ ){ llh += this->evalLikelihood(sample); }
  return llh;
}
double FitSampleSet::evalLikelihood(const FitSample& sample_) const{
  return _jointProbabilityPtr_->eval(sample_);
}

void FitSampleSet::copyMcEventListToDataContainer(){
  for( auto& sample : _fitSampleList_ ){
    LogInfo << "Copying MC events in sample \"" << sample.getName() << "\"" << std::endl;
    sample.getDataContainer().eventList.insert(
        std::end(sample.getDataContainer().eventList),
        std::begin(sample.getMcContainer().eventList),
        std::end(sample.getMcContainer().eventList)
    );
  }
}
void FitSampleSet::clearMcContainers(){
  for( auto& sample : _fitSampleList_ ){
    LogInfo << "Clearing event list for \"" << sample.getName() << "\"" << std::endl;
    sample.getMcContainer().eventList.clear();
  }
}

void FitSampleSet::updateSampleEventBinIndexes() const{
  if( _showTimeStats_ ) GenericToolbox::getElapsedTimeSinceLastCallInMicroSeconds(__METHOD_NAME__);
  GlobalVariables::getParallelWorker().runJob("FitSampleSet::updateSampleEventBinIndexes");
  if( _showTimeStats_ ) LogDebug << __METHOD_NAME__ << " took: " << GenericToolbox::getElapsedTimeSinceLastCallStr(__METHOD_NAME__) << std::endl;
}
void FitSampleSet::updateSampleBinEventList() const{
  if( _showTimeStats_ ) GenericToolbox::getElapsedTimeSinceLastCallInMicroSeconds(__METHOD_NAME__);
  GlobalVariables::getParallelWorker().runJob("FitSampleSet::updateSampleBinEventList");
  if( _showTimeStats_ ) LogDebug << __METHOD_NAME__ << " took: " << GenericToolbox::getElapsedTimeSinceLastCallStr(__METHOD_NAME__) << std::endl;
}
void FitSampleSet::updateSampleHistograms() const {
  if( _showTimeStats_ ) GenericToolbox::getElapsedTimeSinceLastCallInMicroSeconds(__METHOD_NAME__);
  GlobalVariables::getParallelWorker().runJob("FitSampleSet::updateSampleHistograms");
  if( _showTimeStats_ ) LogDebug << __METHOD_NAME__ << " took: " << GenericToolbox::getElapsedTimeSinceLastCallStr(__METHOD_NAME__) << std::endl;
}


