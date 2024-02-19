//
// Created by Nadrino on 22/07/2021.
//


#include "SampleSet.h"

#include "GundamGlobals.h"

#include "GenericToolbox.Json.h"
#include "GenericToolbox.Time.h"
#include "Logger.h"

#include "nlohmann/json.hpp"
#include <TTreeFormulaManager.h>

#include <memory>


LoggerInit([]{ Logger::setUserHeaderStr("[SampleSet]"); });


void SampleSet::readConfigImpl(){
  LogWarning << __METHOD_NAME__ << std::endl;
  LogThrowIf(_config_.empty(), "_config_ is not set." << std::endl);

  LogInfo << "Reading samples definition..." << std::endl;
  auto fitSampleListConfig = GenericToolbox::Json::fetchValue(_config_, "fitSampleList", JsonType());
  for( const auto& fitSampleConfig: fitSampleListConfig ){
    if( not GenericToolbox::Json::fetchValue(fitSampleConfig, "isEnabled", true) ) continue;
    _sampleList_.emplace_back();
    _sampleList_.back().setIndex(int(_sampleList_.size()) - 1);
    _sampleList_.back().setConfig(fitSampleConfig);
    _sampleList_.back().readConfig();
  }

  // TODO: To be moved elsewhere -> nothing to do in sample... -> this should belong to the fitter engine
  std::string llhMethod = "PoissonLLH";
  llhMethod = GenericToolbox::Json::fetchValue(_config_, "llhStatFunction", llhMethod);

  // new config structure
  auto configJointProbability = GenericToolbox::Json::fetchValue(_config_, {{"jointProbability"}, {"llhConfig"}}, JsonType());
  llhMethod = GenericToolbox::Json::fetchValue(configJointProbability, "type", llhMethod);

  LogInfo << "Using \"" << llhMethod << "\" LLH function." << std::endl;
  if     ( llhMethod == "Chi2" ){                    _jointProbabilityPtr_ = std::make_shared<JointProbability::Chi2>(); }
  else if( llhMethod == "PoissonLLH" ){              _jointProbabilityPtr_ = std::make_shared<JointProbability::PoissonLLH>(); }
  else if( llhMethod == "BarlowLLH" ) {              _jointProbabilityPtr_ = std::make_shared<JointProbability::BarlowLLH>(); }
  else if( llhMethod == "Plugin" ) {                 _jointProbabilityPtr_ = std::make_shared<JointProbability::JointProbabilityPlugin>(); }
  else if( llhMethod == "BarlowLLH_BANFF_OA2020" ) { _jointProbabilityPtr_ = std::make_shared<JointProbability::BarlowLLH_BANFF_OA2020>(); }
  else if( llhMethod == "BarlowLLH_BANFF_OA2021" ) { _jointProbabilityPtr_ = std::make_shared<JointProbability::BarlowLLH_BANFF_OA2021>(); }
  else if( llhMethod == "LeastSquares" ) { _jointProbabilityPtr_ = std::make_shared<JointProbability::LeastSquaresLLH>(); }
  else if( llhMethod == "BarlowLLH_BANFF_OA2021_SFGD" ) {  _jointProbabilityPtr_ = std::make_shared<JointProbability::BarlowLLH_BANFF_OA2021_SFGD>(); }
  else{ LogThrow("Unknown LLH Method: " << llhMethod); }

  _jointProbabilityPtr_->readConfig(configJointProbability);
  _jointProbabilityPtr_->initialize();
}
void SampleSet::initializeImpl() {
  LogWarning << __METHOD_NAME__ << std::endl;
  LogThrowIf(_sampleList_.empty(), "No sample is defined.");

  for( auto& sample : _sampleList_ ){ sample.initialize(); }

  // Fill the bin index inside each event
  std::function<void(int)> updateSampleEventBinIndexesFct = [this](int iThread){
    LogInfoIf(iThread <= 0) << "Updating event sample bin indices..." << std::endl;
    for( auto& sample : _sampleList_ ){
      sample.getMcContainer().updateEventBinIndexes(iThread);
      sample.getDataContainer().updateEventBinIndexes(iThread);
    }
  };
  GundamGlobals::getParallelWorker().addJob("FitSampleSet::updateSampleEventBinIndexes", updateSampleEventBinIndexesFct);

  // Fill bin event caches
  std::function<void(int)> updateSampleBinEventListFct = [this](int iThread){
    LogInfoIf(iThread <= 0) << "Updating sample per bin event lists..." << std::endl;
    for( auto& sample : _sampleList_ ){
      sample.getMcContainer().updateBinEventList(iThread);
      sample.getDataContainer().updateBinEventList(iThread);
    }
  };
  GundamGlobals::getParallelWorker().addJob("FitSampleSet::updateSampleBinEventList", updateSampleBinEventListFct);


  // Histogram fills
  std::function<void(int)> refillMcHistogramsFct = [this](int iThread){
    for( auto& sample : _sampleList_ ){
      sample.getMcContainer().refillHistogram(iThread);
      sample.getDataContainer().refillHistogram(iThread);
    }
  };
  std::function<void()> rescaleMcHistogramsFct = [this](){
    for( auto& sample : _sampleList_ ){
      sample.getMcContainer().rescaleHistogram();
      sample.getDataContainer().rescaleHistogram();
    }
  };
  GundamGlobals::getParallelWorker().addJob("FitSampleSet::updateSampleHistograms", refillMcHistogramsFct);
  GundamGlobals::getParallelWorker().setPostParallelJob("FitSampleSet::updateSampleHistograms", rescaleMcHistogramsFct);
}

double SampleSet::evalLikelihood(){
  double llh = 0.;
  for( auto& sample : _sampleList_ ){
    llh += this->evalLikelihood(sample);
    LogThrowIf(std::isnan(llh) or std::isinf(llh), sample.getName() << ": reportde likelihood is invalid:" << llh);
  }
  return llh;
}
double SampleSet::evalLikelihood(Sample& sample_){
  sample_.setLlhStatBuffer(_jointProbabilityPtr_->eval(sample_));
  return sample_.getLlhStatBuffer();
}

void SampleSet::copyMcEventListToDataContainer(){
  for( auto& sample : _sampleList_ ){
    LogInfo << "Copying MC events in sample \"" << sample.getName() << "\"" << std::endl;
    sample.getDataContainer().eventList.clear();
    sample.getDataContainer().eventList.reserve(sample.getMcContainer().eventList.size());
//    sample.getDataContainer().eventList = sample.getMcContainer().eventList;
    sample.getDataContainer().eventList.insert(
        sample.getDataContainer().eventList.begin(),
        std::begin(sample.getMcContainer().eventList),
        std::end(sample.getMcContainer().eventList)
    );
  }
}
void SampleSet::clearMcContainers(){
  for( auto& sample : _sampleList_ ){
    LogInfo << "Clearing event list for \"" << sample.getName() << "\"" << std::endl;
    sample.getMcContainer().eventList.clear();
  }
}

void SampleSet::updateSampleEventBinIndexes() const{
  if( _showTimeStats_ ) GenericToolbox::getElapsedTimeSinceLastCallInMicroSeconds(__METHOD_NAME__);
  GundamGlobals::getParallelWorker().runJob("FitSampleSet::updateSampleEventBinIndexes");
  if( _showTimeStats_ ) LogDebug << __METHOD_NAME__ << " took: " << GenericToolbox::getElapsedTimeSinceLastCallStr(__METHOD_NAME__) << std::endl;
}
void SampleSet::updateSampleBinEventList() const{
  if( _showTimeStats_ ) GenericToolbox::getElapsedTimeSinceLastCallInMicroSeconds(__METHOD_NAME__);
  GundamGlobals::getParallelWorker().runJob("FitSampleSet::updateSampleBinEventList");
  if( _showTimeStats_ ) LogDebug << __METHOD_NAME__ << " took: " << GenericToolbox::getElapsedTimeSinceLastCallStr(__METHOD_NAME__) << std::endl;
}
void SampleSet::updateSampleHistograms() const {
  if( _showTimeStats_ ) GenericToolbox::getElapsedTimeSinceLastCallInMicroSeconds(__METHOD_NAME__);
  GundamGlobals::getParallelWorker().runJob("FitSampleSet::updateSampleHistograms");
  if( _showTimeStats_ ) LogDebug << __METHOD_NAME__ << " took: " << GenericToolbox::getElapsedTimeSinceLastCallStr(__METHOD_NAME__) << std::endl;
}
