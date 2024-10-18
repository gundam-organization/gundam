//
// Created by Nadrino on 22/07/2021.
//

#include "Sample.h"
#include "GundamGlobals.h"
#include "GundamAlmostEqual.h"

#include "Logger.h"
#include "GenericToolbox.Thread.h"
#include "GenericToolbox.Loops.h"

#include <string>
#include <memory>

#ifndef DISABLE_USER_HEADER
LoggerInit([]{ Logger::setUserHeaderStr("[Sample]"); });
#endif


void Sample::configureImpl(){
  GenericToolbox::Json::fillValue(_config_, _name_, "name");
  GenericToolbox::Json::fillValue(_config_, _isEnabled_, "isEnabled");
  GenericToolbox::Json::fillValue(_config_, _disableEventMcThrow_, "disableEventMcThrow");
  GenericToolbox::Json::fillValue(_config_, _binningConfig_, {{"binningFilePath"},{"binningFile"},{"binning"}});
  GenericToolbox::Json::fillValue(_config_, _selectionCutStr_, {{"selectionCutStr"},{"selectionCuts"}});
  GenericToolbox::Json::fillValue(_config_, _enabledDatasetList_, {{"datasets"},{"dataSets"}});

  LogThrowIf(_name_.empty(), "No name was provided for sample #" << _index_ << std::endl << GenericToolbox::Json::toReadableString(_config_));
  LogDebugIf(GundamGlobals::isDebug()) << "Defining sample \"" << _name_ << "\"" << std::endl;
  if( not _isEnabled_ ){
    LogDebugIf(GundamGlobals::isDebug()) << "-> disabled" << std::endl;
    return;
  }

  LogDebugIf(GundamGlobals::isDebug()) << "Reading binning: " << _config_ << std::endl;
  _histogram_.getBinning().configure( _binningConfig_ );
  _histogram_.build();
}

void Sample::writeEventRates(const GenericToolbox::TFilePath& saveDir_) const{
  GenericToolbox::writeInTFile(saveDir_.getSubDir(_name_).getDir(), getSumWeights(), "sumWeights");
}
bool Sample::isDatasetValid(const std::string& datasetName_){
  if( _enabledDatasetList_.empty() ) return true;
  return std::any_of(
      _enabledDatasetList_.begin(), _enabledDatasetList_.end(),
      [&](const std::string& enabled_){
        return (enabled_ == "*" or enabled_ == datasetName_);
      }
  );
}

void Sample::Histogram::build(){

  nBins = int( _binning_.getBinList().size() );
  binContentList.resize( nBins );
  binContextList.resize( nBins );

  // filling bin contexts
  for( int iBin = 0 ; iBin < nBins ; iBin++ ){
    binContextList[iBin].index = iBin;
    binContextList[iBin].binPtr = &_binning_.getBinList()[iBin];
  }

}
void Sample::reserveEventMemory(size_t dataSetIndex_, size_t nEvents, const Event &eventBuffer_) {
  // adding one dataset:
  _loadedDatasetList_.emplace_back();

  // filling up properties:
  auto& datasetProperties{_loadedDatasetList_.back()};
  datasetProperties.dataSetIndex = dataSetIndex_;
  datasetProperties.eventOffSet = _eventList_.size();
  datasetProperties.eventNb = nEvents;

  LogInfo << "Creating " << nEvents << " event slots for sample:" << _name_ << std::endl;

  _eventList_.resize(datasetProperties.eventOffSet + datasetProperties.eventNb, eventBuffer_);
}
void Sample::shrinkEventList(size_t newTotalSize_){

  if( _loadedDatasetList_.empty() and newTotalSize_ == 0 ){
    LogAlert << "Empty dataset list. Nothing to shrink." << std::endl;
    return;
  }

  LogThrowIf(_eventList_.size() < newTotalSize_,
             "Can't shrink since eventList is too small: " << GET_VAR_NAME_VALUE(newTotalSize_)
                                                           << " > " << GET_VAR_NAME_VALUE(_eventList_.size()));

  LogThrowIf(not _loadedDatasetList_.empty() and _loadedDatasetList_.back().eventNb < (_eventList_.size() - newTotalSize_),
             "Can't shrink since eventList of the last dataSet is too small.");

  LogInfo << _name_ << ": shrinking event list from " << _eventList_.size() << " to " << newTotalSize_ << "..." << std::endl;

  _loadedDatasetList_.back().eventNb -= (_eventList_.size() - newTotalSize_);
  _eventList_.resize(newTotalSize_);
  _eventList_.shrink_to_fit();
}
void Sample::updateBinEventList(int iThread_) {

  auto bounds = GenericToolbox::ParallelWorker::getThreadBoundIndices(
      iThread_, GundamGlobals::getNbCpuThreads(), _histogram_.getNbBins()
  );
  
  for( int iBin = bounds.beginIndex ; iBin < bounds.endIndex ; iBin++ ){
    size_t count = std::count_if(_eventList_.begin(), _eventList_.end(), [&]( auto& e) {return e.getIndices().bin == iBin;});
    _histogram_.getBinContextList()[iBin].eventPtrList.resize(count, nullptr);

    // Now filling the event indexes
    size_t index = 0;
    std::for_each(_eventList_.begin(), _eventList_.end(), [&]( auto& e){
      if( e.getIndices().bin == iBin){ _histogram_.getBinContextList()[iBin].eventPtrList[index++] = &e; }
    });
  }
}
void Sample::refillHistogram(int iThread_){

#ifdef GUNDAM_USING_CACHE_MANAGER
  if (_CacheManagerValid_ and not (*_CacheManagerValid_)) {
    // This can be slow (~10 usec for 5000 bins) when data must be copied
    // from the device, but it makes sure that the results are copied from
    // the device when they have changed. The values pointed to by
    // _CacheManagerValue_ and _CacheManagerValid_ are inside the summed
    // index cache (a bit of evil coding here), and are updated by the
    // cache.  The update is triggered by (*_CacheManagerUpdate_)().
    if (_CacheManagerUpdate_) (*_CacheManagerUpdate_)();
  }
#endif

  auto bounds = GenericToolbox::ParallelWorker::getThreadBoundIndices(
      iThread_, GundamGlobals::getNbCpuThreads(), _histogram_.getNbBins()
  );

  // avoid using [] operator for each access. Use the memory address directly
  double weightBuffer;

  for( auto [binContent, binContext] : _histogram_.loop(bounds.beginIndex, bounds.endIndex) ){


#ifdef GUNDAM_USING_CACHE_MANAGER
    bool filledWithManager = false;

    // container used for debugging
    Histogram::BinContent cacheManagerValue;

    if (_CacheManagerValid_ and (*_CacheManagerValid_)
        and _CacheManagerValue_ and _CacheManagerIndex_ >= 0) {
      cacheManagerValue.sumWeights = _CacheManagerValue_[_CacheManagerIndex_+binContext.index];
      cacheManagerValue.sqrtSumSqWeight = _CacheManagerValue2_[_CacheManagerIndex_+binContext.index];
      cacheManagerValue.sqrtSumSqWeight = sqrt(cacheManagerValue.sqrtSumSqWeight);
      filledWithManager = true;
    }
    if( not filledWithManager or GundamGlobals::isForceCpuCalculation() ){
#endif
      // reset
      binContent.sumWeights = 0;
      binContent.sqrtSumSqWeight = 0;
      for( auto *eventPtr: binContext.eventPtrList ){
        weightBuffer = eventPtr->getEventWeight();
        binContent.sumWeights += weightBuffer;
        binContent.sqrtSumSqWeight += weightBuffer * weightBuffer;
      }

      binContent.sqrtSumSqWeight = std::sqrt(binContent.sqrtSumSqWeight);
#ifdef GUNDAM_USING_CACHE_MANAGER
    }
    else{
      // copy the result as
      binContent.sumWeights = cacheManagerValue.sumWeights;
      binContent.sqrtSumSqWeight = cacheManagerValue.sqrtSumSqWeight;
    }

    // Parallel calculations of the histogramming have been run.  Make sure
    // they are the same.
    if( filledWithManager and  GundamGlobals::isForceCpuCalculation() ){
      bool problemFound = false;
      if (not GundamUtils::almostEqual(cacheManagerValue.sumWeights,(binContent.sumWeights))) {
        double magnitude = std::abs(cacheManagerValue.sumWeights) + std::abs(binContent.sumWeights);
        double delta = std::abs(cacheManagerValue.sumWeights - binContent.sumWeights);
        if (magnitude > 0.0) delta /= 0.5*magnitude;
        LogError << "Incorrect histogram content --"
                 << " Content: " << cacheManagerValue.sumWeights << "!=" << binContent.sumWeights
                 << " Error: " << cacheManagerValue.sqrtSumSqWeight << "!=" << binContent.sqrtSumSqWeight
                 << " Precision: " << delta
                 << std::endl;
        problemFound = true;
      }
      if (not GundamUtils::almostEqual(cacheManagerValue.sqrtSumSqWeight,(binContent.sqrtSumSqWeight))) {
        double magnitude = std::abs(cacheManagerValue.sqrtSumSqWeight) + std::abs(binContent.sqrtSumSqWeight);
        double delta = std::abs(cacheManagerValue.sqrtSumSqWeight - binContent.sqrtSumSqWeight);
        if (magnitude > 0.0) delta /= 0.5*magnitude;
        LogError << "Incorrect histogram error --"
                 << " Content: " << cacheManagerValue.sumWeights << "!=" << binContent.sumWeights
                 << " Error: " << cacheManagerValue.sqrtSumSqWeight << "!=" << binContent.sqrtSumSqWeight
                 << " Precision: " << delta
                 << std::endl;
        problemFound = true;
      }
      if( false and problemFound ){ std::exit(EXIT_FAILURE); }// For debugging
    }
#endif

  }

}

void Sample::throwEventMcError(){
  // Take into account the finite number of events

  if( _disableEventMcThrow_ ){
    LogAlert << "MC event throw is disabled for sample: " << this->getName() << std::endl;
    return;
  }

  for( auto [binContent, binContext] : _histogram_.loop() ){

    binContent.sumWeights = 0;
    binContent.sqrtSumSqWeight = 0;
    for (auto *eventPtr: binContext.eventPtrList) {
      // gRandom->Poisson(1) -> returns an INT -> can be 0
      eventPtr->getWeights().current = (gRandom->Poisson(1) * eventPtr->getEventWeight());

      double weight{eventPtr->getEventWeight()};
      binContent.sumWeights += weight;
      binContent.sqrtSumSqWeight += weight * weight;
    }

    binContent.sqrtSumSqWeight = sqrt(binContent.sqrtSumSqWeight);
  }

}
void Sample::throwStatError(bool useGaussThrow_){
  /*
   * This is to convert "Asimov" histogram to toy-experiment (pseudo-data), i.e. with statistical fluctuations
   * */
  int nCounts;
  for( auto [binContent, binContext] : _histogram_.loop() ){
    if( binContent.sumWeights == 0 ){
      // this should not happen.
      continue;
    }

    if( not useGaussThrow_ ){
      nCounts = gRandom->Poisson( binContent.sumWeights );
    }
    else{
      nCounts = std::max(
          int( gRandom->Gaus(binContent.sumWeights, TMath::Sqrt(binContent.sumWeights)) )
          , 0 // if the throw is negative, cap it to 0
      );
    }
    for (auto *eventPtr: binContext.eventPtrList) {
      // make sure refill of the histogram will produce the same hist
      eventPtr->getWeights().current *= (double) nCounts / binContent.sumWeights;
    }
    binContent.sumWeights = nCounts;
  }
}

double Sample::getSumWeights() const{
  double output = std::accumulate(_eventList_.begin(), _eventList_.end(), double(0.),
                                  [](double sum_, const Event& ev_){ return sum_ + ev_.getEventWeight(); });
  return output;
}
size_t Sample::getNbBinnedEvents() const{
  return std::accumulate(
      _eventList_.begin(), _eventList_.end(), size_t(0.),
      []( size_t sum_, const Event &ev_ ){
        return sum_ + (ev_.getIndices().bin != -1);
      });
}
std::shared_ptr<TH1D> Sample::generateRootHistogram() const{
  std::shared_ptr<TH1D> out{nullptr};
  if( _histogram_.getNbBins() == 0 ){ return out; }
  out = std::make_shared<TH1D>(
      Form("%s_bins", _name_.c_str()), Form("%s_bins", _name_.c_str()),
      _histogram_.getNbBins(), 0, _histogram_.getNbBins()
  );
  return out;
}

void Sample::printConfiguration() const{

  LogInfo << "#" << _index_;
  LogInfo << ", name(" << _name_ << ")";
  LogInfo << ", nBins(" << _histogram_.getNbBins() << ")";
  LogInfo << std::endl;

}
[[nodiscard]] std::string Sample::getSummary() const{
  std::stringstream ss;
  ss << "Sample: " << _name_ << std::endl;
  ss << " - " << "Nb bins: " << _histogram_.getNbBins() << std::endl;
  ss << " - " << "Nb events: " << _eventList_.size();
  return ss.str();
}
std::ostream& operator <<( std::ostream& o, const Sample& this_ ){
  o << this_.getSummary(); return o;
}

