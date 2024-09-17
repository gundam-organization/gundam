//
// Created by Nadrino on 22/07/2021.
//

#include "Sample.h"
#include "GundamGlobals.h"
#include "GundamAlmostEqual.h"

#include "GenericToolbox.Json.h"
#include "Logger.h"

#include <vector>
#include <string>
#include <memory>


#ifndef DISABLE_USER_HEADER
LoggerInit([]{ Logger::setUserHeaderStr("[Sample]"); });
#endif


void Sample::readConfigImpl(){
  _name_ = GenericToolbox::Json::fetchValue(_config_, "name", _name_);
  GenericToolbox::replaceSubstringInsideInputString(_name_, "/", " ");

  _isEnabled_ = GenericToolbox::Json::fetchValue(_config_, "isEnabled", _isEnabled_);
  if( not _isEnabled_ ){ return; }

  _binningConfig_ = GenericToolbox::Json::fetchValue(_config_, {{"binningFilePath"}, {"binningFile"}, {"binning"}}, _binningConfig_);
  _selectionCutStr_ = GenericToolbox::Json::fetchValue(_config_, {{"selectionCutStr"}, {"selectionCuts"}}, _selectionCutStr_);
  _enabledDatasetList_ = GenericToolbox::Json::fetchValue(_config_, std::vector<std::string>{"datasets", "dataSets"}, _enabledDatasetList_);

  _binning_.readBinningDefinition( _binningConfig_ );
  _binning_.sortBins();
  this->buildHistogram( _binning_ );
}

void Sample::writeEventRates(const GenericToolbox::TFilePath& saveDir_) const{
  if( not _isEnabled_ ){ return; } // don't write anything, even the containing dir
  GenericToolbox::writeInTFile(saveDir_.getSubDir(_name_).getDir(), getSumWeights(), "sumWeights");
}
bool Sample::isDatasetValid(const std::string& datasetName_){
  if( _enabledDatasetList_.empty() ) return true;
  for( auto& dataSetName : _enabledDatasetList_){
    if( dataSetName == "*" or dataSetName == datasetName_ ){
      return true;
    }
  }
  return false;
}

void Sample::buildHistogram(const DataBinSet& binning_){
  _histogram_.binList.reserve(binning_.getBinList().size() );
  int iBin{0};
  for( auto& bin : binning_.getBinList() ){
    _histogram_.binList.emplace_back();
    _histogram_.binList.back().dataBinPtr = &bin;
    _histogram_.binList.back().index = iBin++;
  }
  _histogram_.nBins = int( _histogram_.binList.size() );
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
  int nbThreads = GundamGlobals::getNumberOfThreads();
  if( iThread_ == -1 ){ iThread_ = 0; nbThreads = 1; }
  // multithread technique with iBin += nbThreads;
  int iBin{iThread_};
  while( iBin < _histogram_.nBins ){
    size_t count = std::count_if(_eventList_.begin(), _eventList_.end(), [&]( auto& e) {return e.getIndices().bin == iBin;});
    _histogram_.binList[iBin].eventPtrList.resize(count, nullptr);

    // Now filling the event indexes
    size_t index = 0;
    std::for_each(_eventList_.begin(), _eventList_.end(), [&]( auto& e){
      if( e.getIndices().bin == iBin){ _histogram_.binList[iBin].eventPtrList[index++] = &e; }
    });

    iBin += nbThreads;
  }
}
void Sample::refillHistogram(int iThread_){
  int nThreads = GundamGlobals::getNumberOfThreads();
  if( iThread_ == -1 ){ nThreads = 1; iThread_ = 0; }

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

  // Faster than pointer shifter. -> would be slower if refillHistogram is
  // handled by the propagator
  int iBin = iThread_; // iBin += nbThreads;
  Histogram::Bin* binPtr;
  double buffer{};
  while( iBin < _histogram_.nBins ){
    bool binFilled = false;
    binPtr = &_histogram_.binList[iBin];
    binPtr->content = std::nan("not-set");
    binPtr->error = std::nan("not-set");
#ifdef GUNDAM_USING_CACHE_MANAGER
    bool filledWithManager = false;
    double value{std::nan("not-set")};
    double error{std::nan("not-set")};
    if (_CacheManagerValid_ and (*_CacheManagerValid_)
        and _CacheManagerValue_ and _CacheManagerIndex_ >= 0) {
      value = _CacheManagerValue_[_CacheManagerIndex_+binPtr->index];
      error = _CacheManagerValue2_[_CacheManagerIndex_+binPtr->index];
      LogThrowIf(std::isnan(value), "Incorrect Cache::Manager initialization");
      binPtr->content = value;
      binPtr->error = error;
      binFilled = not GundamGlobals::getForceDirectCalculation();
      filledWithManager = true;
    }
#endif
    if (not binFilled) {  // Will (should) optimize away w/o Cache::Manager
      binPtr->content = 0;
      binPtr->error = 0;
      for (auto *eventPtr: binPtr->eventPtrList) {
        buffer = eventPtr->getEventWeight();
        binPtr->content += buffer;
        binPtr->error += buffer * buffer;
      }
    }
    LogThrowIf(std::isnan((binPtr->content)), "NaN while filling histogram");
#ifdef GUNDAM_USING_CACHE_MANAGER
    // Parallel calculations of the histogramming have been run.  Make sure
    // they are the same.
    if (GundamGlobals::getForceDirectCalculation() and filledWithManager) {
      bool problemFound = false;
      if (not GundamUtils::almostEqual(value,(binPtr->content))) {
        double magnitude = std::abs(value) + std::abs(binPtr->content);
        double delta = std::abs(value - binPtr->content);
        if (magnitude > 0.0) delta /= 0.5*magnitude;
        LogError << "Incorrect histogram content --"
                 << " Content: " << value << "!=" << binPtr->content
                 << " Error: " << error << "!=" << binPtr->error
                 << " Precision: " << delta
                 << std::endl;
        problemFound = true;
      }
      if (not GundamUtils::almostEqual(error,(binPtr->error))) {
        double magnitude = std::abs(error) + std::abs(binPtr->error);
        double delta = std::abs(error - binPtr->error);
        if (magnitude > 0.0) delta /= 0.5*magnitude;
        LogError << "Incorrect histogram error --"
                 << " Content: " << value << "!=" << binPtr->content
                 << " Error: " << error << "!=" << binPtr->error
                 << " Precision: " << delta
                 << std::endl;
        problemFound = true;
      }
      if (false and problemFound) std::exit(EXIT_FAILURE); // For debugging
    }
#endif
    // We don't use TH1D anymore.  TH1 tracks the variance, while we track the
    // standard deviation (i.e. sqrt(variance)).  This changed from older
    // versions.
    binPtr->error = std::sqrt(binPtr->error); // YIKES!  NOTICE THIS!
    iBin += nThreads;
  }

}

void Sample::throwEventMcError(){
  // Take into account the finite number of events
  double weightSum;
  for( auto& bin : _histogram_.binList ){
    weightSum = 0;
    for (auto *eventPtr: bin.eventPtrList) {
      // gRandom->Poisson(1) -> returns an INT -> can be 0
      eventPtr->getWeights().current = (gRandom->Poisson(1) * eventPtr->getEventWeight());
      weightSum += eventPtr->getEventWeight();
    }
    bin.content = weightSum;
  }
}
void Sample::throwStatError(bool useGaussThrow_){
  /*
   * This is to convert "Asimov" histogram to toy-experiment (pseudo-data), i.e. with statistical fluctuations
   * */
  int nCounts;
  for( auto& bin : _histogram_.binList ){
    if( bin.content == 0 ){ continue; }
    if( not useGaussThrow_ ){
      nCounts = gRandom->Poisson( bin.content );
    }
    else{
      nCounts = std::max(
          int( gRandom->Gaus(bin.content, TMath::Sqrt(bin.content)) )
          , 0 // if the throw is negative, cap it to 0
      );
    }
    for (auto *eventPtr: bin.eventPtrList) {
      // make sure refill of the histogram will produce the same hist
      eventPtr->getWeights().current = ( eventPtr->getEventWeight()*((double) nCounts / bin.content) );
    }
    bin.content = nCounts;
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
  if( _histogram_.nBins == 0 ){ return out; }
  out = std::make_shared<TH1D>(
      Form("%s_bins", _name_.c_str()), Form("%s_bins", _name_.c_str()),
      _histogram_.nBins, 0, _histogram_.nBins
  );
  return out;
}

void Sample::printConfiguration() const{

  LogInfo << "#" << _index_;
  LogInfo << ", name(" << _name_ << ")";
  LogInfo << ", nBins(" << _histogram_.binList.size() << ")";
  LogInfo << std::endl;

}
[[nodiscard]] std::string Sample::getSummary() const{
  std::stringstream ss;
  ss << "Sample: " << _name_ << std::endl;
  ss << " - " << "Nb bins: " << _histogram_.binList.size() << std::endl;
  ss << " - " << "Nb events: " << _eventList_.size();
  return ss.str();
}
std::ostream& operator <<( std::ostream& o, const Sample& this_ ){
  o << this_.getSummary(); return o;
}

