//
// Created by Adrien BLANCHET on 30/07/2021.
//

#include "GundamGlobals.h"
#include "SampleElement.h"

#include "Logger.h"

#include "TRandom.h"


LoggerInit([]{ Logger::setUserHeaderStr("[SampleElement]"); });


void SampleElement::buildHistogram(const DataBinSet& binning_){
  _myhistogram_.binList.reserve( binning_.getBinList().size() );
  int iBin{0};
  for( auto& bin : binning_.getBinList() ){
    _myhistogram_.binList.emplace_back();
    _myhistogram_.binList.back().dataBinPtr = &bin;
    _myhistogram_.binList.back().index = iBin++;
  }
  _myhistogram_.nBins = int( _myhistogram_.binList.size() );
}
void SampleElement::reserveEventMemory(size_t dataSetIndex_, size_t nEvents, const PhysicsEvent &eventBuffer_) {
  // adding one dataset:
  _loadedDatasetList_.emplace_back();

  // filling up properties:
  auto& datasetProperties{_loadedDatasetList_.back()};
  datasetProperties.dataSetIndex = dataSetIndex_;
  datasetProperties.eventOffSet = _eventList_.size();
  datasetProperties.eventNb = nEvents;

  LogScopeIndent;
  LogInfo << _name_ << ": creating " << nEvents << " events ("
          << GenericToolbox::parseSizeUnits( double(nEvents) * sizeof(eventBuffer_) )
          << ")" << std::endl;

  _eventList_.resize(datasetProperties.eventOffSet + datasetProperties.eventNb, eventBuffer_);
}
void SampleElement::shrinkEventList(size_t newTotalSize_){

  if( _loadedDatasetList_.empty() and newTotalSize_ == 0 ){
    LogAlert << "Empty dataset list. Nothing to shrink." << std::endl;
    return;
  }

  LogThrowIf(_eventList_.size() < newTotalSize_,
             "Can't shrink since eventList is too small: " << GET_VAR_NAME_VALUE(newTotalSize_)
             << " > " << GET_VAR_NAME_VALUE(_eventList_.size()));

  LogThrowIf(not _loadedDatasetList_.empty() and _loadedDatasetList_.back().eventNb < (_eventList_.size() - newTotalSize_),
              "Can't shrink since eventList of the last dataSet is too small.");

  LogScopeIndent;
  LogInfo << _name_ << ": shrinking event list from " << _eventList_.size() << " to " << newTotalSize_ << "..."
          << "(+" << GenericToolbox::parseSizeUnits(double(_eventList_.size() - newTotalSize_) * sizeof(_eventList_.back()) ) << ")" << std::endl;

  _loadedDatasetList_.back().eventNb -= (_eventList_.size() - newTotalSize_);
  _eventList_.resize(newTotalSize_);
  _eventList_.shrink_to_fit();
}
void SampleElement::updateBinEventList(int iThread_) {
  int nbThreads = GundamGlobals::getParallelWorker().getNbThreads();
  if( iThread_ == -1 ){ iThread_ = 0; nbThreads = 1; }

  if( iThread_ == 0 ){ LogScopeIndent; LogInfo << "Filling bin event cache for \"" << _name_ << "\"..." << std::endl; }

  // multithread technique with iBin += nbThreads;
  int iBin{iThread_};
  while( iBin < _myhistogram_.nBins ){
    size_t count = std::count_if(_eventList_.begin(), _eventList_.end(), [&]( auto& e) {return e.getSampleBinIndex() == iBin;});
    _myhistogram_.binList[iBin].eventPtrList.resize(count, nullptr);

    // Now filling the event indexes
    size_t index = 0;
    std::for_each(_eventList_.begin(), _eventList_.end(), [&]( auto& e){
      if( e.getSampleBinIndex() == iBin){ _myhistogram_.binList[iBin].eventPtrList[index++] = &e; }
    });

    iBin += nbThreads;
  }
}
void SampleElement::refillHistogram(int iThread_){
  int nThreads = GundamGlobals::getParallelWorker().getNbThreads();
  if( iThread_ == -1 ){ nThreads = 1; iThread_ = 0; }

#ifdef GUNDAM_USING_CACHE_MANAGER
  if (_CacheManagerValid_ and not (*_CacheManagerValid_)) {
      // This can be slowish when data must be copied from the device, but
      // it makes sure that the results are copied from the device when they
      // have changed. The values pointed to by _CacheManagerValue_ and
      // _CacheManagerValid_ are inside the summed index cache (a bit of
      // evil coding here), and are updated by the cache.  The update is
      // triggered by (*_CacheManagerUpdate_)().
      if (_CacheManagerUpdate_) (*_CacheManagerUpdate_)();
  }
#endif

  // Faster that pointer shifter. -> would be slower if refillHistogram is
  // handled by the propagator
  int iBin = iThread_; // iBin += nbThreads;
  Histogram::Bin* binPtr;
  while( iBin < _myhistogram_.nBins ){
    binPtr = &_myhistogram_.binList[iBin];
    binPtr->content = 0;
    binPtr->error = 0;
#ifdef GUNDAM_USING_CACHE_MANAGER
    if (_CacheManagerValue_ !=nullptr and _CacheManagerIndex_ >= 0) {
      const double ew = _CacheManagerValue_[_CacheManagerIndex_+binPtr->index];
      const double ew2 = _CacheManagerValue2_[_CacheManagerIndex_+binPtr->index];
      binPtr->content += ew;
      binPtr->error += ew2;
#ifdef CACHE_MANAGER_SLOW_VALIDATION
      double content = binContentArray[iBin+1];
      double slowValue = 0.0;
      for( auto* eventPtr : perBinEventPtrList.at(iBin)){
        slowValue += eventPtr->getEventWeight();
      }
      double delta = std::abs(slowValue-content);
      if (delta > 1E-6) {
        LogInfo << "VALIDATION: Mismatched bin: " << _CacheManagerIndex_
                << "+" << iBin
                << "(" << name
                << ") gpu: " << content
                << " PhysEvt: " << slowValue
                << " delta: " << delta
                << std::endl;
      }
#endif // CACHE_MANAGER_SLOW_VALIDATION
    }
    else {
#endif
      for (auto *eventPtr: binPtr->eventPtrList) {
        binPtr->content += eventPtr->getEventWeight();
        binPtr->error += eventPtr->getEventWeight() * eventPtr->getEventWeight();
      }
#ifdef GUNDAM_USING_CACHE_MANAGER
    }
#endif // GUNDAM_USING_CACHE_MANAGER
    binPtr->error = sqrt(binPtr->error);
    iBin += nThreads;
  }

}

void SampleElement::throwEventMcError(){
  /*
   * This is to take into account the finite amount of event
   * */
  double weightSum;
  for( auto& bin : _myhistogram_.binList ){
    weightSum = 0;
    for (auto *eventPtr: bin.eventPtrList) {
      // gRandom->Poisson(1) -> returns an INT -> can be 0
      eventPtr->setEventWeight(gRandom->Poisson(1) * eventPtr->getEventWeight());
      weightSum += eventPtr->getEventWeight();
    }
    bin.content = weightSum;
  }
}
void SampleElement::throwStatError(bool useGaussThrow_){
  /*
   * This is to convert "Asimov" histogram to toy-experiment (pseudo-data), i.e. with statistical fluctuations
   * */
  int nCounts;
  for( auto& bin : _myhistogram_.binList ){
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
      eventPtr->setEventWeight( eventPtr->getEventWeight()*((double) nCounts / bin.content) );
    }
    bin.content = nCounts;
  }
}

double SampleElement::getSumWeights() const{
  double output = std::accumulate(_eventList_.begin(), _eventList_.end(), double(0.),
                                  [](double sum_, const PhysicsEvent& ev_){ return sum_ + ev_.getEventWeight(); });

  if( std::isnan(output) ){
    for( auto& event : _eventList_ ){
      if( std::isnan(event.getEventWeight()) ){
        event.print();
      }
    }
    LogThrow("NAN getSumWeights");
  }

  return output;
}
size_t SampleElement::getNbBinnedEvents() const{
  return std::accumulate(_eventList_.begin(), _eventList_.end(), size_t(0.),
                         []( size_t sum_, const PhysicsEvent &ev_ ){ return sum_ + (ev_.getSampleBinIndex() != -1); });
}
std::shared_ptr<TH1D> SampleElement::generateRootHistogram() const{
  std::shared_ptr<TH1D> out{nullptr};
  if( _myhistogram_.nBins == 0 ){ return out; }
  out = std::make_shared<TH1D>(
      Form("%s_bins", _name_.c_str()), Form("%s_bins", _name_.c_str()),
      _myhistogram_.nBins, 0, _myhistogram_.nBins
  );
  return out;
}

[[nodiscard]] std::string SampleElement::getSummary() const{
  std::stringstream ss;
  ss << "SampleElement: " << _name_ << std::endl;
  ss << " - " << "Nb bins: " << _myhistogram_.binList.size() << std::endl;
  ss << " - " << "Nb events: " << _eventList_.size();
  return ss.str();
}
std::ostream& operator <<( std::ostream& o, const SampleElement& this_ ){
  o << this_.getSummary(); return o;
}

//  A Lesser GNU Public License

//  Copyright (C) 2023 GUNDAM DEVELOPERS

//  This library is free software; you can redistribute it and/or
//  modify it under the terms of the GNU Lesser General Public
//  License as published by the Free Software Foundation; either
//  version 2.1 of the License, or (at your option) any later version.

//  This library is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
//  Lesser General Public License for more details.

//  You should have received a copy of the GNU Lesser General Public
//  License along with this library; if not, write to the
//
//  Free Software Foundation, Inc.
//  51 Franklin Street, Fifth Floor,
//  Boston, MA  02110-1301  USA

// Local Variables:
// mode:c++
// c-basic-offset:2
// compile-command:"$(git rev-parse --show-toplevel)/cmake/gundam-build.sh"
// End:
