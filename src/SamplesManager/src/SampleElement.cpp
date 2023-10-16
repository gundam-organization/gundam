//
// Created by Adrien BLANCHET on 30/07/2021.
//

#include "GundamGlobals.h"
#include "SampleElement.h"

#include "Logger.h"
#include "GenericToolbox.h"

#include "TRandom.h"


LoggerInit([]{ Logger::setUserHeaderStr("[SampleElement]"); });

void SampleElement::reserveEventMemory(size_t dataSetIndex_, size_t nEvents, const PhysicsEvent &eventBuffer_) {
  LogScopeIndent;
  LogThrowIf(isLocked, "Can't " << __METHOD_NAME__ << " while locked");
  if( nEvents == 0 ){ return; }
  dataSetIndexList.emplace_back(dataSetIndex_);
  eventOffSetList.emplace_back(eventList.size());
  eventNbList.emplace_back(nEvents);
  LogInfo << name << ": creating " << eventNbList.back() << " events ("
  << GenericToolbox::parseSizeUnits( double(eventNbList.back()) * sizeof(eventBuffer_) )
  << ")" << std::endl;
  eventList.resize(eventOffSetList.back()+eventNbList.back(), eventBuffer_);
}
void SampleElement::shrinkEventList(size_t newTotalSize_){
  LogScopeIndent;
  LogThrowIf(isLocked, "Can't " << __METHOD_NAME__ << " while locked");
  if( eventNbList.empty() and newTotalSize_ == 0 ) return;
  LogThrowIf(eventList.size() < newTotalSize_,
             "Can't shrink since eventList is too small: " << GET_VAR_NAME_VALUE(newTotalSize_)
             << " > " << GET_VAR_NAME_VALUE(eventList.size()));
  LogThrowIf(not eventNbList.empty() and eventNbList.back() < (eventList.size() - newTotalSize_), "Can't shrink since eventList of the last dataSet is too small.");
  LogInfo << name << ": shrinking event list from " << eventList.size() << " to " << newTotalSize_ << "..."
  << "(+" << GenericToolbox::parseSizeUnits( double(eventList.size() - newTotalSize_) * sizeof(eventList.back()) ) << ")" << std::endl;
  eventNbList.back() -= (eventList.size() - newTotalSize_);
  eventList.resize(newTotalSize_);
  eventList.shrink_to_fit();
}
void SampleElement::updateEventBinIndexes(int iThread_){
  if( isLocked ) return;

  int nThreads{GundamGlobals::getParallelWorker().getNbThreads()};
  if( iThread_ == -1 ){ iThread_ = 0; nThreads = 1; }

  PhysicsEvent* eventPtr{nullptr};
  auto bounds = GenericToolbox::ParallelWorker::getThreadBoundIndices( iThread_, nThreads, int( eventList.size() ) );

  if( iThread_ == 0 ){ LogScopeIndent; LogInfo << "Finding bin indexes for \"" << name << "\"..." << std::endl; }

  for( int iEvent = bounds.first ; iEvent < bounds.second ; iEvent++ ){
    eventPtr = &eventList[iEvent];
    for( auto& bin : binning.getBinsList() ){
      bool isInBin = std::all_of(bin.getEdgesList().begin(), bin.getEdgesList().end(), [&](const DataBin::Edges& e){
        return bin.isBetweenEdges( e, eventPtr->getVarAsDouble( e.varName ) );
      });

      if( isInBin ){
        eventPtr->setSampleBinIndex( bin.getIndex() );
        break;
      }
    }
  }
}
void SampleElement::updateBinEventList(int iThread_) {
  if( isLocked ) return;

  if( iThread_ <= 0 ){ LogScopeIndent; LogInfo << "Filling bin event cache for \"" << name << "\"..." << std::endl; }
  int nBins = int(perBinEventPtrList.size());
  int nbThreads = GundamGlobals::getParallelWorker().getNbThreads();
  if( iThread_ == -1 ){
    nbThreads = 1;
    iThread_ = 0;
  }

  int iBin = iThread_;
  size_t count;
  while( iBin < nBins ){
    count = std::count_if(eventList.begin(), eventList.end(), [&](auto& e) {return e.getSampleBinIndex() == iBin;});
    perBinEventPtrList[iBin].resize(count, nullptr);

    // Now filling the event indexes
    size_t index = 0;
    std::for_each(eventList.begin(), eventList.end(), [&](auto& e){ if(e.getSampleBinIndex() == iBin){ perBinEventPtrList[iBin][index++] = &e; } });

    iBin += nbThreads;
  }
}
void SampleElement::refillHistogram(int iThread_){
  if( isLocked ) return;

  int nbThreads = GundamGlobals::getParallelWorker().getNbThreads();
  if( iThread_ == -1 ){ nbThreads = 1; iThread_ = 0; }

#ifdef GUNDAM_USING_CACHE_MANAGER
  if (_CacheManagerValid_ and not (*_CacheManagerValid_)) {
      // This is can be slowish when data must be copied from the device, but
      // it makes sure that the results are copied from the device when they
      // have changed. The values pointed to by _CacheManagerValue_ and
      // _CacheManagerValid_ are inside of the summed index cache (a bit of
      // evil coding here), and are updated by the cache.  The update is
      // triggered by (*_CacheManagerUpdate_)().
      if (_CacheManagerUpdate_) (*_CacheManagerUpdate_)();
  }
#endif

  // Faster that pointer shifter. -> would be slower if refillHistogram is
  // handled by the propagator
  int iBin = iThread_;
  int nBins = int(perBinEventPtrList.size());
  auto* binContentArrayPtr = histogram->GetArray();
  auto* binErrorArrayPtr = histogram->GetSumw2()->GetArray();

  double* binContentPtr{nullptr};
  double* binErrorPtr{nullptr};

  while( iBin < nBins ) {
    binContentPtr = &binContentArrayPtr[iBin+1];
    binErrorPtr = &binErrorArrayPtr[iBin+1];

    (*binContentPtr) = 0;
    (*binErrorPtr) = 0;
#ifdef GUNDAM_USING_CACHE_MANAGER
    if (_CacheManagerValue_ !=nullptr and _CacheManagerIndex_ >= 0) {
      const double ew = _CacheManagerValue_[_CacheManagerIndex_+iBin];
      const double ew2 = _CacheManagerValue2_[_CacheManagerIndex_+iBin];
      (*binContentPtr) += ew;
      (*binErrorPtr) += ew2;
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
#endif // GUNDAM_USING_CACHE_MANAGER
      for (auto *eventPtr: perBinEventPtrList[iBin]) {
        (*binContentPtr) += eventPtr->getEventWeight();
        (*binErrorPtr) += eventPtr->getEventWeight() * eventPtr->getEventWeight();
      }
      LogThrowIf(std::isnan((*binContentPtr)));
#ifdef GUNDAM_USING_CACHE_MANAGER
    }
#endif // GUNDAM_USING_CACHE_MANAGER
    iBin += nbThreads;
  }

}
void SampleElement::rescaleHistogram() {
  if( isLocked ) return;
  if( histScale != 1 ) histogram->Scale(histScale);
}
void SampleElement::saveAsHistogramNominal(){
  histogramNominal = std::make_shared<TH1D>(*histogram);
}

void SampleElement::throwEventMcError(){
  /*
 * This is to take into account the finite amount of event
 * */
  double weightSum;
  for( int iBin = 1 ; iBin <= histogram->GetNbinsX() ; iBin++ ){
    weightSum = 0;
    for (auto *eventPtr: perBinEventPtrList[iBin-1]) {
      // gRandom->Poisson(1) -> returns an INT -> can be 0
      eventPtr->setEventWeight(gRandom->Poisson(1) * eventPtr->getEventWeight());
      weightSum += eventPtr->getEventWeight();
    }
    histogram->SetBinContent(iBin, weightSum);
  }
}
void SampleElement::throwStatError(bool useGaussThrow_){
  /*
   * This is to convert "Asimov" histogram to toy-experiment (pseudo-data), i.e. with statistical fluctuations
   * */
  int nCounts;
  for( int iBin = 1 ; iBin <= histogram->GetNbinsX() ; iBin++ ){
    if( histogram->GetBinContent(iBin) != 0 ){
      if( not useGaussThrow_ ){
        nCounts = gRandom->Poisson(histogram->GetBinContent(iBin));
      }
      else{
        nCounts = std::max(
            int( gRandom->Gaus(histogram->GetBinContent(iBin), TMath::Sqrt(histogram->GetBinContent(iBin))) )
            , 0 // if the throw is negative, cap it to 0
            );
      }
      for (auto *eventPtr: perBinEventPtrList[iBin-1]) {
        // make sure refill of the histogram will produce the same hist
        eventPtr->setEventWeight( eventPtr->getEventWeight()*( (double) nCounts/histogram->GetBinContent(iBin)) );
      }
      histogram->SetBinContent(iBin, nCounts);
    }
  }
}

double SampleElement::getSumWeights() const{
  double output = std::accumulate(eventList.begin(), eventList.end(), double(0.),
                                  [](double sum_, const PhysicsEvent& ev_){ return sum_ + ev_.getEventWeight(); });

  if( std::isnan(output) ){
    for( auto& event : eventList ){
      if( std::isnan(event.getEventWeight()) ){
        event.print();
      }
    }
    LogThrow("NAN getSumWeights");
  }

  return output;
}
size_t SampleElement::getNbBinnedEvents() const{
  return std::accumulate(eventList.begin(), eventList.end(), size_t(0.),
                         [](size_t sum_, const PhysicsEvent& ev_){ return sum_ + (ev_.getSampleBinIndex() != -1); });
}

void SampleElement::print() const{
  LogInfo << "SampleElement: " << name << std::endl;
  LogInfo << " - " << "Nb bins: " << binning.getBinsList().size() << std::endl;
  LogInfo << " - " << "Nb events: " << eventList.size() << std::endl;
  LogInfo << " - " << "Hist rescale: " << histScale << std::endl;
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
