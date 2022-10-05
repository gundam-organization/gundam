//
// Created by Adrien BLANCHET on 30/07/2021.
//

#include "GlobalVariables.h"
#include "SampleElement.h"

#include "Logger.h"
#include "GenericToolbox.h"

#include "TRandom.h"


LoggerInit([]{ Logger::setUserHeaderStr("[SampleElement]"); });

SampleElement::SampleElement() = default;
SampleElement::~SampleElement() = default;

void SampleElement::reserveEventMemory(size_t dataSetIndex_, size_t nEvents, const PhysicsEvent &eventBuffer_) {
  LogThrowIf(isLocked, "Can't " << __METHOD_NAME__ << " while locked");
  if( nEvents == 0 ){ return; }
  dataSetIndexList.emplace_back(dataSetIndex_);
  eventOffSetList.emplace_back(eventList.size());
  eventNbList.emplace_back(nEvents);
  eventList.resize(eventOffSetList.back()+eventNbList.back(), eventBuffer_);
}
void SampleElement::shrinkEventList(size_t newTotalSize_){
  LogThrowIf(isLocked, "Can't " << __METHOD_NAME__ << " while locked");
  if( eventNbList.empty() and newTotalSize_ == 0 ) return;
  LogThrowIf(eventList.size() < newTotalSize_,
             "Can't shrink since eventList is too small: " << GET_VAR_NAME_VALUE(newTotalSize_)
             << " > " << GET_VAR_NAME_VALUE(eventList.size()));
  LogThrowIf(not eventNbList.empty() and eventNbList.back() < (eventList.size() - newTotalSize_), "Can't shrink since eventList of the last dataSet is too small.");
  LogInfo << "-> Shrinking " << eventList.size() << " to " << newTotalSize_ << "..." << std::endl;
  eventNbList.back() -= (eventList.size() - newTotalSize_);
  eventList.resize(newTotalSize_);
  eventList.shrink_to_fit();
}
void SampleElement::updateEventBinIndexes(int iThread_){
  if( isLocked ) return;
  int nBins = int(binning.getBinsList().size());
  if(iThread_ <= 0) LogInfo << "Finding bin indexes for \"" << name << "\"..." << std::endl;
  int toDelete = 0;
  for( size_t iEvent = 0 ; iEvent < eventList.size() ; iEvent++ ){
    if( iThread_ != -1 and iEvent % GlobalVariables::getNbThreads() != iThread_ ) continue;
    auto& event = eventList.at(iEvent);
    for( int iBin = 0 ; iBin < nBins ; iBin++ ){
      auto& bin = binning.getBinsList().at(iBin);
      bool isInBin = true;
      for( size_t iVar = 0 ; iVar < bin.getVariableNameList().size() ; iVar++ ){
        if( not bin.isBetweenEdges(iVar, event.getVarAsDouble(bin.getVariableNameList().at(iVar))) ){
          isInBin = false;
          break;
        }
      } // Var
      if( isInBin ){
        event.setSampleBinIndex(iBin);
        break;
      }
    } // Bin

    if( event.getSampleBinIndex() == -1 ){
      toDelete++;
    }

  } // Event

//  LogTrace << iThread_ << " -> unbinned events: " << toDelete << std::endl;
}
void SampleElement::updateBinEventList(int iThread_) {
  if( isLocked ) return;

  if(iThread_ <= 0) LogInfo << "Filling bin event cache for \"" << name << "\"..." << std::endl;
  int nBins = int(perBinEventPtrList.size());
  int nbThreads = GlobalVariables::getNbThreads();
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

  int nbThreads = GlobalVariables::getNbThreads();
  if( iThread_ == -1 ){ nbThreads = 1; iThread_ = 0; }

  // Faster that pointer shifter. -> would be slower if refillHistogram is
  // handled by the propagator
  int iBin = iThread_;
  int nBins = int(perBinEventPtrList.size());
  auto* binContentArray = histogram->GetArray();
  auto* binErrorArray = histogram->GetSumw2()->GetArray();
  while( iBin < nBins ) {
    binContentArray[iBin + 1] = 0; 
    binErrorArray[iBin + 1] = 0;
#ifdef GUNDAM_USING_CACHE_MANAGER
    if (_CacheManagerValue_!=nullptr and _CacheManagerIndex_ >= 0) {
      binContentArray[iBin + 1] += _CacheManagerValue_[_CacheManagerIndex_+iBin];
      binErrorArray[iBin + 1] += binContentArray[iBin + 1]*binContentArray[iBin + 1];
#ifdef CACHE_MANAGER_SLOW_VALIDATION
      double slowValue = 0.0;
        for( auto* eventPtr : perBinEventPtrList.at(iBin)){
            slowValue += eventPtr->getEventWeight();
        }
        double delta = std::abs(slowValue-content);
        if (delta > 1E-6) {
            LogInfo << "VALIDATION: Bin mismatch " << _CacheManagerIndex_
                    << " " << iBin
                    << " " << name
                    << " " << slowValue
                    << " " << content
                    << " " << delta
                    << std::endl;
        }
#endif // CACHE_MANAGER_SLOW_VALIDATION
    }
    else {
//      LogThrow(GET_VAR_NAME_VALUE(_CacheManagerValue_) << " / " << GET_VAR_NAME_VALUE(_CacheManagerIndex_)); // debug
#endif // GUNDAM_USING_CACHE_MANAGER
      for (auto *eventPtr: perBinEventPtrList[iBin]) {
        binContentArray[iBin + 1] += eventPtr->getEventWeight();
        binErrorArray[iBin + 1] += eventPtr->getEventWeight() * eventPtr->getEventWeight();
      }
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
    gRandom->Poisson(histogram->GetBinContent(iBin));
    for (auto *eventPtr: perBinEventPtrList[iBin-1]) {
      // gRandom->Poisson(1) -> returns an INT -> can be 0
      eventPtr->setEventWeight(gRandom->Poisson(1) * eventPtr->getEventWeight());
      weightSum += eventPtr->getEventWeight();
    }
    histogram->SetBinContent(iBin, weightSum);
  }
}
void SampleElement::throwStatError(){
  /*
   * This is to convert "Asimov" histogram to toy-experiment (pseudo-data), i.e. with statistical fluctuations
   * */
  int nCounts;
  for( int iBin = 1 ; iBin <= histogram->GetNbinsX() ; iBin++ ){
    nCounts = gRandom->Poisson(histogram->GetBinContent(iBin));
    for (auto *eventPtr: perBinEventPtrList[iBin-1]) {
      // make sure refill of the histogram will produce the same hist
      eventPtr->setEventWeight(eventPtr->getEventWeight()*((double)nCounts/histogram->GetBinContent(iBin)));
    }
    histogram->SetBinContent(iBin, nCounts);
  }
}

double SampleElement::getSumWeights() const{
  return std::accumulate(eventList.begin(), eventList.end(), double(0.),
                         [](double sum_, const PhysicsEvent& ev_){ return sum_ + ev_.getEventWeight(); });
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
