//
// Created by Adrien BLANCHET on 30/07/2021.
//

#include "Logger.h"

#include "GlobalVariables.h"
#include "SampleElement.h"

LoggerInit([](){
  Logger::setUserHeaderStr("[SampleElement]");
})


void SampleElement::reserveEventMemory(size_t dataSetIndex_, size_t nEvents, const PhysicsEvent &eventBuffer_) {
  LogThrowIf(isLocked, "Can't " << __METHOD_NAME__ << " while locked");
  if( nEvents == 0 ){ return; }
  dataSetIndexList.emplace_back(dataSetIndex_);
  eventOffSetList.emplace_back(eventList.size());
  eventNbList.emplace_back(nEvents);
  eventList.resize(eventOffSetList.back()+eventNbList.back(), eventBuffer_);
}
void SampleElement::updateEventBinIndexes(int iThread_){
  if( isLocked ) return;
  std::string progressTitle = LogInfo.getPrefixString() + "Finding bin index for each event...";
  int nBins = int(binning.getBinsList().size());
  for( size_t iEvent = 0 ; iEvent < eventList.size() ; iEvent++ ){
    if( iThread_ != -1 and iEvent % GlobalVariables::getNbThreads() != iThread_ ) continue;
    if( iThread_ <= 0 ) GenericToolbox::displayProgressBar(iEvent, eventList.size(), progressTitle);
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
  } // Event
}
void SampleElement::updateBinEventList(int iThread_) {
  if( isLocked ) return;
  std::string progressTitle = LogInfo.getPrefixString() + "Filling bin event cache...";
  int nBins = int(perBinEventPtrList.size());
  for( int iBin = 0 ; iBin < nBins ; iBin++ ){
    if( iThread_ != -1 and iBin % GlobalVariables::getNbThreads() != iThread_ ) continue;
    if( iThread_ <= 0 ) GenericToolbox::displayProgressBar(iBin, nBins, progressTitle);
    auto& thisBinEventList = perBinEventPtrList.at(iBin);
    // Counting first (otherwise the memory allocation will keep moving data around)
    size_t count = 0;
    for( auto& event : eventList ){ if( event.getSampleBinIndex() == iBin ){ count++; } }
    thisBinEventList.resize(count); // allocate once
    // Now filling the event indexes
    size_t index = 0;
    for( auto& event : eventList ){
      if( event.getSampleBinIndex() == iBin ){
        thisBinEventList.at(index++) = &event;
      }
    } // event
  } // hist bin
}
void SampleElement::refillHistogram(int iThread_){
  if( isLocked ) return;
//  histogram->Reset();
  int nBins = int(perBinEventPtrList.size());
  for(int iBin = 0 ; iBin < nBins ; iBin++){
    if( iThread_ != -1 and iBin % GlobalVariables::getNbThreads() != iThread_ ) continue;
    histogram->SetBinContent(iBin+1, 0);
    for( auto* eventPtr : perBinEventPtrList.at(iBin)){
      histogram->AddBinContent(iBin+1, eventPtr->getEventWeight());
    }
  }
}
void SampleElement::rescaleHistogram() {
  if( isLocked ) return;
  if(histScale != 1) histogram->Scale(histScale);
}

