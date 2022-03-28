//
// Created by Adrien BLANCHET on 30/07/2021.
//

#include "Logger.h"
#include "GenericToolbox.h"

#include "GlobalVariables.h"
#include "SampleElement.h"

#include "CacheManager.h"

LoggerInit([](){
  Logger::setUserHeaderStr("[SampleElement]");
})


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
  LogDebug << "Shrinking " << eventList.size() << " to " << newTotalSize_ << "..." << std::endl;
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
    count = 0;
    for( auto& event : eventList ){ if( event.getSampleBinIndex() == iBin ){ count++; } }
    std::vector<PhysicsEvent*> thisBinEventList(count, nullptr);

    // Now filling the event indexes
    size_t index = 0;
    for( auto& event : eventList ){
      if( event.getSampleBinIndex() == iBin ){
        thisBinEventList.at(index++) = &event;
      }
    } // event

    GlobalVariables::getThreadMutex().lock();
    // BETTER TO MAKE SURE THE MEMORY IS NOT MOVED WHILE FILLING UP
    perBinEventPtrList.at(iBin) = thisBinEventList;
    GlobalVariables::getThreadMutex().unlock();

    iBin += nbThreads;
  }
}
void SampleElement::refillHistogram(int iThread_){
  if( isLocked ) return;

  int nbThreads = GlobalVariables::getNbThreads();
  if( iThread_ == -1 ){
    nbThreads = 1;
    iThread_ = 0;
  }

  // Faster that pointer shifter. -> would be slower if refillHistogram is
  // handled by the propagator

  int histIndex = -1;
#ifdef GUNDAM_USING_CUDA
  Cache::Manager* cache = Cache::Manager::Get();
  if (cache) histIndex = getCacheManagerIndex();
#endif

  // Size = Nbins + 2 overflow (0 and last)
  auto* binContentArray = histogram->GetArray();

  int iBin = iThread_;
  int nBins = int(perBinEventPtrList.size());
  while( iBin < nBins ){
    double content = 0.0;
#ifdef GUNDAM_USING_CUDA
    if (0 <= histIndex) {
        content = cache->GetHistogramsCache().GetSum(histIndex+iBin);
    }
#ifdef CACHE_MANAGER_SLOW_VALIDATION
    {
        double slowValue = 0.0;
        for( auto* eventPtr : perBinEventPtrList.at(iBin)){
            slowValue += eventPtr->getEventWeight();
        }
        double delta = std::abs(slowValue-content);
        if (delta > 1E-6) {
            LogInfo << "Bin mismatch " << histIndex
                    << " " << iBin
                    << " " << slowValue
                    << " " << content
                    << " " << delta
                    << std::endl;
        }
    }
#endif
#else
    for( auto* eventPtr : perBinEventPtrList.at(iBin)){
        content += eventPtr->getEventWeight();
    }
#endif
    binContentArray[iBin+1] = content;
    histogram->GetSumw2()->GetArray()[iBin+1] = content;
    double newContent = content;
    iBin += nbThreads;
  }
}
void SampleElement::rescaleHistogram() {
  if( isLocked ) return;
  if(histScale != 1) histogram->Scale(histScale);
}

double SampleElement::getSumWeights() const{
  return std::accumulate(eventList.begin(), eventList.end(), double(0.),
                         [](double sum_, const PhysicsEvent& ev_){ return sum_ + ev_.getEventWeight(); });
}

int SampleElement::getCacheManagerIndex() const {return cacheManagerIndex;}
void SampleElement::setCacheManagerIndex(int i) {cacheManagerIndex = i;}

void SampleElement::print() const{
  LogInfo << "SampleElement: " << name << std::endl;
  LogInfo << " - " << "Nb bins: " << binning.getBinsList().size() << std::endl;
  LogInfo << " - " << "Nb events: " << eventList.size() << std::endl;
  LogInfo << " - " << "Hist rescale: " << histScale << std::endl;
}
