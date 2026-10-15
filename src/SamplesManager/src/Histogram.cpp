//
// Created by Nadrino on 19/10/2024.
//

#include "Histogram.h"
#include "BinSet.h"
#include "GundamGlobals.h"
#include "GundamAlmostEqual.h"

#include "GenericToolbox.Thread.h"
#include "Logger.h"

#include "TRandom3.h"

#include <unordered_set>



void Histogram::build(const ConfigReader& binningConfig_){

  BinSet binning;
  binning.configure( binningConfig_ );

  _outsideBinContent_ = BinContent();
  _outsideContributionList_.clear();
  _responseWorkers_.clear();
  nBins = int( binning.getBinList().size() );
  binContentList.resize( nBins );
  binContextList.resize( nBins );

  // filling bin contexts
  for( int iBin = 0 ; iBin < nBins ; iBin++ ){
    binContextList[iBin].bin = binning.getBinList()[iBin];
  }

}
void Histogram::configureBinningResponse(const ConfigReader& config_){
  _binningResponse_.setConfig(config_);
  _binningResponse_.configure();
  std::vector<Bin> bins;
  bins.reserve(binContextList.size());
  for( const auto& context : binContextList ){ bins.emplace_back(context.bin); }
  _binningResponse_.build(bins);
}
BinningResponse::EventResponse Histogram::evaluateEventBinningResponse(const Event& event_) const{
  if( isBinningResponseEnabled() ){
    BinningResponse::EvaluationBuffer buffer;
    _binningResponse_.prepareEvaluation(event_.getVariables(), buffer);
    _binningResponse_.evaluate(event_.getVariables(), buffer);
    return std::move(buffer.response);
  }
  BinningResponse::EventResponse response;
  int index = findBinIndex(event_);
  if( index >= 0 ){
    BinningResponse::BinFraction fraction;
    fraction.binIndex = index;
    fraction.fraction = 1;
    response.binFractions.emplace_back(fraction);
    response.outsideFraction = 0;
  }
  return response;
}
std::vector<BinningResponse::BinFraction> Histogram::evaluateBinningResponse(const Event& event_) const{
  return evaluateEventBinningResponse(event_).binFractions;
}
std::vector<Histogram::EventContribution>& Histogram::getContributionList(int binIndex_){
  return binIndex_ == nBins ? _outsideContributionList_ : binContextList[binIndex_].eventContributionList;
}
void Histogram::fillBinContent(const std::vector<EventContribution>& contributions_, BinContent& content_){
  content_.sumWeights = 0;
  content_.sqrtSumSqWeights = 0;
  for( const auto& contribution : contributions_ ){
    double weight = contribution.eventPtr->getEventWeight() * contribution.fraction;
    content_.sumWeights += weight;
    content_.sqrtSumSqWeights += weight * weight;
  }
  content_.sqrtSumSqWeights = std::sqrt(content_.sqrtSumSqWeights);
}
int Histogram::findBinIndex(const Event& event_) const{
  for( const auto& binContext : binContextList ){
    if( event_.getVariables().isInBin(binContext.bin) ){
      return binContext.bin.getIndex();
    }
  }
  return -1;
}
void Histogram::throwEventMcError(){
  if( isBinningResponseEnabled() ){
    std::unordered_set<Event*> events;
    for( int iBin = 0 ; iBin <= nBins ; ++iBin ){
      for( const auto& contribution : getContributionList(iBin) ){
        if( events.insert(contribution.eventPtr).second ){
          contribution.eventPtr->getWeights().current *= double(gRandom->Poisson(1));
        }
      }
    }
    refillHistogram();
    return;
  }

  // event by event poisson throw -> takes into account the finite amount of stat in MC

#if HAS_CPP_17
  for( auto [binContent, binContext] : loop() ){
#else
  for( auto element : loop() ){ auto& binContent = std::get<0>(element); auto& binContext = std::get<1>(element);
#endif
    binContent.sumWeights = 0;
    binContent.sqrtSumSqWeights = 0;
    for (auto *eventPtr: binContext.eventPtrList) {
      // gRandom->Poisson(1) -> returns an INT -> can be 0
      eventPtr->getWeights().current = (double(gRandom->Poisson(1)) * eventPtr->getEventWeight());

      double weight{eventPtr->getEventWeight()};
      binContent.sumWeights += weight;
      binContent.sqrtSumSqWeights += weight * weight;
    }

    binContent.sqrtSumSqWeights = sqrt(binContent.sqrtSumSqWeights);
  }

}
void Histogram::throwStatError(bool useGaussThrow_){
  LogThrowIf(isBinningResponseEnabled(),
             "Bin-wise statistical throws cannot be represented by shared event weights with a binning response.");

  /*
   * This is to convert "Asimov" histogram to toy-experiment (pseudo-data), i.e. with statistical fluctuations
   * */
  double nCounts;

#if HAS_CPP_17
  for( auto [binContent, binContext] : loop() ){
#else
  for( auto element : loop() ){ auto& binContent = std::get<0>(element); auto& binContext = std::get<1>(element);
#endif
    if( binContent.sumWeights == 0 ){
      // this should not happen.
      continue;
    }

    if( not useGaussThrow_ ){
      nCounts = double( gRandom->Poisson( binContent.sumWeights ) );
    }
    else{
      nCounts = double( std::max(
          int( gRandom->Gaus(binContent.sumWeights, std::sqrt(binContent.sumWeights)) )
          , 0 // if the throw is negative, cap it to 0
      ) );
    }
    for (auto *eventPtr: binContext.eventPtrList) {
      // make sure refill of the histogram will produce the same hist
      if( binContent.sumWeights == 0 ){ eventPtr->getWeights().current = 0; }
      else{ eventPtr->getWeights().current *= (double) nCounts / binContent.sumWeights; }
    }
    binContent.sumWeights = nCounts;
  }
}

void Histogram::updateBinEventLists(std::vector<Event>& eventList_, GenericToolbox::ParallelWorker& threadPool_, bool updateBinIndices_){
  if( not isBinningResponseEnabled() ){
    if( updateBinIndices_ ){
      threadPool_.runJob([&](int iThread_){
        auto bounds = GenericToolbox::ParallelWorker::getThreadBoundIndices(iThread_, threadPool_.getNbThreads(), eventList_.size());
        for( size_t iEvent = bounds.beginIndex ; iEvent < bounds.endIndex ; ++iEvent ){
          eventList_[iEvent].getIndices().bin = findBinIndex(eventList_[iEvent]);
        }
      });
    }
    threadPool_.runJob([&](int iThread_){ updateBinEventList(eventList_, iThread_, threadPool_.getNbThreads()); });
    return;
  }
  if( eventList_.empty() ){
    _outsideContributionList_.clear();
    for( auto& context : binContextList ){
      context.eventPtrList.clear();
      context.eventContributionList.clear();
    }
    return;
  }

  _responseWorkers_.resize(threadPool_.getNbThreads());
  for( auto& worker : _responseWorkers_ ){
    _binningResponse_.prepareEvaluation(eventList_.front().getVariables(), worker.evaluationBuffer);
    worker.contributions.clear();
    worker.contributionOffsets.assign(nBins + 1, 0);
    worker.eventOffsets.assign(nBins, 0);
  }

  threadPool_.runJob([&](int iThread_){
    auto& worker = _responseWorkers_[iThread_];
    auto bounds = GenericToolbox::ParallelWorker::getThreadBoundIndices(iThread_, threadPool_.getNbThreads(), eventList_.size());
    for( size_t iEvent = bounds.beginIndex ; iEvent < bounds.endIndex ; ++iEvent ){
      auto& event = eventList_[iEvent];
      if( updateBinIndices_ ){ event.getIndices().bin = findBinIndex(event); }
      if( event.getIndices().bin >= 0 ){ ++worker.eventOffsets[event.getIndices().bin]; }
      _binningResponse_.evaluate(event.getVariables(), worker.evaluationBuffer);
      for( const auto& fraction : worker.evaluationBuffer.response.binFractions ){
        IndexedContribution contribution;
        contribution.binIndex = fraction.binIndex;
        contribution.contribution.eventPtr = &event;
        contribution.contribution.fraction = fraction.fraction;
        worker.contributions.emplace_back(contribution);
        ++worker.contributionOffsets[fraction.binIndex];
      }
      if( worker.evaluationBuffer.response.outsideFraction > 0 ){
        IndexedContribution contribution;
        contribution.binIndex = nBins;
        contribution.contribution.eventPtr = &event;
        contribution.contribution.fraction = worker.evaluationBuffer.response.outsideFraction;
        worker.contributions.emplace_back(contribution);
        ++worker.contributionOffsets[nBins];
      }
    }
  });

  for( int iBin = 0 ; iBin <= nBins ; ++iBin ){
    size_t nContributions{0};
    size_t nEvents{0};
    for( auto& worker : _responseWorkers_ ){
      size_t contributionCount = worker.contributionOffsets[iBin];
      worker.contributionOffsets[iBin] = nContributions;
      nContributions += contributionCount;
      if( iBin < nBins ){
        size_t eventCount = worker.eventOffsets[iBin];
        worker.eventOffsets[iBin] = nEvents;
        nEvents += eventCount;
      }
    }
    getContributionList(iBin).resize(nContributions);
    if( iBin < nBins ){ binContextList[iBin].eventPtrList.resize(nEvents); }
  }

  threadPool_.runJob([&](int iThread_){
    auto& worker = _responseWorkers_[iThread_];
    for( const auto& contribution : worker.contributions ){
      int iBin = contribution.binIndex;
      getContributionList(iBin)[worker.contributionOffsets[iBin]++] = contribution.contribution;
    }
    auto bounds = GenericToolbox::ParallelWorker::getThreadBoundIndices(iThread_, threadPool_.getNbThreads(), eventList_.size());
    for( size_t iEvent = bounds.beginIndex ; iEvent < bounds.endIndex ; ++iEvent ){
      auto& event = eventList_[iEvent];
      int iBin = event.getIndices().bin;
      if( iBin >= 0 ){ binContextList[iBin].eventPtrList[worker.eventOffsets[iBin]++] = &event; }
    }
  });
}

void Histogram::updateBinEventList(std::vector<Event>& eventList_, int iThread_, int nThreads_) {

  auto bounds = GenericToolbox::ParallelWorker::getThreadBoundIndices(
      iThread_, nThreads_ == -1 ? GundamGlobals::getNbCpuThreads() : nThreads_, getNbBins() + int(isBinningResponseEnabled())
  );

  for( int iBin = bounds.beginIndex ; iBin < bounds.endIndex ; iBin++ ){
    getContributionList(iBin).clear();
    if( iBin < nBins ){ getBinContextList()[iBin].eventPtrList.clear(); }
  }
  if( bounds.beginIndex == bounds.endIndex ){ return; }

  // Scan the events once per worker, rather than twice per bin. Each bin list
  // has a single writer and retains its capacity and event order across calls.
  BinningResponse::EvaluationBuffer buffer;
  if( isBinningResponseEnabled() and not eventList_.empty() ){
    _binningResponse_.prepareEvaluation(eventList_.front().getVariables(), buffer);
  }
  for( auto& event : eventList_ ){
    if( isBinningResponseEnabled() ){
      _binningResponse_.evaluate(event.getVariables(), buffer);
      for( const auto& fraction : buffer.response.binFractions ){
        if( fraction.binIndex < bounds.beginIndex or fraction.binIndex >= bounds.endIndex ){ continue; }
        EventContribution contribution;
        contribution.eventPtr = &event;
        contribution.fraction = fraction.fraction;
        getBinContextList()[fraction.binIndex].eventContributionList.emplace_back(contribution);
      }
      if( nBins >= bounds.beginIndex and nBins < bounds.endIndex and buffer.response.outsideFraction > 0 ){
        EventContribution contribution;
        contribution.eventPtr = &event;
        contribution.fraction = buffer.response.outsideFraction;
        _outsideContributionList_.emplace_back(contribution);
      }
    }
    int iBin = event.getIndices().bin;
    if( iBin >= bounds.beginIndex and iBin < bounds.endIndex ){
      getBinContextList()[iBin].eventPtrList.emplace_back(&event);
    }
  }
}
void Histogram::refillHistogram(int iThread_){

  auto bounds = GenericToolbox::ParallelWorker::getThreadBoundIndices(
      iThread_, GundamGlobals::getNbCpuThreads(), getNbBins() + int(isBinningResponseEnabled())
  );

  if( isBinningResponseEnabled() and nBins >= bounds.beginIndex and nBins < bounds.endIndex ){
    fillBinContent(_outsideContributionList_, _outsideBinContent_);
  }
  if( bounds.beginIndex >= nBins ){ return; }
  bounds.endIndex = std::min(bounds.endIndex, nBins);

  // avoid using [] operator for each access. Use the memory address directly
  double weightBuffer;

#if HAS_CPP_17
  for( auto [binContent, binContext] : loop(bounds.beginIndex, bounds.endIndex) ){
#else
  for( auto element : loop(bounds.beginIndex, bounds.endIndex) ){ auto& binContent = std::get<0>(element); auto& binContext = std::get<1>(element);
#endif

    // reset
    binContent.sumWeights = 0;
    binContent.sqrtSumSqWeights = 0;
    if( isBinningResponseEnabled() ){
      fillBinContent(binContext.eventContributionList, binContent);
      continue;
    }
    for( auto *eventPtr: binContext.eventPtrList ){
      weightBuffer = eventPtr->getEventWeight();
      binContent.sumWeights += weightBuffer;
      binContent.sqrtSumSqWeights += weightBuffer * weightBuffer;
    }

    binContent.sqrtSumSqWeights = std::sqrt(binContent.sqrtSumSqWeights);

  }

}
