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
std::vector<BinningResponse::BinFraction> Histogram::evaluateBinningResponse(const Event& event_) const{
  std::vector<BinningResponse::BinFraction> fractions;
  if( isBinningResponseEnabled() ){
    BinningResponse::EvaluationBuffer buffer;
    _binningResponse_.prepareEvaluation(event_.getVariables(), buffer);
    _binningResponse_.evaluate(event_.getVariables(), buffer);
    fractions = std::move(buffer.fractions);
  }
  else{
    int index = findBinIndex(event_);
    if( index >= 0 ){
      BinningResponse::BinFraction fraction;
      fraction.binIndex = index;
      fraction.fraction = 1;
      fractions.emplace_back(fraction);
    }
  }
  return fractions;
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
    for( const auto& context : binContextList ){
      for( const auto& contribution : context.eventContributionList ){
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
    worker.contributionOffsets.assign(nBins, 0);
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
      for( const auto& fraction : worker.evaluationBuffer.fractions ){
        IndexedContribution contribution;
        contribution.binIndex = fraction.binIndex;
        contribution.contribution.eventPtr = &event;
        contribution.contribution.fraction = fraction.fraction;
        worker.contributions.emplace_back(contribution);
        ++worker.contributionOffsets[fraction.binIndex];
      }
    }
  });

  for( int iBin = 0 ; iBin < nBins ; ++iBin ){
    size_t nContributions{0};
    size_t nEvents{0};
    for( auto& worker : _responseWorkers_ ){
      size_t contributionCount = worker.contributionOffsets[iBin];
      size_t eventCount = worker.eventOffsets[iBin];
      worker.contributionOffsets[iBin] = nContributions;
      worker.eventOffsets[iBin] = nEvents;
      nContributions += contributionCount;
      nEvents += eventCount;
    }
    binContextList[iBin].eventContributionList.resize(nContributions);
    binContextList[iBin].eventPtrList.resize(nEvents);
  }

  threadPool_.runJob([&](int iThread_){
    auto& worker = _responseWorkers_[iThread_];
    for( const auto& contribution : worker.contributions ){
      int iBin = contribution.binIndex;
      binContextList[iBin].eventContributionList[worker.contributionOffsets[iBin]++] = contribution.contribution;
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
      iThread_, nThreads_ == -1 ? GundamGlobals::getNbCpuThreads() : nThreads_, getNbBins()
  );

  for( int iBin = bounds.beginIndex ; iBin < bounds.endIndex ; iBin++ ){
    getBinContextList()[iBin].eventPtrList.clear();
    getBinContextList()[iBin].eventContributionList.clear();
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
      for( const auto& fraction : buffer.fractions ){
        if( fraction.binIndex < bounds.beginIndex or fraction.binIndex >= bounds.endIndex ){ continue; }
        EventContribution contribution;
        contribution.eventPtr = &event;
        contribution.fraction = fraction.fraction;
        getBinContextList()[fraction.binIndex].eventContributionList.emplace_back(contribution);
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
      iThread_, GundamGlobals::getNbCpuThreads(), getNbBins()
  );

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
      for( const auto& contribution : binContext.eventContributionList ){
        weightBuffer = contribution.eventPtr->getEventWeight() * contribution.fraction;
        binContent.sumWeights += weightBuffer;
        binContent.sqrtSumSqWeights += weightBuffer * weightBuffer;
      }
      binContent.sqrtSumSqWeights = std::sqrt(binContent.sqrtSumSqWeights);
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
