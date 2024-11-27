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



void Histogram::build(const JsonType& binningConfig_){

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
void Histogram::throwEventMcError(){
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

void Histogram::updateBinEventList(std::vector<Event>& eventList_, int iThread_) {

  auto bounds = GenericToolbox::ParallelWorker::getThreadBoundIndices(
      iThread_, GundamGlobals::getNbCpuThreads(), getNbBins()
  );

  for( int iBin = bounds.beginIndex ; iBin < bounds.endIndex ; iBin++ ){
    size_t count = std::count_if(eventList_.begin(), eventList_.end(), [&]( auto& e) {return e.getIndices().bin == iBin;});
    getBinContextList()[iBin].eventPtrList.clear();
    getBinContextList()[iBin].eventPtrList.resize(count, nullptr);

    // Now filling the event indexes
    size_t index = 0;
    std::for_each(eventList_.begin(), eventList_.end(), [&]( auto& e){
      if( e.getIndices().bin == iBin){ getBinContextList()[iBin].eventPtrList[index++] = &e; }
    });
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
    for( auto *eventPtr: binContext.eventPtrList ){
      weightBuffer = eventPtr->getEventWeight();
      binContent.sumWeights += weightBuffer;
      binContent.sqrtSumSqWeights += weightBuffer * weightBuffer;
    }

    binContent.sqrtSumSqWeights = std::sqrt(binContent.sqrtSumSqWeights);

  }

}
