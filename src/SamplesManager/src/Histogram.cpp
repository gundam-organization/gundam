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
int Histogram::findBinIndex(const Event& event_) const{
  for( const auto& binContext : binContextList ){
    if( event_.getVariables().isInBin(binContext.bin) ){
      return binContext.bin.getIndex();
    }
  }
  return -1;
}
void Histogram::throwEventMcError(){
  // event by event poisson throw -> takes into account the finite amount of stat in MC

  for( auto [binContent, binContext] : loop() ){
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

  for( auto [binContent, binContext] : loop() ){
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

void Histogram::updateBinEventList(std::vector<Event>& eventList_, int iThread_, int nThreads_) {

  auto bounds = GenericToolbox::ParallelWorker::getThreadBoundIndices(
      iThread_, nThreads_ == -1 ? GundamGlobals::getNbCpuThreads() : nThreads_, getNbBins()
  );

  for( int iBin = bounds.beginIndex ; iBin < bounds.endIndex ; iBin++ ){
    getBinContextList()[iBin].eventPtrList.clear();
  }
  if( bounds.beginIndex == bounds.endIndex ){ return; }

  // Scan the events once per worker, rather than twice per bin. Each bin list
  // has a single writer and retains its capacity and event order across calls.
  for( auto& event : eventList_ ){
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

  for( auto [binContent, binContext] : loop(bounds.beginIndex, bounds.endIndex) ){

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
