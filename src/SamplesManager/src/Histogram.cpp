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
      eventPtr->getWeights().current *= (double) nCounts / binContent.sumWeights;
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

#ifdef GUNDAM_USING_CACHE_MANAGER
  // avoid checking those variables at each bin
  bool isCacheManagerEnabled{this->isCacheManagerEnabled()};
  bool useCpuCalculation{not isCacheManagerEnabled or GundamGlobals::isForceCpuCalculation()};

  LogDebug << GET_VAR_NAME_VALUE(useCpuCalculation) << std::endl;
  LogDebug << GET_VAR_NAME_VALUE(isCacheManagerEnabled) << std::endl;
  LogDebug << GET_VAR_NAME_VALUE(GundamGlobals::isForceCpuCalculation()) << std::endl;
  LogDebug << GET_VAR_NAME_VALUE(this->isCacheManagerEnabled()) << std::endl;
  LogDebug << GET_VAR_NAME_VALUE(_cacheManagerIndex_) << std::endl;
  LogDebug << GET_VAR_NAME_VALUE(_cacheManagerValidFlagPtr_) << std::endl;
//  LogThrow("debug stop");

  if( isCacheManagerEnabled ){
    // This can be slow (~10 usec for 5000 bins) when data must be copied
    // from the device, but it makes sure that the results are copied from
    // the device when they have changed. The values pointed to by
    // _CacheManagerValue_ and _CacheManagerValid_ are inside the summed
    // index cache (a bit of evil coding here), and are updated by the
    // cache.  The update is triggered by (*_CacheManagerUpdate_)().
    if (_cacheManagerUpdateFctPtr_) (*_cacheManagerUpdateFctPtr_)();
  }
#endif

  for( auto [binContent, binContext] : loop(bounds.beginIndex, bounds.endIndex) ){

#ifdef GUNDAM_USING_CACHE_MANAGER
    if( useCpuCalculation ){
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
#ifdef GUNDAM_USING_CACHE_MANAGER
    }

    if( isCacheManagerEnabled ){

      if( not useCpuCalculation ){
        // copy the result as
        LogThrowIf(_cacheManagerSumWeightsArray_ == nullptr);
        binContent.sumWeights = _cacheManagerSumWeightsArray_[_cacheManagerIndex_ + binContext.bin.getIndex()];
        binContent.sqrtSumSqWeights = _cacheManagerSumSqWeightsArray_[_cacheManagerIndex_ + binContext.bin.getIndex()];
        binContent.sqrtSumSqWeights = sqrt(binContent.sqrtSumSqWeights);
      }
      else{
        // container used for debugging
        Histogram::BinContent cacheManagerValue;

        LogThrowIf(_cacheManagerSumWeightsArray_ == nullptr);
        cacheManagerValue.sumWeights = _cacheManagerSumWeightsArray_[_cacheManagerIndex_ + binContext.bin.getIndex()];
        cacheManagerValue.sqrtSumSqWeights = _cacheManagerSumSqWeightsArray_[_cacheManagerIndex_ + binContext.bin.getIndex()];
        cacheManagerValue.sqrtSumSqWeights = sqrt(cacheManagerValue.sqrtSumSqWeights);

        // Parallel calculations of the histogramming have been run.  Make sure
        // they are the same.
        bool problemFound = false;
        if (not GundamUtils::almostEqual(cacheManagerValue.sumWeights,(binContent.sumWeights))) {
          double magnitude = std::abs(cacheManagerValue.sumWeights) + std::abs(binContent.sumWeights);
          double delta = std::abs(cacheManagerValue.sumWeights - binContent.sumWeights);
          if (magnitude > 0.0) delta /= 0.5*magnitude;
          LogError << "Incorrect histogram content --"
                   << " Content: " << cacheManagerValue.sumWeights << "!=" << binContent.sumWeights
                   << " Error: " << cacheManagerValue.sqrtSumSqWeights << "!=" << binContent.sqrtSumSqWeights
                   << " Precision: " << delta
                   << std::endl;
          problemFound = true;
        }
        if (not GundamUtils::almostEqual(cacheManagerValue.sqrtSumSqWeights, (binContent.sqrtSumSqWeights))) {
          double magnitude = std::abs(cacheManagerValue.sqrtSumSqWeights) + std::abs(binContent.sqrtSumSqWeights);
          double delta = std::abs(cacheManagerValue.sqrtSumSqWeights - binContent.sqrtSumSqWeights);
          if (magnitude > 0.0) delta /= 0.5*magnitude;
          LogError << "Incorrect histogram error --"
                   << " Content: " << cacheManagerValue.sumWeights << "!=" << binContent.sumWeights
                   << " Error: " << cacheManagerValue.sqrtSumSqWeights << "!=" << binContent.sqrtSumSqWeights
                   << " Precision: " << delta
                   << std::endl;
          problemFound = true;
        }
        if( false and problemFound ){ std::exit(EXIT_FAILURE); }// For debugging
      }

    }
#endif

  }

}
