//
// Created by Nadrino on 28/10/2024.
//

#include "CacheSampleHistFiller.h"
#include "GundamAlmostEqual.h"
#include <cmath>

void CacheSampleHistFiller::copyHistogram(const double* fSumHostPtr_, const double* fSum2HostPtr_){

#if HAS_CPP_17
  for( auto [binContent, binContext] : histPtr->loop() ){
#else
    for( auto element : histPtr->loop() ){ auto& binContent = std::get<0>(element); auto& binContext = std::get<1>(element);
#endif
    binContent.sumWeights = fSumHostPtr_[cacheManagerIndexOffset + binContext.bin.getIndex()];
    binContent.sqrtSumSqWeights = sqrt(fSum2HostPtr_[cacheManagerIndexOffset + binContext.bin.getIndex()]);
  }
}

bool CacheSampleHistFiller::validateHistogram(bool quiet,
                                              const double* fSumHostPtr_,
                                              const double* fSum2HostPtr_) {
  bool ok{true};
#if HAS_CPP_17
  for( auto [binContent, binContext] : histPtr->loop() ){
#else
  for( auto element : histPtr->loop() ){ auto& binContent = std::get<0>(element); auto& binContext = std::get<1>(element);
#endif
    double hSum = fSumHostPtr_[cacheManagerIndexOffset + binContext.bin.getIndex()];
    double hErr = std::sqrt(fSum2HostPtr_[cacheManagerIndexOffset + binContext.bin.getIndex()]);

    if (not GundamUtils::almostEqual(binContent.sumWeights,hSum)
        or not GundamUtils::almostEqual(binContent.sqrtSumSqWeights,hErr)) {
      double dSum = binContent.sumWeights - hSum;
      double dErr = binContent.sqrtSumSqWeights - hErr;
      if (not quiet) {
        LogError << "Invalid bin[" << binContext.bin.getIndex()
                 << "+" << cacheManagerIndexOffset << "] --"
                 << " GPU: " << hSum << " +/- " << hErr
                 << " CPU: " << binContent.sumWeights
                 << " +/- " << binContent.sqrtSumSqWeights
                 << " Difference: " << dSum << " +/- " << dErr
                 << std::endl;
      }
      ok = false;
    }
  }
  return ok;
}
