//
// Created by Nadrino on 28/10/2024.
//

#include "CacheSampleHistFiller.h"


void CacheSampleHistFiller::pullHistContent(const double* fSumHostPtr_, const double* fSum2HostPtr_){

#if HAS_CPP_17
  for( auto [binContent, binContext] : histPtr->loop() ){
#else
    for( auto element : histPtr->loop() ){ auto& binContent = std::get<0>(element); auto& binContext = std::get<1>(element);
#endif
    binContent.sumWeights = fSumHostPtr_[cacheManagerIndexOffset + binContext.bin.getIndex()];
    binContent.sqrtSumSqWeights = sqrt(fSum2HostPtr_[cacheManagerIndexOffset + binContext.bin.getIndex()]);

  }

}
