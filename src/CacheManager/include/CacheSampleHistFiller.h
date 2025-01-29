//
// Created by Nadrino on 28/10/2024.
//

#ifndef GUNDAM_CACHESAMPLEHISTFILLER_H
#define GUNDAM_CACHESAMPLEHISTFILLER_H

#include "Histogram.h"


class CacheSampleHistFiller{

public:
  explicit CacheSampleHistFiller(Histogram* histPtr_, int cacheManagerIndexOffset_): histPtr(histPtr_), cacheManagerIndexOffset(cacheManagerIndexOffset_){}

  void pullHistContent(const double* fSumHostPtr_, const double* fSum2HostPtr_);

private:
  Histogram* histPtr{nullptr};
  int cacheManagerIndexOffset{-1};

};


#endif //GUNDAM_CACHESAMPLEHISTFILLER_H
