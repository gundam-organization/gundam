#ifndef GUNDAM_CACHESAMPLEHISTFILLER_H
#define GUNDAM_CACHESAMPLEHISTFILLER_H

#include "Histogram.h"

/// Provide a class to copy histograms from the Cache::Manager into the local
/// GUNDAM histogram class, or verify that the current histogram contents
/// match the calculation in the Cache::Manager.
class CacheSampleHistFiller{

public:
  explicit CacheSampleHistFiller(Histogram* histPtr_, int cacheManagerIndexOffset_): histPtr(histPtr_), cacheManagerIndexOffset(cacheManagerIndexOffset_){}

  /// Copy the results from the Cache::Manager histogram cache into the GUNDAM
  /// Histogram provided in the constructor.
  void copyHistogram(const double* fSumHostPtr_, const double* fSum2HostPtr_);

  /// Check that the GUNDAM Histogram contents matches the Cache::Manager
  /// histogram cache.
  bool validateHistogram(bool quiet, const double* fSumHostPtr_, const double* fSum2HostPtr_);

private:
  Histogram* histPtr{nullptr};
  int cacheManagerIndexOffset{-1};
};

#endif //GUNDAM_CACHESAMPLEHISTFILLER_H

// Copyright (c) 2025 Adrien Blanche

// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.

// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor,
// Boston, MA  02110-1301  USA
