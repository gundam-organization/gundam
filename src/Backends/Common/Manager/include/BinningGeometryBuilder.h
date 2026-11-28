#ifndef GUNDAM_BINNING_GEOMETRY_BUILDER_H
#define GUNDAM_BINNING_GEOMETRY_BUILDER_H

#include "BinningGeometry.h"

class Histogram;

namespace Backends {
  [[nodiscard]] BinningGeometry buildBinningGeometry(const Histogram& histogram_);
}

#endif
