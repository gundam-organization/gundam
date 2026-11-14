#ifndef GUNDAM_RESPONSE_SMOOTHING_PACKING_H
#define GUNDAM_RESPONSE_SMOOTHING_PACKING_H

#include "EngineDescriptors.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace Backends {
  template<typename Descriptor, typename Convert>
  void packResponseLimits(const BackendDialDescriptor& dial_, Descriptor& packed_, Convert convert_) {
    packed_.minResponse = convert_(dial_.hasMinResponse ? dial_.minResponse : -std::numeric_limits<double>::infinity());
    packed_.maxResponse = convert_(dial_.hasMaxResponse ? dial_.maxResponse : std::numeric_limits<double>::infinity());
    const auto& source = dial_.smoothing;
    auto& target = packed_.smoothing;
    target.minResponse = convert_(source.minResponse);
    target.maxResponse = convert_(source.maxResponse);
    target.minWidth = convert_(source.minWidth);
    target.maxWidth = convert_(source.maxWidth);
    target.minInverseWidth = convert_(source.minInverseWidth);
    target.maxInverseWidth = convert_(source.maxInverseWidth);
    // Refuse a collapsed transition rather than silently changing its shape.
    if( (source.minWidth > 0. and (not (packed_.minResponse > target.minResponse)
                                  or not (target.minResponse - target.minWidth < target.minResponse)
                                  or not std::isnormal(float(target.minWidth))
                                  or not std::isnormal(float(target.minInverseWidth))))
        or (source.maxWidth > 0. and (not (packed_.maxResponse < target.maxResponse)
                                     or not (target.maxResponse + target.maxWidth > target.maxResponse)
                                     or not std::isnormal(float(target.maxWidth))
                                     or not std::isnormal(float(target.maxInverseWidth)))) ){
      throw std::runtime_error("Dial response smoothing cannot be represented at device precision.");
    }
  }
}
#endif
