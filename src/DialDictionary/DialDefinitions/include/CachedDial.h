//
// Created by Adrien Blanchet on 08/04/2023.
//

#ifndef GUNDAM_CACHEDDIAL_H
#define GUNDAM_CACHEDDIAL_H

#include "DialInputBuffer.h"

/// This is a template to add caching to a DialBase derived class.
template <typename T> class CachedDial: public T {
public:
  CachedDial() = default;
  double evalResponse(const DialInputBuffer& input_) const override;
  bool isCacheValid(const DialInputBuffer& input_) const;
  void updateInputCache(const DialInputBuffer& input_) const;

protected:
  mutable double _cachedResponse_{std::nan("unset")}; // + 8 bytes
  mutable GenericToolbox::NoCopyWrapper<std::mutex> _evalLock_{}; // + 64 bytes
  mutable std::vector<double> _cachedInputs_{}; // + 24 bytes
};


#include "CachedDial.impl.h"


#endif //GUNDAM_CACHEDDIAL_H
