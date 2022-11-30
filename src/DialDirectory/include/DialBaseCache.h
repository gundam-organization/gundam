//
// Created by Adrien Blanchet on 28/11/2022.
//

#ifndef GUNDAM_DIALBASECACHE_H
#define GUNDAM_DIALBASECACHE_H

#include "DialBase.h"

#include "GenericToolbox.Wrappers.h"


#include <zlib.h>

#include "cmath"
#include "vector"
#include "mutex"




class DialBaseCache : public DialBase {
  // + sizeof(DialBase) = 24 bytes - padding = 20 bytes

public:
  DialBaseCache() = default;
  [[nodiscard]] std::string getDialTypeName() const override { return {"DialBaseCache"}; }

  // Cache handling here:
  double evalResponse(const DialInputBuffer& input_) override;

protected:
  GenericToolbox::NoCopyWrapper<std::mutex> _evalLock_{}; // + 64 bytes
  uint32_t _cachedInputHash_{0}; // + 4 bytes (keeping a vector empty is already 24...)
  double _cachedResponse_{std::nan("unset")}; // + 8 bytes

  // + 8 bytes padding
  // = 96 bytes
};


#endif //GUNDAM_DIALBASECACHE_H
