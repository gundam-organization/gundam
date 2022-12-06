//
// Created by Adrien Blanchet on 28/11/2022.
//

#ifndef GUNDAM_DIALBASECACHE_H
#define GUNDAM_DIALBASECACHE_H

#include "DialBase.h"

#include "GenericToolbox.Wrappers.h"

#include "cmath"
#include "vector"
#include "mutex"


class DialBaseCache : public DialBase {
  // + sizeof(DialBase) = 24 bytes - padding = 20 bytes

public:
  DialBaseCache() = default;
  [[nodiscard]] std::string getDialTypeName() const override { return {"DialBaseCache"}; }

#if USE_MANUAL_CACHE
  void updateCache(const DialInputBuffer& input_){ _cachedResponse_ = this->evalResponseImpl(input_); }
  double evalResponse(const DialInputBuffer& input_) override;
#else
  double evalResponse(const DialInputBuffer& input_) override;
  // Cache handling here:
  bool isCacheValid(const DialInputBuffer& input_);
  void updateInputCache(const DialInputBuffer& input_);
#endif

protected:
  double _cachedResponse_{std::nan("unset")}; // + 8 bytes

#if USE_MANUAL_CACHE
#else
  GenericToolbox::NoCopyWrapper<std::mutex> _evalLock_{}; // + 64 bytes
#if USE_ZLIB
  uint32_t _cachedInputHash_{0}; // + 4 bytes (keeping a vector empty is already 24...)
#else
  std::vector<double> _cachedInputs_{}; // + 24 bytes
#endif
#endif

  // + 8 bytes padding
  // = 96 bytes
};


#endif //GUNDAM_DIALBASECACHE_H
