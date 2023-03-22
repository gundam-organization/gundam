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


class [[deprecated]] DialBaseCache : public DialBase {
  // + sizeof(DialBase) = 24 bytes - padding = 20 bytes

public:
  DialBaseCache() = default;
  [[nodiscard]] std::string getDialTypeName() const override { return {"DialBaseCache"}; }

  double evalResponse(const DialInputBuffer& input_) override;
  // Cache handling here:
  bool isCacheValid(const DialInputBuffer& input_) const;
  void updateInputCache(const DialInputBuffer& input_);

protected:
  double _cachedResponse_{std::nan("unset")}; // + 8 bytes

  GenericToolbox::NoCopyWrapper<std::mutex> _evalLock_{}; // + 64 bytes
#if USE_ZLIB
  uint32_t _cachedInputHash_{0}; // + 4 bytes (keeping a vector empty is already 24...)
#else
  std::vector<double> _cachedInputs_{}; // + 24 bytes
#endif

  // + 8 bytes padding
  // = 96 bytes
};


#endif //GUNDAM_DIALBASECACHE_H
