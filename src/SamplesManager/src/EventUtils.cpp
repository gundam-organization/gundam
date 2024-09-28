//
// Created by Nadrino on 06/03/2024.
//

#include "EventUtils.h"

#include <sstream>

#ifndef DISABLE_USER_HEADER
LoggerInit([]{ Logger::getUserHeader() << "[EventUtils]"; });
#endif


/// Indices
namespace EventUtils{
  std::string Indices::getSummary() const{
    std::stringstream ss;
    ss << "dataset(" << dataset << ")";
    ss << ", " << "entry(" << entry << ")";
    ss << ", " << "sample(" << sample << ")";
    ss << ", " << "bin(" << bin << ")";
    return ss.str();
  }
}


/// Weights
namespace EventUtils{
  std::string Weights::getSummary() const{
    std::stringstream ss;
    ss << "base(" << base << ")";
    ss << ", " << "current(" << current << ")";
    return ss.str();
  }
}

#ifdef GUNDAM_USING_CACHE_MANAGER
/// Cache
namespace EventUtils {
  void Cache::update() const {
    if( valuePtr and isValidPtr and not (*isValidPtr)) {
      // This is slowish, but will make sure that the cached result is updated
      // when the cache has changed.  The value pointed to by isValidPtr is
      // inside of the weights cache (a bit of evil coding here), and are
      // updated by the cache.  The update is triggered by
      // (*updateCallbackPtr)().
      if(updateCallbackPtr) { (*updateCallbackPtr)(); }
    }
  }

  bool Cache::valid() const {
    // Check that the valuePtr points to a value that exists, and is valid.
    return valuePtr and isValidPtr and *isValidPtr;
  }

  double Cache::getWeight() const {
    // The value pointed to by valuePtr limes inside of the weights cache (a
    // bit of evil coding here).
    return *valuePtr;
  }
}
#endif
