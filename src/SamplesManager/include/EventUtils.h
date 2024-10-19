//
// Created by Nadrino on 06/03/2024.
//

#ifndef GUNDAM_EVENT_UTILS_H
#define GUNDAM_EVENT_UTILS_H

#include "GenericToolbox.Utils.h"

#include <iostream>
#include <string>


namespace EventUtils{

  struct Indices{

    // source
    int dataset{-1}; // which DatasetDefinition?
    long long entry{-1}; // which entry of the TChain?

    // destination
    int sample{-1}; // this information is lost in the EventDialCache manager
    int bin{-1}; // which bin of the sample?

    [[nodiscard]] std::string getSummary() const;
    friend std::ostream& operator <<( std::ostream& o, const Indices& this_ ){ o << this_.getSummary(); return o; }
  };

  struct Weights{
    double base{1};
    double current{1};

    void resetCurrentWeight(){ current = base; }
    [[nodiscard]] std::string getSummary() const;
    friend std::ostream& operator <<( std::ostream& o, const Weights& this_ ){ o << this_.getSummary(); return o; }
  };


#ifdef GUNDAM_USING_CACHE_MANAGER
  struct Cache{
    // An "opaque" index into the cache that is used to simplify bookkeeping.
    // This is actually the index of the result in the buffer filled by the
    // Cache::Manager (i.e. the index on the GPU).
    int index{-1};
    // A pointer to the cached result.  Will not be a nullptr if the cache is
    // initialized and should be checked before calling getWeight.  If it is a
    // nullptr, then that means the cache must not be used.
    const double* valuePtr{nullptr};
    // A pointer to the cache validity flag.  When not nullptr it will point
    // to the flag for if the cache needs to be updated.  This is only
    // valid when valuePtr is not null.
    const bool* isValidPtr{nullptr};
    // A pointer to a callback to force the cache to be updated.  This will
    // force the value to be copied from the GPU to the host (if necessary).
    void (*updateCallbackPtr)(){nullptr};
    // Safely update the value.  The value may not be valid after
    // this call.
    void update() const;
    // Check if there is a valid value.  The update might not provide a valid
    // value for the cache, so THIS. MUST. BE. CHECKED.
    [[nodiscard]] bool valid() const;
    // Get the current value of the weight.  Only valid if valid() returned
    // true.
    [[nodiscard]] double getWeight() const;
  };
#endif

}


#endif //GUNDAM_EVENT_UTILS_H
