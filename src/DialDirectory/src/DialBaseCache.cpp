//
// Created by Adrien Blanchet on 28/11/2022.
//

#include "DialBaseCache.h"

#include "zlib.h"


double DialBaseCache::evalResponse(const DialInputBuffer& input_){

  // Already computed ? unlocked cache
  if( _cachedInputHash_ == input_.getCurrentHash() ){ return _cachedResponse_; }

  // lock -> only one at a time pass this point
#if __cplusplus >= 201703L
  // https://stackoverflow.com/questions/26089319/is-there-a-standard-definition-for-cplusplus-in-c14
  std::scoped_lock<std::mutex> g(_evalLock_);
#else
  std::lock_guard<std::mutex> g(_evalLock_);
#endif

  // Still not computed?
  if( _cachedInputHash_ == input_.getCurrentHash() ){ return _cachedResponse_; }

  // Eval
  _cachedResponse_ = this->DialBase::evalResponseImpl(input_);

  // Update hash
  _cachedInputHash_ = input_.getCurrentHash();

  // return it
  return _cachedResponse_;
}
