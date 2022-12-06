//
// Created by Adrien Blanchet on 28/11/2022.
//

#include "DialBaseCache.h"

#include "Logger.h"

LoggerInit([]{
  Logger::setUserHeaderStr("[DialBaseCache]");
});


double DialBaseCache::evalResponse(const DialInputBuffer& input_){

  if( isCacheValid(input_) ){ return _cachedResponse_; }

  // lock -> only one at a time pass this point
#if __cplusplus >= 201703L
  // https://stackoverflow.com/questions/26089319/is-there-a-standard-definition-for-cplusplus-in-c14
  std::scoped_lock<std::mutex> g(_evalLock_);
#else
  std::lock_guard<std::mutex> g(_evalLock_);
#endif

  if( isCacheValid(input_) ){ return _cachedResponse_; }

  // Eval
  _cachedResponse_ = this->evalResponseImpl(input_);

  // Update hash
  updateInputCache(input_);

  // return it
  return _cachedResponse_;
}

bool DialBaseCache::isCacheValid(const DialInputBuffer& input_) const{
#if USE_ZLIB
  return _cachedInputHash_ == input_.getCurrentHash();
#else
  if( _cachedInputs_.size() != input_.getBufferSize() ) return false;
  return ( memcmp(_cachedInputs_.data(), input_.getBuffer(), input_.getBufferSize() * sizeof(*input_.getBuffer())) == 0 );
#endif
}
void DialBaseCache::updateInputCache(const DialInputBuffer& input_){
#if USE_ZLIB
  _cachedInputHash_ = input_.getCurrentHash();
#else
  if( _cachedInputs_.size() != input_.getBufferSize() ){
    _cachedInputs_.resize(input_.getBufferSize(), std::nan("unset"));
  }
  memcpy(_cachedInputs_.data(), input_.getBuffer(), input_.getBufferSize() * sizeof(*input_.getBuffer()));
#endif
}
