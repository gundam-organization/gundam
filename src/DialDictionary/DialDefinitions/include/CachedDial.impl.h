//
// Created by Nadrino on 08/04/2023.
//

#ifndef GUNDAM_CACHEDDIAL_IMPL_H
#define GUNDAM_CACHEDDIAL_IMPL_H

#include "CachedDial.h"

template <typename T> double CachedDial<T>::evalResponse(const DialInputBuffer& input_) const {
  if (isCacheValid(input_)) {return _cachedResponse_;}
  #if HAS_CPP_17
  std::scoped_lock<std::mutex> g(_evalLock_);
  #else
  std::lock_guard<std::mutex> g(_evalLock_);
  #endif
  if (isCacheValid(input_)) {return _cachedResponse_;}
  _cachedResponse_ = this->T::evalResponse(input_);
  updateInputCache(input_);
  return _cachedResponse_;
}
template <typename T> bool CachedDial<T>::isCacheValid(const DialInputBuffer& input_) const {
#if USE_ZLIB
  return _cachedInputHash_ == input_.getCurrentHash();
#else
  if( _cachedInputs_.size() != input_.getBufferSize() ) return false;
  return ( memcmp(
             _cachedInputs_.data(), input_.getBuffer(),
             input_.getBufferSize() * sizeof(*input_.getBuffer())) == 0 );
#endif
}
template <typename T> void CachedDial<T>::updateInputCache(const DialInputBuffer& input_) const {
#if USE_ZLIB
  _cachedInputHash_ = input_.getCurrentHash();
#else
  if( _cachedInputs_.size() != input_.getBufferSize() ){
    _cachedInputs_.resize(input_.getBufferSize(), std::nan("unset"));
  }
  memcpy(_cachedInputs_.data(), input_.getBuffer(),
         input_.getBufferSize() * sizeof(*input_.getBuffer()));
#endif
}

#endif //GUNDAM_CACHEDDIAL_IMPL_H
