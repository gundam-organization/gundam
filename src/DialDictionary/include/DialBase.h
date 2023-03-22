//
// Created by Adrien BLANCHET on 13/04/2022.
//

#ifndef GUNDAM_DIALBASE_H
#define GUNDAM_DIALBASE_H

#include "DialInputBuffer.h"

#include "vector"
#include "string"
#include "memory"

// should be thread safe -> add lock?
// any number of inputs (provided doubles) -> set input size
// fast -> no checks while eval

// as light as possible: minimize members
// keep in mind: class size is n*8 bytes (64 bits chunks)
// virtual layer adds 8 bytes for vftable pointer (virtual function table)
// adds members memory ->
// https://stackoverflow.com/questions/9439240/sizeof-class-with-int-function-virtual-function-in-c

class DialBase {
public:
  DialBase() = default;
  virtual ~DialBase() = default;

  // virtual layer + 8 bytes

  // to override
  [[nodiscard]] virtual std::unique_ptr<DialBase> clone() const = 0; // can't allocate pure virtual class
  [[nodiscard]] virtual std::string getDialTypeName() const { return {"DialBase"}; }
  virtual double evalResponseImpl(const DialInputBuffer& input_) = 0;
  virtual bool isBinned(){ return false; }

  // other virtual
  virtual double evalResponse(const DialInputBuffer& input_) {return this->evalResponseImpl(input_); }

  //! class size = 8 bytes (no padding!)
};

/// This is a template to add caching to a DialBase derived class.
template <typename T>
class CachedDial: public T {
public:
  CachedDial() = default;
  double evalResponse(const DialInputBuffer& input_) override;
  bool isCacheValid(const DialInputBuffer& input_) const;
  void updateInputCache(const DialInputBuffer& input_);

protected:
  double _cachedResponse_{std::nan("unset")}; // + 8 bytes
  GenericToolbox::NoCopyWrapper<std::mutex> _evalLock_{}; // + 64 bytes
#if USE_ZLIB
  // + 4 bytes (keeping a vector empty is already 24...)
  uint32_t _cachedInputHash_{0};
#else
  std::vector<double> _cachedInputs_{}; // + 24 bytes
#endif
};

/// This is a template to add caching to a DialBase derived class.
template <typename T>
double CachedDial<T>::evalResponse(const DialInputBuffer& input_) {
  if (isCacheValid(input_)) {return _cachedResponse_;}
#if HAS_CPP_17
  std::scoped_lock<std::mutex> g(_evalLock_);
#else
  std::lock_guard<std::mutex> g(_evalLock_);
#endif
  if (isCacheValid(input_)) {return _cachedResponse_;}
  _cachedResponse_ = T::evalResponse(input_);
  updateInputCache(input_);
  return _cachedResponse_;
}

template <typename T>
bool CachedDial<T>::isCacheValid(const DialInputBuffer& input_) const {
#if USE_ZLIB
  return _cachedInputHash_ == input_.getCurrentHash();
#else
  if( _cachedInputs_.size() != input_.getBufferSize() ) return false;
  return ( memcmp(
             _cachedInputs_.data(), input_.getBuffer(),
             input_.getBufferSize() * sizeof(*input_.getBuffer())) == 0 );
#endif
}

template <typename T>
void CachedDial<T>::updateInputCache(const DialInputBuffer& input_) {
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
#endif //GUNDAM_DIALBASE_H
// Local Variables:
// mode:c++
// c-basic-offset:2
// compile-command:"$(git rev-parse --show-toplevel)/cmake/gundam-build.sh"
// End:
