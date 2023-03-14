//
// Created by Adrien BLANCHET on 24/05/2022.
//

#ifndef GUNDAM_DIALWRAPPER_H
#define GUNDAM_DIALWRAPPER_H

#include "Dial.h"

#include <memory>
#include <string>
#include <vector>
#include "type_traits"

#ifdef USE_NEW_DIALS
#define DEPRECATED [[deprecated("Not used with new dial implementation")]]
#else
#define DEPRECATED /*[[deprecated("Not used with new dial implementation")]]*/
#endif

class DEPRECATED DialWrapper{
public:
  DialWrapper() = default;

  // Handling copy
  DialWrapper(const DialWrapper& src_): dialPtr{src_.dialPtr->clone()} {  }
  DialWrapper& operator=(const DialWrapper& other) { if (this != &other) { dialPtr = other.dialPtr->clone(); } return *this; }
  DialWrapper(DialWrapper&&)  noexcept = default;
  DialWrapper& operator=(DialWrapper&&)  noexcept = default;

  template<typename DerivedDial, std::enable_if_t<std::is_base_of<Dial, DerivedDial>::value, int> = 0>
  explicit DialWrapper(const DerivedDial& src_): dialPtr{src_.clone()} {  }

  template<typename DerivedDial, std::enable_if_t<std::is_base_of<Dial, DerivedDial>::value, int> = 0>
  explicit DialWrapper(std::unique_ptr<DerivedDial> def): dialPtr{std::move(def)} {};

  Dial& operator*() const { return *dialPtr; }
  Dial* operator->() const { return dialPtr.get(); }
  Dial* get() const{ return dialPtr.get(); }

  std::unique_ptr<Dial> dialPtr;
};



#endif //GUNDAM_DIALWRAPPER_H
