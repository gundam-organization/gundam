//
// Created by Nadrino on 27/09/2024.
//

#ifndef GUNDAM_VARIABLE_HOLDER_H
#define GUNDAM_VARIABLE_HOLDER_H

#include "GundamUtils.h"

#include "GenericToolbox.Utils.h"


class VariableHolder{
  /*
   * This class aims at preserving the type of loaded variables
   * It uses GenericToolbox::AnyType as a container.
   *
   * Casting back the variable to its original type takes time.
   * This is why a "double" cache is used for fast evaluations in formulas for instance
   *
   * */

public:
  VariableHolder() = default;

  // setters
  template<typename T>void set(const T& value_){ var = value_; updateCache(); }
  void set(const void *src_, size_t size_);

  // const-getters
  [[nodiscard]] const GenericToolbox::AnyType& get() const { return var; }
  [[nodiscard]] double getVarAsDouble() const { return cache; }

  // mutable getters
  GenericToolbox::AnyType& get(){ return var; }

protected:
  // user should not have to worry about the cache
  void updateCache(){ cache = var.getValueAsDouble(); }

private:
  GenericToolbox::AnyType var{};
  double cache{std::nan("unset")};

};


inline void VariableHolder::set(const void *src_, size_t size_){
  memcpy(var.getPlaceHolderPtr()->getVariableAddress(), src_, size_);
  updateCache();
}



#endif //GUNDAM_VARIABLE_HOLDER_H
