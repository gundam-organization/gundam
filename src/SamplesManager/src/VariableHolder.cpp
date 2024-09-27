//
// Created by Nadrino on 27/09/2024.
//

#include "VariableHolder.h"

void VariableHolder::set(const GenericToolbox::LeafForm& leafForm_){
  if( leafForm_.getTreeFormulaPtr() != nullptr ){ leafForm_.fillLocalBuffer(); }
  memcpy(
      var.getPlaceHolderPtr()->getVariableAddress(),
      leafForm_.getDataAddress(), leafForm_.getDataSize()
  );
  updateCache();
}


