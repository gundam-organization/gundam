//
// Created by Adrien Blanchet on 30/11/2022.
//

#include "DialBinned.h"

void DialBinned::setApplyConditionBin(const DataBin &applyConditionBin) {
  _applyConditionBin_ = applyConditionBin;
}

const DataBin &DialBinned::getApplyConditionBin() const {
  return _applyConditionBin_;
}
DataBin &DialBinned::getApplyConditionBin() {
  return _applyConditionBin_;
}
