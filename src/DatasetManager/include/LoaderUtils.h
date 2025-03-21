//
// Created by Nadrino on 28/09/2024.
//

#ifndef GUNDAM_LOADER_UTILS_H
#define GUNDAM_LOADER_UTILS_H

#include "Event.h"
#include "Sample.h"
#include "Histogram.h"

#include "EventVarTransformLib.h"

#include "GenericToolbox.Root.h"

#include "TFormula.h"

#include <vector>


namespace LoaderUtils{

  void copyData(const Event& src_, Event& dst_);
  void copyData(Event& event_, const std::vector<const GenericToolbox::TreeBuffer::ExpressionBuffer*>& expList_);
  void fillBinIndex(Event& event_, const std::vector<Histogram::BinContext>& binList_);
  double evalFormula(const Event& event_, const TFormula* formulaPtr_, std::vector<int>* indexDict_ = nullptr);
  void applyVarTransforms(Event& event_, const std::vector<EventVarTransformLib*>& transformList_);

}

#endif //GUNDAM_LOADER_UTILS_H
