//
// Created by Nadrino on 28/09/2024.
//

#ifndef GUNDAM_LOADER_UTILS_H
#define GUNDAM_LOADER_UTILS_H

#include "Event.h"

#include "GenericToolbox.Root.h"

#include "TFormula.h"

#include <vector>


namespace LoaderUtils{

  void allocateMemory(Event& event_, const std::vector<const GenericToolbox::LeafForm*>& leafFormList_);
  void copyData(Event& event_, const std::vector<const GenericToolbox::LeafForm*>& leafFormList_);
  double evalFormula(const Event& event_, const TFormula* formulaPtr_, std::vector<int>* indexDict_ = nullptr);

}

#endif //GUNDAM_LOADER_UTILS_H
