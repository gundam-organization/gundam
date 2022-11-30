//
// Created by Adrien Blanchet on 30/11/2022.
//

#ifndef GUNDAM_DIALBINNED_H
#define GUNDAM_DIALBINNED_H

#include "DataBin.h"


class DialBinned {

public:
  DialBinned() = default;
  virtual ~DialBinned() = default;

  void setApplyConditionBin(const DataBin &applyConditionBin);

  [[nodiscard]] const DataBin &getApplyConditionBin() const;

protected:
  DataBin _applyConditionBin_;

};


#endif //GUNDAM_DIALBINNED_H
