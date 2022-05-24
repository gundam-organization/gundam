//
// Created by Adrien BLANCHET on 23/03/2022.
//

#ifndef GUNDAM_APPLYCONDITION_H
#define GUNDAM_APPLYCONDITION_H

#include "DataBin.h"

class ApplyCondition {

public:
  ApplyCondition();
  virtual ~ApplyCondition();

private:
  DataBin _binCondition_;
  TFormula _formulaCondition_;

};


#endif //GUNDAM_APPLYCONDITION_H
