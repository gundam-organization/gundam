//
// Created by Adrien BLANCHET on 13/04/2022.
//

#ifndef GUNDAM_DIALCOLLECTION_H
#define GUNDAM_DIALCOLLECTION_H

#include "DialBase.h"

#include "vector"

// A DialCollection provide at most ONE dial per event.
// Own by DialDirectory

class DialCollection {

public:
  DialCollection();
  virtual ~DialCollection();

private:
  std::vector<DialBase> _dialApplierList_{};

};


#endif //GUNDAM_DIALCOLLECTION_H
