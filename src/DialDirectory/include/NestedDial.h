//
// Created by Adrien BLANCHET on 13/04/2022.
//

#ifndef GUNDAM_NESTEDDIAL_H
#define GUNDAM_NESTEDDIAL_H

#include "DialBase.h"

class NestedDial : public DialBase {

public:
  NestedDial();
  ~NestedDial() override;


private:
  std::vector<std::shared_ptr<DialBase>> _subDialList_{};

};


#endif //GUNDAM_NESTEDDIAL_H
