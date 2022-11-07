//
// Created by Adrien BLANCHET on 13/04/2022.
//

#include "DialBase.h"

DialBase::DialBase(DialType dialType_)
  : _dialType_{dialType_}, _evalMutex_{std::make_shared<std::mutex>()} {}
DialBase::~DialBase() = default;
