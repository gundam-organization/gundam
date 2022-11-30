//
// Created by Adrien Blanchet on 29/11/2022.
//

#ifndef GUNDAM_DIALINTERFACE_H
#define GUNDAM_DIALINTERFACE_H

#include "DialBase.h"
#include "DialInputBuffer.h"
#include "DialResponseSupervisor.h"

#include "PhysicsEvent.h"

#include "string"
#include "memory"


class DialInterface {

public:
  DialInterface() = default;

  double evalResponse();
  void propagateToTargets(int iThread_=-1);

private:
  DialBase* _dialBasePtr_{nullptr};
  const DialInputBuffer* _inputBuffer_{nullptr};
  const DialResponseSupervisor* _responseSupervisor_{nullptr};

  std::vector<PhysicsEvent*> _responseTargetsList_{};

};


#endif //GUNDAM_DIALINTERFACE_H
