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

  void setDialBaseRef(DialBase *dialBasePtr);
  void setInputBufferRef(const DialInputBuffer *inputBufferRef);
  void setResponseSupervisorRef(const DialResponseSupervisor *responseSupervisorRef);

  double evalResponse();
  void propagateToTargets(int event_=-1);

private:
  DialBase* _dialBaseRef_{nullptr}; // should be filled while init
  const DialInputBuffer* _inputBufferRef_{nullptr};
  const DialResponseSupervisor* _responseSupervisorRef_{nullptr};

  std::vector<PhysicsEvent*> _responseTargetsList_{};

};


#endif //GUNDAM_DIALINTERFACE_H
