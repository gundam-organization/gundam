//
// Created by Adrien Blanchet on 29/11/2022.
//

#ifndef GUNDAM_DIALINTERFACE_H
#define GUNDAM_DIALINTERFACE_H

#include "DialBase.h"
#include "DialInputBuffer.h"
#include "DialResponseSupervisor.h"
#include "DialUtils.h"

#include "PhysicsEvent.h"

#include "string"
#include "memory"


class DialInterface {

public:
  DialInterface() = default;

  void setDialBaseRef(DialBase *dialBasePtr);
  void setInputBufferRef(const DialInputBuffer *inputBufferRef);
  void setResponseSupervisorRef(const DialResponseSupervisor *responseSupervisorRef);

  std::vector<PhysicsEvent *> &getTargetEventList();

  double evalResponse();
  void propagateToTargets(int event_=-1);
  void addTargetEvent(PhysicsEvent* event_);
  [[nodiscard]] std::string getSummary(bool shallow_=true) const;

private:
  DialBase* _dialBaseRef_{nullptr}; // should be filled while init
  const DialInputBuffer* _inputBufferRef_{nullptr};
  const DialResponseSupervisor* _responseSupervisorRef_{nullptr};

  GenericToolbox::NoCopyWrapper<std::mutex> _mutex_{};
  std::vector<PhysicsEvent*> _targetEventList_{};

};


#endif //GUNDAM_DIALINTERFACE_H
