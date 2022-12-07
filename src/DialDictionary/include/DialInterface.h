//
// Created by Adrien Blanchet on 29/11/2022.
//

#ifndef GUNDAM_DIALINTERFACE_H
#define GUNDAM_DIALINTERFACE_H

#include "DialBase.h"
#include "DialInputBuffer.h"
#include "DialResponseSupervisor.h"
#include "DialUtils.h"
#include "SplineCache.h"

#include "FitSampleSet.h"
#include "FitParameterSet.h"

#include "string"
#include "memory"
#include "vector"


class DialInterface {

public:
  DialInterface() = default;

  void setDialBaseRef(DialBase *dialBasePtr);
  void setInputBufferRef(DialInputBuffer *inputBufferRef);
  void setResponseSupervisorRef(const DialResponseSupervisor *responseSupervisorRef);

  double evalResponse();
  [[nodiscard]] std::string getSummary(bool shallow_=true) const;

private:
  DialBase* _dialBaseRef_{nullptr}; // should be filled while init
  DialInputBuffer* _inputBufferRef_{nullptr};
  const DialResponseSupervisor* _responseSupervisorRef_{nullptr};

};


#endif //GUNDAM_DIALINTERFACE_H
