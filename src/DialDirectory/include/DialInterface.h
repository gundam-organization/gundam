//
// Created by Adrien Blanchet on 29/11/2022.
//

#ifndef GUNDAM_DIALINTERFACE_H
#define GUNDAM_DIALINTERFACE_H

#include "DialBase.h"
#include "DialInputBuffer.h"
#include "DialResponseSupervisor.h"
#include "DialUtils.h"

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
  void addTarget(const std::pair<size_t, size_t>& sampleEventIndices_);
  [[nodiscard]] std::string getSummary(bool shallow_=true) const;

private:
  DialBase* _dialBaseRef_{nullptr}; // should be filled while init
  DialInputBuffer* _inputBufferRef_{nullptr};
  const DialResponseSupervisor* _responseSupervisorRef_{nullptr};

  GenericToolbox::NoCopyWrapper<std::mutex> _mutex_{};
  std::vector<std::pair<size_t, size_t>> _targetSampleEventIndicesList_{};

};


#endif //GUNDAM_DIALINTERFACE_H
