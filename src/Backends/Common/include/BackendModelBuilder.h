#ifndef GUNDAM_BACKEND_MODEL_BUILDER_H
#define GUNDAM_BACKEND_MODEL_BUILDER_H

#include "BackendModel.h"

class EventDialCache;
class SampleSet;

namespace Backends {

  class BackendEngineViewBuilder {
  public:
    static BackendEngineView build(SampleSet& sampleSet_, const EventDialCache& eventDialCache_);
  };

}

#endif // GUNDAM_BACKEND_MODEL_BUILDER_H
