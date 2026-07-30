#ifndef GUNDAM_BACKEND_ENGINE_BINDINGS_H
#define GUNDAM_BACKEND_ENGINE_BINDINGS_H

#include <vector>

class DialInterface;
class Event;
class Histogram;
class LikelihoodInterface;
class Parameter;

namespace Backends {

  struct BackendEventBinding {
    Event* event{nullptr};
  };

  struct BackendDialBinding {
    const DialInterface* interface{nullptr};
  };

  struct BackendSampleBinding {
    Histogram* histogram{nullptr};
    int sampleIndex{-1};
  };

  struct BackendParameterBinding {
    Parameter* parameter{nullptr};
  };

  struct BackendEngineBindings {
    std::vector<BackendEventBinding> events{};
    std::vector<BackendDialBinding> eventDials{};
    std::vector<BackendSampleBinding> samples{};
    std::vector<BackendParameterBinding> parameters{};

    void clear();
    void build(LikelihoodInterface& likelihoodInterface_);
    [[nodiscard]] bool empty() const {
      return events.empty()
             and eventDials.empty()
             and samples.empty()
             and parameters.empty();
    }
  };

}

#endif // GUNDAM_BACKEND_ENGINE_BINDINGS_H
