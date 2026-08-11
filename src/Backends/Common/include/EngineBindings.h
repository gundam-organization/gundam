#ifndef GUNDAM_BACKEND_ENGINE_BINDINGS_H
#define GUNDAM_BACKEND_ENGINE_BINDINGS_H

#include <vector>

class DialInterface;
class Event;
class Histogram;
class LikelihoodInterface;
class Parameter;

namespace Backends {

  struct EventBinding {
    Event* event{nullptr};
  };

  struct DialBinding {
    const DialInterface* interface{nullptr};
  };

  struct SampleBinding {
    Histogram* histogram{nullptr};
    int sampleIndex{-1};
  };

  struct ParameterBinding {
    Parameter* parameter{nullptr};
  };

  struct EngineBindings {
    std::vector<EventBinding> events{};
    std::vector<DialBinding> eventDials{};
    std::vector<SampleBinding> samples{};
    std::vector<ParameterBinding> parameters{};

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
