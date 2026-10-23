#ifndef GUNDAM_BACKEND_ENGINE_BINDINGS_H
#define GUNDAM_BACKEND_ENGINE_BINDINGS_H

#include <vector>
#include <memory>

class ExternalWeightBuffer;
class DialInputBuffer;
class Event;
class Histogram;
class LikelihoodInterface;
class Parameter;
class Sample;

namespace Backends {

  struct EventBinding {
    Event* event{nullptr};
  };

  struct VariableBinding {
    Event* event{nullptr};
    int variableIndex{-1};
    std::size_t valueIndex{0};
    bool readFromHost{false};
    bool isDialTarget{false};
  };

  struct SampleBinding {
    Histogram* histogram{nullptr};
    int sampleIndex{-1};
    Sample* sample{nullptr};
  };

  struct ParameterBinding {
    Parameter* parameter{nullptr};
  };

  struct EngineBindings {
    std::vector<EventBinding> events{};
    std::vector<VariableBinding> variables{};
    std::vector<const DialInputBuffer*> variableInputBuffers{};
    std::vector<SampleBinding> samples{};
    std::vector<ParameterBinding> parameters{};
    std::vector<std::shared_ptr<const ExternalWeightBuffer>> externalWeights{};

    void clear();
    void build(LikelihoodInterface& likelihoodInterface_);
    [[nodiscard]] bool empty() const {
      return events.empty()
             and samples.empty()
             and parameters.empty();
    }
  };

}

#endif // GUNDAM_BACKEND_ENGINE_BINDINGS_H
