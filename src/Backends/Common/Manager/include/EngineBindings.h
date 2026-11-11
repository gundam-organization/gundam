#ifndef GUNDAM_BACKEND_ENGINE_BINDINGS_H
#define GUNDAM_BACKEND_ENGINE_BINDINGS_H

#include <cstdint>
#include <vector>
#include <memory>

class ExternalWeightBuffer;
class DialBase;
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

  // Tables are owned by dial factories, whose lifetime covers the engine layout.
  // Each preparation refills them, so each published snapshot gets a new generation.
  struct ExternalWeightBinding {
    std::shared_ptr<const ExternalWeightBuffer> source{};
    const std::vector<double>* table{nullptr};
    std::uint64_t generation{0};
  };

  struct EngineBindings {
    std::vector<EventBinding> events{};
    std::vector<VariableBinding> variables{};
    std::vector<const DialInputBuffer*> variableInputBuffers{};
    std::vector<SampleBinding> samples{};
    std::vector<ParameterBinding> parameters{};
    std::vector<ExternalWeightBinding> externalWeights{};
    struct GradientBinding {
      const DialBase* dial{nullptr};
      const DialInputBuffer* inputs{nullptr};
      std::size_t dialIndex{0};
    };
    // Only external/table producers, in increasing backend dial-index order.
    std::vector<GradientBinding> gradientDials{};

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
