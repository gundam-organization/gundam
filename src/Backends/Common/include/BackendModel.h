#ifndef GUNDAM_BACKEND_MODEL_H
#define GUNDAM_BACKEND_MODEL_H

#include <cstddef>
#include <vector>

class DialInterface;
class DialInputBuffer;
class Event;
class Histogram;
class Parameter;

namespace Backends {

  struct BackendDialRef {
    const DialInterface* interface{nullptr};
  };

  struct BackendEventRef {
    Event* event{nullptr};
    int sampleIndex{-1};
    int binIndex{-1};
    int globalBinIndex{-1};
    double baseWeight{1};
    std::size_t firstDial{0};
    std::size_t dialCount{0};
    std::size_t resultIndex{0};
  };

  struct BackendSampleRef {
    Histogram* histogram{nullptr};
    int sampleIndex{-1};
    int binOffset{0};
    int binCount{0};
  };

  struct BackendModel {
    std::vector<BackendEventRef> events{};
    std::vector<BackendDialRef> eventDials{};
    std::vector<const DialInputBuffer*> inputBuffers{};
    std::vector<BackendSampleRef> samples{};
    std::vector<const Parameter*> parameters{};
    int totalBins{0};

    void clear();
    [[nodiscard]] bool empty() const { return events.empty(); }
  };

}

#endif // GUNDAM_BACKEND_MODEL_H
