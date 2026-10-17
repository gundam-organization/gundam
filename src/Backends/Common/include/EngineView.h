#ifndef GUNDAM_BACKEND_MODEL_H
#define GUNDAM_BACKEND_MODEL_H

#include "EngineDescriptors.h"
#include "BinningGeometry.h"

#include <cstdint>
#include <functional>
#include <vector>

namespace Backends {

  // Passive, flattened view of the fitter engine state for backend consumption.
  // It owns no engine logic and only exposes backend-friendly event, dial, sample,
  // and likelihood descriptors built upstream by BackendEngineLayout.
  struct EventView {
    int sampleIndex{-1};
    int binIndex{-1};
    int globalBinIndex{-1};
    BackendEventWeightDescriptor weight{};
    std::size_t resultIndex{0};
    std::size_t samplePosition{std::size_t(-1)};
    std::size_t firstVariable{0};
    std::size_t variableCount{0};
    std::size_t firstVariableResponse{0};
    std::size_t variableResponseCount{0};
  };

  struct SampleView {
    int sampleIndex{-1};
    int binOffset{0};
    int binCount{0};
    bool useDynamicBinning{false};
    bool readHostVariables{false};
    BinningGeometry binning{};
    std::vector<std::string> variableNames{};
  };

  struct LikelihoodSampleView {
    int binOffset{0};
    std::vector<double> dataSums{};
    std::vector<bool> ignoredBins{};
    std::function<double(double, double, double, int)> evalBin{};
  };

  struct PropagationView {
    std::vector<EventView> events{};
    // Each dial is stored once. Events reference their dial occurrences through
    // eventDialIndices, using EventWeightDescriptor::firstDial/dialCount.
    std::vector<BackendDialDescriptor> dials{};
    std::vector<std::uint32_t> eventDialIndices{};
    std::vector<BackendDialInputDescriptor> dialInputs{};
    std::vector<double> dialPayloads{};
    std::vector<ExternalWeightBlockDescriptor> externalWeightBlocks{};
    std::vector<SampleView> samples{};
    std::size_t parameterCount{0};
    int totalBins{0};
    std::vector<double> variableValues{};
    std::vector<BackendVariableResponseDescriptor> variableResponses{};
    std::vector<std::uint32_t> variableDialIndices{};
    bool hasVariableResponses{false};
    bool hasReweightCap{false};
    double maxReweight{0};

    [[nodiscard]] bool requiresEventBinning() const {
      if( hasVariableResponses ){ return true; }
      for( const auto& sample : samples ){
        if( sample.useDynamicBinning or sample.binning.isSmoothed ){ return true; }
      }
      return false;
    }

    void clear() {
      events.clear();
      dials.clear();
      eventDialIndices.clear();
      dialInputs.clear();
      dialPayloads.clear();
      externalWeightBlocks.clear();
      samples.clear();
      parameterCount = 0;
      totalBins = 0;
      variableValues.clear();
      variableResponses.clear();
      variableDialIndices.clear();
      hasVariableResponses = false;
      hasReweightCap = false;
      maxReweight = 0;
    }
    [[nodiscard]] bool empty() const { return events.empty(); }
  };

  struct LikelihoodView {
    std::vector<LikelihoodSampleView> samples{};

    [[nodiscard]] bool empty() const { return samples.empty(); }
  };

  struct EventBinContribution {
    int binIndex{-1}; // Sample-local; -1 denotes the outside contribution.
    double fraction{0};
  };

  struct EventState {
    std::uint64_t revision{1};
    std::uint64_t variableRevision{1};
    std::vector<std::uint64_t> sampleRevisions{};
    std::vector<double> variables{};
    std::vector<int> bins{};
    std::vector<std::vector<EventBinContribution>> contributions{};
  };

  struct EngineView {
    PropagationView propagation{};
    LikelihoodView likelihood{};

    void clear() {
      propagation.clear();
      likelihood.samples.clear();
    }
    [[nodiscard]] bool empty() const { return propagation.empty() and likelihood.empty(); }
  };

}

#endif // GUNDAM_BACKEND_MODEL_H
