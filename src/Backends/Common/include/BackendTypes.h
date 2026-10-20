#ifndef GUNDAM_BACKEND_TYPES_H
#define GUNDAM_BACKEND_TYPES_H

#include <cstdint>
#include <cstddef>
#include <sstream>
#include <string>
#include <vector>

namespace Backends {

#define ENUM_NAME OutputRequest
#define ENUM_FIELDS \
ENUM_FIELD( EventWeights, 0 ) \
ENUM_FIELD( Histograms ) \
ENUM_FIELD( SampleLikelihoods ) \
ENUM_FIELD( StatLikelihood ) \
ENUM_FIELD( EventState )
#include "GenericToolbox.MakeEnum.h"

#define ENUM_NAME OutputState
#define ENUM_FIELDS \
ENUM_FIELD( NotRequested, 0 ) \
ENUM_FIELD( Scheduled ) \
ENUM_FIELD( ReadyOnDevice ) \
ENUM_FIELD( ReadyOnHost ) \
ENUM_FIELD( Failed )
#include "GenericToolbox.MakeEnum.h"

#define ENUM_NAME BackendStatus
#define ENUM_FIELDS \
ENUM_FIELD( Unconfigured, 0 ) \
ENUM_FIELD( Ready ) \
ENUM_FIELD( Running ) \
ENUM_FIELD( Failed ) \
ENUM_FIELD( Unavailable )
#include "GenericToolbox.MakeEnum.h"

  struct BackendCapabilities {
    bool supportsCpu{false};
    bool supportsGpu{false};
    bool supportsEventWeights{false};
    bool supportsHistograms{false};
    bool supportsLikelihood{false};
    bool supportsDynamicBinning{false};
    bool supportsObservableTransforms{false};
    // Relative objective-function precision recommended to numerical fitters.
    // Zero leaves the fitter's precision default unchanged.
    double recommendedFunctionPrecision{0};
    std::string deviceName{};
  };

  struct BackendDeviceView {
    const void* device{nullptr};
    const void* eventWeights{nullptr};
    std::size_t eventWeightsBytes{0};
    const void* histSums{nullptr};
    const void* histSumSquares{nullptr};
    std::size_t histogramBytes{0};
  };

  struct BackendTimingSummary {
    double buildCompatibilityScanSeconds{0};
    double buildParameterLookupSeconds{0};
    double buildFirstPassSeconds{0};
    double buildSecondPassSeconds{0};
    double buildFinalFlattenSeconds{0};
    double buildHistogramIndexSeconds{0};
    double buildBufferUploadSeconds{0};
    double externalWeightUploadSeconds{0};
    std::size_t externalWeightUploadBytes{0};
    std::size_t externalWeightUploadBlocks{0};
    double parameterUploadSeconds{0};
    double cachedDialStageSeconds{0};
    double cachePreparationSeconds{0};
    double cacheExecutionSeconds{0};
    std::size_t cacheCandidateCount{0};
    std::size_t cacheInputTransformCount{0};
    bool sharedCacheInputs{false};
    bool linearCacheTraversal{false};
    double eventWeightsStageSeconds{0};
    double histogramStageSeconds{0};
    double variableStageSeconds{0};
    double binningStageSeconds{0};
    std::size_t rebinnedEventCount{0};
    std::size_t binCandidateCount{0};
    std::size_t binContributionCount{0};
    double commandEncodeSeconds{0};
    double deviceWaitSeconds{0};
    double devicePropagationSeconds{0};
    double eventStateReadbackSeconds{0};
    std::size_t eventStateReadbackBytes{0};
    std::size_t contributionBufferBytes{0};
    double histogramReadbackSeconds{0};
    double eventWeightReadbackSeconds{0};
    double eventWeightMaterializationSeconds{0};
    double histogramMaterializationSeconds{0};
    double likelihoodHostSeconds{0};
    std::size_t uniqueDialCount{0};
    std::size_t cachedDialCount{0};
    std::size_t eventDialIndexCount{0};
    std::size_t splineScalarCount{0};
    std::size_t histogramReadbackBytes{0};
    std::size_t eventWeightReadbackBytes{0};
  };

  // Exact accumulated stage durations since the last backend build.
  struct BackendTimingTotals {
    std::uint64_t propagationCount{0};
    double propagationSeconds{0};
    double cachedDialSeconds{0};
    double eventWeightsSeconds{0};
    double histogramSeconds{0};
    double likelihoodSeconds{0};
  };

  struct PropagationToken {
    std::uint64_t id{0};
    bool isValid{false};
  };

  struct PropagationStatus {
    BackendStatus backend{BackendStatus::Failed};
    OutputState eventState{OutputState::NotRequested};
    OutputState eventWeights{OutputState::NotRequested};
    OutputState histograms{OutputState::NotRequested};
    OutputState sampleLikelihoods{OutputState::NotRequested};
    OutputState statLikelihood{OutputState::NotRequested};

    OutputState& state(OutputRequest request_);
    [[nodiscard]] OutputState state(OutputRequest request_) const;
  };

}

#endif // GUNDAM_BACKEND_TYPES_H
