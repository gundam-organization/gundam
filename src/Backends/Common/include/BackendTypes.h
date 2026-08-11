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
ENUM_FIELD( StatLikelihood )
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
    double parameterUploadSeconds{0};
    double cachedDialStageSeconds{0};
    double eventWeightsStageSeconds{0};
    double histogramStageSeconds{0};
    double commandEncodeSeconds{0};
    double deviceWaitSeconds{0};
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

  struct PropagationToken {
    std::uint64_t id{0};
    bool isValid{false};
  };

  struct PropagationStatus {
    BackendStatus backend{BackendStatus::Failed};
    OutputState eventWeights{OutputState::NotRequested};
    OutputState histograms{OutputState::NotRequested};
    OutputState sampleLikelihoods{OutputState::NotRequested};
    OutputState statLikelihood{OutputState::NotRequested};

    OutputState& state(OutputRequest request_);
    [[nodiscard]] OutputState state(OutputRequest request_) const;
  };

}

#endif // GUNDAM_BACKEND_TYPES_H
