#ifndef GUNDAM_DEVICE_BACKEND_MODEL_H
#define GUNDAM_DEVICE_BACKEND_MODEL_H

#include "BackendTypes.h"
#include "EngineView.h"

#include <cstdint>
#include <string>
#include <vector>

namespace Backends {

  constexpr std::uint32_t kDeviceDialTypeNorm{0};
  constexpr std::uint32_t kDeviceDialTypeCompactSpline{1};
  constexpr std::uint32_t kDeviceDialTypeUniformSpline{2};
  constexpr std::uint32_t kDeviceDialTypeMonotonicSpline{3};
  constexpr std::uint32_t kDeviceDialTypeGeneralSpline{4};
  constexpr std::uint32_t kDeviceDialTypeGraph{5};
  constexpr std::uint32_t kDeviceDialFlagAllowExtrapolation{1u << 0};
  constexpr std::uint32_t kDeviceDialFlagCached{1u << 1};
  constexpr std::uint32_t kDeviceCachedDialReuseThreshold{8};
  constexpr std::uint32_t kDeviceHistogramChunkSize{256};

  struct DeviceEventDialRanges {
    std::uint32_t normOffset{0};
    std::uint32_t normCount{0};
    std::uint32_t compactOffset{0};
    std::uint32_t compactCount{0};
    std::uint32_t uniformOffset{0};
    std::uint32_t uniformCount{0};
    std::uint32_t monotonicOffset{0};
    std::uint32_t monotonicCount{0};
    std::uint32_t generalOffset{0};
    std::uint32_t generalCount{0};
    std::uint32_t graphOffset{0};
    std::uint32_t graphCount{0};
  };

  struct DeviceNormDialOccurrence {
    std::uint32_t parameterIndex{0};
    float minResponse{1.0F};
    float maxResponse{1.0F};
  };

  struct DeviceSplineDialDescriptor {
    std::uint32_t parameterIndex{0};
    std::uint32_t splineOffset{0};
    std::uint32_t splineSize{0};
    std::uint32_t flags{0};
    float minResponse{1.0F};
    float maxResponse{1.0F};
  };

  struct DevicePackedDialRef {
    std::uint32_t type{0};
    std::uint32_t localIndex{0};
  };

  struct DevicePackedModel {
    std::vector<float> baseWeights{};
    std::vector<DeviceEventDialRanges> eventDialRanges{};
    std::vector<DeviceNormDialOccurrence> normDialOccurrences{};
    std::vector<std::uint32_t> compactDialIndices{};
    std::vector<std::uint32_t> uniformDialIndices{};
    std::vector<std::uint32_t> monotonicDialIndices{};
    std::vector<std::uint32_t> generalDialIndices{};
    std::vector<std::uint32_t> graphDialIndices{};
    std::vector<DeviceSplineDialDescriptor> compactDialDescriptors{};
    std::vector<DeviceSplineDialDescriptor> uniformDialDescriptors{};
    std::vector<DeviceSplineDialDescriptor> monotonicDialDescriptors{};
    std::vector<DeviceSplineDialDescriptor> generalDialDescriptors{};
    std::vector<DeviceSplineDialDescriptor> graphDialDescriptors{};
    std::vector<int> globalBins{};
    std::vector<std::uint32_t> binEventOffsets{};
    std::vector<std::uint32_t> binEventIndices{};
    std::vector<float> splineData{};

    std::uint32_t cachedDialCount{0};
    std::uint32_t compactCachedDialCount{0};
    std::uint32_t uniformCachedDialCount{0};
    std::uint32_t monotonicCachedDialCount{0};
    std::uint32_t generalCachedDialCount{0};
    std::uint32_t graphCachedDialCount{0};
    std::uint32_t compactDialDescriptorCount{0};
    std::uint32_t uniformDialDescriptorCount{0};
    std::uint32_t monotonicDialDescriptorCount{0};
    std::uint32_t generalDialDescriptorCount{0};
    std::uint32_t graphDialDescriptorCount{0};
    std::uint32_t maxHistogramChunksPerBin{1};
    std::uint32_t totalDynamicDialOccurrences{0};
    std::uint32_t maxEventsPerBin{0};

    void clear();
    [[nodiscard]] std::uint32_t uniqueDialCount() const;
  };

  [[nodiscard]] bool packDeviceBackendModel(const PropagationView& model_,
                                            bool enableCachedDialResponses_,
                                            DevicePackedModel& packedModel_,
                                            BackendTimingSummary& buildTiming_,
                                            std::string& fallbackReason_,
                                            const std::string& logPrefix_);

}

#endif // GUNDAM_DEVICE_BACKEND_MODEL_H
