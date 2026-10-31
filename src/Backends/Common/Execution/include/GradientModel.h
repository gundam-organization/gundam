#ifndef GUNDAM_GRADIENT_MODEL_H
#define GUNDAM_GRADIENT_MODEL_H
#include "LikelihoodModel.h"
#include "GradientTypes.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace Backends {
  inline void validateDeviceGradient(const EngineView& view_) {
    const auto& model = view_.propagation;
    if( model.requiresEventBinning() ) throw std::runtime_error("Backend gradients require fixed, unsmoothed bins without observable-response dials.");
    if( view_.likelihood.empty() ) throw std::runtime_error("Backend gradient requires a statistical likelihood.");
    for( const auto& sample : view_.likelihood.samples ){
      if( sample.portableOptions.kind != LikelihoodKind::BarlowBeestonLiteGaussian )
        throw std::runtime_error("Backend gradients currently require BarlowBeestonLiteGaussian.");
    }
    for( const auto& dial : model.dials ){
      switch( dial.type ){
        case BackendDialType::Norm: case BackendDialType::Graph:
        case BackendDialType::CompactSpline: case BackendDialType::UniformSpline:
        case BackendDialType::MonotonicSpline: case BackendDialType::GeneralSpline:
          if( dial.inputCount != 1 ) throw std::runtime_error("Backend weight gradient requires a single dial input.");
          break;
        case BackendDialType::Shift: break;
        default: throw std::runtime_error("Unsupported backend gradient dial: " + std::to_string(static_cast<unsigned>(dial.type)));
      }
      if( dial.smoothing.minWidth != 0. || dial.smoothing.maxWidth != 0. )
        throw std::runtime_error("Backend gradients do not yet support response smoothing.");
      for( std::size_t i = 0; i < dial.inputCount; ++i ){
        const auto& input = model.dialInputs.at(dial.firstInput + i);
        if( input.source != DialInputSource::Parameter || input.parameterIndex >= model.parameterCount )
          throw std::runtime_error("Backend gradient requires a valid parameter input.");
        if( input.useMirror && (!std::isfinite(input.mirrorRange) || input.mirrorRange <= 0.) )
          throw std::runtime_error("Backend gradient requires a positive finite mirror range.");
      }
    }
  }
  inline DeviceIndex gradientIndex(std::size_t value_) {
    if( value_ >= std::numeric_limits<DeviceIndex>::max() ) throw std::runtime_error("Backend gradient exceeds device index range.");
    return static_cast<DeviceIndex>(value_);
  }
  // Immutable sparse occurrence layout. No event-by-parameter Jacobian.
  template<typename S> struct GradientModel {
    GradientWork<S> work{};
    std::vector<LikelihoodBin<S>> bins;
    std::vector<GradientDial<S>> dials;
    std::vector<GradientEvent<S>> events;
    std::vector<S> payloads;
    std::vector<DeviceIndex> indices, parameterOffsets, parameterOccurrences;
    void build(const EngineView& view_) {
      validateDeviceGradient(view_);
      const auto& m = view_.propagation;
      work = {gradientIndex(m.events.size()), gradientIndex(m.dials.size()), gradientIndex(m.totalBins),
              gradientIndex(m.parameterCount), m.hasReweightCap ? likelihoodScalar<S>(m.maxReweight, true) : S(0), DeviceIndex(m.hasReweightCap), 0};
      bins.assign(m.totalBins, {});
      for( auto& bin : bins ) bin.ignored = 1;
      for( const auto& bin : packLikelihood<S>(view_.likelihood, m.totalBins) ) bins.at(bin.globalBin) = bin;
      dials.clear(); events.clear(); payloads.clear(); indices.clear(); parameterOffsets.clear(); parameterOccurrences.clear();
      for( auto value : m.dialPayloads ) payloads.push_back(likelihoodScalar<S>(value));
      for( const auto& dial : m.dials ){
        GradientDial<S> out;
        out.type = DeviceIndex(dial.type); out.parameter = std::numeric_limits<DeviceIndex>::max();
        out.payload = gradientIndex(dial.payloadOffset); out.payloadSize = gradientIndex(dial.payloadSize);
        out.extrapolate = dial.allowExtrapolation; out.hasMin = dial.hasMinResponse; out.hasMax = dial.hasMaxResponse;
        if( out.hasMin ) out.minResponse = likelihoodScalar<S>(dial.minResponse);
        if( out.hasMax ) out.maxResponse = likelihoodScalar<S>(dial.maxResponse);
        if( dial.type != BackendDialType::Shift ){
          const auto& input = m.dialInputs.at(dial.firstInput);
          out.parameter = gradientIndex(input.parameterIndex); out.mirror = input.useMirror;
          if( out.mirror ) { out.mirrorMin = likelihoodScalar<S>(input.mirrorMin); out.mirrorRange = likelihoodScalar<S>(input.mirrorRange); }
        }
        dials.push_back(out);
      }
      std::vector<std::vector<DeviceIndex>> byParameter(m.parameterCount);
      for( const auto& event : m.events ){
        const auto& weight = m.eventWeights.at(event.resultIndex);
        events.push_back({gradientIndex(indices.size()), gradientIndex(weight.dialCount),
                          event.globalBinIndex < 0 ? std::numeric_limits<DeviceIndex>::max() : gradientIndex(event.globalBinIndex),
                          gradientIndex(event.resultIndex), likelihoodScalar<S>(weight.baseWeight)});
        for( std::size_t j = 0; j < weight.dialCount; ++j ){
          const auto index = m.eventDialIndices.at(weight.firstDial+j);
          const auto parameter = dials.at(index).parameter;
          if( parameter != std::numeric_limits<DeviceIndex>::max() ) byParameter.at(parameter).push_back(gradientIndex(indices.size()));
          indices.push_back(index);
        }
      }
      for( const auto& occurrences : byParameter ){
        parameterOffsets.push_back(gradientIndex(parameterOccurrences.size()));
        parameterOccurrences.insert(parameterOccurrences.end(), occurrences.begin(), occurrences.end());
      }
      parameterOffsets.push_back(gradientIndex(parameterOccurrences.size()));
    }
  };
}
#endif
