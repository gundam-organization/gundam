#include "CpuBackend.h"

#include "Semantics/BackendDialGradient.h"
#include "Semantics/BackendDialSemantics.h"
#include "Semantics/BackendLikelihood.h"
#include "CalculateBilinearInterpolation.h"
#include "CalculateBicubicSpline.h"

#include "Logger.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace {
  std::array<double, 2> surfaceGradient(const Backends::BackendDialDescriptor& dial_,
      const Backends::BackendDialInputDescriptor* inputs_, const double* data_, const double* parameters_) {
    const int nx = int(data_[0]), ny = int(data_[1]);
    const double* axes[]{data_ + 2, data_ + 2 + nx};
    const int sizes[]{nx, ny};
    double coordinates[2], factors[2]{1., 1.};
    for( int axis = 0; axis < 2; ++axis ){
      const auto& input = inputs_[axis];
      const double raw = parameters_[input.parameterIndex];
      coordinates[axis] = Backends::Semantics::transformDialInput(input, raw);
      if( input.useMirror ){
        const double folded = std::fmod(raw - input.mirrorMin, 2. * input.mirrorRange);
        factors[axis] = folded < 0. ? -1. : 1.;
        if( std::abs(folded) > input.mirrorRange ){ factors[axis] = -factors[axis]; }
      }
      if( not dial_.allowExtrapolation ){
        const double low = axes[axis][0], high = axes[axis][sizes[axis] - 1];
        if( coordinates[axis] <= low or coordinates[axis] >= high ){ factors[axis] = 0.; }
        coordinates[axis] = std::max(low, std::min(high, coordinates[axis]));
      }
    }
    std::array<double, 2> gradient{};
    const auto* knots = axes[1] + ny;
    const double response = dial_.type == Backends::BackendDialType::Bilinear
        ? CalculateBilinearInterpolation(gradient.data(), coordinates[0], coordinates[1], -1E20, 1E20,
                                        knots, nx, ny, axes[0], nx, axes[1], ny)
        : CalculateBicubicSpline(gradient.data(), coordinates[0], coordinates[1], -1E20, 1E20,
                                knots, nx, ny, axes[0], nx, axes[1], ny);
    if( (dial_.hasMinResponse and response < dial_.minResponse)
        or (dial_.hasMaxResponse and response > dial_.maxResponse) ){ return {}; }
    for( int axis = 0; axis < 2; ++axis ){ gradient[axis] *= factors[axis]; }
    return gradient;
  }
}

void Backends::CpuBackend::validateLikelihoodGradient() const {
  LogThrowIf(not _isBuilt_, "CPU gradient requires a built backend.");
  if( _isGradientValidated_ ){ return; }
  const auto& model = _engineView_->propagation;
  LogThrowIf(model.requiresEventBinning(), "CPU backend gradients require fixed, unsmoothed bins without observable-response dials.");
  LogThrowIf(_engineView_->likelihood.empty(), "CPU backend gradient requires a statistical likelihood.");
  for( const auto& sample : _engineView_->likelihood.samples ){
    LogThrowIf(sample.portableOptions.kind != LikelihoodKind::BarlowBeestonLiteGaussian,
               "CPU backend gradients currently require BarlowBeestonLiteGaussian.");
  }
  for( const auto& dial : model.dials ){
    switch( dial.type ){
      case BackendDialType::Norm:
      case BackendDialType::Graph:
      case BackendDialType::CompactSpline:
      case BackendDialType::UniformSpline:
      case BackendDialType::MonotonicSpline:
      case BackendDialType::GeneralSpline:
        LogThrowIf(dial.inputCount != 1, "CPU backend weight gradient requires a single dial input.");
        break;
      case BackendDialType::Bilinear:
      case BackendDialType::Bicubic:
        Semantics::validateSurfaceDial(model, dial);
        break;
      case BackendDialType::Shift: break;
      default: LogThrow("Unsupported CPU backend gradient dial: " << static_cast<unsigned>(dial.type));
    }
    LogThrowIf(dial.smoothing.minWidth != 0. or dial.smoothing.maxWidth != 0.,
               "CPU backend gradients do not yet support response smoothing.");
    for( std::size_t i = 0; i < dial.inputCount; ++i ){
      const auto& input = model.dialInputs.at(dial.firstInput + i);
      LogThrowIf(input.source != DialInputSource::Parameter or input.parameterIndex >= model.parameterCount,
                 "CPU backend gradient requires a valid parameter input.");
      LogThrowIf(input.useMirror and (not std::isfinite(input.mirrorRange) or input.mirrorRange <= 0.),
                 "CPU backend gradient requires a positive finite mirror range.");
    }
  }
  _isGradientValidated_ = true;
}

std::vector<double> Backends::CpuBackend::getLikelihoodGradient(const PropagationToken& token_) {
  validateLikelihoodGradient();
  LogThrowIf(not isCurrentToken(token_), "Invalid CPU gradient propagation token.");
  LogThrowIf(not std::isfinite(getLikelihood(token_)), "Cannot differentiate a non-finite backend likelihood.");
  const auto& model = _engineView_->propagation;
  LogThrowIf(_lastParameterValues_.size() != model.parameterCount, "CPU gradient parameter snapshot is unavailable.");
  std::vector<double> predictionScales(model.totalBins, 0.), uncertaintyScales(model.totalBins, 0.);
  for( const auto& sample : _engineView_->likelihood.samples ){
    for( std::size_t i = 0; i < sample.dataSums.size(); ++i ){
      if( i < sample.ignoredBins.size() and sample.ignoredBins[i] ){ continue; }
      const auto bin = sample.binOffset + i;
      const double sigma = std::sqrt(_lastResult_.histSumSquares.at(bin));
      const auto profile = LikelihoodSemantics::evaluateLiteGaussian<LikelihoodSemantics::DoubleMath>(
          sample.dataSums[i], _lastResult_.histSums.at(bin), sigma);
      LogThrowIf(not std::isfinite(profile.predictionGradient) or not std::isfinite(profile.uncertaintyGradient),
                 "Non-finite CPU backend bin gradient.");
      predictionScales.at(bin) = profile.predictionGradient;
      uncertaintyScales.at(bin) = sigma > 0. ? profile.uncertaintyGradient / sigma : 0.;
    }
  }
  // Evaluate each unique dial derivative once; events retain only dial indices.
  std::vector<double> responses(model.dials.size());
  std::vector<std::array<double, 2>> derivatives(model.dials.size());
  for( std::size_t i = 0; i < model.dials.size(); ++i ){
    const auto& dial = model.dials[i];
    responses[i] = _isCachedDial_[i] ? _cachedDialResponses_[i]
        : Semantics::evalDialResponse(dial, model.dialInputs.data(), model.dialPayloads.data(), _lastParameterValues_.data());
    if( dial.type == BackendDialType::Shift ){ continue; }
    const auto& input = model.dialInputs.at(dial.firstInput);
    if( dial.type == BackendDialType::Bilinear or dial.type == BackendDialType::Bicubic ){
      derivatives[i] = surfaceGradient(dial, model.dialInputs.data() + dial.firstInput,
          model.dialPayloads.data() + dial.payloadOffset, _lastParameterValues_.data());
    }
    else {
      derivatives[i][0] = Semantics::evalWeightDialGradient(dial, input,
          model.dialPayloads.data() + dial.payloadOffset, _lastParameterValues_[input.parameterIndex]);
    }
  }
  const auto threadCount = _threadPool_.getNbThreads();
  std::vector<std::vector<double>> partials(threadCount, std::vector<double>(model.parameterCount, 0.));
  _threadPool_.runJob([&](int thread_){
    auto& sums = partials[thread_];
    std::vector<double> prefix;
    const auto bounds = GenericToolbox::ParallelWorker::getThreadBoundIndices(thread_, threadCount, model.events.size());
    for( auto i = bounds.beginIndex; i < bounds.endIndex; ++i ){
      const auto& event = model.events[i];
      if( event.globalBinIndex < 0 ){ continue; }
      const auto& weight = model.eventWeights[event.resultIndex];
      const auto first = weight.firstDial;
      prefix.resize(weight.dialCount + 1);
      prefix[0] = 1.;
      for( std::size_t j = 0; j < weight.dialCount; ++j ){
        prefix[j+1] = prefix[j] * responses[model.eventDialIndices[first+j]];
      }
      if( model.hasReweightCap and prefix.back() > model.maxReweight ){ continue; }
      const double scale = weight.baseWeight * (predictionScales[event.globalBinIndex]
          + _lastResult_.eventWeights[event.resultIndex] * uncertaintyScales[event.globalBinIndex]);
      if( scale == 0. ){ continue; }
      double suffix = 1.;
      for( std::size_t j = weight.dialCount; j-- > 0; ){
        const auto dialIndex = model.eventDialIndices[first+j];
        const auto& dial = model.dials[dialIndex];
        if( dial.type != BackendDialType::Shift ){
          for( std::size_t axis = 0; axis < dial.inputCount; ++axis ){
            sums[model.dialInputs[dial.firstInput + axis].parameterIndex] += scale * prefix[j] * suffix * derivatives[dialIndex][axis];
          }
        }
        suffix *= responses[dialIndex];
      }
    }
  });
  std::vector<double> gradient(model.parameterCount, 0.);
  for( const auto& partial : partials ){
    for( std::size_t i = 0; i < gradient.size(); ++i ){ gradient[i] += partial[i]; }
  }
  for( const auto value : gradient ){ LogThrowIf(not std::isfinite(value), "Non-finite CPU backend parameter gradient."); }
  return gradient;
}
