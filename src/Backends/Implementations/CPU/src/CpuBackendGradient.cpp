#include "CpuBackend.h"

#include "Semantics/BackendDialGradient.h"
#include "Semantics/BackendLikelihood.h"

#include "Logger.h"

#include <cmath>
#include <vector>

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
  std::vector<double> responses(model.dials.size()), derivatives(model.dials.size());
  for( std::size_t i = 0; i < model.dials.size(); ++i ){
    const auto& dial = model.dials[i];
    responses[i] = _isCachedDial_[i] ? _cachedDialResponses_[i]
        : Semantics::evalDialResponse(dial, model.dialInputs.data(), model.dialPayloads.data(), _lastParameterValues_.data());
    if( dial.type == BackendDialType::Shift ){ continue; }
    const auto& input = model.dialInputs.at(dial.firstInput);
    derivatives[i] = Semantics::evalWeightDialGradient(dial, input,
        model.dialPayloads.data() + dial.payloadOffset, _lastParameterValues_[input.parameterIndex]);
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
          sums[model.dialInputs[dial.firstInput].parameterIndex] += scale * prefix[j] * suffix * derivatives[dialIndex];
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
