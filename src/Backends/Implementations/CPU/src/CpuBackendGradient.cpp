#include "CpuBackend.h"
#include "CpuFormulaProgram.h"

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
  double mirrorSlope(const Backends::BackendDialInputDescriptor& input_, double raw_) {
    if( not input_.useMirror ){ return 1.; }
    const double folded = std::fmod(raw_ - input_.mirrorMin, 2. * input_.mirrorRange);
    double slope = folded < 0. ? -1. : 1.;
    return std::abs(folded) > input_.mirrorRange ? -slope : slope;
  }

  struct DialGradient {
    double value{0.};
    std::vector<double> slopes;
  };

  // Forward differentiation of the formula bytecode. Zero tangents avoid
  // evaluating singular partials belonging to constant operands (e.g. x^2).
  double formulaSlope(const Backends::FormulaProgram& program_, const std::vector<double>& values_,
                      std::size_t input_) {
    using namespace Backends;
    struct Dual { double value, slope; };
    std::vector<Dual> stack;
    for( const auto& instruction : program_.instructions ){
      const auto op = instruction.opcode;
      if( op == FormulaOpcode::LoadInput ){
        stack.push_back({values_.at(instruction.operand), instruction.operand == input_ ? 1. : 0.}); continue;
      }
      if( op == FormulaOpcode::LoadConstant ){
        stack.push_back({program_.constants.at(instruction.operand), 0.}); continue;
      }
      auto& a = stack.back();
      const double x = a.value;
      double partial = 0.;
      bool unary = true;
      switch( op ){
        case FormulaOpcode::Negate: partial = -1.; break;
        case FormulaOpcode::Exp: partial = std::exp(x); break;
        case FormulaOpcode::Log: partial = 1./x; break;
        case FormulaOpcode::Log10: partial = 1./(x*std::log(10.)); break;
        case FormulaOpcode::Sqrt: partial = .5/std::sqrt(x); break;
        case FormulaOpcode::Abs: partial = x < 0. ? -1. : (x > 0. ? 1. : 0.); break;
        case FormulaOpcode::Sin: partial = std::cos(x); break;
        case FormulaOpcode::Cos: partial = -std::sin(x); break;
        case FormulaOpcode::Tan: partial = 1./std::pow(std::cos(x), 2); break;
        case FormulaOpcode::Asin: partial = 1./std::sqrt(1.-x*x); break;
        case FormulaOpcode::Acos: partial = -1./std::sqrt(1.-x*x); break;
        case FormulaOpcode::Atan: partial = 1./(1.+x*x); break;
        case FormulaOpcode::Sinh: partial = std::cosh(x); break;
        case FormulaOpcode::Cosh: partial = std::sinh(x); break;
        case FormulaOpcode::Tanh: partial = 1./std::pow(std::cosh(x), 2); break;
        case FormulaOpcode::Erf: partial = 2./std::sqrt(std::acos(-1.))*std::exp(-x*x); break;
        case FormulaOpcode::Erfc: partial = -2./std::sqrt(std::acos(-1.))*std::exp(-x*x); break;
        case FormulaOpcode::Floor: case FormulaOpcode::Ceil: break;
        default: unary = false;
      }
      if( unary ){
        a.slope = a.slope == 0. ? 0. : a.slope * partial;
        switch( op ){
#define GUNDAM_FORMULA_UNARY(opcode, operation) \
          case FormulaOpcode::opcode: a.value = NativeFormulaMath::operation(x); break;
#include "DeviceMath/SharedFormulaOperations.h"
#undef GUNDAM_FORMULA_UNARY
          default: throw std::runtime_error("Invalid formula gradient opcode");
        }
        continue;
      }
      const auto right = a;
      stack.pop_back();
      auto& left = stack.back();
      const double l = left.value, r = right.value;
      double dl = 0., dr = 0.;
      switch( op ){
        case FormulaOpcode::Add: dl = dr = 1.; break;
        case FormulaOpcode::Subtract: dl = 1.; dr = -1.; break;
        case FormulaOpcode::Multiply: dl = r; dr = l; break;
        case FormulaOpcode::Divide: dl = 1./r; dr = -l/(r*r); break;
        case FormulaOpcode::Pow: dl = r == 0. ? 0. : r*std::pow(l, r-1.); dr = std::pow(l,r)*std::log(l); break;
        case FormulaOpcode::Atan2: dl = r/(l*l+r*r); dr = -l/(l*l+r*r); break;
        case FormulaOpcode::Min: dl = l <= r ? 1. : 0.; dr = 1.-dl; break;
        case FormulaOpcode::Max: dl = l >= r ? 1. : 0.; dr = 1.-dl; break;
        default: throw std::runtime_error("Invalid formula gradient opcode");
      }
      left.slope = (left.slope == 0. ? 0. : left.slope*dl) + (right.slope == 0. ? 0. : right.slope*dr);
      switch( op ){
#define GUNDAM_FORMULA_BINARY(opcode, operation) \
        case FormulaOpcode::opcode: left.value = NativeFormulaMath::operation(l, r); break;
#include "DeviceMath/SharedFormulaOperations.h"
#undef GUNDAM_FORMULA_BINARY
        default: throw std::runtime_error("Invalid formula gradient opcode");
      }
    }
    return stack.back().slope;
  }

  DialGradient nodeGradient(const Backends::PropagationView& model_, const Backends::BackendDialDescriptor& dial_,
                            const Backends::BackendDialInputDescriptor* inputs_, const double* parameters_) {
    using namespace Backends;
    DialGradient result;
    result.slopes.resize(dial_.inputCount);
    const double* payload = dial_.payloadSize ? model_.dialPayloads.data() + dial_.payloadOffset : nullptr;
    if( dial_.type == BackendDialType::Formula ){
      const auto& program = model_.formulaPrograms.at(dial_.formulaProgramIndex);
      std::vector<double> values(program.inputCount), scratch(program.stackSize + program.inputCount);
      for( std::size_t i = 0; i < values.size(); ++i ){
        const auto& binding = model_.formulaInputs.at(dial_.firstFormulaInput + i);
        values[i] = binding.parameterIndex < 0 ? binding.constant : Semantics::transformDialInput(
            inputs_[binding.parameterIndex], parameters_[inputs_[binding.parameterIndex].parameterIndex]);
      }
      result.value = evaluateBoundFormula(program, [&](std::size_t i_){ return values[i_]; }, scratch.data());
      for( std::size_t i = 0; i < values.size(); ++i ){
        const int input = model_.formulaInputs[dial_.firstFormulaInput+i].parameterIndex;
        if( input >= 0 ){
          result.slopes[input] += formulaSlope(program, values, i)
              * mirrorSlope(inputs_[input], parameters_[inputs_[input].parameterIndex]);
        }
      }
    }
    else {
      auto local = dial_;
      local.firstInput = 0;
      local.hasMinResponse = local.hasMaxResponse = false;
      result.value = Semantics::evalDialResponse(local, inputs_, model_.dialPayloads.data(), parameters_);
      if( dial_.type == BackendDialType::Bilinear or dial_.type == BackendDialType::Bicubic ){
        const auto slopes = surfaceGradient(local, inputs_, payload, parameters_);
        result.slopes.assign(slopes.begin(), slopes.end());
      }
      else if( dial_.type != BackendDialType::Shift ){
        result.slopes[0] = Semantics::evalWeightDialGradient(local, inputs_[0], payload, parameters_[inputs_[0].parameterIndex]);
      }
    }
    if( (dial_.hasMinResponse and result.value < dial_.minResponse)
        or (dial_.hasMaxResponse and result.value > dial_.maxResponse) ){
      std::fill(result.slopes.begin(), result.slopes.end(), 0.);
    }
    result.value = Semantics::clampDialResponse(dial_, result.value);
    return result;
  }

  DialGradient compositeGradient(const Backends::PropagationView& model_, const Backends::BackendDialDescriptor& root_,
                                 const double* parameters_) {
    using namespace Backends;
    const auto& program = model_.compositePrograms.at(root_.compositeProgram);
    std::vector<double> values(program.scratchSize()), adjoints(program.scratchSize());
    std::vector<DialGradient> nodes;
    for( std::size_t i = 0; i < program.inputCount; ++i ){
      const auto& input = model_.dialInputs[root_.firstInput+i];
      values[i] = Semantics::transformDialInput(input, parameters_[input.parameterIndex]);
    }
    for( std::size_t i = 0; i < program.instructionCount; ++i ){
      const auto& dial = model_.compositeInstructions[program.firstInstruction+i].dial;
      nodes.push_back(nodeGradient(model_, dial, model_.compositeInputs.data()+dial.firstInput, values.data()));
      values[program.inputCount+i] = nodes.back().value;
    }
    const double raw = values.at(program.outputSlot);
    if( not ((root_.hasMinResponse and raw < root_.minResponse) or (root_.hasMaxResponse and raw > root_.maxResponse)) ){
      adjoints[program.outputSlot] = 1.;
    }
    for( std::size_t i = program.instructionCount; i-- > 0; ){
      const auto& dial = model_.compositeInstructions[program.firstInstruction+i].dial;
      const double scale = adjoints[program.inputCount+i];
      if( scale == 0. ){ continue; }
      for( std::size_t j = 0; j < dial.inputCount; ++j ){
        adjoints[model_.compositeInputs[dial.firstInput+j].parameterIndex] += scale*nodes[i].slopes[j];
      }
    }
    DialGradient result{Semantics::clampDialResponse(root_, raw), std::vector<double>(program.inputCount)};
    for( std::size_t i = 0; i < result.slopes.size(); ++i ){
      const auto& input = model_.dialInputs[root_.firstInput+i];
      result.slopes[i] = adjoints[i]*mirrorSlope(input, parameters_[input.parameterIndex]);
    }
    return result;
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
      case BackendDialType::Composite:
      case BackendDialType::Tabulated:
      case BackendDialType::Kriged:
      case BackendDialType::ExternalWeight:
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
  for( const auto& instruction : model.compositeInstructions ){
    const auto& dial = instruction.dial;
    LogThrowIf(dial.smoothing.minWidth != 0. or dial.smoothing.maxWidth != 0.,
               "CPU composite gradients do not support response smoothing.");
    for( std::size_t i = 0; i < dial.inputCount; ++i ){
      const auto& input = model.compositeInputs.at(dial.firstInput+i);
      LogThrowIf(input.useMirror and (not std::isfinite(input.mirrorRange) or input.mirrorRange <= 0.),
                 "CPU composite gradient requires a positive finite mirror range.");
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
  std::vector<std::vector<double>> derivatives(model.dials.size());
  for( std::size_t i = 0; i < model.dials.size(); ++i ){
    const auto& dial = model.dials[i];
    if( dial.type == BackendDialType::ExternalWeight or isTableDial(dial.type) ){
      responses[i] = _externalGradientResponses_.at(i);
      LogThrowIf(_externalDialGradients_.size() != model.dials.size()
                 or _externalDialGradients_[i].size() != dial.inputCount,
                 "CPU external/table gradient is missing or has invalid dimensions; publish it with propagation.");
      derivatives[i] = _externalDialGradients_[i];
      const bool clamped = (dial.hasMinResponse and responses[i] < dial.minResponse)
          or (dial.hasMaxResponse and responses[i] > dial.maxResponse);
      responses[i] = Semantics::clampDialResponse(dial, responses[i]);
      for( std::size_t j = 0; j < dial.inputCount; ++j ){
        LogThrowIf(not std::isfinite(derivatives[i][j]), "Non-finite CPU external/table derivative.");
        const auto& input = model.dialInputs[dial.firstInput+j];
        derivatives[i][j] *= clamped ? 0. : mirrorSlope(input, _lastParameterValues_[input.parameterIndex]);
      }
    }
    else {
      auto result = dial.type == BackendDialType::Composite
          ? compositeGradient(model, dial, _lastParameterValues_.data())
          : nodeGradient(model, dial, model.dialInputs.data()+dial.firstInput, _lastParameterValues_.data());
      responses[i] = _isCachedDial_[i] ? _cachedDialResponses_[i] : result.value;
      derivatives[i] = std::move(result.slopes);
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
