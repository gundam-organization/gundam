#include "CpuBackend.h"
#include "CpuFormulaProgram.h"
#include "GradientValidation.h"

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
        factors[axis] = Backends::Semantics::mirrorSlopeFromFolded(folded,input.mirrorRange);
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
    return Backends::Semantics::mirrorSlopeFromFolded(folded,input_.mirrorRange);
  }

  struct DialGradient {
    double value{0.};
    std::vector<double> slopes;
  };

  // Forward differentiation of the formula bytecode. Zero tangents avoid
  // evaluating singular partials belonging to constant operands (e.g. x^2).
  double formulaSlope(const Backends::FormulaProgram& program_, const std::vector<double>& values_,
                      std::size_t input_) {
#define GUNDAM_FORMULA_GRADIENT_MATH NativeFormulaMath
#define GUNDAM_FORMULA_GRADIENT_INV_SQRT_PI (1./std::sqrt(std::acos(-1.)))
    using S = double;
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
#define GUNDAM_FORMULA_UNARY_GRADIENT(opcode, operation) \
        case FormulaOpcode::opcode: partial = operation; break;
#include "DeviceMath/SharedFormulaGradientOperations.h"
#undef GUNDAM_FORMULA_UNARY_GRADIENT
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
#define GUNDAM_FORMULA_BINARY_GRADIENT(opcode, left, right) \
        case FormulaOpcode::opcode: dl = left; dr = right; break;
#include "DeviceMath/SharedFormulaGradientOperations.h"
#undef GUNDAM_FORMULA_BINARY_GRADIENT
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
#undef GUNDAM_FORMULA_GRADIENT_MATH
#undef GUNDAM_FORMULA_GRADIENT_INV_SQRT_PI
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
    const double responseSlope = Semantics::dialResponseSlope(dial_, result.value);
    for( auto& slope : result.slopes ){ slope = responseSlope == 0. ? 0. : slope * responseSlope; }
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
    adjoints[program.outputSlot] = Semantics::dialResponseSlope(root_, raw);
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

  struct FractionGradient {
    int binIndex;
    double fraction;
    std::vector<double> slopes;
  };

  // Differentiate the same compact-support geometry used by propagation.
  // Strict selection coordinates are constant; moving them is rejected below.
  std::vector<FractionGradient> binningGradient(const Backends::BinningGeometry& layout_,
      const Backends::BinningIndex& index_, const double* values_, std::vector<std::size_t>& candidates_) {
    index_.findCandidates(values_, candidates_);
    std::vector<FractionGradient> rows;
    std::vector<double> factors, slopes, prefix;
    double total = 0.;
    std::size_t largest = 0;
    for( auto position : candidates_ ){
      const auto& bin = layout_.bins[position];
      factors.resize(bin.intervalCount); slopes.assign(bin.intervalCount, 0.);
      prefix.assign(bin.intervalCount+1, 1.);
      for( std::size_t j = 0; j < bin.intervalCount; ++j ){
        const auto& interval = layout_.intervals[bin.firstInterval+j];
        const double value = values_[interval.variableIndex];
        factors[j] = GundamBinningMath::intervalFraction(interval.lowerBound, interval.upperBound,
            interval.lowerInverseWidth, interval.upperInverseWidth, value);
        if( factors[j] > 0. and interval.lowerInverseWidth > 0. ){
          slopes[j] = GundamBinningMath::cumulativeKernelSlope((interval.lowerBound-value)*interval.lowerInverseWidth)*interval.lowerInverseWidth
              - GundamBinningMath::cumulativeKernelSlope((interval.upperBound-value)*interval.upperInverseWidth)*interval.upperInverseWidth;
        }
        prefix[j+1] = prefix[j]*factors[j];
      }
      if( prefix.back() == 0. ){ continue; }
      rows.push_back({bin.binIndex, prefix.back(), std::vector<double>(layout_.variableNames.size(), 0.)});
      auto& row = rows.back();
      double suffix = 1.;
      for( std::size_t j = bin.intervalCount; j-- > 0; ){
        row.slopes[layout_.intervals[bin.firstInterval+j].variableIndex] += prefix[j]*slopes[j]*suffix;
        suffix *= factors[j];
      }
      total += row.fraction;
      if( row.fraction > rows[largest].fraction ){ largest = rows.size()-1; }
    }
    // Propagation removes tiny excesses from the largest contribution. Apply
    // the same correction to its Jacobian; the outside remainder has no LLH.
    if( total > 1. and not rows.empty() ){
      rows[largest].fraction -= total-1.;
      for( std::size_t j = 0; j < layout_.variableNames.size(); ++j ){
        double sum = 0.;
        for( const auto& row : rows ){ sum += row.slopes[j]; }
        rows[largest].slopes[j] -= sum;
      }
    }
    return rows;
  }

}

void Backends::CpuBackend::validateLikelihoodGradient() const {
  LogThrowIf(not _isBuilt_, "CPU gradient requires a built backend.");
  if( _isGradientValidated_ ){ return; }
  validateGradientModel(*_engineView_);
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
      const auto profile = LikelihoodSemantics::evaluateGradient<LikelihoodSemantics::DoubleMath>(
          sample.dataSums[i], _lastResult_.histSums.at(bin), sigma, sample.portableOptions);
      LogThrowIf(not std::isfinite(profile.predictionGradient) or not std::isfinite(profile.uncertaintyGradient),
                 "Non-finite CPU backend bin gradient.");
      predictionScales.at(bin) = profile.predictionGradient;
      uncertaintyScales.at(bin) = sigma > 0. ? profile.uncertaintyGradient / sigma : 0.;
    }
  }
  // Evaluate each unique dial derivative once; events retain only dial indices.
  std::vector<double> responses(model.dials.size());
  std::vector<std::vector<double>> derivatives(model.dials.size());
  std::size_t externalPosition = 0;
  for( std::size_t i = 0; i < model.dials.size(); ++i ){
    const auto& dial = model.dials[i];
    if( dial.type == BackendDialType::ExternalWeight or isTableDial(dial.type) ){
      const auto position = externalPosition++;
      responses[i] = _externalGradientResponses_.at(position);
      LogThrowIf(not _externalGradientPublished_.at(position),
                 "CPU external/table gradient is missing, non-finite or has invalid dimensions; publish it with propagation.");
      derivatives[i] = _externalDialGradients_[position];
      const double responseSlope = Semantics::dialResponseSlope(dial, responses[i]);
      responses[i] = Semantics::clampDialResponse(dial, responses[i]);
      for( std::size_t j = 0; j < dial.inputCount; ++j ){
        const auto& input = model.dialInputs[dial.firstInput+j];
        derivatives[i][j] = responseSlope == 0. ? 0. : derivatives[i][j] * responseSlope
            * mirrorSlope(input, _lastParameterValues_[input.parameterIndex]);
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
    std::vector<double> prefix, observableScales;
    std::vector<std::size_t> candidates;
    const auto addDial = [&](std::size_t index_, double scale_){
      if( scale_ == 0. ){ return; }
      const auto& dial = model.dials[index_];
      for( std::size_t j = 0; j < dial.inputCount; ++j ){
        sums[model.dialInputs[dial.firstInput+j].parameterIndex] += scale_*derivatives[index_][j];
      }
    };
    const auto bounds = GenericToolbox::ParallelWorker::getThreadBoundIndices(thread_, threadCount, model.events.size());
    for( auto i = bounds.beginIndex; i < bounds.endIndex; ++i ){
      const auto& event = model.events[i];
      const double eventWeight = _lastResult_.eventWeights[event.resultIndex];
      double weightScale = 0.;
      const bool smoothed = event.samplePosition < model.samples.size()
          and model.samples[event.samplePosition].binning.isSmoothed;
      if( smoothed ){
        const auto& sample = model.samples[event.samplePosition];
        observableScales.assign(sample.binning.variableNames.size(), 0.);
        const auto rows = binningGradient(sample.binning, _binningIndices_[event.samplePosition],
            _eventState_.variables.data()+event.firstVariable, candidates);
        for( const auto& row : rows ){
          if( row.binIndex < 0 ){ continue; }
          const auto bin = sample.binOffset+row.binIndex;
          const double scale = predictionScales[bin]+eventWeight*row.fraction*uncertaintyScales[bin];
          weightScale += scale*row.fraction;
          for( std::size_t j = 0; j < observableScales.size(); ++j ){
            observableScales[j] += scale*eventWeight*row.slopes[j];
          }
        }
        for( std::size_t j = 0; j < event.variableResponseCount; ++j ){
          const auto& target = model.variableResponses[event.firstVariableResponse+j];
          const auto column = target.variableIndex-event.firstVariable;
          if( column >= observableScales.size() ){ continue; }
          const double scale = observableScales[column];
          // (override or reference) * product(multipliers) + sum(additions).
          // Prefix/suffix products preserve zero and repeated multipliers.
          prefix.resize(target.multiplyDialCount+1);
          prefix[0] = target.overrideDial == std::size_t(-1) ? target.referenceValue : responses[target.overrideDial];
          for( std::size_t k = 0; k < target.multiplyDialCount; ++k ){
            prefix[k+1] = prefix[k]*responses[model.variableDialIndices[target.firstMultiplyDial+k]];
          }
          double suffix = 1.;
          for( std::size_t k = target.multiplyDialCount; k-- > 0; ){
            const auto dial = model.variableDialIndices[target.firstMultiplyDial+k];
            addDial(dial, scale*prefix[k]*suffix);
            suffix *= responses[dial];
          }
          if( target.overrideDial != std::size_t(-1) ){ addDial(target.overrideDial, scale*suffix); }
          for( std::size_t k = 0; k < target.addDialCount; ++k ){
            addDial(model.variableDialIndices[target.firstAddDial+k], scale);
          }
        }
      }
      else if( event.globalBinIndex >= 0 ){
        weightScale = predictionScales[event.globalBinIndex]+eventWeight*uncertaintyScales[event.globalBinIndex];
      }
      const auto& weight = model.eventWeights[event.resultIndex];
      prefix.resize(weight.dialCount+1);
      prefix[0] = 1.;
      for( std::size_t j = 0; j < weight.dialCount; ++j ){
        prefix[j+1] = prefix[j]*responses[model.eventDialIndices[weight.firstDial+j]];
      }
      // A weight cap suppresses weight derivatives, but never migration.
      if( model.hasReweightCap and prefix.back() > model.maxReweight ){ continue; }
      double suffix = 1.;
      for( std::size_t j = weight.dialCount; j-- > 0; ){
        const auto dial = model.eventDialIndices[weight.firstDial+j];
        addDial(dial, weight.baseWeight*weightScale*prefix[j]*suffix);
        suffix *= responses[dial];
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
