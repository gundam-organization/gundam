#include "CudaWeightModel.h"
#include "CudaBinning.h"
#include "DialDependencyGroups.h"
#include "DialProgramValidation.h"
#include "HistogramEventIndex.h"
#include "Semantics/BackendDialSemantics.h"

#include <algorithm>
#include <map>
#include <numeric>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace {
  void require(bool condition_, const char* message_) {
    if( not condition_ ){ throw std::runtime_error(std::string("CudaBackend: ") + message_); }
  }

  bool isExternal(Backends::BackendDialType type_) {
    return type_ == Backends::BackendDialType::ExternalWeight or Backends::isTableDial(type_);
  }

  void validateDial(const Backends::PropagationView& model_, const Backends::BackendDialDescriptor& dial_, bool compositeNode_ = false) {
    const auto& inputs = compositeNode_ ? model_.compositeInputs : model_.dialInputs;
    using namespace Backends;
    require(dial_.payloadOffset <= model_.dialPayloads.size()
            and dial_.payloadSize <= model_.dialPayloads.size() - dial_.payloadOffset,
            "dial payload is outside the model.");
    require(dial_.payloadSize <= std::size_t(std::numeric_limits<int>::max() - 16), "dial payload is too large.");
    const auto size = dial_.payloadSize;
    switch( dial_.type ){
      case BackendDialType::Formula:
        validateFormulaDial(model_, dial_, inputs.size()); break;
      case BackendDialType::Composite: require(not compositeNode_, "nested composites must be flattened."); break;
      case BackendDialType::Norm: break;
      case BackendDialType::Shift: require(size >= 1, "Shift requires a constant payload."); break;
      case BackendDialType::CompactSpline:
      case BackendDialType::MonotonicSpline: require(size >= 4, "spline requires at least two knots."); break;
      case BackendDialType::UniformSpline: require(size >= 6 and (size - 2) % 2 == 0, "invalid uniform spline payload."); break;
      case BackendDialType::GeneralSpline: require(size >= 8 and (size - 2) % 3 == 0, "invalid general spline payload."); break;
      case BackendDialType::Graph: require(size >= 2 and size % 2 == 0, "invalid graph payload."); break;
      case BackendDialType::Bilinear:
      case BackendDialType::Bicubic: Semantics::validateSurfaceDial(model_, dial_, inputs.size()); break;
      case BackendDialType::ExternalWeight:
        require(dial_.externalBlockIndex < model_.externalWeightBlocks.size(), "invalid external block index.");
        require(dial_.externalWeightIndex < model_.externalWeightBlocks[dial_.externalBlockIndex].count,
                "invalid external weight index.");
        break;
      case BackendDialType::Tabulated:
      case BackendDialType::Kriged: Semantics::validateTableDial(model_, dial_); break;
      default: throw std::runtime_error("CudaBackend: unsupported dial type.");
    }
    if( dial_.type != BackendDialType::Shift and not isExternal(dial_.type) ){
      const auto count = dial_.type == BackendDialType::Formula or dial_.type == BackendDialType::Composite
          ? dial_.inputCount : (dial_.type == BackendDialType::Bilinear or dial_.type == BackendDialType::Bicubic ? 2u : 1u);
      require(dial_.inputCount == count and dial_.firstInput <= inputs.size()
              and count <= inputs.size() - dial_.firstInput, "invalid dial parameter input range.");
      for( std::size_t i = 0; i < count; ++i ){
        const auto& input = inputs[dial_.firstInput + i];
        require(input.source == DialInputSource::Parameter and (compositeNode_ or input.parameterIndex < model_.parameterCount),
                "event-variable dial inputs are not supported; parameter index must be valid.");
        require(not input.useMirror or (std::isfinite(input.mirrorMin) and std::isfinite(input.mirrorRange)
                and input.mirrorRange > 0.), "mirror range must be finite and positive.");
      }
    }
    for( std::size_t i = 0; i < size; ++i ){
      require(std::isfinite(model_.dialPayloads[dial_.payloadOffset + i]), "dial payload must be finite.");
    }
    if( dial_.type == BackendDialType::CompactSpline or dial_.type == BackendDialType::MonotonicSpline
        or dial_.type == BackendDialType::UniformSpline or dial_.type == BackendDialType::GeneralSpline ){
      require(model_.dialPayloads[dial_.payloadOffset + 1] > 0., "spline step must be positive.");
      const auto knots = dial_.type == BackendDialType::UniformSpline ? (size - 2) / 2
          : (dial_.type == BackendDialType::GeneralSpline ? (size - 2) / 3 : size - 2);
      require(std::isfinite(model_.dialPayloads[dial_.payloadOffset]
              + model_.dialPayloads[dial_.payloadOffset + 1] * double(knots - 1)), "spline extent must be finite.");
    }
    if( dial_.type == BackendDialType::Graph or dial_.type == BackendDialType::GeneralSpline ){
      const std::size_t stride = dial_.type == BackendDialType::Graph ? 2 : 3;
      const std::size_t first = dial_.type == BackendDialType::Graph ? 1 : 4;
      for( std::size_t i = first + stride; i < size; i += stride ){
        require(model_.dialPayloads[dial_.payloadOffset + i] > model_.dialPayloads[dial_.payloadOffset + i - stride],
                "graph and general-spline coordinates must be strictly increasing.");
      }
    }
    require(not dial_.hasMinResponse or std::isfinite(dial_.minResponse), "minimum response must be finite.");
    require(not dial_.hasMaxResponse or std::isfinite(dial_.maxResponse), "maximum response must be finite.");
    require(not (dial_.hasMinResponse and dial_.hasMaxResponse) or dial_.minResponse <= dial_.maxResponse,
            "response bounds are reversed.");
    const auto& s = dial_.smoothing;
    require(std::isfinite(s.minWidth) and std::isfinite(s.maxWidth) and s.minWidth >= 0. and s.maxWidth >= 0.,
            "smoothing widths must be finite and nonnegative.");
    require(s.minWidth == 0. or (std::isfinite(s.minResponse) and std::isfinite(s.minInverseWidth) and s.minInverseWidth > 0.),
            "invalid lower response smoothing.");
    require(s.maxWidth == 0. or (std::isfinite(s.maxResponse) and std::isfinite(s.maxInverseWidth) and s.maxInverseWidth > 0.),
            "invalid upper response smoothing.");
  }
}

Backends::CudaWeightModel Backends::makeCudaWeightModel(const EngineView& view_, bool interleaveEventDials_) {
  static_assert(std::is_trivially_copyable<CudaFormulaProgram>::value);
  static_assert(std::is_trivially_copyable<FormulaInstruction>::value);
  static_assert(std::is_trivially_copyable<FormulaInput>::value);
  static_assert(std::is_trivially_copyable<CompositeProgram>::value);
  static_assert(std::is_trivially_copyable<CompositeInstruction>::value);
  static_assert(std::is_trivially_copyable<BackendDialDescriptor>::value);
  static_assert(std::is_trivially_copyable<BackendDialInputDescriptor>::value);
  static_assert(std::is_trivially_copyable<BackendEventWeightDescriptor>::value);
  static_assert(std::is_trivially_copyable<CudaResponseUpdate>::value);
  static_assert(std::is_trivially_copyable<BackendTableTerm>::value);
  static_assert(std::is_trivially_copyable<ExternalWeightBlockDescriptor>::value);
  static_assert(std::is_trivially_copyable<CudaBinContribution>::value);
  static_assert(std::is_trivially_copyable<CudaBinningResult>::value);
  static_assert(std::is_trivially_copyable<EventView>::value);
  static_assert(std::is_trivially_copyable<CudaBinningSample>::value);
  static_assert(std::is_trivially_copyable<CudaBinningNode>::value);
  static_assert(std::is_trivially_copyable<CudaBinningBound>::value);
  static_assert(std::is_trivially_copyable<BackendVariableResponseDescriptor>::value);
  const auto& model = view_.propagation;

  // Packed blocks may contain gaps, but must not overlap or overflow.
  std::vector<std::pair<std::size_t, std::size_t>> externalIntervals;
  std::size_t externalWeightCount = 0;
  for( const auto& block : model.externalWeightBlocks ){
    require(block.count <= std::numeric_limits<std::size_t>::max() - block.offset, "external block extent overflow.");
    if( block.count ){
      externalIntervals.emplace_back(block.offset, block.offset + block.count);
      externalWeightCount = std::max(externalWeightCount, block.offset + block.count);
    }
  }
  require(externalWeightCount <= std::numeric_limits<std::size_t>::max() / sizeof(double), "external buffer size overflow.");
  std::sort(externalIntervals.begin(), externalIntervals.end());
  for( std::size_t i = 1; i < externalIntervals.size(); ++i ){
    require(externalIntervals[i - 1].second <= externalIntervals[i].first, "external blocks overlap.");
  }
  require(model.totalBins >= 0, "totalBins must be nonnegative.");
  require(model.events.size() == model.eventWeights.size(), "event weight layout size mismatch.");
  require(not model.hasReweightCap or std::isfinite(model.maxReweight), "reweight cap must be finite.");
  for( const auto& program : model.formulaPrograms ){
    program.validate();
    for( double value : program.constants ){ require(std::isfinite(value), "formula literals must be finite."); }
  }
  for( const auto& input : model.formulaInputs ){
    require(std::isfinite(input.constant), "formula constants must be finite.");
  }
  validateCompositePrograms(model);
  for( const auto& instruction : model.compositeInstructions ){
    validateDial(model, instruction.dial, true);
    require(not std::isnan(instruction.inputMin) and not std::isnan(instruction.inputMax)
            and instruction.inputMin <= instruction.inputMax, "invalid composite spline input bounds.");
  }
  for( const auto& dial : model.dials ){ validateDial(model, dial); }
  for( auto dial : model.eventDialIndices ){ require(dial < model.dials.size(), "event dial index is out of range."); }
  for( const auto& event : model.eventWeights ){
    require(std::isfinite(event.baseWeight), "base weights must be finite.");
    require(event.firstDial <= model.eventDialIndices.size()
            and event.dialCount <= model.eventDialIndices.size() - event.firstDial, "event dial range is invalid.");
  }
  CudaWeightModel out;
  require(model.dials.size() <= std::numeric_limits<std::uint32_t>::max(), "too many CUDA dials.");
  require(model.parameterCount < std::numeric_limits<std::size_t>::max(), "parameter count overflow.");
  auto isSurface = [](const auto& dial_){
    return dial_.type == BackendDialType::Bilinear or dial_.type == BackendDialType::Bicubic;
  };
  out.surfaceOffset = std::count_if(model.dials.begin(), model.dials.end(),
                                  [&](const auto& dial_){ return not isSurface(dial_); });
  std::vector<std::uint32_t> dialSlots;
  if( out.surfaceOffset != model.dials.size() ){
    out.dials.resize(model.dials.size());
    dialSlots.resize(model.dials.size());
    std::size_t scalar = 0, surface = out.surfaceOffset;
    for( std::size_t i = 0; i < model.dials.size(); ++i ){
      const auto slot = isSurface(model.dials[i]) ? surface++ : scalar++;
      out.dials[slot] = model.dials[i];
      dialSlots[i] = static_cast<std::uint32_t>(slot);
    }
  }
  else{ out.dials = model.dials; }
  NativeDialProgramModel native;
  native.build(model, out.dials);
  if( not native.dials.empty() ){ out.dials = std::move(native.dials); }
  if( not native.compositeInstances.empty() or not native.formulaPrograms.empty() ){
    // Keep ordinary reweights and the surface suffix unchanged. Program roots
    // become contiguous by dependencies/template, enabling direct range dispatch.
    using GroupKey = std::pair<std::vector<std::size_t>, std::pair<BackendDialType, std::size_t>>;
    std::map<GroupKey, std::vector<std::size_t>> programGroups;
    std::vector<std::size_t> order;
    order.reserve(out.dials.size());
    for( std::size_t i = 0; i < out.surfaceOffset; ++i ){
      const auto& dial = out.dials[i];
      if( dial.type != BackendDialType::Composite and dial.type != BackendDialType::Formula ){ order.push_back(i); continue; }
      std::vector<std::size_t> parameters;
      for( std::size_t j = 0; j < dial.inputCount; ++j ){ parameters.push_back(model.dialInputs[dial.firstInput + j].parameterIndex); }
      std::sort(parameters.begin(), parameters.end());
      const auto program = dial.type == BackendDialType::Composite
          ? native.compositeInstances[dial.compositeProgram].programIndex : dial.formulaProgramIndex;
      programGroups[{std::move(parameters), {dial.type, program}}].push_back(i);
    }
    for( const auto& group : programGroups ){ order.insert(order.end(), group.second.begin(), group.second.end()); }
    for( std::size_t i = out.surfaceOffset; i < out.dials.size(); ++i ){ order.push_back(i); }
    std::vector<BackendDialDescriptor> sorted;
    std::vector<std::uint32_t> destinations(order.size());
    sorted.reserve(order.size());
    for( std::size_t i = 0; i < order.size(); ++i ){
      sorted.push_back(out.dials[order[i]]); destinations[order[i]] = static_cast<std::uint32_t>(i);
    }
    if( dialSlots.empty() ){
      dialSlots.resize(order.size()); std::iota(dialSlots.begin(), dialSlots.end(), 0);
    }
    for( auto& slot : dialSlots ){ slot = destinations[slot]; }
    out.dials = std::move(sorted);
  }
  auto groups = buildDialDependencyGroups(model, out.dials);
  out.parameterDialOffsets = std::move(groups.parameterOffsets);
  out.parameterDialIndices = std::move(groups.parameterDialIndices);
  for( const auto& group : groups.multiParameterDials ){
    const auto first = out.parameterDialIndices.size();
    out.parameterDialIndices.insert(out.parameterDialIndices.end(), group.second.begin(), group.second.end());
    if( group.first.size() == 2 ){
      out.surfaceGroups.push_back({group.first[0], group.first[1], first, out.parameterDialIndices.size()});
    }
    else{ out.multiInputGroups.push_back({group.first, first, out.parameterDialIndices.size()}); }
  }
  for( const auto& group : groups.externalDials ){
    const auto first = out.parameterDialIndices.size();
    out.parameterDialIndices.insert(out.parameterDialIndices.end(), group.second.begin(), group.second.end());
    out.externalGroups.push_back({group.first, first, out.parameterDialIndices.size()});
    out.externalDialCount += group.second.size();
  }
  out.externalWeightBlocks = model.externalWeightBlocks;
  out.externalWeightCount = externalWeightCount;
  out.tableTerms = model.tableTerms;
  out.hasEventState = model.requiresEventBinning() or not model.variableResponses.empty() or not model.variableDialIndices.empty();
  if( out.hasEventState ){
    out.stateEvents = model.events;
    out.variableValues = model.variableValues;
    for( double value : out.variableValues ){ require(std::isfinite(value), "initial observables must be finite."); }
    out.variableResponses = model.variableResponses;
    out.variableDialIndices = model.variableDialIndices;
    auto remap = [&](std::size_t dial_){
      require(dial_ < model.dials.size(), "observable dial index is out of range.");
      return dialSlots.empty() ? dial_ : std::size_t(dialSlots[dial_]);
    };
    for( auto& index : out.variableDialIndices ){ index = static_cast<std::uint32_t>(remap(index)); }
    for( auto& target : out.variableResponses ){
      require(target.variableIndex < model.variableValues.size() and std::isfinite(target.referenceValue),
              "invalid observable target or reference value.");
      if( target.overrideDial != std::size_t(-1) ){ target.overrideDial = remap(target.overrideDial); }
      require(target.firstMultiplyDial <= out.variableDialIndices.size()
              and target.multiplyDialCount <= out.variableDialIndices.size() - target.firstMultiplyDial
              and target.firstAddDial <= out.variableDialIndices.size()
              and target.addDialCount <= out.variableDialIndices.size() - target.firstAddDial,
              "observable dial range is out of bounds.");
    }
    for( const auto& sample : model.samples ){

      require(sample.binOffset >= 0 and sample.binCount >= 0 and sample.binOffset <= model.totalBins
              and sample.binCount <= model.totalBins - sample.binOffset, "sample bin range is out of bounds.");
      CudaBinningSample packed;
      packed.binOffset = sample.binOffset;
      packed.binCount = sample.binCount;
      packed.firstBin = out.bins.size();
      packed.variableCount = sample.binning.variableNames.size();
      packed.isDynamic = sample.useDynamicBinning;
      packed.isSmoothed = sample.binning.isSmoothed;
      out.hasSmoothedBinning |= packed.isSmoothed;
      packed.readHostVariables = sample.readHostVariables;
      for( auto bin : sample.binning.bins ){
        require(bin.binIndex >= 0 and bin.binIndex < sample.binCount
                and bin.firstInterval <= sample.binning.intervals.size()
                and bin.intervalCount <= sample.binning.intervals.size() - bin.firstInterval,
                "invalid strict bin descriptor.");
        const auto first = bin.firstInterval;
        bin.firstInterval = out.intervals.size();
        for( std::size_t j = 0; j < bin.intervalCount; ++j ){
          const auto& interval = sample.binning.intervals[first + j];
          require(interval.variableIndex < packed.variableCount and interval.lowerBound <= interval.upperBound
                  and std::isfinite(interval.lowerInverseWidth) and std::isfinite(interval.upperInverseWidth)
                  and interval.lowerInverseWidth >= 0. and interval.upperInverseWidth >= 0.
                  and (interval.lowerInverseWidth == 0.) == (interval.upperInverseWidth == 0.)
                  and (packed.isSmoothed or interval.lowerInverseWidth == 0.), "invalid bin interval or smoothing widths.");
          out.intervals.push_back(interval);
        }
        out.bins.push_back(bin);
      }
      packed.endBin = out.bins.size();
      packed.firstNode = out.binningNodes.size();
      if( packed.isDynamic or packed.isSmoothed ){
        // Keep original bin order, including overlapping selections. Bounds
        // include both smoothing shoulders and are rounded outwards.
        const double infinity = std::numeric_limits<double>::infinity();
        auto buildNode = [&](auto&& self_, std::size_t first_, std::size_t end_) -> void {
          const auto nodeIndex = out.binningNodes.size();
          const auto firstBound = out.binningBounds.size();
          out.binningNodes.push_back({first_, end_, firstBound, 0});
          out.binningBounds.resize(firstBound + packed.variableCount, {infinity, -infinity});
          for( std::size_t binIndex = first_; binIndex < end_; ++binIndex ){
            std::vector<CudaBinningBound> bounds(packed.variableCount, {-infinity, infinity});
            const auto& bin = out.bins[binIndex];
            for( std::size_t j = bin.firstInterval; j < bin.firstInterval + bin.intervalCount; ++j ){
              const auto& interval = out.intervals[j];
              double lower = interval.lowerBound, upper = interval.upperBound;
              if( interval.lowerInverseWidth != 0. ){
                lower -= 1. / interval.lowerInverseWidth;
                upper += 1. / interval.upperInverseWidth;
              }
              auto& bound = bounds[interval.variableIndex];
              bound.lower = std::max(bound.lower, std::nextafter(lower, -infinity));
              bound.upper = std::min(bound.upper, std::nextafter(upper, infinity));
            }
            for( std::size_t axis = 0; axis < packed.variableCount; ++axis ){
              auto& bound = out.binningBounds[firstBound + axis];
              bound.lower = std::min(bound.lower, bounds[axis].lower);
              bound.upper = std::max(bound.upper, bounds[axis].upper);
            }
          }
          if( end_ - first_ > 4 ){
            const auto middle = first_ + (end_ - first_) / 2;
            self_(self_, first_, middle);
            self_(self_, middle, end_);
            out.binningNodes[nodeIndex].endBin = first_; // Internal node.
          }
          out.binningNodes[nodeIndex].skip = out.binningNodes.size();
        };
        if( packed.firstBin != packed.endBin ){ buildNode(buildNode, packed.firstBin, packed.endBin); }
      }
      packed.endNode = out.binningNodes.size();
      out.samples.push_back(packed);
    }
    std::vector<std::pair<std::size_t, std::size_t>> variableRanges;
    std::vector<bool> targets(model.variableValues.size(), false);
    for( const auto& event : model.events ){
      require(event.firstVariable <= model.variableValues.size()
              and event.variableCount <= model.variableValues.size() - event.firstVariable
              and event.firstVariableResponse <= model.variableResponses.size()
              and event.variableResponseCount <= model.variableResponses.size() - event.firstVariableResponse,
              "invalid event observable range.");
      if( event.samplePosition >= out.samples.size() ){
        require(event.variableCount == 0 and event.variableResponseCount == 0, "observable event requires a valid sample.");
        continue;
      }
      const auto& sample = out.samples[event.samplePosition];
      require(not (sample.isDynamic or sample.isSmoothed) or event.variableCount >= sample.variableCount, "missing binning variables.");
      require(event.binIndex >= -1 and event.binIndex < sample.binCount, "fixed local bin is out of bounds.");
      if( event.variableCount ){ variableRanges.emplace_back(event.firstVariable, event.firstVariable + event.variableCount); }
      for( std::size_t j = 0; j < event.variableResponseCount; ++j ){
        const auto index = model.variableResponses[event.firstVariableResponse + j].variableIndex;
        require(index >= event.firstVariable and index - event.firstVariable < event.variableCount,
                "observable response targets another event.");
        require(not targets[index], "observable target has multiple writers.");
        targets[index] = true;
      }
    }
    if( out.hasSmoothedBinning ){
      require(model.samples.size() <= std::size_t(std::numeric_limits<int>::max() - model.totalBins),
              "outside bins exceed the CUDA index range.");
      require(model.events.size() < std::numeric_limits<std::size_t>::max() / sizeof(std::size_t), "contribution offsets overflow.");
      std::size_t maximumContributions = 0;
      for( const auto& event : model.events ){
        std::size_t capacity = 1;
        if( event.samplePosition < out.samples.size() and out.samples[event.samplePosition].isSmoothed ){
          const auto& sample = out.samples[event.samplePosition];
          capacity += sample.endBin - sample.firstBin;
        }
        require(capacity <= std::numeric_limits<std::size_t>::max() / sizeof(CudaBinContribution) - maximumContributions,
                "contribution buffer size overflow.");
        maximumContributions += capacity;
      }
    }
    std::sort(variableRanges.begin(), variableRanges.end());
    for( std::size_t i = 1; i < variableRanges.size(); ++i ){
      require(variableRanges[i - 1].second <= variableRanges[i].first, "event observable ranges overlap.");
    }
  }
  out.binOffsets.assign(std::size_t(model.totalBins) + 1, 0);
  std::vector<bool> seen(model.events.size(), false);
  for( const auto& event : model.events ){
    require(event.resultIndex < seen.size() and not seen[event.resultIndex], "event result slots must be a permutation.");
    seen[event.resultIndex] = true;
    require(event.globalBinIndex < model.totalBins, "event global bin is out of range.");
    if( event.globalBinIndex >= 0 ){ ++out.binOffsets[std::size_t(event.globalBinIndex) + 1]; }
  }
  buildHistogramEventIndex(model.events, out.binOffsets, out.binEvents);
  for( const auto& sample : view_.likelihood.samples ){
    require(sample.binOffset >= 0 and std::size_t(sample.binOffset) <= std::size_t(model.totalBins)
            and sample.dataSums.size() <= std::size_t(model.totalBins) - std::size_t(sample.binOffset),
            "likelihood bin range is outside the histogram.");
    require(sample.dataSums.empty() or sample.portableOptions.kind != LikelihoodKind::Host or bool(sample.evalBin), "likelihood evaluator is missing.");
  }
  out.events = model.eventWeights;
  out.formulaInputs = std::move(native.formulaInputs);
  out.compositePrograms = std::move(native.compositePrograms);
  out.compositeInstances = std::move(native.compositeInstances);
  out.compositeNodeData = std::move(native.compositeNodeData);
  out.compositeInstructions = std::move(native.compositeInstructions);
  out.compositeInputs = std::move(native.compositeInputs);
  for( const auto& program : native.formulaPrograms ){
    out.formulaPrograms.push_back({out.formulaInstructions.size(), program.instructions.size(),
                                   out.formulaConstants.size(), program.stackSize, program.nativePlan});
    out.formulaInstructions.insert(out.formulaInstructions.end(), program.instructions.begin(), program.instructions.end());
    out.formulaConstants.insert(out.formulaConstants.end(), program.constants.begin(), program.constants.end());
  }
  for( std::size_t i = 0; i < out.dials.size(); ++i ){
    const auto& dial = out.dials[i];
    if( dial.type != BackendDialType::Composite and dial.type != BackendDialType::Formula ){ continue; }
    const bool fast = dial.type == BackendDialType::Formula
        ? out.formulaPrograms[dial.formulaProgramIndex].nativePlan.kernel != NativeFormulaKernel::Interpreter
        : out.compositePrograms[out.compositeInstances[dial.compositeProgram].programIndex].isChain;
    if( fast ){ continue; }
    if( out.programScratchOffsets.empty() ){ out.programScratchOffsets.resize(out.dials.size()); }
    out.programScratchOffsets[i] = out.programScratchSize;
    std::size_t slots = dial.inputCount, stack = 0;
    if( dial.type == BackendDialType::Formula ){ stack = out.formulaPrograms[dial.formulaProgramIndex].stackSize; }
    else{
      const auto& program = out.compositePrograms[out.compositeInstances[dial.compositeProgram].programIndex];
      slots = program.scratchSize();
      for( std::size_t j = 0; j < program.instructionCount; ++j ){
        const auto& node = out.compositeInstructions[program.firstInstruction + j].dial;
        if( node.type == BackendDialType::Formula ){
          stack = std::max(stack, out.formulaPrograms[node.formulaProgramIndex].stackSize);
        }
      }
    }
    // Reserve at least one slot so constant/empty graphs have valid pointers.
    require(slots <= std::numeric_limits<std::size_t>::max() - stack, "program scratch overflow.");
    slots = std::max<std::size_t>(1, slots + stack);
    require(slots <= std::numeric_limits<std::size_t>::max() / sizeof(double) - out.programScratchSize,
            "program scratch allocation overflow.");
    out.programScratchSize += slots;
  }
  out.onlyFastPrograms = (not out.formulaPrograms.empty() or not out.compositePrograms.empty())
      and std::all_of(out.dials.begin(), out.dials.end(), [&](const auto& dial_){
        if( dial_.type == BackendDialType::Formula ){
          return out.formulaPrograms[dial_.formulaProgramIndex].nativePlan.kernel != NativeFormulaKernel::Interpreter;
        }
        if( dial_.type == BackendDialType::Composite ){
          return out.compositePrograms[out.compositeInstances[dial_.compositeProgram].programIndex].isChain;
        }
        return true;
      });
  out.inputs = model.dialInputs;
  out.payloads = model.dialPayloads;
  if( interleaveEventDials_ and not out.events.empty() ){
    // Adjacent warp lanes read adjacent indices at every multiplication step.
    // Cap padding at twice the original index storage; highly ragged models
    // retain the event-major layout instead of inflating device memory.
    constexpr std::size_t width = 32;
    const auto originalSize = model.eventDialIndices.size();
    const auto budget = originalSize > std::numeric_limits<std::size_t>::max() / 2
        ? std::numeric_limits<std::size_t>::max() : 2 * originalSize;
    std::size_t packedSize = 0;
    bool fits = true;
    for( std::size_t first = 0; first < out.events.size(); first += width ){
      std::size_t maxDials = 0;
      for( std::size_t i = first; i < std::min(first + width, out.events.size()); ++i ){
        maxDials = std::max(maxDials, out.events[i].dialCount);
      }
      if( maxDials > (budget - packedSize) / width ){ fits = false; break; }
      packedSize += width * maxDials;
    }
    if( fits and packedSize ){
      out.eventDialIndices.resize(packedSize);
      out.eventDialStride = width;
      std::size_t offset = 0;
      for( std::size_t first = 0; first < out.events.size(); first += width ){
        std::size_t maxDials = 0;
        for( std::size_t i = first; i < std::min(first + width, out.events.size()); ++i ){
          auto& event = out.events[i];
          maxDials = std::max(maxDials, event.dialCount);
          for( std::size_t j = 0; j < event.dialCount; ++j ){
            out.eventDialIndices[offset + width * j + i - first] = model.eventDialIndices[event.firstDial + j];
          }
          event.firstDial = offset + i - first;
        }
        offset += width * maxDials;
      }
    }
  }
  if( out.eventDialStride == 1 ){ out.eventDialIndices = model.eventDialIndices; }
  if( not dialSlots.empty() ){
    for( auto& index : out.eventDialIndices ){ index = dialSlots[index]; }
  }
  out.parameterCount = model.parameterCount;
  out.hasReweightCap = model.hasReweightCap;
  out.maxReweight = model.maxReweight;
  return out;
}

void Backends::CudaResponseUpdatePlan::select(const std::vector<std::size_t>& offsets_,
    const std::vector<double>& previous_, const std::vector<double>& current_,
    const std::vector<CudaSurfaceGroup>& surfaceGroups_,
    const std::vector<CudaMultiInputGroup>& multiInputGroups_,
    const std::vector<std::uint32_t>& indices_) {
  require(previous_.size() == current_.size() and offsets_.size() == current_.size() + 1,
          "response update parameter layout mismatch.");
  ranges.clear();
  dialCount = blockCount = 0;
  auto append = [&](std::size_t first_, std::size_t end_) {
    const auto count = end_ - first_;
    const bool direct = not indices_.empty() and end_ <= indices_.size()
        and std::size_t(indices_[first_]) + count - 1 == indices_[end_ - 1];
    ranges.push_back({direct ? std::size_t(indices_[first_]) : first_,
                      direct ? std::size_t(indices_[first_]) + count : end_, blockCount, direct});
    dialCount += count;
    blockCount += (count - 1) / cudaResponseBlockSize + 1;
  };
  for( std::size_t i = 0; i < current_.size(); ++i ){
    if( previous_[i] == current_[i] or offsets_[i] == offsets_[i + 1] ){ continue; }
    append(offsets_[i], offsets_[i + 1]);
  }
  for( const auto& group : surfaceGroups_ ){
    if( previous_[group.firstParameter] == current_[group.firstParameter]
        and previous_[group.secondParameter] == current_[group.secondParameter] ){ continue; }
    append(group.firstDial, group.endDial);
  }
  for( const auto& group : multiInputGroups_ ){
    const bool changed = std::any_of(group.parameters.begin(), group.parameters.end(),
        [&](std::size_t index_){ return previous_[index_] != current_[index_]; });
    if( not changed ){ continue; }
    append(group.firstDial, group.endDial);
  }
}

void Backends::CudaResponseUpdatePlan::selectExternal(const std::vector<CudaExternalGroup>& groups_,
    const std::vector<std::uint64_t>& previous_, const std::vector<std::uint64_t>& current_, bool force_) {
  require(previous_.size() == current_.size(), "external generation layout mismatch.");
  ranges.clear();
  dialCount = blockCount = 0;
  for( const auto& group : groups_ ){
    require(group.blockIndex < current_.size(), "invalid external dependency block.");
    if( not force_ and previous_[group.blockIndex] == current_[group.blockIndex] ){ continue; }
    const auto count = group.endDial - group.firstDial;
    if( count == 0 ){ continue; }
    ranges.push_back({group.firstDial, group.endDial, blockCount});
    dialCount += count;
    blockCount += (count - 1) / cudaResponseBlockSize + 1;
  }
}
