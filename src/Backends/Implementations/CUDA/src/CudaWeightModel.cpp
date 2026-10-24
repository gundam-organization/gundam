#include "CudaWeightModel.h"
#include "Semantics/BackendDialSemantics.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace {
  void require(bool condition_, const char* message_) {
    if( not condition_ ){ throw std::runtime_error(std::string("CudaBackend: ") + message_); }
  }

  bool isExternal(Backends::BackendDialType type_) {
    return type_ == Backends::BackendDialType::ExternalWeight or Backends::isTableDial(type_);
  }

  void validateDial(const Backends::PropagationView& model_, const Backends::BackendDialDescriptor& dial_) {
    using namespace Backends;
    require(dial_.payloadOffset <= model_.dialPayloads.size()
            and dial_.payloadSize <= model_.dialPayloads.size() - dial_.payloadOffset,
            "dial payload is outside the model.");
    require(dial_.payloadSize <= std::size_t(std::numeric_limits<int>::max() - 16), "dial payload is too large.");
    const auto size = dial_.payloadSize;
    switch( dial_.type ){
      case BackendDialType::Norm: break;
      case BackendDialType::Shift: require(size >= 1, "Shift requires a constant payload."); break;
      case BackendDialType::CompactSpline:
      case BackendDialType::MonotonicSpline: require(size >= 4, "spline requires at least two knots."); break;
      case BackendDialType::UniformSpline: require(size >= 6 and (size - 2) % 2 == 0, "invalid uniform spline payload."); break;
      case BackendDialType::GeneralSpline: require(size >= 8 and (size - 2) % 3 == 0, "invalid general spline payload."); break;
      case BackendDialType::Graph: require(size >= 2 and size % 2 == 0, "invalid graph payload."); break;
      case BackendDialType::Bilinear:
      case BackendDialType::Bicubic: Semantics::validateSurfaceDial(model_, dial_); break;
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
      const auto count = dial_.type == BackendDialType::Bilinear or dial_.type == BackendDialType::Bicubic ? 2u : 1u;
      require(dial_.inputCount == count and dial_.firstInput <= model_.dialInputs.size()
              and count <= model_.dialInputs.size() - dial_.firstInput, "invalid dial parameter input range.");
      for( std::size_t i = 0; i < count; ++i ){
        const auto& input = model_.dialInputs[dial_.firstInput + i];
        require(input.source == DialInputSource::Parameter and input.parameterIndex < model_.parameterCount,
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
  static_assert(std::is_trivially_copyable<BackendDialDescriptor>::value);
  static_assert(std::is_trivially_copyable<BackendDialInputDescriptor>::value);
  static_assert(std::is_trivially_copyable<BackendEventWeightDescriptor>::value);
  static_assert(std::is_trivially_copyable<CudaResponseUpdate>::value);
  static_assert(std::is_trivially_copyable<BackendTableTerm>::value);
  static_assert(std::is_trivially_copyable<ExternalWeightBlockDescriptor>::value);
  const auto& model = view_.propagation;
  require(not model.requiresEventBinning() and model.variableResponses.empty() and model.variableDialIndices.empty(),
          "observable responses, dynamic binning and smoothed binning are not implemented.");
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
  out.parameterDialOffsets.assign(model.parameterCount + 1, 0);
  for( const auto& dial : out.dials ){
    if( dial.type != BackendDialType::Shift and not isSurface(dial) and not isExternal(dial.type) ){
      ++out.parameterDialOffsets[model.dialInputs[dial.firstInput].parameterIndex + 1];
    }
  }
  for( std::size_t i = 1; i < out.parameterDialOffsets.size(); ++i ){
    out.parameterDialOffsets[i] += out.parameterDialOffsets[i - 1];
  }
  out.parameterDialIndices.resize(out.parameterDialOffsets.back());
  auto dialPositions = out.parameterDialOffsets;
  for( std::size_t i = 0; i < out.surfaceOffset; ++i ){
    const auto& dial = out.dials[i];
    if( dial.type != BackendDialType::Shift and not isExternal(dial.type) ){
      out.parameterDialIndices[dialPositions[model.dialInputs[dial.firstInput].parameterIndex]++]
          = static_cast<std::uint32_t>(i);
    }
  }
  std::map<std::pair<std::size_t, std::size_t>, std::vector<std::uint32_t>> surfaceGroups;
  for( std::size_t i = out.surfaceOffset; i < out.dials.size(); ++i ){
    const auto first = model.dialInputs[out.dials[i].firstInput].parameterIndex;
    const auto second = model.dialInputs[out.dials[i].firstInput + 1].parameterIndex;
    surfaceGroups[{std::min(first, second), std::max(first, second)}].push_back(static_cast<std::uint32_t>(i));
  }
  for( const auto& group : surfaceGroups ){
    const auto first = out.parameterDialIndices.size();
    out.parameterDialIndices.insert(out.parameterDialIndices.end(), group.second.begin(), group.second.end());
    out.surfaceGroups.push_back({group.first.first, group.first.second, first, out.parameterDialIndices.size()});
  }
  std::map<std::size_t, std::vector<std::uint32_t>> externalGroups;
  for( std::size_t i = 0; i < out.dials.size(); ++i ){
    if( isExternal(out.dials[i].type) ){
      externalGroups[out.dials[i].externalBlockIndex].push_back(static_cast<std::uint32_t>(i));
      ++out.externalDialCount;
    }
  }
  for( const auto& group : externalGroups ){
    const auto first = out.parameterDialIndices.size();
    out.parameterDialIndices.insert(out.parameterDialIndices.end(), group.second.begin(), group.second.end());
    out.externalGroups.push_back({group.first, first, out.parameterDialIndices.size()});
  }
  out.externalWeightBlocks = model.externalWeightBlocks;
  out.externalWeightCount = externalWeightCount;
  out.tableTerms = model.tableTerms;
  out.binOffsets.assign(std::size_t(model.totalBins) + 1, 0);
  std::vector<bool> seen(model.events.size(), false);
  for( const auto& event : model.events ){
    require(event.resultIndex < seen.size() and not seen[event.resultIndex], "event result slots must be a permutation.");
    seen[event.resultIndex] = true;
    require(event.globalBinIndex < model.totalBins, "event global bin is out of range.");
    if( event.globalBinIndex >= 0 ){ ++out.binOffsets[std::size_t(event.globalBinIndex) + 1]; }
  }
  for( std::size_t i = 1; i < out.binOffsets.size(); ++i ){ out.binOffsets[i] += out.binOffsets[i - 1]; }
  out.binEvents.resize(out.binOffsets.back());
  auto positions = out.binOffsets;
  for( const auto& event : model.events ){
    if( event.globalBinIndex >= 0 ){ out.binEvents[positions[event.globalBinIndex]++] = event.resultIndex; }
  }
  for( const auto& sample : view_.likelihood.samples ){
    require(sample.binOffset >= 0 and std::size_t(sample.binOffset) <= std::size_t(model.totalBins)
            and sample.dataSums.size() <= std::size_t(model.totalBins) - std::size_t(sample.binOffset),
            "likelihood bin range is outside the histogram.");
    require(sample.dataSums.empty() or bool(sample.evalBin), "likelihood evaluator is missing.");
  }
  out.events = model.eventWeights;
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
    const std::vector<CudaSurfaceGroup>& surfaceGroups_) {
  require(previous_.size() == current_.size() and offsets_.size() == current_.size() + 1,
          "response update parameter layout mismatch.");
  ranges.clear();
  dialCount = blockCount = 0;
  for( std::size_t i = 0; i < current_.size(); ++i ){
    if( previous_[i] == current_[i] or offsets_[i] == offsets_[i + 1] ){ continue; }
    const auto count = offsets_[i + 1] - offsets_[i];
    ranges.push_back({offsets_[i], offsets_[i + 1], blockCount});
    dialCount += count;
    blockCount += (count - 1) / cudaResponseBlockSize + 1;
  }
  for( const auto& group : surfaceGroups_ ){
    if( previous_[group.firstParameter] == current_[group.firstParameter]
        and previous_[group.secondParameter] == current_[group.secondParameter] ){ continue; }
    const auto count = group.endDial - group.firstDial;
    ranges.push_back({group.firstDial, group.endDial, blockCount});
    dialCount += count;
    blockCount += (count - 1) / cudaResponseBlockSize + 1;
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
