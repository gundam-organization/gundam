#ifndef GUNDAM_BIN_FACTORS_H
#define GUNDAM_BIN_FACTORS_H

#include "DialDependencies.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <unordered_map>

namespace Backends {
struct BinFactors {
  std::vector<std::uint32_t> offsets, dials;
};

inline bool isPublishedWeightDial(const BackendDialDescriptor& dial_) {
  return dial_.type == BackendDialType::ExternalWeight or isTableDial(dial_.type);
}

inline bool canFactorWeightDial(const BackendDialDescriptor& dial_) {
  return isPublishedWeightDial(dial_) or (dial_.type != BackendDialType::Shift and dial_.inputCount != 0);
}

struct BinFactorDependencies {
  std::vector<std::size_t> weightParameters, weightExternalBlocks;
  std::vector<bool> residualParameters, residualExternalBlocks;
  std::size_t factoredBins{0}, multiFactorBins{0}, reusableParameters{0}, reusableExternalBlocks{0};
};

template<typename VisitDial>
BinFactorDependencies classifyBinFactorDependencies(const PropagationView& model_,
                                                    const BinFactors& factors_, VisitDial visitDial_) {
  BinFactorDependencies out;
  std::vector<bool> used(model_.dials.size()), factored(factors_.dials.empty() ? 0 : model_.dials.size());
  for( auto dial : model_.eventDialIndices ){ used.at(dial) = true; }
  for( auto dial : factors_.dials ){ factored[dial] = true; }
  for( std::size_t i = 1; i < factors_.offsets.size(); ++i ){
    const auto count = factors_.offsets[i] - factors_.offsets[i - 1];
    out.factoredBins += count != 0;
    out.multiFactorBins += count > 1;
  }
  std::vector<bool> parameters(model_.parameterCount), blocks(model_.externalWeightBlocks.size());
  out.residualParameters.resize(parameters.size());
  out.residualExternalBlocks.resize(blocks.size());
  for( std::size_t i = 0; i < used.size(); ++i ){
    if( not used[i] ){ continue; }
    const bool residual = factored.empty() or not factored[i];
    if( not visitDial_(std::uint32_t(i), residual) ){ return {}; }
    const auto& dial = model_.dials[i];
    if( dial.type == BackendDialType::Shift ){ continue; }
    visitDialDependencies(model_, dial, [&](const auto& input_){
      parameters.at(input_.parameterIndex) = true;
      if( residual ){ out.residualParameters[input_.parameterIndex] = true; }
    }, [&](std::size_t block_){
      blocks.at(block_) = true;
      if( residual ){ out.residualExternalBlocks[block_] = true; }
    });
  }
  for( std::size_t i = 0; i < parameters.size(); ++i ){
    if( not parameters[i] ){ continue; }
    out.weightParameters.push_back(i);
    out.reusableParameters += not out.residualParameters[i];
  }
  for( std::size_t i = 0; i < blocks.size(); ++i ){
    if( not blocks[i] ){ continue; }
    out.weightExternalBlocks.push_back(i);
    out.reusableExternalBlocks += not out.residualExternalBlocks[i];
  }
  return out;
}

// Use the executor's fixed bin mapping. A bin outside this layout cannot
// supply a factor, and its dial occurrences still count toward global coverage.
template<typename BinForEvent, typename IsCandidate>
BinFactors findBinFactors(const PropagationView& model_, std::size_t binCount_,
                          BinForEvent binForEvent_, IsCandidate isCandidate_) {
  if( model_.requiresEventBinning() or model_.hasReweightCap ){ return {}; }
  std::vector<std::vector<std::uint32_t>> commonDials(binCount_);
  std::vector<std::size_t> eventsPerBin(binCount_);
  const auto firstIndex = model_.eventDialIndices.begin();
  for( const auto& event : model_.events ){
    const auto bin = binForEvent_(event);
    if( bin == std::size_t(-1) ){ continue; }
    if( bin >= binCount_ ){ return {}; }
    auto& candidates = commonDials[bin];
    const auto& weight = model_.eventWeights.at(event.resultIndex);
    const auto first = firstIndex + weight.firstDial;
    const auto last = first + weight.dialCount;
    if( eventsPerBin[bin]++ == 0 ){
      for( auto it = first; it != last; ++it ){
        if( isCandidate_(*it) ){ candidates.push_back(*it); }
      }
    }
    // A factor must occur exactly once in every product for this bin.
    candidates.erase(std::remove_if(candidates.begin(), candidates.end(), [&](std::uint32_t dial_){
      return std::count(first, last, dial_) != 1;
    }), candidates.end());
  }
  // A dial must be factored everywhere it occurs, including when a bin
  // shares its event's dial range with an event outside the histogram.
  std::unordered_map<std::uint32_t, std::int64_t> coveredOccurrences;
  for( std::size_t bin = 0; bin < binCount_; ++bin ){
    for( auto dial : commonDials[bin] ){ coveredOccurrences[dial] += eventsPerBin[bin]; }
  }
  if( coveredOccurrences.empty() ){ return {}; }
  for( const auto& event : model_.eventWeights ){
    for( auto it = firstIndex + event.firstDial, end = it + event.dialCount; it != end; ++it ){
      const auto found = coveredOccurrences.find(*it);
      if( found != coveredOccurrences.end() ){ --found->second; }
    }
  }
  BinFactors factors;
  factors.offsets.reserve(binCount_ + 1);
  factors.offsets.push_back(0);
  for( const auto& candidates : commonDials ){
    for( auto dial : candidates ){
      if( coveredOccurrences.at(dial) != 0 ){ continue; }
      if( factors.dials.size() == std::numeric_limits<std::uint32_t>::max() ){ return {}; }
      factors.dials.push_back(dial);
    }
    factors.offsets.push_back(static_cast<std::uint32_t>(factors.dials.size()));
  }
  if( factors.dials.empty() ){ return {}; }
  return factors;
}
}

#endif
