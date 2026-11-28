#ifndef GUNDAM_BIN_FACTORS_H
#define GUNDAM_BIN_FACTORS_H

#include "DialDependencies.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <unordered_map>

namespace Backends {
inline constexpr auto noBinFactor = std::numeric_limits<std::uint32_t>::max();

inline bool isPublishedWeightDial(const BackendDialDescriptor& dial_) {
  return dial_.type == BackendDialType::ExternalWeight or isTableDial(dial_.type);
}

inline bool canFactorWeightDial(const BackendDialDescriptor& dial_) {
  return isPublishedWeightDial(dial_) or (dial_.type != BackendDialType::Shift and dial_.inputCount != 0);
}

struct BinFactorDependencies {
  std::vector<std::size_t> weightParameters, weightExternalBlocks;
  std::vector<bool> residualParameters, residualExternalBlocks;
  std::size_t factoredBins{0}, reusableParameters{0}, reusableExternalBlocks{0};
};

template<typename VisitDial>
BinFactorDependencies classifyBinFactorDependencies(const PropagationView& model_,
                                                    const std::vector<std::uint32_t>& factors_, VisitDial visitDial_) {
  BinFactorDependencies out;
  std::vector<bool> used(model_.dials.size()), factored(factors_.empty() ? 0 : model_.dials.size());
  for( auto dial : model_.eventDialIndices ){ used.at(dial) = true; }
  for( auto dial : factors_ ){
    if( dial == noBinFactor ){ continue; }
    factored[dial] = true;
    ++out.factoredBins;
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
std::vector<std::uint32_t> findBinFactors(const PropagationView& model, std::size_t binCount,
                                        BinForEvent binForEvent_, IsCandidate isCandidate_) {
  if( model.requiresEventBinning() or model.hasReweightCap ){ return {}; }
  std::vector<std::vector<std::uint32_t>> commonDials(binCount);
  std::vector<std::size_t> eventsPerBin(binCount);
  const auto firstIndex = model.eventDialIndices.begin();
  for( const auto& event : model.events ){
    const auto bin = binForEvent_(event);
    if( bin == std::size_t(-1) ){ continue; }
    if( bin >= binCount ){ return {}; }
    auto& candidates = commonDials[bin];
    const auto& weight = model.eventWeights.at(event.resultIndex);
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
  // Prefer dials for which every occurrence can be factored. This also checks
  // references from events outside the histogram and from mixed bins.
  std::unordered_map<std::uint32_t, std::int64_t> coveredOccurrences;
  for( std::size_t bin = 0; bin < binCount; ++bin ){
    for( auto dial : commonDials[bin] ){ coveredOccurrences[dial] += eventsPerBin[bin]; }
  }
  if( coveredOccurrences.empty() ){ return {}; }
  for( auto dial : model.eventDialIndices ){
    auto found = coveredOccurrences.find(dial);
    if( found != coveredOccurrences.end() ){ --found->second; }
  }
  std::vector<std::uint32_t> factors(binCount, noBinFactor);
  for( std::size_t bin = 0; bin < binCount; ++bin ){
    const auto& candidates = commonDials[bin];
    const auto covered = [&](std::uint32_t dial_){
      return coveredOccurrences.at(dial_) == 0;
    };
    auto factor = std::find_if(candidates.begin(), candidates.end(), [&](std::uint32_t dial_){
      return model.dials[dial_].type == BackendDialType::Norm and covered(dial_);
    });
    // Preserve the existing Norm choice when other dial types also qualify.
    if( factor == candidates.end() ){ factor = std::find_if(candidates.begin(), candidates.end(), covered); }
    if( factor != candidates.end() ){ factors[bin] = *factor; }
  }
  // If a dial lost a bin to a different factor, its response still affects
  // residual products in that bin. Discard it everywhere.
  for( std::size_t bin = 0; bin < binCount; ++bin ){
    for( auto dial : commonDials[bin] ){
      if( dial != factors[bin] ){ coveredOccurrences.at(dial) = -1; }
    }
  }
  for( auto& dial : factors ){
    if( dial != noBinFactor and coveredOccurrences.at(dial) != 0 ){ dial = noBinFactor; }
  }
  return factors;
}
}

#endif
