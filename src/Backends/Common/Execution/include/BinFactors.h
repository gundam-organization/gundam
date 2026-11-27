#ifndef GUNDAM_BIN_FACTORS_H
#define GUNDAM_BIN_FACTORS_H

#include "EngineView.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <unordered_map>

namespace Backends {
inline constexpr auto noBinFactor = std::numeric_limits<std::uint32_t>::max();

// Use the executor's fixed bin mapping. A bin outside this layout cannot
// supply a factor, and its dial occurrences still count toward global coverage.
template<typename BinForEvent>
std::vector<std::uint32_t> findBinFactors(const PropagationView& model, std::size_t binCount, BinForEvent binForEvent_) {
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
        if( model.dials[*it].type == BackendDialType::Norm ){ candidates.push_back(*it); }
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
    const auto factor = std::find_if(candidates.begin(), candidates.end(), [&](std::uint32_t dial_){
      return coveredOccurrences.at(dial_) == 0;
    });
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
