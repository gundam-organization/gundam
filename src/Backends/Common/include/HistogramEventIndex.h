#ifndef GUNDAM_HISTOGRAM_EVENT_INDEX_H
#define GUNDAM_HISTOGRAM_EVENT_INDEX_H

#include <cstddef>
#include <vector>

namespace Backends {
  // offsets_[bin + 1] initially holds the event count for that bin, with
  // offsets_[0] == 0. Preserve traversal order within each bin. Validation
  // and counting remain with the caller (MPS counts during weight packing).
  template<typename Events, typename Index>
  void buildHistogramEventIndex(const Events& events_, std::vector<Index>& offsets_, std::vector<Index>& indices_) {
    for( std::size_t i = 1; i < offsets_.size(); ++i ){ offsets_[i] += offsets_[i - 1]; }
    indices_.resize(offsets_.back());
    auto positions = offsets_;
    for( const auto& event : events_ ){
      if( event.globalBinIndex >= 0 ){
        indices_[positions[event.globalBinIndex]++] = static_cast<Index>(event.resultIndex);
      }
    }
  }
}
#endif
