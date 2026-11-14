#include "BinningGeometry.h"

#include <limits>
#include <numeric>
#include <utility>

namespace Backends {

namespace {

  bool strictBinsOverlap(const Backends::BinningGeometry& layout_, const Backends::BinningBin& first_,
                         const Backends::BinningBin& second_) {
    for( std::size_t i = first_.firstInterval; i < first_.firstInterval + first_.intervalCount; ++i ){
      const auto& first = layout_.intervals[i];
      for( std::size_t j = second_.firstInterval; j < second_.firstInterval + second_.intervalCount; ++j ){
        const auto& second = layout_.intervals[j];
        if( first.variableIndex != second.variableIndex ){ continue; }
        if( first.lowerBound == first.upperBound ){
          if( second.lowerBound == second.upperBound ){
            if( first.lowerBound != second.lowerBound ){ return false; }
          }
          else if( first.lowerBound < second.lowerBound or first.lowerBound >= second.upperBound ){ return false; }
        }
        else if( second.lowerBound == second.upperBound ){
          if( second.lowerBound < first.lowerBound or second.lowerBound >= first.upperBound ){ return false; }
        }
        else if( first.upperBound <= second.lowerBound or second.upperBound <= first.lowerBound ){ return false; }
      }
    }
    return true;
  }

}

void BinningIndex::build(const BinningGeometry& layout_) {
  _dimensionCount_ = layout_.variableNames.size();
  _nodes_.clear();
  _nodeBounds_.clear();
  _gridEdges_.clear();
  _gridBins_.clear();
  _binPositions_.resize(layout_.bins.size());
  std::iota(_binPositions_.begin(), _binPositions_.end(), std::size_t(0));
  const double infinity = std::numeric_limits<double>::infinity();
  _binBounds_.assign(layout_.bins.size() * _dimensionCount_, {-infinity, infinity});
  for( std::size_t iBin = 0; iBin < layout_.bins.size(); ++iBin ){
    const auto& bin = layout_.bins[iBin];
    for( std::size_t i = bin.firstInterval; i < bin.firstInterval + bin.intervalCount; ++i ){
      const auto& interval = layout_.intervals[i];
      auto& bound = _binBounds_[iBin * _dimensionCount_ + interval.variableIndex];
      double lower = interval.lowerBound;
      double upper = interval.upperBound;
      if( interval.lowerInverseWidth != 0 ){
        // Fraction is F(upper-x)-F(lower-x): its support extends below
        // lower by the *lower* width and above upper by the *upper* width.
        lower -= 1 / interval.lowerInverseWidth;
        upper += 1 / interval.upperInverseWidth;
      }
      // Outward rounding makes pruning conservative at transition boundaries.
      bound = {std::nextafter(lower, -infinity), std::nextafter(upper, infinity)};
    }
  }
  if( not layout_.bins.empty() ){ buildNode(0, layout_.bins.size()); }
  if( not layout_.isSmoothed ){ buildGrid(layout_); }
}

std::vector<std::size_t> BinningIndex::buildReusableBinPositions(const BinningGeometry& layout_, std::size_t binCount_) const {
  std::vector<std::size_t> positions(binCount_, std::size_t(-1));
  if( layout_.isSmoothed ){ return positions; }
  const auto dimensions = layout_.variableNames.size();
  const auto& binBounds = getBinBounds();
  // Certify reuse once at build time. The existing tree prunes distant bins;
  // exact interval checks distinguish adjacent bins from actual overlaps.
  for( std::size_t position = 0; position < layout_.bins.size(); ++position ){
    const auto& bin = layout_.bins[position];
    if( bin.binIndex < 0 or std::size_t(bin.binIndex) >= positions.size() ){ continue; }
    auto overlapsEarlier = [&](const auto& self_, int nodeIndex_) -> bool {
      const auto& node = getNodes()[nodeIndex_];
      for( std::size_t axis = 0; axis < dimensions; ++axis ){
        const auto& bound = binBounds[position * dimensions + axis];
        const auto& other = getNodeBounds()[nodeIndex_ * dimensions + axis];
        if( bound.upper < other.lower or other.upper < bound.lower ){ return false; }
      }
      if( node.left >= 0 ){ return self_(self_, node.left) or self_(self_, node.right); }
      for( std::size_t i = node.first; i < node.first + node.count; ++i ){
        auto earlier = getBinPositions()[i];
        if( earlier < position and strictBinsOverlap(layout_, layout_.bins[earlier], bin) ){ return true; }
      }
      return false;
    };
    if( not overlapsEarlier(overlapsEarlier, 0) ){ positions[bin.binIndex] = position; }
  }
  return positions;
}

int BinningIndex::buildNode(std::size_t first_, std::size_t count_) {
  int index = int(_nodes_.size());
  _nodes_.push_back({first_, count_, -1, -1});
  const double infinity = std::numeric_limits<double>::infinity();
  _nodeBounds_.resize(_nodes_.size() * _dimensionCount_, {infinity, -infinity});
  for( std::size_t i = first_; i < first_ + count_; ++i ){
    for( std::size_t axis = 0; axis < _dimensionCount_; ++axis ){
      auto& bounds = _nodeBounds_[index * _dimensionCount_ + axis];
      const auto& bin = _binBounds_[_binPositions_[i] * _dimensionCount_ + axis];
      bounds.lower = std::min(bounds.lower, bin.lower);
      bounds.upper = std::max(bounds.upper, bin.upper);
    }
  }
  if( count_ <= 4 or _dimensionCount_ == 0 ){ return index; }
  std::size_t splitAxis{0};
  double largestSpan{-1};
  for( std::size_t axis = 0; axis < _dimensionCount_; ++axis ){
    const auto& bounds = _nodeBounds_[index * _dimensionCount_ + axis];
    double span = bounds.upper - bounds.lower;
    if( span > largestSpan ){ largestSpan = span; splitAxis = axis; }
  }
  auto center = [&](std::size_t bin_){
    const auto& bounds = _binBounds_[bin_ * _dimensionCount_ + splitAxis];
    // Infinite/open ranges do not provide a useful sorting coordinate.
    return std::isfinite(bounds.lower) and std::isfinite(bounds.upper)
        ? bounds.lower / 2 + bounds.upper / 2 : 0.;
  };
  std::size_t middle = first_ + count_ / 2;
  std::nth_element(_binPositions_.begin() + first_, _binPositions_.begin() + middle,
                   _binPositions_.begin() + first_ + count_,
                   [&](std::size_t a_, std::size_t b_){ return center(a_) < center(b_); });
  int left = buildNode(first_, middle - first_);
  int right = buildNode(middle, first_ + count_ - middle);
  _nodes_[index].left = left;
  _nodes_[index].right = right;
  return index;
}

bool BinningIndex::buildGrid(const BinningGeometry& layout_) {
  if( _dimensionCount_ == 0 or layout_.bins.empty() ){ return false; }
  std::vector<std::vector<double>> edges(_dimensionCount_);
  for( const auto& bin : layout_.bins ){
    if( bin.intervalCount != _dimensionCount_ ){ return false; }
    for( std::size_t i = bin.firstInterval; i < bin.firstInterval + bin.intervalCount; ++i ){
      const auto& interval = layout_.intervals[i];
      if( not std::isfinite(interval.lowerBound) or not std::isfinite(interval.upperBound)
          or interval.lowerBound >= interval.upperBound ){ return false; }
      auto& axis = edges[interval.variableIndex];
      axis.push_back(interval.lowerBound);
      axis.push_back(interval.upperBound);
    }
  }
  std::size_t cellCount{1};
  for( auto& axis : edges ){
    std::sort(axis.begin(), axis.end());
    axis.erase(std::unique(axis.begin(), axis.end()), axis.end());
    if( axis.size() < 2 or cellCount > layout_.bins.size() / (axis.size() - 1) ){ return false; }
    cellCount *= axis.size() - 1;
  }
  if( cellCount != layout_.bins.size() ){ return false; }
  std::vector<std::size_t> cells(cellCount, std::size_t(-1));
  for( std::size_t iBin = 0; iBin < layout_.bins.size(); ++iBin ){
    const auto& bin = layout_.bins[iBin];
    std::size_t cell{0};
    std::size_t stride{1};
    for( std::size_t axis = 0; axis < _dimensionCount_; ++axis ){
      auto begin = layout_.intervals.begin() + bin.firstInterval;
      auto interval = std::find_if(begin, begin + bin.intervalCount,
                                   [&](const BinningInterval& interval_){ return interval_.variableIndex == axis; });
      if( interval == begin + bin.intervalCount ){ return false; }
      auto lower = std::lower_bound(edges[axis].begin(), edges[axis].end(), interval->lowerBound);
      if( lower + 1 == edges[axis].end() or *(lower + 1) != interval->upperBound ){ return false; }
      cell += std::size_t(lower - edges[axis].begin()) * stride;
      stride *= edges[axis].size() - 1;
    }
    if( cells[cell] != std::size_t(-1) ){ return false; }
    cells[cell] = iBin;
  }
  _gridEdges_ = std::move(edges);
  _gridBins_ = std::move(cells);
  return true;
}

void BinningIndex::findCandidates(const double* values_, std::vector<std::size_t>& candidates_) const {
  candidates_.clear();
  // Strict legacy selections have defined comparison behaviour for NaN/Inf.
  // Preserve it by bypassing spatial pruning for nonfinite values.
  for( std::size_t axis = 0; axis < _dimensionCount_; ++axis ){
    if( not std::isfinite(values_[axis]) ){
      candidates_.resize(_binPositions_.size());
      std::iota(candidates_.begin(), candidates_.end(), std::size_t(0));
      return;
    }
  }
  if( not _gridBins_.empty() ){
    std::size_t cell{0};
    std::size_t stride{1};
    for( std::size_t axis = 0; axis < _dimensionCount_; ++axis ){
      const auto& edges = _gridEdges_[axis];
      auto upper = std::upper_bound(edges.begin(), edges.end(), values_[axis]);
      if( upper == edges.begin() or upper == edges.end() ){ return; }
      cell += std::size_t(upper - edges.begin() - 1) * stride;
      stride *= edges.size() - 1;
    }
    candidates_.push_back(_gridBins_[cell]);
    return;
  }
  if( not _nodes_.empty() ){ findCandidates(0, values_, candidates_); }
  std::sort(candidates_.begin(), candidates_.end());
}

void BinningIndex::findCandidates(int node_, const double* values_, std::vector<std::size_t>& candidates_) const {
  for( std::size_t axis = 0; axis < _dimensionCount_; ++axis ){
    const auto& bounds = _nodeBounds_[node_ * _dimensionCount_ + axis];
    if( values_[axis] < bounds.lower or values_[axis] > bounds.upper ){ return; }
  }
  const auto& node = _nodes_[node_];
  if( node.left < 0 ){
    for( std::size_t i = node.first; i < node.first + node.count; ++i ){ candidates_.push_back(_binPositions_[i]); }
    return;
  }
  findCandidates(node.left, values_, candidates_);
  findCandidates(node.right, values_, candidates_);
}

} // namespace Backends
