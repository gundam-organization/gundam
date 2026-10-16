#ifndef GUNDAM_BINNING_LAYOUT_H
#define GUNDAM_BINNING_LAYOUT_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

// Compiled bin geometry. Offsets and numeric descriptors can also be packed
// for device execution; no Event, ROOT or configuration objects are retained.
struct BinningInterval {
  std::size_t variableIndex{0};
  double lowerBound{0};
  double upperBound{0};
  // Zero denotes a strict selection, otherwise the two transition widths.
  double lowerInverseWidth{0};
  double upperInverseWidth{0};
};

struct BinningBin {
  int binIndex{-1};
  std::size_t firstInterval{0};
  std::size_t intervalCount{0};
};

struct BinningLayout {
  std::vector<std::string> variableNames;
  std::vector<BinningInterval> intervals;
  std::vector<BinningBin> bins;
  bool isSmoothed{false};
};

namespace BinningSemantics {
  inline double cumulativeKernel(double position_) {
    if( position_ <= -1 ){ return 0; }
    if( position_ >= 1 ){ return 1; }
    double t = (position_ + 1) / 2;
    return t * t * t * (10 + t * (-15 + 6 * t));
  }

  inline double fraction(const BinningLayout& layout_, const BinningBin& bin_, const double* values_) {
    double result{1};
    for( std::size_t i = bin_.firstInterval; i < bin_.firstInterval + bin_.intervalCount; ++i ){
      const auto& interval = layout_.intervals[i];
      double value = values_[interval.variableIndex];
      if( interval.lowerInverseWidth == 0 ){
        if( interval.lowerBound == interval.upperBound ){
          if( value != interval.lowerBound ){ return 0; }
        }
        else if( value < interval.lowerBound or value >= interval.upperBound ){ return 0; }
      }
      else{
        result *= std::max(0., cumulativeKernel((interval.upperBound - value) * interval.upperInverseWidth)
                              - cumulativeKernel((interval.lowerBound - value) * interval.lowerInverseWidth));
        if( result == 0 ){ return 0; }
      }
    }
    return result;
  }
}

// Immutable spatial index over conservative bin support bounds. Queries return
// bin positions in original order, preserving first-match semantics for strict
// overlapping bins. The scratch vector belongs to the caller/thread.
class BinningIndex {
public:
  void build(const BinningLayout& layout_);
  void findCandidates(const double* values_, std::vector<std::size_t>& candidates_) const;
  [[nodiscard]] std::size_t getNbNodes() const { return _nodes_.size(); }

private:
  struct Bounds { double lower{0}; double upper{0}; };
  struct Node {
    std::size_t first{0};
    std::size_t count{0};
    int left{-1};
    int right{-1};
  };
  int buildNode(std::size_t first_, std::size_t count_);
  void findCandidates(int node_, const double* values_, std::vector<std::size_t>& candidates_) const;
  bool buildGrid(const BinningLayout& layout_);

  std::size_t _dimensionCount_{0};
  std::vector<std::size_t> _binPositions_;
  std::vector<Bounds> _binBounds_;
  std::vector<Node> _nodes_;
  std::vector<Bounds> _nodeBounds_;
  std::vector<std::vector<double>> _gridEdges_;
  std::vector<std::size_t> _gridBins_;
};

#endif
