#ifndef GUNDAM_BACKEND_BINNING_GEOMETRY_H
#define GUNDAM_BACKEND_BINNING_GEOMETRY_H

#include "Semantics/BackendBinningSemantics.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

class Histogram;

namespace Backends {

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

struct BinningGeometry {
  std::vector<std::string> variableNames;
  std::vector<BinningInterval> intervals;
  std::vector<BinningBin> bins;
  bool isSmoothed{false};
};

[[nodiscard]] BinningGeometry buildBinningGeometry(const Histogram& histogram_);

namespace BinningSemantics {
  inline double cumulativeKernel(double position_) {
    return GundamBinningMath::cumulativeKernel(position_);
  }

  inline double fraction(const BinningGeometry& layout_, const BinningBin& bin_, const double* values_) {
    return binFraction(bin_, layout_.intervals.data(), values_);
  }
}

// Immutable spatial index over conservative bin support bounds. Queries return
// bin positions in original order, preserving first-match semantics for strict
// overlapping bins. The scratch vector belongs to the caller/thread.
class BinningIndex {
public:
  void build(const BinningGeometry& layout_);
  void findCandidates(const double* values_, std::vector<std::size_t>& candidates_) const;
  // Bin-index lookup for strict bins with no overlapping earlier selection.
  // Missing/unsafe identifiers map to size_t(-1); smoothed bins are never reusable.
  // layout_ must be the geometry used to build this index.
  [[nodiscard]] std::vector<std::size_t> buildReusableBinPositions(const BinningGeometry& layout_, std::size_t binCount_) const;
  [[nodiscard]] std::size_t getNbNodes() const { return _nodes_.size(); }

  struct Bounds { double lower{0}; double upper{0}; };
  struct Node {
    std::size_t first{0};
    std::size_t count{0};
    int left{-1};
    int right{-1};
  };
  [[nodiscard]] const auto& getNodes() const { return _nodes_; }
  [[nodiscard]] const auto& getNodeBounds() const { return _nodeBounds_; }
  [[nodiscard]] const auto& getBinBounds() const { return _binBounds_; }
  [[nodiscard]] const auto& getBinPositions() const { return _binPositions_; }

private:
  int buildNode(std::size_t first_, std::size_t count_);
  void findCandidates(int node_, const double* values_, std::vector<std::size_t>& candidates_) const;
  bool buildGrid(const BinningGeometry& layout_);

  std::size_t _dimensionCount_{0};
  std::vector<std::size_t> _binPositions_;
  std::vector<Bounds> _binBounds_;
  std::vector<Node> _nodes_;
  std::vector<Bounds> _nodeBounds_;
  std::vector<std::vector<double>> _gridEdges_;
  std::vector<std::size_t> _gridBins_;
};

} // namespace Backends

#endif
