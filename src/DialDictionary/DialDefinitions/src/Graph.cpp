//
// Created by Adrien Blanchet on 19/01/2023.
//

/// @file Graph.cpp
/// @brief Implements Graph: packing the knots and evaluating the
///        piecewise-linear response.
/// @ingroup DialDictionary
///
/// The public contract of each method is documented in Graph.h. The
/// comments here describe how each method does its job.

#include "Graph.h"

#include "CalculateGraph.h"


/// @par Implementation
/// Stores the flag in `_allowExtrapolation_`.
void Graph::setAllowExtrapolation(bool allowExtrapolation) {
  _allowExtrapolation_ = allowExtrapolation;
}

/// @par Implementation
/// Returns `_allowExtrapolation_`.
bool Graph::getAllowExtrapolation() const {
  return _allowExtrapolation_;
}

/// @par Implementation
/// Copies the graph and sorts the copy with TGraph::Sort() (by increasing
/// x), then interleaves y and x into `_data_`.
void Graph::buildDial(const TGraph &grf, const std::string& option_) {
  LogThrowIf(grf.GetN() == 0, "Invalid input graph");
  TGraph graph(grf);
  graph.Sort();

  int nPoints = graph.GetN();

  _data_.reserve(2 * nPoints);
  _data_.clear();
  for (int i=0; i< nPoints; ++i) {
      _data_.push_back(graph.GetY()[i]);
      _data_.push_back(graph.GetX()[i]);
  }
}
/// @par Implementation
/// Interleaves y and x of each knot into `_data_`, in the given order.
void Graph::buildDial(const std::vector<DialUtils::DialPoint>& pointList_){
  _data_.reserve(2 * pointList_.size());
  _data_.clear();
  for( auto& point : pointList_){
    _data_.emplace_back(point.y);
    _data_.emplace_back(point.x);
  }
}

/// @par Implementation
/// The out-of-range cases are handled here, before CalculateGraph(), because
/// that kernel always extends the end segments. CalculateGraph() then finds
/// the segment around x by a binary search over the knot positions (it uses
/// the last segment above the last knot and the first segment below the
/// first one). Its output bounds are set to +-1e20, so it does not clamp;
/// clamping is left to the DialResponseSupervisor.
double Graph::evalResponse(const DialInputBuffer& input_) const {
  double dialInput{input_.getInputBuffer()[0]};

#ifndef NDEBUG
  LogThrowIf(not std::isfinite(dialInput), "Invalid input for Graph");
#endif

  if( not _allowExtrapolation_ ){
    if     ( dialInput <= _data_[1])     { return _data_[0]; }
    else if( dialInput >= _data_.back()) { return _data_[_data_.size() - 2]; }
  }

  return CalculateGraph(dialInput, -1E20, 1E20, _data_.data(), _data_.size());
}

/// @par Implementation
/// `_data_` is laid out as `{y0, x0, y1, x1, ...}`, so the positions are the
/// entries at odd indices.
std::vector<double> Graph::getSourceKnotPositions() const {
  std::vector<double> positions;
  positions.reserve(_data_.size() / 2);
  for( size_t i = 1; i < _data_.size(); i += 2 ){
    positions.emplace_back(_data_[i]);
  }
  return positions;
}
