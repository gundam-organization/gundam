//
// Created by Adrien Blanchet on 07/12/2022.
//

#include "RootGraph.h"

#include <algorithm>


double RootGraph::evalResponse(const DialInputBuffer& input_) const {
  double dialInput{input_.getInputBuffer()[0]};

#ifndef NDEBUG
  LogThrowIf(not std::isfinite(dialInput), "Invalid input for Graph");
#endif

  if( not _allowExtrapolation_ ){
    if     (dialInput <= _graph_.GetX()[0])                { return _graph_.GetY()[0]; }
    else if(dialInput >= _graph_.GetX()[_graph_.GetN()-1]) { return _graph_.GetY()[_graph_.GetN() - 1]; }
  }
  return _graph_.Eval(dialInput);
}


double RootGraph::evalGradient(const DialInputBuffer& input_, int iInput_) const {
  if( iInput_ != 0 or _graph_.GetN() < 2 ){ return 0.; }
  const double x = input_.getInputBuffer()[0];
  const auto* points = _graph_.GetX();
  const auto* values = _graph_.GetY();
  const int count = _graph_.GetN();
  if( not _allowExtrapolation_ and (x <= points[0] or x >= points[count - 1]) ){ return 0.; }

  // TGraph::Eval also accepts unsorted points. At knots use the right-hand
  // segment, except at the final knot where the last segment is used.
  if( _graph_.TestBit(TGraph::kIsSortedX) ){
    auto upperPoint = std::upper_bound(points, points + count, x);
    int lower = int(upperPoint - points) - 1;
    lower = std::max(0, std::min(count - 2, lower));
    const double width = points[lower + 1] - points[lower];
    return width == 0. ? 0. : (values[lower + 1] - values[lower]) / width;
  }
  int lower = -1, upper = -1;
  for( int i = 0; i < count; ++i ){
    if( points[i] <= x and (lower < 0 or points[i] > points[lower]) ){ lower = i; }
    if( points[i] > x and (upper < 0 or points[i] < points[upper]) ){ upper = i; }
  }
  if( lower < 0 ){
    lower = upper;
    upper = -1;
    for( int i = 0; i < count; ++i ){
      if( points[i] > points[lower] and (upper < 0 or points[i] < points[upper]) ){ upper = i; }
    }
  }
  else if( upper < 0 ){
    upper = lower;
    lower = -1;
    for( int i = 0; i < count; ++i ){
      if( points[i] < points[upper] and (lower < 0 or points[i] > points[lower]) ){ lower = i; }
    }
  }
  // Identical abscissas describe a constant response away from that abscissa.
  if( lower < 0 or upper < 0 ){ return 0.; }
  return (values[upper] - values[lower]) / (points[upper] - points[lower]);
}

std::vector<double> RootGraph::getSourceKnotPositions() const {
  std::vector<double> positions(_graph_.GetN());
  for( size_t i = 0; i < positions.size(); ++i ){
    positions[i] = _graph_.GetX()[i];
  }
  return positions;
}
