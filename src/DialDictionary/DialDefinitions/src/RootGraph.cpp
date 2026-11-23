//
// Created by Adrien Blanchet on 07/12/2022.
//

/// @file RootGraph.cpp
/// @brief Implements RootGraph::evalResponse().
/// @ingroup DialDictionary
///
/// The public contract of each method is documented in RootGraph.h. The
/// comments here describe how each method does its job.

#include "RootGraph.h"


/// @par Implementation
/// The range check reads the first and last entries of TGraph::GetX() and
/// TGraph::GetY() directly; inside the range, TGraph::Eval() is called with
/// its default (linear) interpolation. The debug-build error message says
/// "Graph", not "RootGraph".
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

/// @par Implementation
/// Copies TGraph::GetX() into a vector.
std::vector<double> RootGraph::getSourceKnotPositions() const {
  std::vector<double> positions(_graph_.GetN());
  for( size_t i = 0; i < positions.size(); ++i ){
    positions[i] = _graph_.GetX()[i];
  }
  return positions;
}
