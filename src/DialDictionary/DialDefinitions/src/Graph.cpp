//
// Created by Adrien Blanchet on 19/01/2023.
//

#include "Graph.h"

#include "CalculateGraph.h"


void Graph::setAllowExtrapolation(bool allowExtrapolation) {
  _allowExtrapolation_ = allowExtrapolation;
}

bool Graph::getAllowExtrapolation() const {
  return _allowExtrapolation_;
}

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
