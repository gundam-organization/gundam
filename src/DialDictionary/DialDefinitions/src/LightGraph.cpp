//
// Created by Adrien Blanchet on 19/01/2023.
//

#include "LightGraph.h"

#include "CalculateGraph.h"


void LightGraph::setAllowExtrapolation(bool allowExtrapolation) {
  _allowExtrapolation_ = allowExtrapolation;
}

bool LightGraph::getAllowExtrapolation() const {
  return _allowExtrapolation_;
}

void LightGraph::buildDial(const TGraph &grf, const std::string& option_) {
  LogThrowIf(grf.GetN() == 0, "Invalid input graph");
  TGraph graph(grf);
  graph.Sort();

  int nPoints = graph.GetN();
  LogThrowIf(nPoints>15, "Light graphs must have fewer than 15 points");

  _Data_.reserve(2*nPoints);
  _Data_.clear();
  for (int i=0; i< nPoints; ++i) {
      _Data_.push_back(graph.GetY()[i]);
      _Data_.push_back(graph.GetX()[i]);
  }
}

double LightGraph::evalResponse(const DialInputBuffer& input_) const {
  double dialInput{input_.getInputBuffer()[0]};

#ifndef NDEBUG
  LogThrowIf(not std::isfinite(dialInput), "Invalid input for LightGraph");
#endif

  if( not _allowExtrapolation_ ){
    if     (dialInput <= _Data_[1])     { return _Data_[0]; }
    else if(dialInput >= _Data_.back()) { return _Data_[_Data_.size()-2]; }
  }

  return CalculateGraph(dialInput,-1E20,1E20,_Data_.data(),_Data_.size());
}
