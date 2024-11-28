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

  _data_.reserve(2 * nPoints);
  _data_.clear();
  for (int i=0; i< nPoints; ++i) {
      _data_.emplace_back(graph.GetY()[i]);
      _data_.emplace_back(graph.GetX()[i]);
  }
}

double LightGraph::evalResponse(const DialInputBuffer& input_) const {
  double dialInput{input_.getInputBuffer()[0]};

#ifndef NDEBUG
  LogThrowIf(not std::isfinite(dialInput), "Invalid input for LightGraph");
#endif

  if( not _allowExtrapolation_ ){
    if     ( dialInput <= _data_[1])     { return _data_[0]; }
    else if( dialInput >= _data_.back()) { return _data_[_data_.size() - 2]; }
  }

  double response{CalculateGraph(dialInput, -1E20, 1E20, _data_.data(), int(_data_.size()))};

  if( std::isnan(response) ){
    LogError << "NAN RESPONSE of dial " << this << std::endl;
    LogError << "_Data_: " << GenericToolbox::toString(_data_) << std::endl;
    LogError << "dialInput: " << dialInput << std::endl;
  }

  return response;
}
