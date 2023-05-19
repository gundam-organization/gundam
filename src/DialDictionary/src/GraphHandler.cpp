//
// Created by Adrien Blanchet on 07/12/2022.
//

#include "GraphHandler.h"

void GraphHandler::setAllowExtrapolation(bool allowExtrapolation) {
  _allowExtrapolation_ = allowExtrapolation;
}
void GraphHandler::setGraph(const TGraph &graph) {
  LogThrowIf(_graph_.GetN() != 0, "Graph already set.");
  LogThrowIf(graph.GetN() == 0, "Invalid input graph");
  _graph_ = graph;
  _graph_.Sort();
}

double GraphHandler::evaluateGraph(const DialInputBuffer& input_) const{
  if( not _allowExtrapolation_ ){
    if     (input_.getBuffer()[0] <= _graph_.GetX()[0])                { return _graph_.GetY()[0]; }
    else if(input_.getBuffer()[0] >= _graph_.GetX()[_graph_.GetN()-1]) { return _graph_.GetY()[_graph_.GetN() - 1]; }
  }
  return _graph_.Eval(input_.getBuffer()[0]);
}
