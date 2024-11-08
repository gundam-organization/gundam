//
// Created by Adrien Blanchet on 07/12/2022.
//

#include "Graph.h"

#ifndef DISABLE_USER_HEADER
LoggerInit([]{ Logger::setUserHeaderStr("[Graph-ROOT]"); });
#endif

void Graph::setAllowExtrapolation(bool allowExtrapolation) {
  _allowExtrapolation_ = allowExtrapolation;
}

bool Graph::getAllowExtrapolation() const {
  return _allowExtrapolation_;
}

void Graph::buildDial(const TGraph &graph, const std::string& option_) {
    LogExitIf(_graph_.GetN() != 0, "Graph already set.");
  LogExitIf(graph.GetN() == 0, "Invalid input graph");
  _graph_ = graph;
  _graph_.Sort();
}

double Graph::evalResponse(const DialInputBuffer& input_) const {
  double dialInput{input_.getInputBuffer()[0]};

#ifndef NDEBUG
  LogExitIf(not std::isfinite(dialInput), "Invalid input for Graph");
#endif

  if( not _allowExtrapolation_ ){
    if     (dialInput <= _graph_.GetX()[0])                { return _graph_.GetY()[0]; }
    else if(dialInput >= _graph_.GetX()[_graph_.GetN()-1]) { return _graph_.GetY()[_graph_.GetN() - 1]; }
  }
  return _graph_.Eval(dialInput);
}
