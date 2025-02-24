//
// Created by Adrien Blanchet on 07/12/2022.
//

#include "RootGraph.h"


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
