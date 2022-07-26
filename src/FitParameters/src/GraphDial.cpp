//
// Created by Adrien BLANCHET on 02/12/2021.
//

#include "GraphDial.h"
#include "DialSet.h"

#include "Logger.h"


LoggerInit([]{
  Logger::setUserHeaderStr("[GraphDial]");
});

GraphDial::GraphDial() : Dial{DialType::Graph} {
  this->GraphDial::reset();
}

void GraphDial::reset() {
  this->Dial::reset();
  _graph_ = TGraph();
}

void GraphDial::initialize() {
  this->Dial::initialize();
  LogThrowIf( _graph_.GetN() == 0 )
}

std::string GraphDial::getSummary() {
  std::stringstream ss;
  ss << Dial::getSummary();
  ss << " (graph{nPt=" << _graph_.GetN() << "})";
  return ss.str();
}


double GraphDial::calcDial(double parameterValue_) {
  if( not _owner_->isAllowDialExtrapolation() ){
    if     (parameterValue_ <= _graph_.GetX()[0])                { return _graph_.GetY()[0]; }
    else if(parameterValue_ >= _graph_.GetX()[_graph_.GetN()-1]) { return _graph_.GetY()[_graph_.GetN() - 1]; }
  }
  return _graph_.Eval(parameterValue_);
}

void GraphDial::setGraph(const TGraph &graph) {
  LogThrowIf(_graph_.GetN() != 0, "Graph already set.")
  LogThrowIf(graph.GetN() == 0, "Invalid input graph")
  _graph_ = graph;
  _graph_.Sort();
}

