//
// Created by Adrien BLANCHET on 02/12/2021.
//

#include "Logger.h"
#include "GenericToolbox.h"

#include "GraphDialLin.h"
#include "GlobalVariables.h"

LoggerInit([]{
  Logger::setUserHeaderStr("[GraphDialLin]");
})

GraphDialLin::GraphDialLin() : Dial{DialType::GraphLin} {
  this->GraphDialLin::reset();
}

void GraphDialLin::reset() {
  this->Dial::reset();
  _graph_ = TGraph();
}

void GraphDialLin::initialize() {
  this->Dial::initialize();
  LogThrowIf( _graph_.GetN() == 0 )
}

std::string GraphDialLin::getSummary() {
  std::stringstream ss;
  ss << Dial::getSummary();
  ss << "g{n=" << _graph_.GetN() << "}";
  return ss.str();
}

// disable cacahe?
//double GraphDial::evalResponse(const double &parameterValue_) {
//  return _graph_.Eval(parameterValue_);
//}
void GraphDialLin::fillResponseCache() {
  if     (_effectiveDialParameterValue_ <= _graph_.GetX()[0])                { _dialResponseCache_ = _graph_.GetY()[0]; }
  else if(_effectiveDialParameterValue_ >= _graph_.GetX()[_graph_.GetN()-1]) { _dialResponseCache_ = _graph_.GetY()[_graph_.GetN() - 1]; }
  else{
    _dialResponseCache_ = _graph_.Eval(_effectiveDialParameterValue_, NULL, "");
  }

    // Checks
  //if(_minDialResponse_ == _minDialResponse_ and _dialResponseCache_ < _minDialResponse_ ){
  //  _dialResponseCache_ = _minDialResponse_;
  //}

  if (_dialResponseCache_ < 0){ 
    _dialResponseCache_ = 0;
  }

  if(_dialResponseCache_ < 0){
    GlobalVariables::getThreadMutex().lock();
    _graph_.Print();
    LogError << GET_VAR_NAME_VALUE(_effectiveDialParameterValue_) << " -> " << _dialResponseCache_ << std::endl;
    LogThrow("NEGATIVE GRAPH RESPONSE");
  }

}

void GraphDialLin::setGraph(const TGraph &graph) {
  LogThrowIf(_graph_.GetN() != 0, "Graph already set.")
  LogThrowIf(graph.GetN() == 0, "Invalid input graph")
  _graph_ = graph;
  _graph_.Sort();
}

