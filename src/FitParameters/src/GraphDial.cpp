//
// Created by Adrien BLANCHET on 02/12/2021.
//

#include "Logger.h"
#include "GenericToolbox.h"

#include "GraphDial.h"
#include "GlobalVariables.h"

LoggerInit([]{
  Logger::setUserHeaderStr("[GraphDial]");
})

GraphDial::GraphDial() {
  this->GraphDial::reset();
}

void GraphDial::reset() {
  this->Dial::reset();
  _dialType_ = DialType::Graph;
  _graph_ = TGraph();
}

void GraphDial::initialize() {
  this->Dial::initialize();
  LogThrowIf(
      _dialType_!=DialType::Graph,
      "_dialType_ is not Graph: " << DialType::DialTypeEnumNamespace::toString(_dialType_)
  )
  LogThrowIf( _graph_.GetN() == 0 )
  _isInitialized_ = true;
}

std::string GraphDial::getSummary() {
  std::stringstream ss;
  ss << Dial::getSummary();
  ss << "g{n=" << _graph_.GetN() << "}";
  return ss.str();
}


// disable cacahe?
//double GraphDial::evalResponse(const double &parameterValue_) {
//  return _graph_.Eval(parameterValue_);
//}
void GraphDial::fillResponseCache() {
  if     (_effectiveDialParameterValue_ <= _graph_.GetX()[0])                { _dialResponseCache_ = _graph_.GetY()[0]; }
  else if(_effectiveDialParameterValue_ >= _graph_.GetX()[_graph_.GetN()-1]) { _dialResponseCache_ = _graph_.GetY()[_graph_.GetN() - 1]; }
  else{
    _dialResponseCache_ = _graph_.Eval(_effectiveDialParameterValue_);
  }

  if(_dialResponseCache_ < 0){
    GlobalVariables::getThreadMutex().lock();
    _graph_.Print();
    LogError << GET_VAR_NAME_VALUE(_effectiveDialParameterValue_) << " -> " << _dialResponseCache_ << std::endl;
    LogThrow("NEGATIVE GRAPH RESPONSE");
  }

}

void GraphDial::setGraph(const TGraph &graph) {
  LogThrowIf(_graph_.GetN() != 0, "Graph already set.")
  LogThrowIf(_isInitialized_, "GraphDial already initialized")
  LogThrowIf(graph.GetN() == 0, "Invalid input graph")
  _graph_ = graph;
  _graph_.Sort();
}

