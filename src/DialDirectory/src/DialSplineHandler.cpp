//
// Created by Adrien Blanchet on 30/11/2022.
//

#include "DialSplineHandler.h"



void DialSplineHandler::setAllowExtrapolation(bool allowExtrapolation) {
  _allowExtrapolation_ = allowExtrapolation;
}
void DialSplineHandler::setSpline(const TSpline3 &spline) {
  _spline_ = spline;
}

const TSpline3 &DialSplineHandler::getSpline() const {
  return _spline_;
}

void DialSplineHandler::copySpline(const TSpline3* splinePtr_){
  // Don't check for override: when loading toy + mc data, these placeholders has to be filled up twice
//  LogThrowIf(_spline_.GetXmin() != _spline_.GetXmax(), "Spline already set");
  _spline_ = *splinePtr_;
}
void DialSplineHandler::createSpline(TGraph* grPtr_){
//  LogThrowIf(_spline_.GetXmin() != _spline_.GetXmax(), "Spline already set");
  _spline_ = TSpline3(grPtr_->GetName(), grPtr_);
#ifdef ENABLE_SPLINE_DIAL_FAST_EVAL
  fs.stepsize = (_spline_.GetXmax() - _spline_.GetXmin())/((double) grPtr_->GetN());
#endif
}

double DialSplineHandler::calculateSplineResponse(const DialInputBuffer& input_) const{
  if( not _allowExtrapolation_ ){
    if     (input_.getBuffer()[0] <= _spline_.GetXmin()) { return _spline_.Eval( _spline_.GetXmin() ); }
    else if(input_.getBuffer()[0] >= _spline_.GetXmax()) { return _spline_.Eval( _spline_.GetXmax() ); }
  }
  return _spline_.Eval( input_.getBuffer()[0] );
}
