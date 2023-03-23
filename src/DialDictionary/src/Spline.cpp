//
// Created by Adrien Blanchet on 30/11/2022.
//

#include "Spline.h"

void Spline::setAllowExtrapolation(bool allowExtrapolation) {
  _allowExtrapolation_ = allowExtrapolation;
}
void Spline::setSpline(const TSpline3 &spline) {
  _spline_ = spline;
}

const TSpline3 &Spline::getSpline() const {
  return _spline_;
}

void Spline::copySpline(const TSpline3* splinePtr_){
  // Don't check for override: when loading toy + mc data, these placeholders has to be filled up twice
//  LogThrowIf(_graph_.GetXmin() != _graph_.GetXmax(), "Spline already set");
  _spline_ = *splinePtr_;
}
void Spline::createSpline(TGraph* grPtr_){
//  LogThrowIf(_graph_.GetXmin() != _graph_.GetXmax(), "Spline already set");
  _spline_ = TSpline3(grPtr_->GetName(), grPtr_);
#ifdef ENABLE_SPLINE_DIAL_FAST_EVAL
  fs.stepsize = (_graph_.GetXmax() - _graph_.GetXmin())/((double) grPtr_->GetN());
#endif
}

double Spline::evaluateSpline(const DialInputBuffer& input_) const{
  if( not _allowExtrapolation_ ){
    if     (input_.getBuffer()[0] <= _spline_.GetXmin()) { return _spline_.Eval( _spline_.GetXmin() ); }
    else if(input_.getBuffer()[0] >= _spline_.GetXmax()) { return _spline_.Eval( _spline_.GetXmax() ); }
  }
  return _spline_.Eval( input_.getBuffer()[0] );
}
