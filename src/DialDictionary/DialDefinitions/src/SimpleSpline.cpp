//
// Created by Adrien Blanchet on 24/01/2023.
//

#include "SimpleSpline.h"
#include "CalculateGeneralSpline.h"
#include "CalculateUniformSpline.h"
#include "CalculateMonotonicSpline.h"
#include "CalculateCompactSpline.h"

#include "Logger.h"
#include "GenericToolbox.Root.h"


void SimpleSpline::setAllowExtrapolation(bool allowExtrapolation) {
  _allowExtrapolation_ = allowExtrapolation;
}

bool SimpleSpline::getAllowExtrapolation() const {
  return _allowExtrapolation_;
}

void SimpleSpline::buildDial(const TGraph& grf, const std::string& option_){
  LogThrowIf(not _splineData_.empty(), "Spline data already set.");
  TGraph graph_ = grf;

  // Copy the spline data into local storage.
  graph_.Sort();

  if( GenericToolbox::hasUniformlySpacedKnots(&graph_) ){
    _isUniform_ = true;
    _splineBounds_.min = graph_.GetX()[0];
    _splineBounds_.max = graph_.GetX()[graph_.GetN()-1];

    // If FAKE_UNIFORM_SPLINE is undefined, then reproduce TSpline3 assuming
    // uniform point spacing.
#ifndef FAKE_UNIFORM_SPLINE
    TSpline3 sp(Form("%p", &graph_), &graph_);
    _splineBounds_.min = sp.GetXmin();
    _splineBounds_.max = sp.GetXmax();

    _splineData_.resize(2 + sp.GetNp() * 2);
    _splineData_[0] = sp.GetXmin();
    _splineData_[1] = ((sp.GetXmax() - sp.GetXmin() ) / (sp.GetNp() - 1.0 ) );

    // Copy the spline data into local storage.
    for (int i = 0; i < sp.GetNp(); ++i) {
      double x;
      double y;
      sp.GetKnot(i, x, y);
      _splineData_[2 + 2*i + 0] = y;
      _splineData_[2 + 2*i + 1] = sp.Derivative(x);
    }
#else
    // If FAKE_UNIFORM_SPLINE is defined, then use a compact spline
    // representation.
    _splineBounds_.min = graph_.GetX()[0];
    _splineBounds_.max = graph_.GetX()[graph_.GetN()-1];

    _splineData_.resize(2 + graph_.GetN());
    _splineData_[0] = graph_.GetX()[0];
    _splineData_[1] = (graph_.GetX()[graph_.GetN()-1] - graph_.GetX()[0])/(graph_.GetN()-1.0);

    memcpy(&_splineData_[2], graph_.GetY(), graph_.GetN() * sizeof(double));
#endif
  }
  else{
    TSpline3 sp(Form("%p", &graph_), &graph_);
    _splineBounds_.min = sp.GetXmin();
    _splineBounds_.max = sp.GetXmax();

    _splineData_.resize(2 + sp.GetNp() * 3);
    _splineData_[0] = sp.GetXmin();
    _splineData_[1] = ((sp.GetXmax() - sp.GetXmin() ) / (sp.GetNp() - 1.0 ) );

    // Copy the spline data into local storage.
    for (int i = 0; i < sp.GetNp(); ++i) {
      double x;
      double y;
      sp.GetKnot(i, x, y);
      _splineData_[2 + 3*i + 0] = y;
      _splineData_[2 + 3*i + 1] = sp.Derivative(x);
      _splineData_[2 + 3*i + 2] = x;
    }
  }

}

void SimpleSpline::buildDial(const TSpline3& sp_, const std::string& option_) {
  LogThrow("NOT IMPLEMENTED");


  LogThrowIf(not _splineData_.empty(), "Spline data already set.");


}

double SimpleSpline::evalResponse(const DialInputBuffer& input_) const {
  double dialInput{input_.getInputBuffer()[0]};

#ifndef NDEBUG
  LogThrowIf(not std::isfinite(dialInput), "Invalid input for SimpleSpline");
#endif

  if( not _allowExtrapolation_ ){
    if     (dialInput <= _splineBounds_.min) { dialInput = _splineBounds_.min; }
    else if(dialInput >= _splineBounds_.max){ dialInput = _splineBounds_.max; }
  }

  if( _isUniform_ ){
#ifndef FAKE_UNIFORM_SPLINE
    // Use the same knots and slopes as TSpline3.
    return CalculateUniformSpline( dialInput, -1E20, 1E20, _splineData_.data(), int(_splineData_.size()) );
#else
#ifndef FAKE_UNIFORM_SPLINE_WITH_COMPACT_SPLINE
    // Fake the TSpline3 spline but impose a monotonic constraint.
    return CalculateMonotonicSpline( dialInput, -1E20, 1E20, _splineData_.data(), int(_splineData_.size()) );
#else
    // Fake the TSpline3 spline.  This does a pretty good of reproducing
    // TSpline3 as long as it uses the default boundary conditions.
    return CalculateCompactSpline( dialInput, -1E20, 1E20, _splineData_.data(), int(_splineData_.size()) );
#endif
#endif
  }
  return CalculateGeneralSpline( dialInput, -1E20, 1E20, _splineData_.data(), int(_splineData_.size()) );
}
