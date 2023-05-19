//
// Created by Adrien Blanchet on 30/11/2022.
//

#include "Spline.h"

LoggerInit([]{
  Logger::setUserHeaderStr("[Spline]");
});

void Spline::setAllowExtrapolation(bool allowExtrapolation) {
  _allowExtrapolation_ = allowExtrapolation;
}

void Spline::buildDial(const TSpline3& spline, const std::string& option_) {
  std::vector<double> xPoint(spline.GetNp());
  std::vector<double> yPoint(spline.GetNp());
  std::vector<double> dummy;
  xPoint.clear(); yPoint.clear();
  for (int i = 0; i<spline.GetNp(); ++i) {
    double x; double y;
    spline.GetKnot(i,x,y);
    xPoint.push_back(x);
    yPoint.push_back(y);
  }
  buildDial(xPoint,yPoint,dummy,option_);
  _spline_.SetTitle(spline.GetTitle());
}

void Spline::buildDial(const TGraph& grf, const std::string& option_) {
  std::vector<double> xPoint(grf.GetN());
  std::vector<double> yPoint(grf.GetN());
  std::vector<double> dummy;
  xPoint.clear(); yPoint.clear();
  for (int i = 0; i<grf.GetN(); ++i) {
      double x; double y;
      grf.GetPoint(i,x,y);
      xPoint.push_back(x);
      yPoint.push_back(y);
  }
  buildDial(xPoint,yPoint,dummy,option_);
  _spline_.SetTitle(grf.GetName());
}

void Spline::buildDial(const std::vector<double>& v1,
                       const std::vector<double>& v2,
                       const std::vector<double>& v3,
                       const std::string& option_) {
  _spline_ = TSpline3("",const_cast<double*>(v1.data()),
                      const_cast<double*>(v2.data()), v1.size());
}

const TSpline3 &Spline::getSpline() const {return _spline_;}

double Spline::evalResponse(const DialInputBuffer& input_) const {
  const double dialInput{input_.getBuffer()[0]};

#ifndef NDEBUG
  LogThrowIf(not std::isfinite(dialInput), "Invalid input for Spline");
#endif

  if( not _allowExtrapolation_ ){
    if (dialInput <= _spline_.GetXmin()) {
      return _spline_.Eval( _spline_.GetXmin() );
    }
    else if (dialInput >= _spline_.GetXmax()) {
      return _spline_.Eval( _spline_.GetXmax() );
    }
  }
  return _spline_.Eval( dialInput );
}
