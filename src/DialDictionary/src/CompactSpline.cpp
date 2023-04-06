//
// Created by Adrien Blanchet on 22/01/2023.
//

#include "CompactSpline.h"
#include "CalculateCompactSpline.h"

#include "GenericToolbox.Root.h"
#include "Logger.h"


LoggerInit([]{
  Logger::setUserHeaderStr("[CompactSpline]");
});

void CompactSpline::setAllowExtrapolation(bool allowExtrapolation) {
  _allowExtrapolation_ = allowExtrapolation;
}

bool CompactSpline::getAllowExtrapolation() const {
  return _allowExtrapolation_;
}

void CompactSpline::buildDial(const TSpline3& spline, std::string option) {
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
  buildDial(xPoint,yPoint,dummy,option);
}

void CompactSpline::buildDial(const TGraph& grf, std::string option) {
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
  buildDial(xPoint,yPoint,dummy,option);
}


void CompactSpline::buildDial(const std::vector<double>& v1,
                              const std::vector<double>& v2,
                              const std::vector<double>& v3,
                              std::string option) {
  LogThrowIf(not _splineData_.empty(), "Spline data already set.");

  _splineBounds_.first = v1.front();
  _splineBounds_.second = v1.back();

  _splineData_.resize(2 + v1.size());
  _splineData_[0] = v1.front();
  _splineData_[1] = (v1.back() - v1.front())/(v1.size()-1.0);

  for (int i=0; i<v1.size()-1; ++i) {
      double d = std::abs(v1[i] - _splineData_[0] - i*_splineData_[1]);
      LogThrowIf(d>1E-7, "CompactSplines require uniform knots: " << GET_VAR_NAME_VALUE(d));
  }

  for(int i=0; i<v2.size(); ++i) _splineData_[2+i] = v2[i];

}

double CompactSpline::evalResponse(const DialInputBuffer& input_) const {
  double dialInput{input_.getBuffer()[0]};

  if( not _allowExtrapolation_ ){
    if     (input_.getBuffer()[0] <= _splineBounds_.first) { dialInput = _splineBounds_.first; }
    else if(input_.getBuffer()[0] >= _splineBounds_.second){ dialInput = _splineBounds_.second; }
  }

  return CalculateCompactSpline( dialInput, -1E20, 1E20, _splineData_.data(), int(_splineData_.size()) );
}
