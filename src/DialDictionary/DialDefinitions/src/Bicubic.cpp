#include "Bicubic.h"
#include "CalculateBicubicSpline.h"

#include "GenericToolbox.Root.h"
#include "Logger.h"


#ifndef DISABLE_USER_HEADER
LoggerInit([]{ Logger::setUserHeaderStr("[Bicubic]"); });
#endif

void Bicubic::setAllowExtrapolation(bool allowExtrapolation) {
  _allowExtrapolation_ = allowExtrapolation;
}

bool Bicubic::getAllowExtrapolation() const {
  return _allowExtrapolation_;
}

void Bicubic::buildDial(const TH2& h2_, const std::string& option_){
    // Copy the spline data into local storage.  The local storage should be
    // easily packable for the GPU.
    //
    // data[0] -- nx (must be an integer value)
    // data[1] -- ny (must be an integer)
    // data[2]..data[2+nx-1] -- The x values for the knots
    // data[2+data[0]]..data[2+data[0]+data[1]-1] -- The y values for the knots
    // data[2+data[0]+data[1]]..data[2+data[0]+data[1]+data[0]*data[1]-1] -- The knots.

    int nx = h2_.GetNbinsX();
    int ny = h2_.GetNbinsY();

    _splineData_.reserve(2+nx+ny+nx*ny);
    _splineData_.emplace_back(nx);
    _splineData_.emplace_back(ny);
    for (int i=1; i<=nx; ++i) {
        _splineData_.emplace_back(h2_.GetXaxis()->GetBinCenter(i));
    }
    _splineBounds_.emplace_back(h2_.GetXaxis()->GetBinCenter(1), h2_.GetXaxis()->GetBinCenter(nx));
    for (int i=1; i<=ny; ++i) {
        _splineData_.emplace_back(h2_.GetYaxis()->GetBinCenter(i));
    }
    _splineBounds_.emplace_back(h2_.GetYaxis()->GetBinCenter(1), h2_.GetYaxis()->GetBinCenter(nx));
    for (int i = 1; i <= nx; ++i) {
        for (int j = 1; j <= ny; ++j) {
            _splineData_.emplace_back(h2_.GetBinContent(i,j));
        }
    }

}

double Bicubic::evalResponse(const DialInputBuffer& input_) const {
    double input0{input_.getInputBuffer()[0]};
    double input1{input_.getInputBuffer()[1]};

    if( not _allowExtrapolation_ ){
        if (input0 < _splineBounds_[0].min) input0 = _splineBounds_[0].min;
        if (input0 > _splineBounds_[0].max) input0 = _splineBounds_[0].max;
        if (input1 < _splineBounds_[1].min) input0 = _splineBounds_[1].min;
        if (input1 > _splineBounds_[1].max) input0 = _splineBounds_[1].max;
    }

    const double *data = _splineData_.data();
    const int nx = *(data++);
    const int ny = *(data++);
    const double* xx = data;
    data += nx;
    const double* yy = data;
    data += ny;
    const double* knots = data;
    return CalculateBicubicSpline(input0, input1, -1E20, 1E20,
                                  knots, nx, ny,
                                  xx, nx,
                                  yy, ny);
}
