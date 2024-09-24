#include "Bilinear.h"
#include "CalculateBilinearInterpolation.h"

#include "GenericToolbox.Root.h"
#include "Logger.h"


#ifndef DISABLE_USER_HEADER
LoggerInit([]{ Logger::setUserHeaderStr("[Bilinear]"); });
#endif

void Bilinear::setAllowExtrapolation(bool allowExtrapolation) {
  _allowExtrapolation_ = allowExtrapolation;
}

bool Bilinear::getAllowExtrapolation() const {
  return _allowExtrapolation_;
}

void Bilinear::buildDial(const TH2& h2_, const std::string& option_){
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
    _splineBounds_.push_back(std::make_pair(h2_.GetXaxis()->GetBinCenter(1),
                                            h2_.GetXaxis()->GetBinCenter(nx)));
    for (int i=1; i<=ny; ++i) {
        _splineData_.emplace_back(h2_.GetYaxis()->GetBinCenter(i));
    }
    _splineBounds_.push_back(std::make_pair(h2_.GetYaxis()->GetBinCenter(1),
                                            h2_.GetYaxis()->GetBinCenter(nx)));
    for (int i = 1; i <= nx; ++i) {
        for (int j = 1; j <= ny; ++j) {
            _splineData_.emplace_back(h2_.GetBinContent(i,j));
        }
    }

}

double Bilinear::evalResponse(const DialInputBuffer& input_) const {
    double input0{input_.getInputBuffer()[0]};
    double input1{input_.getInputBuffer()[1]};

    if( not _allowExtrapolation_ ){
        if (input0 < _splineBounds_[0].first) input0 = _splineBounds_[0].first;
        if (input0 > _splineBounds_[0].second) input0 = _splineBounds_[0].second;
        if (input1 < _splineBounds_[1].first) input1 = _splineBounds_[1].first;
        if (input1 > _splineBounds_[1].second) input1 = _splineBounds_[1].second;
    }

    const double *data = _splineData_.data();
    int nx = *(data++);
    int ny = *(data++);
    const double* xx = data;
    data += nx;
    const double* yy = data;
    data += ny;
    const double* knots = data;
    return CalculateBilinearInterpolation(
        input0, input1, -1E20, 1E20,
        knots, nx, ny,
        xx, nx,
        yy, ny);
}
