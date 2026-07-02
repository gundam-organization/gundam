#include "Bicubic.h"
#include "CalculateBicubicSpline.h"

#include "GenericToolbox.Root.h"
#include "Logger.h"

void Bicubic::buildDial(const TH2& h2_){
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
    _splineBounds_.emplace_back(h2_.GetYaxis()->GetBinCenter(1), h2_.GetYaxis()->GetBinCenter(ny));
    for (int i = 1; i <= nx; ++i) {
        for (int j = 1; j <= ny; ++j) {
            _splineData_.emplace_back(h2_.GetBinContent(i,j));
        }
    }

}

Bicubic::PreparedBicubicCall Bicubic::prepareBicubicCall(
    const DialInputBuffer& input_, const bool forGradient_) const {

    PreparedBicubicCall call{};
    call.input0 = input_.getInputBuffer()[0];
    call.input1 = input_.getInputBuffer()[1];

    if( not _allowExtrapolation_ ){
        if( forGradient_ ){
            if( call.input0 <= _splineBounds_[0].min or call.input0 >= _splineBounds_[0].max ) call.valid = false;
            if( call.input1 <= _splineBounds_[1].min or call.input1 >= _splineBounds_[1].max ) call.valid = false;
        }
        else {
            if( call.input0 < _splineBounds_[0].min ) call.input0 = _splineBounds_[0].min;
            if( call.input0 > _splineBounds_[0].max ) call.input0 = _splineBounds_[0].max;
            if( call.input1 < _splineBounds_[1].min ) call.input1 = _splineBounds_[1].min;
            if( call.input1 > _splineBounds_[1].max ) call.input1 = _splineBounds_[1].max;
        }
    }

    const double *data = _splineData_.data();
    call.nx = *(data++);
    call.ny = *(data++);
    call.xx = data;
    data += call.nx;
    call.yy = data;
    data += call.ny;
    call.knots = data;

    return call;
}

double Bicubic::evalResponse(const DialInputBuffer& input_) const {
    const auto call = this->prepareBicubicCall(input_, false);
    return CalculateBicubicSpline(call.input0, call.input1, -1E20, 1E20,
                                  call.knots, call.nx, call.ny,
                                  call.xx, call.nx,
                                  call.yy, call.ny);
}

double Bicubic::evalGradient(const DialInputBuffer& input_, int iInput_) const {
    if( iInput_ < 0 or iInput_ > 1 ){ return 0.; }

    const auto call = this->prepareBicubicCall(input_, true);
    if( not call.valid ){ return 0.; }

    return CalculateBicubicSplineGradient(
        call.input0, call.input1, iInput_, -1E20, 1E20,
        call.knots, call.nx, call.ny,
        call.xx, call.nx,
        call.yy, call.ny);
}
