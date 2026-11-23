/// @file Bicubic.cpp
/// @brief Implements Bicubic: packs a TH2 grid and evaluates it with the
///        shared CalculateBicubicSpline() kernel.
/// @ingroup DialDictionary
///
/// The public contract of each method is documented in Bicubic.h. The
/// comments here describe how each method does its job.

#include "Bicubic.h"
#include "CalculateBicubicSpline.h"

#include "GenericToolbox.Root.h"
#include "Logger.h"


/// @par Implementation
/// Same packing as Bilinear::buildDial(). Appends to `_splineData_` in layout
/// order after reserving its final size: the two counts, the X bin centres,
/// the Y bin centres, then the bin contents with the X index in the outer
/// loop, which gives the X-major order `ix*ny + iy` that the kernel indexes.
/// The bounds are the first and last bin centres of each axis. ROOT bin
/// numbers start at 1, so the under- and overflow bins are skipped.
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

/// @par Implementation
/// Clamps each input to its own axis range unless extrapolation is allowed,
/// then unpacks `_splineData_` into the counts and the pointers to the X
/// positions, Y positions and knot values, and calls CalculateBicubicSpline()
/// with output bounds of -1E20 and 1E20 (no effective clamp). The position
/// arrays are passed with the same counts as the grid (`nnx = nx`,
/// `nny = ny`), the non-uniform form of the kernel call. The kernel finds the
/// cell with a fixed four-step binary search, so an input outside the knot
/// range uses the edge cell; there each 1D step sets both end slopes to the
/// edge interval's slope, which makes the extrapolation linear.
/// @note The kernel's search always reads element 8 of each position array,
///       even when the axis has fewer knots (the value is then multiplied by
///       zero). For very small grids, those with `ny*(nx+1) <= 8` such as
///       2x2, this read falls past the end of `_splineData_`.
double Bicubic::evalResponse(const DialInputBuffer& input_) const {
    double input0{input_.getInputBuffer()[0]};
    double input1{input_.getInputBuffer()[1]};

    if( not _allowExtrapolation_ ){
        if (input0 < _splineBounds_[0].min) input0 = _splineBounds_[0].min;
        if (input0 > _splineBounds_[0].max) input0 = _splineBounds_[0].max;
        if (input1 < _splineBounds_[1].min) input1 = _splineBounds_[1].min;
        if (input1 > _splineBounds_[1].max) input1 = _splineBounds_[1].max;
    }

    const double *data = _splineData_.data(); // read cursor over _splineData_, advanced past each section
    const int nx = *(data++);
    const int ny = *(data++);
    const double* xx = data; // X positions of the knots (bin centres)
    data += nx;
    const double* yy = data; // Y positions of the knots (bin centres)
    data += ny;
    const double* knots = data; // knot values, X-major (ix*ny + iy)
    return CalculateBicubicSpline(input0, input1, -1E20, 1E20,
                                  knots, nx, ny,
                                  xx, nx,
                                  yy, ny);
}
