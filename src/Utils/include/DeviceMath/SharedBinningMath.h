#ifndef GUNDAM_SHARED_BINNING_MATH_H
#define GUNDAM_SHARED_BINNING_MATH_H

// Scalar semantics compiled unchanged for host C++ and Metal.
namespace GundamBinningMath {
  template<typename Scalar> inline Scalar cumulativeKernel(Scalar position_) {
    if( position_ <= Scalar(-1) ){ return Scalar(0); }
    if( position_ >= Scalar(1) ){ return Scalar(1); }
    Scalar t = (position_ + Scalar(1)) / Scalar(2);
    return t * t * t * (Scalar(10) + t * (Scalar(-15) + Scalar(6) * t));
  }
  template<typename Scalar> inline bool contains(Scalar lower_, Scalar upper_, Scalar value_) {
    return lower_ == upper_ ? value_ == lower_ : !(value_ < lower_ || value_ >= upper_);
  }
  template<typename Scalar> inline Scalar intervalFraction(Scalar lower_, Scalar upper_, Scalar lowerInverse_, Scalar upperInverse_, Scalar value_) {
    if( lowerInverse_ == Scalar(0) ){ return contains(lower_, upper_, value_) ? Scalar(1) : Scalar(0); }
    Scalar result = cumulativeKernel((upper_ - value_) * upperInverse_) - cumulativeKernel((lower_ - value_) * lowerInverse_);
    return result > Scalar(0) ? result : Scalar(0);
  }
}
#endif
