#ifndef SHARED_SPLINE_COMMON_H_SEEN
#define SHARED_SPLINE_COMMON_H_SEEN

#ifndef GUNDAM_DEVICE_MATH_INLINE
#ifdef DEVICE_CALLABLE_INLINE
#define GUNDAM_DEVICE_MATH_INLINE static DEVICE_CALLABLE_INLINE
#else
#define GUNDAM_DEVICE_MATH_INLINE static inline
#endif
#endif

#ifndef GUNDAM_DEVICE_MATH_FLOAT
#ifdef DEVICE_FLOATING_POINT
#define GUNDAM_DEVICE_MATH_FLOAT DEVICE_FLOATING_POINT
#else
#define GUNDAM_DEVICE_MATH_FLOAT double
#endif
#endif

namespace GundamDeviceMath {
  GUNDAM_DEVICE_MATH_INLINE
  int ClampSplineSegmentIndex(int index_, int dim_) {
    if( index_ < 0 ){ return 0; }
    if( index_ > dim_ - 2 ){ return dim_ - 2; }
    return index_;
  }

  GUNDAM_DEVICE_MATH_INLINE
  double ClampResponse(double value_, double lowerBound_, double upperBound_) {
    if( value_ < lowerBound_ ){ return lowerBound_; }
    if( value_ > upperBound_ ){ return upperBound_; }
    return value_;
  }

  GUNDAM_DEVICE_MATH_INLINE
  double Abs(double value_) {
    return value_ < 0.0 ? -value_ : value_;
  }

  GUNDAM_DEVICE_MATH_INLINE
  double Min(double left_, double right_) {
    return left_ < right_ ? left_ : right_;
  }
}

#endif
