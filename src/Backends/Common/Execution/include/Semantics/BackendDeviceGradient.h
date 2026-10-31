#ifndef GUNDAM_BACKEND_DEVICE_GRADIENT_H
#define GUNDAM_BACKEND_DEVICE_GRADIENT_H
#ifndef __METAL_VERSION__
#include "GradientTypes.h"
#include "Semantics/BackendLikelihood.h"
#define GUNDAM_GRADIENT_DEVICE
#ifdef __CUDACC__
#define GUNDAM_GRADIENT_INLINE __host__ __device__ inline
#else
#define GUNDAM_GRADIENT_INLINE inline
#endif
#else
#define GUNDAM_GRADIENT_DEVICE device
#define GUNDAM_GRADIENT_INLINE inline
#endif

namespace Backends::GradientSemantics {
  // Hermite interpolation and its exact first derivative. Same knot packing,
  // monotonic slope limiter and piecewise conventions as SharedSpline*.h.
  template<typename S> struct DialEvaluation { S value{}, derivative{}; };
  template<typename S> GUNDAM_GRADIENT_INLINE S absolute(S x_) { return x_ < S(0) ? -x_ : x_; }
  template<typename S> GUNDAM_GRADIENT_INLINE S minimum(S a_, S b_) { return a_ < b_ ? a_ : b_; }
  GUNDAM_GRADIENT_INLINE int segment(int i_, int count_) { return i_ < 0 ? 0 : (i_ > count_-2 ? count_-2 : i_); }

  template<typename Math> GUNDAM_GRADIENT_INLINE DialEvaluation<typename Math::Scalar> evaluateDial(
      GradientDial<typename Math::Scalar> d_, GUNDAM_GRADIENT_DEVICE const typename Math::Scalar* data_,
      typename Math::Scalar raw_) {
    using S = typename Math::Scalar;
    const auto kind = BackendDialType(d_.type);

    S x = raw_, inputSlope(1);
    if( d_.mirror ){
      x = Math::remainder(x - d_.mirrorMin, S(2) * d_.mirrorRange);
      inputSlope = x < S(0) ? S(-1) : S(1);
      x = absolute(x);
      if( x > d_.mirrorRange ){ x = S(2) * d_.mirrorRange - x; inputSlope = -inputSlope; }
      x += d_.mirrorMin;
    }
    DialEvaluation<S> out{x, S(1)};
    if( kind == BackendDialType::Shift || (kind == BackendDialType::Graph && d_.payloadSize < 4) )
      out = {data_[d_.payload], S(0)};
    else if( kind != BackendDialType::Norm ){
      auto data = data_ + d_.payload;
      int knots;
      S low, high;
      if( kind == BackendDialType::Graph ){
        knots = int(d_.payloadSize/2); low = data[1]; high = data[d_.payloadSize-1];
      }
      else if( kind == BackendDialType::GeneralSpline ){
        knots = int((d_.payloadSize-2)/3); low = data[4]; high = data[2+3*(knots-1)+2];
      }
      else{
        knots = kind == BackendDialType::UniformSpline ? int((d_.payloadSize-2)/2) : int(d_.payloadSize-2);
        low = data[0]; high = low + data[1]*S(knots-1);
      }
      if( !d_.extrapolate ){
        if( x <= low ){ x = low; inputSlope = S(0); }
        if( x >= high ){ x = high; inputSlope = S(0); }
      }
      S p1, p2, m1(0), m2(0), fx, step;
      if( kind == BackendDialType::Graph || kind == BackendDialType::GeneralSpline ){
        int left = 0, right = knots-1;
        while( right-left > 1 ){
          int middle = (left+right)/2;
          S edge = kind == BackendDialType::Graph ? data[2*middle+1] : data[2+3*middle+2];
          if( x > edge ) left = middle; else right = middle;
        }
        if( kind == BackendDialType::Graph ){
          step = data[2*(left+1)+1]-data[2*left+1]; fx = (x-data[2*left+1])/step;
          p1 = data[2*left]; p2 = data[2*(left+1)];
          out = {p1 + fx*(p2-p1), (p2-p1)/step};
        }
        else{
          step = data[2+3*(left+1)+2]-data[2+3*left+2]; fx = (x-data[2+3*left+2])/step;
          p1 = data[2+3*left]; p2 = data[2+3*(left+1)];
          m1 = data[2+3*left+1]*step; m2 = data[2+3*(left+1)+1]*step;
        }
      }
      else{
        step = data[1]; S xx = (x-low)/step;
        // Clamp before conversion; extrapolation still uses the outer segment.
        int raw = xx < S(-1) ? -2 : (xx < S(0) ? Math::integer(xx)-1 : (xx >= S(knots) ? knots : Math::integer(xx)));
        int i = kind == BackendDialType::UniformSpline
            ? segment(xx < S(0) ? 0 : (xx >= S(knots) ? knots : Math::integer(xx)), knots) : segment(raw, knots);
        fx = xx-S(i);
        if( kind == BackendDialType::UniformSpline ){
          p1 = data[2+2*i]; p2 = data[2+2*(i+1)];
          m1 = data[2+2*i+1]*step; m2 = data[2+2*(i+1)+1]*step;
        }
        else{
          p1 = data[2+i]; p2 = data[2+i+1];
          const int previous = segment(raw-1, knots), next = segment(raw+1, knots);
          S a = data[2+previous+1]-data[2+previous], b = p2-p1, c = data[2+next+1]-data[2+next];
          m1 = S(.5)*(a+b); m2 = S(.5)*(b+c);
          if( kind == BackendDialType::MonotonicSpline ){
            if( b*a <= S(0) ) m1 = S(0);
            if( c*b <= S(0) ) m2 = S(0);
            S bound1 = S(3)*minimum(absolute(a), absolute(b)), bound2 = S(3)*minimum(absolute(b), absolute(c));
            if( m1 > bound1 ) m1 = bound1;
            if( m1 < -bound1 ) m1 = -bound1;
            if( m2 > bound2 ) m2 = bound2;
            if( m2 < -bound2 ) m2 = -bound2;
          }
        }
      }
      if( kind != BackendDialType::Graph ){
        S a = S(2)*p1-S(2)*p2+m1+m2, b = S(3)*p2-S(3)*p1-m2-S(2)*m1;
        out = {((a*fx+b)*fx+m1)*fx+p1, ((S(3)*a*fx+S(2)*b)*fx+m1)/step};
      }
    }
    out.derivative *= inputSlope;
    if( d_.hasMin && out.value < d_.minResponse ) out = {d_.minResponse, S(0)};
    if( d_.hasMax && out.value > d_.maxResponse ) out = {d_.maxResponse, S(0)};
    return out;
  }

  template<typename Math> GUNDAM_GRADIENT_INLINE DialEvaluation<typename Math::Scalar> binScales(
      LikelihoodBin<typename Math::Scalar> bin_, typename Math::Scalar sum_, typename Math::Scalar square_) {
    using S = typename Math::Scalar;
    if( bin_.ignored ) return {S(0), S(0)};
    const S sigma = Math::sqrt(square_);
    const auto profile = LikelihoodSemantics::evaluateLiteGaussian<Math>(bin_.data, sum_, sigma);
    return {profile.predictionGradient, sigma > S(0) ? profile.uncertaintyGradient/sigma : S(0)};
  }
  template<typename S> GUNDAM_GRADIENT_INLINE void eventContributions(
      GradientEvent<S> event_, GradientWork<S> work_,
      GUNDAM_GRADIENT_DEVICE const DeviceIndex* indices_,
      GUNDAM_GRADIENT_DEVICE const DialEvaluation<S>* responses_,
      GUNDAM_GRADIENT_DEVICE const DialEvaluation<S>* scales_,
      GUNDAM_GRADIENT_DEVICE const S* weights_, GUNDAM_GRADIENT_DEVICE S* contributions_) {
    S prefix(1);
    for( DeviceIndex j = 0; j < event_.count; ++j ){
      contributions_[event_.first+j] = prefix;
      prefix *= responses_[indices_[event_.first+j]].value;
    }
    S scale(0);
    if( event_.bin < work_.bins && !(work_.useCap && prefix > work_.cap) ){
      const auto bin = scales_[event_.bin];
      scale = event_.base * (bin.value + weights_[event_.result]*bin.derivative);
    }
    S suffix(1);
    for( DeviceIndex j = event_.count; j-- > 0; ){
      const auto dial = responses_[indices_[event_.first+j]];
      contributions_[event_.first+j] = scale == S(0) ? S(0) : scale * contributions_[event_.first+j] * suffix * dial.derivative;
      suffix *= dial.value;
    }
  }
}
#endif
