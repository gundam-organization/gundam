#ifndef GUNDAM_BACKEND_LIKELIHOOD_H
#define GUNDAM_BACKEND_LIKELIHOOD_H
#ifndef __METAL_VERSION__
#include "LikelihoodTypes.h"
#include <cmath>
#include <limits>
#endif

namespace Backends::LikelihoodSemantics {
#ifndef __METAL_VERSION__
  struct DoubleMath {
    using Scalar = double;
    GUNDAM_LLH_INLINE static double sqrt(double x_) { return ::sqrt(x_); }
    GUNDAM_LLH_INLINE static double log(double x_) { return ::log(x_); }
    GUNDAM_LLH_INLINE static double log1p(double x_) { return ::log1p(x_); }
    GUNDAM_LLH_INLINE static bool finite(double x_) { return x_ <= std::numeric_limits<double>::max() && x_ >= -std::numeric_limits<double>::max(); }
    GUNDAM_LLH_INLINE static double minimum() { return std::numeric_limits<double>::min(); }
    GUNDAM_LLH_INLINE static double logMinimum() { return ::log(std::numeric_limits<double>::min()); }
    GUNDAM_LLH_INLINE static double infinity() { return std::numeric_limits<double>::infinity(); }
    GUNDAM_LLH_INLINE static double invalid() { return std::numeric_limits<double>::quiet_NaN(); }
  };
#endif
  template<typename Scalar> GUNDAM_LLH_INLINE Scalar absolute(Scalar x_) { return x_ < Scalar(0) ? -x_ : x_; }
  // Deliberately preserve std::min/max ordering, including NaN behavior.
  template<typename Scalar> GUNDAM_LLH_INLINE Scalar minimum(Scalar a_, Scalar b_) { return b_ < a_ ? b_ : a_; }
  template<typename Scalar> GUNDAM_LLH_INLINE Scalar maximum(Scalar a_, Scalar b_) { return a_ < b_ ? b_ : a_; }

  // Mirrors the validated 2022 implementation, including its compatibility
  // options and its stabilized expansion near the Asimov minimum. The caller
  // supplies a variance (not an error). NaN signals an error requiring a throw.
  template<typename Math> GUNDAM_LLH_INLINE typename Math::Scalar evaluate(
      typename Math::Scalar data_, typename Math::Scalar pred_, typename Math::Scalar variance_,
      LikelihoodOptions<typename Math::Scalar> options_) {
    using S = typename Math::Scalar;
    if( options_.kind == LikelihoodKind::Poisson ){
      if( pred_ <= S(0) ){ return Math::infinity(); }
      if( data_ <= S(0) ){ return S(2) * pred_; }
      const S residual = (pred_ - data_) / data_;
      if( absolute(residual) < S(1)/S(100) ){
        // Algebraically identical deviance; avoid cancellation at the minimum
        // where Metal's compensated logarithm error would dominate the signal.
        S series = S(1)/S(2) + residual * (-S(1)/S(3) + residual * (S(1)/S(4)
            + residual * (-S(1)/S(5) + residual * (S(1)/S(6) + residual * (-S(1)/S(7)
            + residual * (S(1)/S(8) + residual * (-S(1)/S(9) + residual/S(10))))))));
        return S(2) * data_ * residual * residual * series;
      }
      return S(2) * (pred_ - data_ + data_ * Math::log(data_ / pred_));
    }
    if( !Math::finite(variance_) || variance_ < S(0) ){
      return options_.throwIfInfLlh ? Math::invalid() : Math::infinity();
    }
    if( pred_ <= options_.expectedValueMinimum ){
      return pred_ <= S(0) && data_ > S(0) ? Math::infinity() : S(0);
    }
    S newmc = pred_, penalty(0), stableResidual(0);
    bool useStableResidual = false;
    const S epsilon = S(0x1p-52f);
    if( variance_ > S(0) ){
      S fractional = minimum(Math::sqrt(variance_) / pred_, options_.fractionalErrorLimit);
      S relativeDifference = (data_ - pred_) / pred_;
      S scaledVariance = pred_ * fractional * fractional;
      if( data_ > epsilon && pred_ > S(0) && absolute(relativeDifference) < S(1) / S(100)
          && fractional > S(0) && Math::finite(scaledVariance) ){
        S linear = S(1) + scaledVariance;
        S scaledShift = relativeDifference * (scaledVariance / linear);
        S delta = S(2) * scaledShift / (S(1) + Math::sqrt(S(1) + S(4) * scaledShift / linear));
        newmc = maximum(pred_ * (S(1) + delta), Math::minimum());
        S pull = delta / fractional;
        penalty = S(1) / S(2) * pull * pull;
        stableResidual = ((pred_ - data_) / data_) * ((S(1) + delta) / (linear + delta));
        useStableResidual = true;
      }
      else{
        S temp = pred_ * fractional * fractional - S(1);
        S discriminant = temp * temp + S(4) * data_ * fractional * fractional;
        if( discriminant < S(0) ){ return options_.throwIfInfLlh ? Math::invalid() : Math::infinity(); }
        S beta = (-temp + Math::sqrt(discriminant)) / S(2);
        newmc = maximum(pred_ * beta, Math::minimum());
        penalty = (beta - S(1)) * (beta - S(1)) / (S(2) * fractional * fractional);
      }
    }
    S stat(0);
    if( data_ <= epsilon ){ stat = newmc; }
    else if( newmc > Math::minimum() ){
      S residual = useStableResidual ? stableResidual : (newmc - data_) / data_;
      if( absolute(residual) < S(1) / S(100) ){
        S series = S(1)/S(2) + residual * (-S(1)/S(3) + residual * (S(1)/S(4)
            + residual * (-S(1)/S(5) + residual * (S(1)/S(6) + residual * (-S(1)/S(7)
            + residual * (S(1)/S(8) + residual * (-S(1)/S(9) + residual/S(10))))))));
        stat = data_ * residual * residual * series;
      }
      else if( absolute(residual) < S(1)/S(2) ){ stat = data_ * (residual - Math::log1p(residual)); }
      else{ stat = newmc - data_ + data_ * (Math::log(data_) - Math::log(newmc)); }
    }
    else{ stat = Math::minimum() - data_ + data_ * (Math::log(data_) - Math::logMinimum()); }
    S result = S(2) * stat;
    if( !options_.usePoissonLikelihood ){ result += S(2) * penalty; }
    return options_.throwIfInfLlh && !Math::finite(result) ? Math::invalid() : result;
  }
}
#endif
