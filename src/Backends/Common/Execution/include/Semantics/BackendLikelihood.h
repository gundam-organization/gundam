#ifndef GUNDAM_BACKEND_LIKELIHOOD_H
#define GUNDAM_BACKEND_LIKELIHOOD_H
#ifndef __METAL_VERSION__
#include "LikelihoodTypes.h"
#include <cfloat>
#include <cmath>
#endif

namespace Backends::LikelihoodSemantics {
#ifndef __METAL_VERSION__
  struct DoubleMath {
    using Scalar = double;
    GUNDAM_LLH_INLINE static double sqrt(double x_) { return ::sqrt(x_); }
    GUNDAM_LLH_INLINE static double log(double x_) { return ::log(x_); }
    GUNDAM_LLH_INLINE static double log1p(double x_) { return ::log1p(x_); }
    GUNDAM_LLH_INLINE static double exp(double x_) { return ::exp(x_); }
    GUNDAM_LLH_INLINE static double expm1(double x_) { return ::expm1(x_); }
    GUNDAM_LLH_INLINE static double hypot(double a_, double b_) { return ::hypot(a_, b_); }
    // C constants and CUDA-supported math functions avoid host-only constexpr
    // calls from the device pass (no --expt-relaxed-constexpr required).
    GUNDAM_LLH_INLINE static bool finite(double x_) { return x_ <= DBL_MAX && x_ >= -DBL_MAX; }
    GUNDAM_LLH_INLINE static double minimum() { return DBL_MIN; }
    GUNDAM_LLH_INLINE static double logMinimum() { return ::log(DBL_MIN); }
    GUNDAM_LLH_INLINE static double infinity() { return HUGE_VAL; }
    GUNDAM_LLH_INLINE static double invalid() { return ::nan(""); }
  };
#endif
  template<typename Scalar> GUNDAM_LLH_INLINE Scalar absolute(Scalar x_) { return x_ < Scalar(0) ? -x_ : x_; }
  // Deliberately preserve std::min/max ordering, including NaN behavior.
  template<typename Scalar> GUNDAM_LLH_INLINE Scalar minimum(Scalar a_, Scalar b_) { return b_ < a_ ? b_ : a_; }
  template<typename Scalar> GUNDAM_LLH_INLINE Scalar maximum(Scalar a_, Scalar b_) { return a_ < b_ ? b_ : a_; }

  // Portable counterparts of the Utils deviance and the two lite references.
  // Inputs are current standard deviations; the resident dispatch converts the
  // histogram variance once. Derivatives are partials with the other input fixed.
  template<typename S> struct LiteEvaluation {
    S value{0};
    S predictionGradient{0};
    S uncertaintyGradient{0};
  };
  template<typename S> GUNDAM_LLH_INLINE S copySign(S magnitude_, S sign_) {
    return sign_ < S(0) ? -absolute(magnitude_) : absolute(magnitude_);
  }
  template<typename Math> GUNDAM_LLH_INLINE typename Math::Scalar halfPoissonDeviance(
      typename Math::Scalar observed_, typename Math::Scalar mean_, typename Math::Scalar residual_) {
    using S = typename Math::Scalar;
    if( observed_ == S(0) ) return mean_;
    if( mean_ <= S(0) ) return Math::infinity();
    if( absolute(residual_) < S(1)/S(100) ) {
      S series(0);
      for( int order = 20; order >= 2; --order )
        series = S(order % 2 == 0 ? 1 : -1) / S(order) + residual_ * series;
      return observed_ * residual_ * residual_ * series;
    }
    if( absolute(residual_) < S(1)/S(2) ) return observed_ * (residual_ - Math::log1p(residual_));
    return mean_ - observed_ + observed_ * (Math::log(observed_) - Math::log(mean_));
  }
  template<typename Math> GUNDAM_LLH_INLINE typename Math::Scalar halfPoissonDeviance(
      typename Math::Scalar observed_, typename Math::Scalar mean_) {
    using S = typename Math::Scalar;
    return halfPoissonDeviance<Math>(observed_, mean_, observed_ > S(0) ? (mean_-observed_)/observed_ : S(0));
  }
  template<typename Math> GUNDAM_LLH_INLINE typename Math::Scalar logAdd(
      typename Math::Scalar first_, typename Math::Scalar second_) {
    const auto larger = maximum(first_, second_);
    return larger + Math::log1p(Math::exp(minimum(first_, second_) - larger));
  }

  template<typename Math> GUNDAM_LLH_INLINE LiteEvaluation<typename Math::Scalar> evaluateLiteGaussian(
      typename Math::Scalar data_, typename Math::Scalar pred_, typename Math::Scalar err_) {
    using S = typename Math::Scalar;
    const S inf = Math::infinity();
    const S nan = Math::invalid();
    if( not Math::finite(data_) or not Math::finite(pred_) or not Math::finite(err_)
        or data_ < 0 or pred_ < 0 or err_ < 0 ) return {inf, nan, nan};
    const S n = data_, mu = pred_, sigma = err_;
    if( sigma == 0 ) {
      if( mu == 0 and n > 0 ) return {inf, nan, nan};
      return {S(2 * halfPoissonDeviance<Math>(n, mu)),
              S(n == 0 ? 2 : 2 * (mu - n) / mu), 0};
    }
    S x, pull, predictionGradient, residual;
    if( sigma >= Math::sqrt(maximum(mu, n)) ) {
      // Normalize by sigma^2 without ever forming it (it may overflow).
      const S scaledMu = (mu / sigma) / sigma;
      const S scaledN = (n / sigma) / sigma;
      const S discriminant = Math::hypot(scaledMu - 1, 2 * Math::sqrt(scaledN));
      x = n == 0 ? 0 : n / ((1 - scaledMu + discriminant) / 2);
      const S shift = (n - mu) / ((scaledMu + 1 + discriminant) / 2);
      pull = shift / sigma;
      predictionGradient = -2 * (pull / sigma);
      residual = n > 0 ? (((mu - n) / sigma) * (x / sigma) / n)
          / (1 + (x / sigma) / sigma) : 0;
    }
    else {
      const S variance = sigma * sigma;
      const S scale = maximum(mu, n);
      const S m = mu / scale, v = variance / scale, d = n / scale;
      const S scaledSigma = sigma / Math::sqrt(scale);
      const S discriminant = Math::hypot(m - v, 2 * scaledSigma * Math::sqrt(d));
      x = m >= v ? scale * ((m - v + discriminant) / 2)
          : (n == 0 ? 0 : n * (2 * v / (v - m + discriminant)));
      const S denominator = x / scale + v;
      // Envelope theorem: (x-mu)/sigma = sigma*(n-mu)/(x+sigma^2).
      // This also preserves the prediction derivative for tiny sigma.
      const S slope = denominator > 0 ? ((mu - n) / scale) / denominator : 0;
      pull = -sigma * slope;
      predictionGradient = 2 * slope;
      const S fraction = x > variance ? 1 / (1 + variance / x)
          : (x / variance) / (1 + x / variance);
      residual = n > 0 ? ((mu - n) / n) * fraction : 0;
    }
    return {S(2 * halfPoissonDeviance<Math>(n, x, residual) + pull * pull),
            S(predictionGradient), S(-2 * pull * (pull / sigma))};
  }

  template<typename Math> GUNDAM_LLH_INLINE LiteEvaluation<typename Math::Scalar> evaluateLitePoisson(
      typename Math::Scalar data_, typename Math::Scalar pred_, typename Math::Scalar err_) {
    using S = typename Math::Scalar;
    const S inf = Math::infinity();
    const S nan = Math::invalid();
    if( not Math::finite(data_) or not Math::finite(pred_) or not Math::finite(err_)
        or data_ < 0 or pred_ < 0 or err_ < 0 ) return {inf, nan, nan};
    const S n = data_, mu = pred_, sigma = err_;
    if( mu == 0 ) {
      // Zero total MC weight with nonzero variance is outside this model.
      if( sigma != 0 or n != 0 ) return {inf, nan, nan};
      return {0, 2, 0};
    }
    if( sigma == 0 ) return {S(2 * halfPoissonDeviance<Math>(n, mu)),
                             S(n == 0 ? 2 : 2 * (mu - n) / mu), 0};
    if( n == mu ) return {0, 0, 0};

    // Work with logarithmic scales: a and beta can exceed the floating-point
    // range even when their combined contribution to the objective is finite.
    const S logMu = Math::log(mu), logSigma = Math::log(sigma);
    const S logA = 2 * (logMu - logSigma);
    const S logDenominator = logAdd<Math>(logMu, logA);
    const S difference = n - mu;
    const S logDifference = Math::log(absolute(difference));
    const S logDelta = logDifference - logDenominator;
    const S delta = copySign(Math::exp(logDelta), difference);
    S logBeta, logPenalty;
    if( absolute(delta) < (S(1)/S(100)) ) {
      // h(1+delta)=delta-log1p(delta). Preserve a*delta^2 without
      // underflowing delta^2 or overflowing a at a precise Asimov minimum.
      S series = 0;
      for( int order = 20; order >= 2; --order )
        series = (order % 2 == 0 ? S(1) : -S(1)) / S(order) + delta * series;
      logBeta = Math::log1p(delta);
      logPenalty = logA + 2 * logDelta + Math::log(series);
    }
    else {
      logBeta = (n == 0 ? logA : logAdd<Math>(Math::log(n), logA)) - logDenominator;
      const S logH = logBeta > 1
          ? logBeta + Math::log1p(-(1 + logBeta) * Math::exp(-logBeta))
          : Math::log(Math::expm1(logBeta) - logBeta);
      logPenalty = logA + logH;
    }
    const S logMean = logMu + logBeta;
    const S mean = Math::exp(logMean);
    // mean-n=(mu-n)*a/(mu+a); retain this residual independently of mean.
    const S logWeight = logA - logDenominator;
    const S residual = n == 0 ? 0 : copySign(
        Math::exp(logDifference + logWeight - Math::log(n)), -difference);
    const S dataTerm = mean == 0 and n > 0
        ? -n + n * (Math::log(n) - logMean)
        : halfPoissonDeviance<Math>(n, mean, residual);
    const S penalty = Math::exp(logPenalty);
    const S predictionGradient = 2 * copySign(
        Math::exp(logDifference + logWeight - logMu), -difference)
        + 4 * Math::exp(logPenalty - logMu);
    return {S(2 * (dataTerm + penalty)), S(predictionGradient),
            S(-4 * Math::exp(logPenalty - logSigma))};
  }

  // Dispatches the two lite models before the legacy evaluators.
  // The legacy Barlow path mirrors the validated 2022 implementation, including its compatibility
  // options and its stabilized expansion near the Asimov minimum. The caller
  // supplies a variance (not an error). NaN signals an error requiring a throw.
  template<typename Math> GUNDAM_LLH_INLINE typename Math::Scalar evaluate(
      typename Math::Scalar data_, typename Math::Scalar pred_, typename Math::Scalar variance_,
      LikelihoodOptions<typename Math::Scalar> options_) {
    using S = typename Math::Scalar;
    if( options_.kind == LikelihoodKind::BarlowBeestonLiteGaussian ){
      return evaluateLiteGaussian<Math>(data_, pred_, Math::sqrt(variance_)).value;
    }
    if( options_.kind == LikelihoodKind::BarlowBeestonLitePoisson ){
      return evaluateLitePoisson<Math>(data_, pred_, Math::sqrt(variance_)).value;
    }
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
