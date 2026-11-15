#ifndef GUNDAM_POISSON_DEVIANCE_H
#define GUNDAM_POISSON_DEVIANCE_H

#include <cmath>
#include <limits>

namespace GundamUtils {
  // mu - n + n*log(n/mu): half the saturated Poisson deviance. The residual can be supplied
  // independently to preserve a small displacement lost when forming mean.
  inline long double halfPoissonDeviance(long double observed_, long double mean_, long double residual_) {
    if( observed_ == 0 ) return mean_;
    if( mean_ <= 0 ) return std::numeric_limits<long double>::infinity();
    if( std::abs(residual_) < .01L ) {
      long double series = 0;
      for( int order = 20; order >= 2; --order ) {
        series = (order % 2 == 0 ? 1.L : -1.L) / order + residual_ * series;
      }
      return observed_ * residual_ * residual_ * series;
    }
    if( std::abs(residual_) < .5L ) return observed_ * (residual_ - std::log1p(residual_));
    return mean_ - observed_ + observed_ * (std::log(observed_) - std::log(mean_));
  }
  inline long double halfPoissonDeviance(long double observed_, long double mean_) {
    return halfPoissonDeviance(observed_, mean_, observed_ > 0 ? (mean_ - observed_) / observed_ : 0);
  }
}
#endif
