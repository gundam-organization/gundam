#ifndef GUNDAM_BARLOW_BEESTON_H
#define GUNDAM_BARLOW_BEESTON_H

#include "JointProbabilityBase.h"
#include "PoissonDeviance.h"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace JointProbability {

  /// Barlow & Beeston, Comput. Phys. Commun. 77 (1993) 219-228,
  /// doi:10.1016/0010-4655(93)90005-W, equations (8)-(9).
  /// Exact Poisson model: n~Pois(b+sum(p_j*A_j)), a_j~Pois(A_j).
  /// Counts must be unweighted MC counts; p_j are nonnegative scale factors.
  /// Arbitrarily weighted templates cannot be reconstructed from two moments.
  class BarlowBeeston : public JointProbabilityBase {
  public:
    using JointProbabilityBase::eval;
    struct Source {
      double count{0};
      double scale{0};
    };
    struct Evaluation {
      double value{0};
      double prediction{0};
      std::vector<double> fittedCounts{};
      std::vector<double> scaleGradients{};
    };

    [[nodiscard]] std::string getType() const override { return "BarlowBeeston"; }

    /// Profile all source means using a bracketed monotonic scalar solve.
    /// Empty MC sources remain in the model and can acquire a fitted mean.
    /// fixedPrediction_ is an optional contribution with no MC uncertainty.
    [[nodiscard]] static Evaluation evaluateSources(double data_, const std::vector<Source>& sources_, double fixedPrediction_ = 0) {
      if( not std::isfinite(data_) or data_ < 0 or not std::isfinite(fixedPrediction_) or fixedPrediction_ < 0 )
        throw std::invalid_argument("BarlowBeeston: finite nonnegative data and fixed prediction required");
      long double maxScale = 0;
      for( const auto& source : sources_ ) {
        if( not std::isfinite(source.count) or source.count < 0 or not std::isfinite(source.scale) or source.scale < 0
            or std::floor(source.count) != source.count )
          throw std::invalid_argument("BarlowBeeston: integer MC counts and finite nonnegative scales required");
        maxScale = std::max(maxScale, static_cast<long double>(source.scale));
      }
      const long double n = data_, fixed = fixedPrediction_;
      Evaluation result;
      result.fittedCounts.resize(sources_.size());
      result.scaleGradients.resize(sources_.size());
      // z=1+p_max*t, t=1-n/x. Domain z in [0,1+p_max].
      // This change of variable preserves distance from the MC pole.
      auto meanAt = [&](long double z_) {
        long double mean = fixed;
        for( const auto& source : sources_ ) {
          if( source.scale == 0 or source.count == 0 ) continue;
          const long double ratio = source.scale / maxScale;
          const long double denominator = (maxScale - source.scale) / maxScale + ratio * z_;
          if( denominator == 0 ) return std::numeric_limits<long double>::infinity();
          mean += source.scale * static_cast<long double>(source.count) / denominator;
        }
        return mean;
      };
      long double z = 1 + maxScale;
      bool boundary = false;
      if( maxScale > 0 and n > 0 ) {
        const long double boundaryMean = n * (maxScale / (1 + maxScale));
        boundary = meanAt(0) <= boundaryMean;
        if( boundary ) z = 0;
        else {
          long double lower = 0, upper = 1 + maxScale;
          bool converged = false;
          for( int iteration = 0; iteration < 4096; ++iteration ) {
            z = lower + (upper - lower) / 2;
            if( z == lower or z == upper ) { converged = true; break; }
            // Monotonic: predicted mean decreases, required mean increases.
            const long double required = n * (maxScale / (1 + maxScale - z));
            if( meanAt(z) > required ) lower = z;
            else upper = z;
          }
          if( not converged ) throw std::runtime_error("BarlowBeeston: source profiling did not converge");
        }
      }
      std::vector<long double> counts(sources_.size());
      long double prediction = fixed;
      for( std::size_t i = 0; i < sources_.size(); ++i ) {
        const auto& source = sources_[i];
        counts[i] = source.count;
        if( source.scale > 0 ) {
          const long double denominator = (maxScale - source.scale) / maxScale + (source.scale / maxScale) * z;
          counts[i] = source.count == 0 ? 0 : source.count / denominator;
        }
        prediction += source.scale * counts[i];
      }
      if( boundary ) {
        const long double extra = std::max(0.L, n * (maxScale / (1 + maxScale)) - prediction);
        for( std::size_t i = 0; i < sources_.size(); ++i ) {
          if( sources_[i].count == 0 and sources_[i].scale == maxScale ) {
            counts[i] = extra / maxScale;
            prediction += extra;
            break;
          }
        }
      }
      long double value = GundamUtils::halfPoissonDeviance(n, prediction);
      for( std::size_t i = 0; i < sources_.size(); ++i ) {
        value += GundamUtils::halfPoissonDeviance(sources_[i].count, counts[i]);
        result.fittedCounts[i] = double(counts[i]);
        result.scaleGradients[i] = double(2 * counts[i] * (n == 0 ? 1.L : prediction > 0 ? (prediction - n) / prediction
            : std::numeric_limits<long double>::quiet_NaN()));
      }
      result.value = double(2 * value);
      result.prediction = double(prediction);
      return result;
    }

    [[nodiscard]] double eval(double data_, double pred_, double err_, int) const override {
      if( not std::isfinite(data_) or not std::isfinite(pred_) or not std::isfinite(err_)
          or data_ < 0 or pred_ < 0 or err_ < 0 ) return std::numeric_limits<double>::infinity();
      if( err_ != 0 ) throw std::logic_error(
          "BarlowBeeston requires per-source MC counts and scales: use evaluateSources; histogram moments are insufficient");
      return double(2 * GundamUtils::halfPoissonDeviance(data_, pred_));
    }
  };
}
#endif
