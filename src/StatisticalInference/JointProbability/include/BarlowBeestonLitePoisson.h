#ifndef GUNDAM_BARLOW_BEESTON_LITE_POISSON_H
#define GUNDAM_BARLOW_BEESTON_LITE_POISSON_H

#include "JointProbabilityBase.h"
#include "PoissonDeviance.h"

#include <algorithm>

namespace JointProbability {

  /// Dembinski & Abdelmotteleb, arXiv:2206.12346v2, equations (17)-(18).
  /// Unweighted Poisson data and current weighted-MC moments:
  /// a = mu^2/sigma^2, beta = (n+a)/(mu+a),
  /// q = D(n,beta*mu) + D(a,beta*a), where D is the Poisson deviance.
  /// This effective-count model approximates mixtures and varying MC weights.
  /// All overloads take sigma=sqrt(sum(w^2)), not its square.
  class BarlowBeestonLitePoisson : public JointProbabilityBase {
  public:
    using JointProbabilityBase::eval;
    [[nodiscard]] std::string getType() const override { return "BarlowBeestonLitePoisson"; }
    struct Evaluation {
      double value{0};
      double predictionGradient{0};
      double uncertaintyGradient{0};
    };

    /// Partial derivatives include the dependence of the effective count a on
    /// both mu and sigma, in addition to profiling beta (envelope theorem).
    [[nodiscard]] static Evaluation evaluate(double data_, double pred_, double err_) {
      const double inf = std::numeric_limits<double>::infinity();
      const double nan = std::numeric_limits<double>::quiet_NaN();
      if( not std::isfinite(data_) or not std::isfinite(pred_) or not std::isfinite(err_)
          or data_ < 0 or pred_ < 0 or err_ < 0 ) return {inf, nan, nan};
      const long double n = data_, mu = pred_, sigma = err_;
      if( mu == 0 ) {
        // Zero total MC weight with nonzero variance is outside this model.
        if( sigma != 0 or n != 0 ) return {inf, nan, nan};
        return {0, 2, 0};
      }
      if( sigma == 0 ) return {double(2 * GundamUtils::halfPoissonDeviance(n, mu)),
                               double(n == 0 ? 2 : 2 * (mu - n) / mu), 0};
      if( n == mu ) return {0, 0, 0};

      // Work with logarithmic scales: a and beta can exceed the floating-point
      // range even when their combined contribution to the objective is finite.
      const long double logMu = std::log(mu), logSigma = std::log(sigma);
      const long double logA = 2 * (logMu - logSigma);
      const long double logDenominator = logAdd(logMu, logA);
      const long double difference = n - mu;
      const long double logDifference = std::log(std::abs(difference));
      const long double logDelta = logDifference - logDenominator;
      const long double delta = std::copysign(std::exp(logDelta), difference);
      long double logBeta, logPenalty;
      if( std::abs(delta) < .01L ) {
        // h(1+delta)=delta-log1p(delta). Preserve a*delta^2 without
        // underflowing delta^2 or overflowing a at a precise Asimov minimum.
        long double series = 0;
        for( int order = 20; order >= 2; --order )
          series = (order % 2 == 0 ? 1.L : -1.L) / order + delta * series;
        logBeta = std::log1p(delta);
        logPenalty = logA + 2 * logDelta + std::log(series);
      }
      else {
        logBeta = (n == 0 ? logA : logAdd(std::log(n), logA)) - logDenominator;
        const long double logH = logBeta > 1
            ? logBeta + std::log1p(-(1 + logBeta) * std::exp(-logBeta))
            : std::log(std::expm1(logBeta) - logBeta);
        logPenalty = logA + logH;
      }
      const long double logMean = logMu + logBeta;
      const long double mean = std::exp(logMean);
      // mean-n=(mu-n)*a/(mu+a); retain this residual independently of mean.
      const long double logWeight = logA - logDenominator;
      const long double residual = n == 0 ? 0 : std::copysign(
          std::exp(logDifference + logWeight - std::log(n)), -difference);
      const long double dataTerm = mean == 0 and n > 0
          ? -n + n * (std::log(n) - logMean)
          : GundamUtils::halfPoissonDeviance(n, mean, residual);
      const long double penalty = std::exp(logPenalty);
      const long double predictionGradient = 2 * std::copysign(
          std::exp(logDifference + logWeight - logMu), -difference)
          + 4 * std::exp(logPenalty - logMu);
      return {double(2 * (dataTerm + penalty)), double(predictionGradient),
              double(-4 * std::exp(logPenalty - logSigma))};
    }

    [[nodiscard]] double eval(double data_, double pred_, double err_, int) const override {
      return evaluate(data_, pred_, err_).value;
    }
    [[nodiscard]] double evalPredictionGradient(double data_, double pred_, double err_, int) const override {
      return evaluate(data_, pred_, err_).predictionGradient;
    }
    [[nodiscard]] double evalSqrtSumSqWeightsGradient(double data_, double pred_, double err_, int) const override {
      return evaluate(data_, pred_, err_).uncertaintyGradient;
    }
    [[nodiscard]] double evalWithPrediction(const SamplePair& pair_, double data_, double pred_, double err_, int bin_) const override {
      return isBinIgnored(pair_, bin_) ? 0 : eval(data_, pred_, err_, bin_);
    }
    [[nodiscard]] double evalPredictionGradient(const SamplePair& pair_, int bin_) const override {
      if( isBinIgnored(pair_, bin_) ) return 0;
      const auto& bin = pair_.model->getHistogram().getBinContentList().at(bin_);
      return evalPredictionGradient(pair_.data->getHistogram().getBinContentList().at(bin_).sumWeights,
                                    bin.sumWeights, bin.sqrtSumSqWeights, bin_);
    }
    [[nodiscard]] double evalSqrtSumSqWeightsGradient(const SamplePair& pair_, int bin_) const override {
      if( isBinIgnored(pair_, bin_) ) return 0;
      const auto& bin = pair_.model->getHistogram().getBinContentList().at(bin_);
      return evalSqrtSumSqWeightsGradient(pair_.data->getHistogram().getBinContentList().at(bin_).sumWeights,
                                         bin.sumWeights, bin.sqrtSumSqWeights, bin_);
    }

  protected:
    void configureImpl() override {
      _config_.clearFields();
      JointProbabilityBase::configureImpl();
      _config_.checkConfiguration();
    }

  private:
    static long double logAdd(long double first_, long double second_) {
      const long double larger = std::max(first_, second_);
      return larger + std::log1p(std::exp(std::min(first_, second_) - larger));
    }
  };
}
#endif
