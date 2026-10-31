#ifndef GUNDAM_BARLOW_BEESTON_LITE_GAUSSIAN_H
#define GUNDAM_BARLOW_BEESTON_LITE_GAUSSIAN_H

#include "JointProbabilityBase.h"
#include "PoissonDeviance.h"

#include <algorithm>

namespace JointProbability {

  /// Conway, arXiv:1103.0354, section 5, equations (10)-(11).
  /// One Gaussian-constrained nuisance per bin, profiled analytically:
  /// q = min_{x>=0} [PoissonDeviance(n,x) + (x-mu)^2 / sigma^2].
  /// For mu>0, x=beta*mu is Conway's multiplicative parameterization.
  /// All scalar overloads take sigma=sqrt(sum(w^2)), NOT its square.
  /// Uses the current MC uncertainty on every evaluation; no nominal cache.
  /// Negative weights/expectations are outside this model's domain.
  class BarlowBeestonLiteGaussian : public JointProbabilityBase {
  public:
    using JointProbabilityBase::eval;
    [[nodiscard]] std::string getType() const override { return "BarlowBeestonLiteGaussian"; }
    [[nodiscard]] bool hasAnalyticBinGradient() const override { return true; }

    struct Evaluation {
      double value{0};
      double predictionGradient{0};
      double uncertaintyGradient{0};
    };

    /// Envelope derivatives hold the other scalar input fixed. At a boundary
    /// they are one-sided; an undefined derivative is returned as NaN.
    [[nodiscard]] static Evaluation evaluate(double data_, double pred_, double err_) {
      const double inf = std::numeric_limits<double>::infinity();
      const double nan = std::numeric_limits<double>::quiet_NaN();
      if( not std::isfinite(data_) or not std::isfinite(pred_) or not std::isfinite(err_)
          or data_ < 0 or pred_ < 0 or err_ < 0 ) return {inf, nan, nan};
      const long double n = data_, mu = pred_, sigma = err_;
      if( sigma == 0 ) {
        if( mu == 0 and n > 0 ) return {inf, nan, nan};
        return {double(2 * GundamUtils::halfPoissonDeviance(n, mu)),
                double(n == 0 ? 2 : 2 * (mu - n) / mu), 0};
      }
      long double x, pull, predictionGradient, residual;
      if( sigma >= std::sqrt(std::max(mu, n)) ) {
        // Normalize by sigma^2 without ever forming it (it may overflow).
        const long double scaledMu = (mu / sigma) / sigma;
        const long double scaledN = (n / sigma) / sigma;
        const long double discriminant = std::hypot(scaledMu - 1, 2 * std::sqrt(scaledN));
        x = n == 0 ? 0 : n / ((1 - scaledMu + discriminant) / 2);
        const long double shift = (n - mu) / ((scaledMu + 1 + discriminant) / 2);
        pull = shift / sigma;
        predictionGradient = -2 * (pull / sigma);
        residual = n > 0 ? (((mu - n) / sigma) * (x / sigma) / n)
            / (1 + (x / sigma) / sigma) : 0;
      }
      else {
        const long double variance = sigma * sigma;
        const long double scale = std::max(mu, n);
        const long double m = mu / scale, v = variance / scale, d = n / scale;
        const long double scaledSigma = sigma / std::sqrt(scale);
        const long double discriminant = std::hypot(m - v, 2 * scaledSigma * std::sqrt(d));
        x = m >= v ? scale * ((m - v + discriminant) / 2)
            : (n == 0 ? 0 : n * (2 * v / (v - m + discriminant)));
        const long double denominator = x / scale + v;
        // Envelope theorem: (x-mu)/sigma = sigma*(n-mu)/(x+sigma^2).
        // This also preserves the prediction derivative for tiny sigma.
        const long double slope = denominator > 0 ? ((mu - n) / scale) / denominator : 0;
        pull = -sigma * slope;
        predictionGradient = 2 * slope;
        const long double fraction = x > variance ? 1 / (1 + variance / x)
            : (x / variance) / (1 + x / variance);
        residual = n > 0 ? ((mu - n) / n) * fraction : 0;
      }
      return {double(2 * GundamUtils::halfPoissonDeviance(n, x, residual) + pull * pull),
              double(predictionGradient), double(-2 * pull * (pull / sigma))};
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
      if( isBinIgnored(pair_, bin_) ) return 0;
      return eval(data_, pred_, err_, bin_);
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

    [[nodiscard]] BinGradient evalBinGradient(const SamplePair& pair_, int bin_) const override {
      if( isBinIgnored(pair_, bin_) ){ return {}; }
      const auto& bin = pair_.model->getHistogram().getBinContentList().at(bin_);
      const auto result = evaluate(pair_.data->getHistogram().getBinContentList().at(bin_).sumWeights,
                                   bin.sumWeights, bin.sqrtSumSqWeights);
      return {result.predictionGradient, result.uncertaintyGradient};
    }

  protected:
    void configureImpl() override {
      _config_.clearFields();
      JointProbabilityBase::configureImpl();
      _config_.checkConfiguration();
    }
  };
}
#endif
