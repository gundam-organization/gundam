//
// Created by Nadrino on 22/02/2024.
//

#ifndef GUNDAM_BARLOW_BEESTON_H
#define GUNDAM_BARLOW_BEESTON_H

#include "JointProbabilityBase.h"


namespace JointProbability{

  class BarlowBeeston : public JointProbabilityBase {
  public:
    [[nodiscard]] std::string getType() const override { return "BarlowBeeston"; }
    [[nodiscard]] double eval(const SamplePair& samplePair_, int bin_) const override;

    struct Buffer{ double rel_var, b, c, beta, mc_hat, chi2; };
    mutable Buffer _buf_{};
  };

  double BarlowBeeston::eval(const SamplePair& samplePair_, int bin_) const {
    _buf_.rel_var = samplePair_.model->getHistogram().getBinContentList()[bin_].sqrtSumSqWeights / TMath::Sq(samplePair_.model->getHistogram().getBinContentList()[bin_].sumWeights);
    _buf_.b       = (samplePair_.model->getHistogram().getBinContentList()[bin_].sumWeights * _buf_.rel_var) - 1;
    _buf_.c       = 4 * samplePair_.data->getHistogram().getBinContentList()[bin_].sumWeights * _buf_.rel_var;

    _buf_.beta   = (-_buf_.b + std::sqrt(_buf_.b * _buf_.b + _buf_.c)) / 2.0;
    _buf_.mc_hat = samplePair_.model->getHistogram().getBinContentList()[bin_].sumWeights * _buf_.beta;

    // Calculate the following LLH:
    //-2lnL = 2 * beta*mc - data + data * ln(data / (beta*mc)) + (beta-1)^2 / sigma^2
    // where sigma^2 is the same as above.
    _buf_.chi2 = 0.0;
    if(samplePair_.data->getHistogram().getBinContentList()[bin_].sumWeights <= 0.0) {
      _buf_.chi2 = 2 * _buf_.mc_hat;
      _buf_.chi2 += (_buf_.beta - 1) * (_buf_.beta - 1) / _buf_.rel_var;
    }
    else{
      _buf_.chi2 = 2 * (_buf_.mc_hat - samplePair_.data->getHistogram().getBinContentList()[bin_].sumWeights);
      if(samplePair_.data->getHistogram().getBinContentList()[bin_].sumWeights > 0.0) {
        _buf_.chi2 += 2 * samplePair_.data->getHistogram().getBinContentList()[bin_].sumWeights *
                      std::log(samplePair_.data->getHistogram().getBinContentList()[bin_].sumWeights / _buf_.mc_hat);
      }
      _buf_.chi2 += (_buf_.beta - 1) * (_buf_.beta - 1) / _buf_.rel_var;
    }
    return _buf_.chi2;
  }

}

#endif // GUNDAM_BARLOW_BEESTON_H
