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
    [[nodiscard]] double eval(const Sample& sample_, int bin_) const override;

    struct Buffer{ double rel_var, b, c, beta, mc_hat, chi2; };
    mutable Buffer _buf_{};
  };

  double BarlowBeeston::eval(const Sample& sample_, int bin_) const {
    _buf_.rel_var = sample_.getMcContainer().getHistogram()->GetBinError(bin_) / TMath::Sq(sample_.getMcContainer().getHistogram()->GetBinContent(bin_));
    _buf_.b       = (sample_.getMcContainer().getHistogram()->GetBinContent(bin_) * _buf_.rel_var) - 1;
    _buf_.c       = 4 * sample_.getDataContainer().getHistogram()->GetBinContent(bin_) * _buf_.rel_var;

    _buf_.beta   = (-_buf_.b + std::sqrt(_buf_.b * _buf_.b + _buf_.c)) / 2.0;
    _buf_.mc_hat = sample_.getMcContainer().getHistogram()->GetBinContent(bin_) * _buf_.beta;

    // Calculate the following LLH:
    //-2lnL = 2 * beta*mc - data + data * ln(data / (beta*mc)) + (beta-1)^2 / sigma^2
    // where sigma^2 is the same as above.
    _buf_.chi2 = 0.0;
    if(sample_.getDataContainer().getHistogram()->GetBinContent(bin_) <= 0.0) {
      _buf_.chi2 = 2 * _buf_.mc_hat;
      _buf_.chi2 += (_buf_.beta - 1) * (_buf_.beta - 1) / _buf_.rel_var;
    }
    else{
      _buf_.chi2 = 2 * (_buf_.mc_hat - sample_.getDataContainer().getHistogram()->GetBinContent(bin_));
      if(sample_.getDataContainer().getHistogram()->GetBinContent(bin_) > 0.0) {
        _buf_.chi2 += 2 * sample_.getDataContainer().getHistogram()->GetBinContent(bin_) *
                      std::log(sample_.getDataContainer().getHistogram()->GetBinContent(bin_) / _buf_.mc_hat);
      }
      _buf_.chi2 += (_buf_.beta - 1) * (_buf_.beta - 1) / _buf_.rel_var;
    }
    return _buf_.chi2;
  }

}

#endif // GUNDAM_BARLOW_BEESTON_H
