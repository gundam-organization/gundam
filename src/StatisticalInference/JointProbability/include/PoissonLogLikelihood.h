//
// Created by Nadrino on 22/02/2024.
//

#ifndef GUNDAM_POISSON_LOG_LIKELIHOOD_H
#define GUNDAM_POISSON_LOG_LIKELIHOOD_H

#include "JointProbabilityBase.h"


namespace JointProbability{

  class PoissonLogLikelihood : public JointProbabilityBase {

  public:
    [[nodiscard]] std::string getType() const override { return "PoissonLogLikelihood"; }
    [[nodiscard]] double eval(const Sample& sample_, int bin_) const override {
      double predVal = sample_.getMcContainer().histogram->GetBinContent(bin_);
      double dataVal = sample_.getDataContainer().histogram->GetBinContent(bin_);

      if(predVal <= 0){
        LogAlert << "Zero MC events in bin " << bin_ << ". predVal = " << predVal << ", dataVal = " << dataVal
                 << ". Setting llh = +inf for this bin." << std::endl;
        return std::numeric_limits<double>::infinity();
      }

      if(dataVal <= 0){
        // lim x -> 0 : x ln(x) = 0
        return 2.0 * predVal;
      }

      // LLH calculation
      return 2.0 * (predVal - dataVal + dataVal * TMath::Log(dataVal / predVal));
    }
  };

}

#endif //GUNDAM_POISSON_LOG_LIKELIHOOD_H
