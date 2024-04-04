//
// Created by Nadrino on 22/02/2024.
//

#ifndef GUNDAM_CHI_SQUARED_H
#define GUNDAM_CHI_SQUARED_H

#include "JointProbabilityBase.h"


namespace JointProbability{

  class ChiSquared : public JointProbabilityBase {
  public:
    [[nodiscard]] std::string getType() const override { return "ChiSquared"; }
    [[nodiscard]] double eval(const Sample& sample_, int bin_) const override;
  };

  double ChiSquared::eval(const Sample& sample_, int bin_) const {
    double predVal = sample_.getMcContainer().getHistogram().binList[bin_].content;
    double dataVal = sample_.getDataContainer().getHistogram().binList[bin_].content;
    if( predVal == 0 ){
      // should not be the case right?
      LogAlert << "Zero MC events in bin " << bin_ << ". predVal = " << predVal << ", dataVal = " << dataVal
               << ". Setting llh = +inf for this bin." << std::endl;
      return std::numeric_limits<double>::infinity();
    }
    return TMath::Sq(predVal - dataVal)/predVal;
  }

}


#endif //GUNDAM_CHI_SQUARED_H
