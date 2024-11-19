//
// Created by Nadrino on 22/02/2024.
//

#ifndef GUNDAM_POISSON_LOG_LIKELIHOOD_H
#define GUNDAM_POISSON_LOG_LIKELIHOOD_H

#include "JointProbabilityBase.h"

#include "Logger.h"


namespace JointProbability{

  class PoissonLogLikelihood : public JointProbabilityBase {

  public:
    [[nodiscard]] std::string getType() const override { return "PoissonLogLikelihood"; }
    [[nodiscard]] double eval(const SamplePair& samplePair_, int bin_) const override {
      double predVal = samplePair_.model->getHistogram().getBinContentList()[bin_].sumWeights;
      double dataVal = samplePair_.data->getHistogram().getBinContentList()[bin_].sumWeights;

      double llhValue{0};


      if(predVal <= 0){
        LogAlert << "Zero MC events in bin " << bin_ << ". predVal = " << predVal << ", dataVal = " << dataVal
                 << ". Setting llh = +inf for this bin." << std::endl;
        llhValue = std::numeric_limits<double>::infinity();
      }
      else if(dataVal <= 0){
        // lim x -> 0 : x ln(x) = 0
        llhValue = 2.0 * predVal;
      }
      else{
        // LLH calculation
        llhValue = 2.0 * (predVal - dataVal + dataVal * TMath::Log(dataVal / predVal));

//        if( llhValue < 0 and std::abs(llhValue) < 1E-12 ){
//          llhValue = -llhValue;
//        }
      }

//      LogThrowIf(
//          llhValue < 0,
//          "Negative poisson llh: " << llhValue << " / "
//          << GET_VAR_NAME_VALUE(predVal) << " / "
//          << GET_VAR_NAME_VALUE(dataVal) << " / "
//          << GET_VAR_NAME_VALUE(dataVal-predVal)
//      );

      return llhValue;
    }
  };

}

#endif //GUNDAM_POISSON_LOG_LIKELIHOOD_H
