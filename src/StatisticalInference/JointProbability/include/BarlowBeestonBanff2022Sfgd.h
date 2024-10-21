//
// Created by Nadrino on 22/02/2024.
//

#ifndef GUNDAM_BARLOW_BEESTON_BANFF_2022_SFGD_H
#define GUNDAM_BARLOW_BEESTON_BANFF_2022_SFGD_H

#include "JointProbabilityBase.h"

namespace JointProbability{


  class BarlowBeestonBanff2022Sfgd : public JointProbabilityBase {
  public:
    [[nodiscard]] std::string getType() const override { return "BarlowBeestonBanff2022Sfgd"; }
    [[nodiscard]] double eval(const SamplePair& samplePair_, int bin_) const override;
  };

  double BarlowBeestonBanff2022Sfgd::eval(const SamplePair& samplePair_, int bin_) const {

    double dataVal = samplePair_.data->getHistogram().getBinContentList()[bin_].sumWeights;
    double predVal = samplePair_.model->getHistogram().getBinContentList()[bin_].sumWeights;
    double mcuncert = samplePair_.model->getHistogram().getBinContentList()[bin_].sqrtSumSqWeights;

    double chisq = 0.0;

    bool usePoissonLikelihood = false;

    double newmc = predVal;

    // The penalty from MC statistics
    double penalty = 0;

    // SFGD detector uncertainty
    double sfgd_det_uncert = 0.;
    if (samplePair_.model->getName().find("SFGD") != std::string::npos){
      // to be applied on SFGD samples only
      sfgd_det_uncert = 0.;
      if (samplePair_.model->getName().find("FHC") != std::string::npos){
        if (samplePair_.model->getName().find("0p") != std::string::npos){
          sfgd_det_uncert = 0.02;
        }
        else if (samplePair_.model->getName().find("Np") != std::string::npos){
          sfgd_det_uncert = 0.04;
        }
      }
      else if (samplePair_.model->getName().find("RHC") != std::string::npos){
        if (samplePair_.model->getName().find("0n") != std::string::npos){
          sfgd_det_uncert = 0.025;
        }
        else if (samplePair_.model->getName().find("Nn") != std::string::npos){
          sfgd_det_uncert = 0.05;
        }
      }
    }

    // DETECTOR UNCERTAINTY FOR WAGASCI
    double wg_det_uncert = 0.;
    if (samplePair_.model->getName().find("WAGASCI") != std::string::npos){
      wg_det_uncert = 0.;
      if(samplePair_.model->getName().find("#0pi") != std::string::npos) {
        if(samplePair_.model->getName().find("PM-BM") != std::string::npos) wg_det_uncert = 0.05;
        if(samplePair_.model->getName().find("PM-WMRD") != std::string::npos) wg_det_uncert = 0.1;
        if(samplePair_.model->getName().find("DWG-BM") != std::string::npos) wg_det_uncert = 0.1;
        if(samplePair_.model->getName().find("UWG-BM") != std::string::npos) wg_det_uncert = 0.12;
        if(samplePair_.model->getName().find("UWG-WMRD") != std::string::npos) wg_det_uncert = 0.1;
      }
      if(samplePair_.model->getName().find("#1pi") != std::string::npos)  {
        if(samplePair_.model->getName().find("PM") != std::string::npos) wg_det_uncert = 0.1;
        if(samplePair_.model->getName().find("WG") != std::string::npos) wg_det_uncert = 0.08;
      }
    }

    // Barlow-Beeston uses fractional uncertainty on MC, so sqrt(sum[w^2])/mc
    double fractional = mcuncert / predVal + sfgd_det_uncert + wg_det_uncert; // Add SFGD detector uncertainty
    // -b/2a in quadratic equation
    double temp = predVal * fractional * fractional - 1;
    // b^2 - 4ac in quadratic equation
    double temp2 = temp * temp + 4 * dataVal * fractional * fractional;

    LogThrowIf(temp2 < 0, "Negative square root in Barlow Beeston coefficient calculation!");


    // Solve for the positive beta
    double beta = (-1 * temp + sqrt(temp2)) / 2.;
    newmc = predVal * beta;
    // And penalise the movement in beta relative the mc uncertainty
    penalty = (beta - 1) * (beta - 1) / (2 * fractional * fractional);
    // And calculate the new Poisson likelihood
    // For Barlow-Beeston newmc is modified, so can only calculate Poisson likelihood after Barlow-Beeston
    double stat = 0;
    if (dataVal == 0)
      stat = newmc;
    else if (newmc > 0)
      stat = newmc - dataVal + dataVal * TMath::Log(dataVal / newmc);

    if ((predVal > 0.0) && (dataVal > 0.0))
    {
      if (usePoissonLikelihood)
        chisq += 2.0*(predVal - dataVal +dataVal*TMath::Log( dataVal/predVal) );
      else // Barlow-Beeston likelihood
        chisq += 2.0 * (stat + penalty);
    }

    else if (predVal > 0.0)
    {
      if (usePoissonLikelihood)
        chisq += 2.0*predVal;
      else // Barlow-Beeston likelihood
        chisq += 2.0 * (stat + penalty);
    }

    if (std::isinf(chisq))
    {
      LogAlert << "Infinite chi2 " << predVal << " " << dataVal
               << samplePair_.model->getHistogram().getBinContentList()[bin_].sqrtSumSqWeights << " "
               << samplePair_.model->getHistogram().getBinContentList()[bin_].sumWeights << std::endl;
    }

    return chisq;
  }

}


#endif // GUNDAM_BARLOW_BEESTON_BANFF_2022_SFGD_H
