//
// Created by Nadrino on 22/02/2024.
//

#ifndef GUNDAM_BARLOW_BEESTON_BANFF_2020_H
#define GUNDAM_BARLOW_BEESTON_BANFF_2020_H

#include "JointProbabilityBase.h"


namespace JointProbability{

  class BarlowBeestonBanff2020 : public JointProbabilityBase {
  public:
    [[nodiscard]] std::string getType() const override { return "BarlowBeestonBanff2020"; }
    [[nodiscard]] double eval(const SamplePair& samplePair_, int bin_) const override;
  };

  double BarlowBeestonBanff2020::eval(const SamplePair& samplePair_, int bin_) const {
    // From BANFF: origin/OA2020 branch -> BANFFBinnedSample::CalcLLRContrib()

    //Loop over all the bins one by one using their unique bin index.
    //Use the stored nBins value and bins array so avoid trying to calculate
    //over underflow or overflow bins.
    double chisq{0};

    double dataVal = samplePair_.data->getHistogram().getBinContentList()[bin_].sumWeights;
    double predVal = samplePair_.model->getHistogram().getBinContentList()[bin_].sumWeights;
    double mcuncert = samplePair_.model->getHistogram().getBinContentList()[bin_].sqrtSumSqWeights;

    //implementing Barlow-Beeston correction for LH calculation the
    //following comments are inspired/copied from Clarence's comments in the
    //MaCh3 implementation of the same feature

    // The MC used in the likeliihood calculation
    // Is allowed to be changed by Barlow Beeston beta parameters
    double newmc = predVal;
    // Not full Barlow-Beeston or what is referred to as "light": we're not introducing any more parameters
    // Assume the MC has a Gaussian distribution around generated
    // As in https://arxiv.org/abs/1103.0354 eq 10, 11

    // The penalty from MC statistics
    double penalty = 0;
    // Barlow-Beeston uses fractional uncertainty on MC, so sqrt(sum[w^2])/mc
    double fractional = sqrt(mcuncert)/predVal;
    // -b/2a in quadratic equation
    double temp = predVal*fractional*fractional-1;
    // b^2 - 4ac in quadratic equation
    double temp2 = temp*temp + 4*dataVal*fractional*fractional;
    LogExitIf(temp2 < 0, "Negative square root in Barlow Beeston coefficient calculation!");

    // Solve for the positive beta
    double beta = (-1*temp+sqrt(temp2))/2.;
    newmc = predVal*beta;
    // And penalise the movement in beta relative the mc uncertainty
    penalty = (beta-1)*(beta-1)/(2*fractional*fractional);

    // And calculate the new Poisson likelihood
    // For Barlow-Beeston newmc is modified, so can only calculate Poisson likelihood after Barlow-Beeston
    double stat = 0;
    if (dataVal == 0) stat = newmc;
    else if (newmc > 0) stat = newmc-dataVal+dataVal*TMath::Log(dataVal/newmc);

    if( std::isnan(penalty) ){ return std::numeric_limits<double>::infinity(); }

    if((predVal > 0.0) && (dataVal > 0.0)){

      chisq += 2.0*(stat+penalty);
      //chisq += 2.0*(predVal - dataVal +dataVal*TMath::Log( dataVal/predVal) );    //this is what was in BANFF before BB

    }

    else if(predVal > 0.0){

      chisq += 2.0*(stat+penalty);
      //chisq += 2.0*predVal;    //this is what was in BANFF before BB

    }

    if(std::isinf(chisq)){
      LogAlert << "Infinite chi2 " << predVal << " " << dataVal << " "
               << samplePair_.model->getHistogram().getBinContentList()[bin_].sqrtSumSqWeights << " "
               << samplePair_.model->getHistogram().getBinContentList()[bin_].sumWeights << std::endl;
    }

    LogExitIf(std::isnan(chisq), "NaN chi2 " << predVal << " " << dataVal
                                              << samplePair_.model->getHistogram().getBinContentList()[bin_].sqrtSumSqWeights << " "
                                              << samplePair_.model->getHistogram().getBinContentList()[bin_].sumWeights);

    return chisq;
  }

}

#endif //GUNDAM_BARLOW_BEESTON_BANFF_2020_H
