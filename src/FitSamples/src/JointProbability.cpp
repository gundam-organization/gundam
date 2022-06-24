//
// Created by Adrien BLANCHET on 23/06/2022.
//

#include "JointProbability.h"

#include "Logger.h"

#include "TMath.h"


LoggerInit([]{
  Logger::setUserHeaderStr("[JointProbability]");
});

namespace JointProbability{

  double PoissonLLH::eval(const FitSample& sample_){
    double out{0};

    int nBins = int(sample_.getBinning().getBinsList().size());
    const TH1D* mcHist = sample_.getMcContainer().histogram.get();
    const TH1D* dataHist = sample_.getDataContainer().histogram.get();

    for( int iBin = 0 ; iBin < nBins ; iBin++ ){
      if( mcHist->GetBinContent(iBin+1) <= 0 ) continue;
      out += - (
          dataHist->GetBinContent(iBin+1) * TMath::Log(mcHist->GetBinContent(iBin+1))
          - mcHist->GetBinContent(iBin+1)
          - TMath::LnGamma(dataHist->GetBinContent(iBin+1)+1.)
      );
    }

    return out;
  }
  double BarlowLLH::eval(const FitSample& sample_){
    double out{0};

    // DOCS?
    // https://indico.cern.ch/event/107747/contributions/32677/attachments/24367/35056/Conway-PhyStat2011.pdf

    int nBins = int(sample_.getBinning().getBinsList().size());
    const TH1D* mcHist = sample_.getMcContainer().histogram.get();
    const TH1D* dataHist = sample_.getDataContainer().histogram.get();

    double rel_var, b, c, beta, mc_hat, chi2;
    for( int iBin = 1 ; iBin <= nBins ; iBin++ ){

      rel_var = mcHist->GetBinError(iBin) / TMath::Sq(mcHist->GetBinContent(iBin));
      b       = (mcHist->GetBinContent(iBin) * rel_var) - 1;
      c       = 4 * dataHist->GetBinContent(iBin) * rel_var;

      beta   = (-b + std::sqrt(b * b + c)) / 2.0;
      mc_hat = mcHist->GetBinContent(iBin) * beta;

      // Calculate the following LLH:
      //-2lnL = 2 * beta*mc - data + data * ln(data / (beta*mc)) + (beta-1)^2 / sigma^2
      // where sigma^2 is the same as above.
      chi2 = 0.0;
      if(dataHist->GetBinContent(iBin) <= 0.0) {
          chi2 = 2 * mc_hat;
          chi2 += (beta - 1) * (beta - 1) / rel_var;
      }
      else{
        chi2 = 2 * (mc_hat - dataHist->GetBinContent(iBin));
        if(dataHist->GetBinContent(iBin) > 0.0)
          chi2 += 2 * dataHist->GetBinContent(iBin) * std::log(dataHist->GetBinContent(iBin) / mc_hat);

        chi2 += (beta - 1) * (beta - 1) / rel_var;
      }

      out += chi2;
    }

    return out;
  }
  double BarlowLLH_BANFF_OA2020::eval(const FitSample& sample_){
    int nBins = int(sample_.getBinning().getBinsList().size());
    const TH1D* mcHist = sample_.getMcContainer().histogram.get();
    const TH1D* dataHist = sample_.getDataContainer().histogram.get();

    double chisq = 0.0;
    double dataVal, predVal;

    //Loop over all the bins one by one using their unique bin index.
    //Use the stored nBins value and bins array so avoid trying to calculate
    //over underflow or overflow bins.
    for( int iBin = 1 ; iBin <= nBins ; iBin++ ){

      dataVal = dataHist->GetBinContent(iBin);
      predVal = mcHist->GetBinContent(iBin);
      double mcuncert = mcHist->GetBinError(iBin);

      //implementing Barlow-Beeston correction for LH calculation
      //the following comments are inspired/copied from Clarence's comments in the MaCh3
      //implementation of the same feature

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

      LogThrowIf(temp2 < 0, "Negative square root in Barlow Beeston coefficient calculation!");

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

      if((predVal > 0.0) && (dataVal > 0.0)){

        chisq += 2.0*(stat+penalty);
        //chisq += 2.0*(predVal - dataVal +dataVal*TMath::Log( dataVal/predVal) );    //this is what was in BANFF before BB

      }

      else if(predVal > 0.0){

        chisq += 2.0*(stat+penalty);
        //chisq += 2.0*predVal;    //this is what was in BANFF before BB

      }

      if(std::isinf(chisq)){
        LogAlert << "Infinite chi2 " << predVal << " " << dataVal
                  << mcHist->GetBinError(iBin) << " "
                  << mcHist->GetBinContent(iBin) << std::endl;
      }
    }

    return chisq;
  }

}
