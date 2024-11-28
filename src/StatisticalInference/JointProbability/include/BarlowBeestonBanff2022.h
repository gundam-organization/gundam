//
// Created by Nadrino on 22/02/2024.
//

#ifndef GUNDAM_BARLOW_BEESTON_BANFF_2022_H
#define GUNDAM_BARLOW_BEESTON_BANFF_2022_H

#include "JointProbabilityBase.h"
#include "GundamUtils.h"

#include "GenericToolbox.Map.h"


namespace JointProbability{

  class BarlowBeestonBanff2022 : public JointProbabilityBase {

  protected:
    void configureImpl() override;

  public:
    [[nodiscard]] std::string getType() const override { return "BarlowBeestonBanff2022"; }
    [[nodiscard]] double eval(const SamplePair& samplePair_, int bin_) const override;

    void createNominalMc(const Sample& modelSample_) const;
    void printConfiguration() const;

    mutable int verboseLevel{0};
    bool throwIfInfLlh{true};
    bool allowZeroMcWhenZeroData{true};
    bool usePoissonLikelihood{false};
    bool BBNoUpdateWeights{false}; // OA 2021 bug reimplementation
    // OA2021 and BANFF fractional error limitation is only relevent with
    // BBNoUpdateWeights is true, and is needed to reproduce bugs when it is
    // true.  When BBNoUpdateWeights is false, the fractional error will
    // naturally be limited to less than 100%.  Physically, the fractional
    // uncertainty should be less than 100% since one MC event in a bin would
    // have 100% fractional uncertainty [under the "Gaussian" approximation,
    // so sqrt(1.0)/1.0].  The OA2021 behavior lets the fractional error grow,
    // but the entire likelihood became discontinuous around a predicted value
    // of 1E-16. Setting fractionalErrorLimit to 1E+19 matches OA2021 before
    // the discontinuity.  The BANFF behavior has the likelihood failed around
    // 1E-154.  Setting fractionalErrorLimit to 1E+150 matchs BANFF.  In both
    // cases, the new likelihood behaves reasonably all the way to zero.
    double fractionalErrorLimit{1.0E+150};
    // OA2021 bug reimplmentation (set to numeric_limits::min() to reproduce
    // the bug).
    double expectedValueMinimum{-1.};

    mutable std::map<const Sample*, std::vector<double>> nomMcUncertList{}; // OA 2021 bug reimplementation
    mutable GenericToolbox::NoCopyWrapper<std::mutex> _mutex_{}; // for creating the nomMC
  };

  void BarlowBeestonBanff2022::configureImpl(){

    GenericToolbox::Json::fillValue(_config_, allowZeroMcWhenZeroData, "allowZeroMcWhenZeroData");
    GenericToolbox::Json::fillValue(_config_, usePoissonLikelihood, "usePoissonLikelihood");
    GenericToolbox::Json::fillValue(_config_, BBNoUpdateWeights, "BBNoUpdateWeights");
    GenericToolbox::Json::fillValue(_config_, fractionalErrorLimit, "fractionalErrorLimit");
    GenericToolbox::Json::fillValue(_config_, expectedValueMinimum, "expectedValueMinimum");
    GenericToolbox::Json::fillValue(_config_, verboseLevel, {{"verboseLevel"},{"isVerbose"}});
    GenericToolbox::Json::fillValue(_config_, throwIfInfLlh, "throwIfInfLlh");

    // Place a hard limit on the fractional error to prevent numeric issues.
    if( fractionalErrorLimit > 1.0E+152 ){
      LogAlert << "Placing a hard limit on the fractional error to prevent numeric issues." << std::endl;
      fractionalErrorLimit = 1.0E+152;
    }

  }
  double BarlowBeestonBanff2022::eval(const SamplePair& samplePair_, int bin_) const {
    double dataVal = samplePair_.data->getHistogram().getBinContentList()[bin_].sumWeights;
    double predVal = samplePair_.model->getHistogram().getBinContentList()[bin_].sumWeights;

    if( predVal == 0 ){
      LogError << bin_ << " bin predicting 0 rate -> llh not defined / inf" << std::endl;
      LogError << samplePair_.model->getSummary() << std::endl;
    }

    {
      /// the first time we reach this point, we assume the predMC is at its nominal value
      std::lock_guard<std::mutex> g(_mutex_);
      if( not GenericToolbox::isIn((const Sample*) samplePair_.model, nomMcUncertList) ){ createNominalMc(*samplePair_.model); }
    }

    // it should exist past this point
    auto& nomHistErr = nomMcUncertList[samplePair_.model];
    double mcuncert{0.0};

    // From OA2021_Eb branch -> BANFFBinnedSample::CalcLLRContrib
    // https://github.com/t2k-software/BANFF/blob/OA2021_Eb/src/BANFFSample/BANFFBinnedSample.cxx

    // Why SQUARE?? -> GetBinError is returning the sqrt(Sum^2) but the BANFF
    // let the BBH make the sqrt
    // https://github.com/t2k-software/BANFF/blob/9140ec11bd74606c10ab4af9ec525352de119c06/src/BANFFSample/BANFFBinnedSample.cxx#L374
    if (BBNoUpdateWeights) {
      mcuncert = nomHistErr[bin_];
      mcuncert *= mcuncert;

      if (not std::isfinite(mcuncert) or mcuncert < 0.0) {
        if( throwIfInfLlh ){
          LogError << "BBNoUpdateWeights mcuncert is not valid "
                   << mcuncert
                   << std::endl;
          LogError << "nomMC bin " << bin_
                   << " error is " << nomHistErr[bin_];
          LogThrow("The mc uncertainty is not a usable number");
        }
        else{
          return std::numeric_limits<double>::infinity();
        }
      }
    }
    else {
      mcuncert = samplePair_.model->getHistogram().getBinContentList()[bin_].sqrtSumSqWeights;
      mcuncert *= mcuncert;

      if(not std::isfinite(mcuncert) or mcuncert < 0.0) {
        if( throwIfInfLlh ){
          LogError << "The mcuncert is not finite " << mcuncert << std::endl;
          LogError << "predMC bin " << bin_
                   << " error is " << samplePair_.model->getHistogram().getBinContentList()[bin_].sqrtSumSqWeights;
          LogThrow("The mc uncertainty is not a usable number");
        }
        else{
          return std::numeric_limits<double>::infinity();
        }
      }
    }

    // Add the (broken) expected value threshold that we saw in the old
    // likelihood. This is here to help understand the old behavior.
    if (predVal <= expectedValueMinimum) {
      LogAlert << "Use incorrect expectation behavior --"
               << " predVal: " << predVal
               << " dataVal: " << dataVal
               << std::endl;
      if (predVal <= 0.0 and dataVal > 0.0) {
        return std::numeric_limits<double>::infinity();
      }
      return 0.0;
    }

    // Implementing Barlow-Beeston correction for LH calculation. The
    // following comments are inspired/copied from Clarence's comments in the
    // MaCh3 implementation of the same feature

    // The MC used in the likelihood calculation is allowed to be changed by
    // Barlow Beeston beta parameters
    double newmc = predVal;

    // Not full Barlow-Beeston or what is referred to as "light": we're not
    // introducing any more parameters.  Assume the MC has a Gaussian
    // distribution generated as in https://arxiv.org/abs/1103.0354 eq 10, 11

    // The penalty from MC statistics
    double penalty = 0.0;

    // Barlow-Beeston uses fractional uncertainty on MC, so sqrt(sum[w^2])/mc
    // -b/2a in quadratic equation
    if ( mcuncert > 0.0 ) {
      // Physically, the
      // fractional uncertainty should be less than 100% since one MC event in
      // a bin would have 100% fractional uncertainty [under the "Gaussian"
      // approximation, so sqrt(1.0)/1.0].  The OA2021 behavior lets the
      // fractional error grow, but the entire likelihood became discontinuous
      // around a predicted value of 1E-16. Setting fractionalLimit to 1E+19
      // matches OA2021 before the discontinuity.  The BANFF behavior has the
      // likelihood failed around 1E-154.  Setting fractionalLimit to 1E+150
      // matchs BANFF.  In both cases, the new likelihood behaves reasonably
      // all the way to zero.
      double fractional = std::min( std::sqrt(mcuncert) / predVal, fractionalErrorLimit);

      double temp = predVal * fractional * fractional - 1;
      // b^2 - 4ac in quadratic equation
      double temp2 = temp * temp + 4 * dataVal * fractional * fractional;
      if(temp2 < 0){
        if( throwIfInfLlh ){ LogThrow("Negative square root in Barlow Beeston"); }
        else{ return std::numeric_limits<double>::infinity(); }
      }

      // Solve for the positive beta
      double beta = (-1 * temp + sqrt(temp2)) / 2.;
      newmc = predVal * beta;
      // And penalise the movement in beta relative the mc uncertainty
      penalty = (beta - 1) * (beta - 1) / (2 * fractional * fractional);
    }

    // And calculate the new Poisson likelihood.  For Barlow-Beeston newmc is
    // modified, so can only calculate Poisson likelihood after Barlow-Beeston
    double stat = 0;
    if (dataVal <= std::numeric_limits<double>::epsilon()) {
      // dataVal should be roughly an integer, so this safer, but roughly
      // equivalent to "dataVal == 0"
      stat = newmc;
    }
    else if (newmc > std::numeric_limits<double>::epsilon()) {
      // The newMC value should not be a lot less than O(1), This protects
      // against small values (intentionally use epsilon() instead of min()
      // since this is relative to 1.0).
      stat = newmc - dataVal + dataVal * TMath::Log(dataVal / newmc);
    }
    else {
      // The mc predicted value is zero, and the data value is not zero.
      // Inconceivable!
      LogErrorIf(verboseLevel>=1) << "Data and predicted value give infinite statistical LLH / "
                                  << "Data: " << dataVal
                                  << " / Barlow Beeston adjusted MC: " << newmc
                                  << std::endl;
    }

    double chisq = 0.0;
    if ((predVal > 0.0) && (dataVal > 0.0)) GUNDAM_LIKELY_COMPILER_FLAG {
      if (usePoissonLikelihood) GUNDAM_UNLIKELY_COMPILER_FLAG {
        // Not quite safe, but not used much, so don't protect
        chisq += 2.0*(predVal - dataVal + dataVal*TMath::Log( dataVal/predVal));
      }
      else GUNDAM_LIKELY_COMPILER_FLAG {
        // Barlow-Beeston likelihood.
        chisq += 2.0 * (stat + penalty);
      }
    }
    else if( predVal > 0.0 ) {
      if (usePoissonLikelihood) GUNDAM_UNLIKELY_COMPILER_FLAG {
        chisq += 2.0*predVal;
      }
      else GUNDAM_LIKELY_COMPILER_FLAG {
        // Barlow-Beeston likelihood
        chisq += 2.0 * (stat + penalty);
      }
    }
    else { // predVal == 0
      if( allowZeroMcWhenZeroData and dataVal == 0 ){
        // need to warn the user something is wrong with the binning definition
        // This might indicate that more MC stat would be needed
        LogErrorOnce << "allowZeroMcWhenZeroData=true: MC bin(s) with 0 as predicted value. "
                     << "Data is also 0, null penalty is applied."
                     << "This is an ill conditioned problem. Please check your inputs."
                     << std::endl;
        chisq = 0.;
      }
      else{
        chisq = std::numeric_limits<double>::infinity();
      }
    }

    if( throwIfInfLlh and not std::isfinite(chisq) ) GUNDAM_UNLIKELY_COMPILER_FLAG {
      LogError << "Infinite chi2: " << chisq << std::endl
               << GET_VAR_NAME_VALUE(samplePair_.model->getName()) << std::endl
               << GET_VAR_NAME_VALUE(bin_) << std::endl
               << GET_VAR_NAME_VALUE(predVal) << std::endl
               << GET_VAR_NAME_VALUE(dataVal) << std::endl
               << GET_VAR_NAME_VALUE(newmc) << std::endl
               << GET_VAR_NAME_VALUE(stat) << std::endl
               << GET_VAR_NAME_VALUE(penalty) << std::endl
               << GET_VAR_NAME_VALUE(mcuncert) << std::endl;
      LogThrow("Bad chisq for bin");
    }

    if(verboseLevel>=3){
      LogTrace << "Bin #" << bin_ << ": chisq(" << chisq << ") / predVal(" << predVal << ") / dataVal(" << dataVal << ")" << std::endl;
    }

    return chisq;
  }
  void BarlowBeestonBanff2022::createNominalMc(const Sample& modelSample_) const {
    LogWarning << "Creating nominal MC histogram for sample \"" << modelSample_.getName() << "\"" << std::endl;
    auto& nomHistErr = nomMcUncertList[&modelSample_];
    nomHistErr.reserve( modelSample_.getHistogram().getNbBins() );
#if HAS_CPP_17
    for( auto [binContent, binContext] : modelSample_.getHistogram().loop() ){
#else
    for( auto element : modelSample_.getHistogram().loop() ){ auto& binContent = std::get<0>(element); auto& binContext = std::get<1>(element);
#endif
      nomHistErr.emplace_back( binContent.sqrtSumSqWeights );
      LogTraceIf(verboseLevel >= 2) << modelSample_.getName() << ": " << binContext.bin.getIndex() << " -> " << binContent.sumWeights << " / " << binContent.sqrtSumSqWeights << std::endl;
    }
  }
  void BarlowBeestonBanff2022::printConfiguration() const{

    LogInfo << "Using BarlowLLH_BANFF_OA2021 parameters:" << std::endl;
    {
      LogScopeIndent;
      LogInfo << GET_VAR_NAME_VALUE(allowZeroMcWhenZeroData) << std::endl;
      LogInfo << GET_VAR_NAME_VALUE(usePoissonLikelihood) << std::endl;
      LogInfo << GET_VAR_NAME_VALUE(BBNoUpdateWeights) << std::endl;
      LogInfo << GET_VAR_NAME_VALUE(verboseLevel) << std::endl;
      LogInfo << GET_VAR_NAME_VALUE(throwIfInfLlh) << std::endl;
      LogInfo << GET_VAR_NAME_VALUE(expectedValueMinimum) << std::endl;
      LogInfo << GET_VAR_NAME_VALUE(fractionalErrorLimit) << std::endl;
    }

  }

}



#endif //GUNDAM_BARLOW_BEESTON_BANFF_2022_H
