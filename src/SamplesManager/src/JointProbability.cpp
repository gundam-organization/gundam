//
// Created by Adrien BLANCHET on 23/06/2022.
//

#include "JointProbability.h"

#include "Logger.h"
#include "GenericToolbox.h"
#include "GenericToolbox.Json.h"

#include "TMath.h"
#include <cmath>

LoggerInit([]{
  Logger::setUserHeaderStr("[JointProbability]");
});

namespace JointProbability{

  // JointProbabilityPlugin
  void JointProbabilityPlugin::readConfigImpl() {
    llhPluginSrc = GenericToolbox::Json::fetchValue<std::string>(_config_, "llhPluginSrc");
    llhSharedLib = GenericToolbox::Json::fetchValue<std::string>(_config_, "llhSharedLib");
  }
  void JointProbabilityPlugin::initializeImpl(){
    if( not llhSharedLib.empty() ) this->load();
    else if( not llhPluginSrc.empty() ){ this->compile(); this->load(); }
    else{ LogThrow("Can't initialize JointProbabilityPlugin without llhSharedLib nor llhPluginSrc."); }
  }
  double JointProbabilityPlugin::eval(const Sample& sample_, int bin_) {
    LogThrowIf(evalFcn == nullptr, "Library not loaded properly.");
    return reinterpret_cast<double(*)(double, double, double)>(evalFcn)(
      sample_.getDataContainer().histogram->GetBinContent(bin_),
      sample_.getMcContainer().histogram->GetBinContent(bin_),
      sample_.getMcContainer().histogram->GetBinError(bin_)
      );
  }
  void JointProbabilityPlugin::compile(){
    LogInfo << "Compiling: " << llhPluginSrc << std::endl;
    llhSharedLib = GenericToolbox::replaceFileExtension(llhPluginSrc, "so");

    // create library
    std::stringstream ss;
    LogThrowIf( getenv("CXX") == nullptr, "CXX env is not set. Can't compile." );
    ss << "$CXX -std=c++11 -shared " << llhPluginSrc << " -o " << llhSharedLib;
    LogThrowIf( system( ss.str().c_str() ) != 0, "Compile command failed." );
  }
  void JointProbabilityPlugin::load(){
    LogInfo << "Loading shared lib: " << llhSharedLib << std::endl;
    fLib = dlopen( llhSharedLib.c_str(), RTLD_LAZY );
    LogThrowIf(fLib == nullptr, "Cannot open library: " << dlerror());
    evalFcn = (dlsym(fLib, "evalFct"));
    LogThrowIf(evalFcn == nullptr, "Cannot open evalFcn");
  }

  // BarlowLLH_BANFF_OA2021
  void BarlowLLH_BANFF_OA2021::readConfigImpl(){
    allowZeroMcWhenZeroData = GenericToolbox::Json::fetchValue(_config_, "allowZeroMcWhenZeroData", allowZeroMcWhenZeroData);
    usePoissonLikelihood = GenericToolbox::Json::fetchValue(_config_, "usePoissonLikelihood", usePoissonLikelihood);
    BBNoUpdateWeights = GenericToolbox::Json::fetchValue(_config_, "BBNoUpdateWeights", BBNoUpdateWeights);
    verboseLevel = GenericToolbox::Json::fetchValue(_config_, {{"verboseLevel"}, {"isVerbose"}}, verboseLevel);

    LogInfo << "Using BarlowLLH_BANFF_OA2021 parameters:" << std::endl;
    {
      LogScopeIndent;
      LogInfo << GET_VAR_NAME_VALUE(allowZeroMcWhenZeroData) << std::endl;
      LogInfo << GET_VAR_NAME_VALUE(usePoissonLikelihood) << std::endl;
      LogInfo << GET_VAR_NAME_VALUE(BBNoUpdateWeights) << std::endl;
      LogInfo << GET_VAR_NAME_VALUE(verboseLevel) << std::endl;
    }
  }

  double BarlowLLH_BANFF_OA2021::eval(const Sample& sample_, int bin_) {
    TH1D* data = sample_.getDataContainer().histogram.get();
    TH1D* predMC = sample_.getMcContainer().histogram.get();
    TH1D* nomMC = sample_.getMcContainer().histogramNominal.get();

    // From OA2021_Eb branch -> BANFFBinnedSample::CalcLLRContrib
    // https://github.com/t2k-software/BANFF/blob/OA2021_Eb/src/BANFFSample/BANFFBinnedSample.cxx

    double dataVal = data->GetBinContent(bin_);
    double predVal = predMC->GetBinContent(bin_);

    // Why SQUARE?? -> GetBinError is returning the sqrt(Sum^2) but the BANFF
    // let the BBH make the sqrt
    // https://github.com/t2k-software/BANFF/blob/9140ec11bd74606c10ab4af9ec525352de119c06/src/BANFFSample/BANFFBinnedSample.cxx#L374
    double mcuncert{0.0};
    if (BBNoUpdateWeights) {
      mcuncert = nomMC->GetBinError(bin_) * nomMC->GetBinError(bin_);
      if (not std::isfinite(mcuncert) or mcuncert < 0.0) {
        LogError << "BBNoUpdateWeights mcuncert is not valid "
                 << mcuncert
                 << std::endl;
        LogError << "nomMC bin " << bin_
                 << " error is " << nomMC->GetBinError(bin_);
        LogThrow("The mc uncertainty is not a usable number");
      }
    }
    else {
      mcuncert = predMC->GetBinError(bin_) * predMC->GetBinError(bin_);
      if (not std::isfinite(mcuncert) or mcuncert < 0.0) {
        LogError << "The mcuncert is not finite " << mcuncert << std::endl;
        LogError << "nomMC bin " << bin_
                 << " error is " << predMC->GetBinError(bin_);
        LogThrow("The mc uncertainty is not a usable number");
      }
    }
    mcuncert = std::max(mcuncert, std::numeric_limits<double>::epsilon());

    // The MC used in the likelihood calculation is allowed to be changed by
    // Barlow Beeston beta parameters.
    double newmc = predVal;

    // The penalty from MC statistics.  This will remain zero if
    // Barlow-Beeston is not applied.
    double penalty = 0.0;

    // Implementing Barlow-Beeston correction for LLH calculation. The
    // following comments are inspired/copied from Clarence's comments in the
    // MaCh3 implementation of the same feature.  This is not full
    // Barlow-Beeston or what is referred to as "light": we're not introducing
    // any more parameters.  Assume the MC has a Gaussian distribution
    // generated as in https://arxiv.org/abs/1103.0354 eq 10, 11
    if (not usePoissonLikelihood and mcuncert > 0.0) {
      // Physically, the fractional uncertainty should be less than 100% since
      // one MC event in a bin would have 100% fractional uncertainty [under
      // the "Gaussian" approximation, so sqrt(1.0)/1.0].  The OA2021 behavior
      // lets the fractional error grow, but the entire likelihood became
      // discontinuous around a predicted value of 1E-16.  Setting
      // fractionalLimit to 1E+20 exactly matches OA2021 before the
      // discontinuity and behaves reasonably below it.
      const double fractionalLimit = 1E+19;  // Match OA2021 behavior
      // Barlow-Beeston uses fractional uncertainty on MC, so
      // sqrt(sum[w^2])/mc -b/2a in quadratic equation.
      double fractional = std::min(std::sqrt(mcuncert)/newmc, fractionalLimit);
      double temp = newmc * fractional * fractional - 1;
      // b^2 - 4ac in quadratic equation
      double temp2 = temp * temp + 4 * dataVal * fractional * fractional;
      LogThrowIf((temp2 < 0),
                 "Negative square root in Barlow Beeston");
      // Solve for the positive beta
      double beta = (-1 * temp + std::sqrt(temp2)) / 2.;
      // Update the "expected" value for Barlow-Beeston (protect against
      // underflow).
      newmc = std::max(newmc * beta, std::numeric_limits<double>::min());
      // And penalise the movement relative the mc uncertainty
      penalty = (beta - 1) * (beta - 1) / (2 * fractional * fractional);
    }

    // And calculate the Poisson likelihood.  The "stat" variable is valid for
    // both Poisson and the Barlow-Beeston correctin.  The newmc value is
    // either the original predVal, or if Barlow-Beeston is applied it is
    // modified to calculate statistical part of the Barlow-Beeston
    // likelihood.
    double stat = 0;
    if (dataVal < std::numeric_limits<double>::epsilon()) {
      // dataVal should be roughly an integer, so this equivalent to "dataVal
      // == 0", but safer.
      stat = newmc;
    }
    else if (newmc > std::numeric_limits<double>::min()) {
      // The newmc value is normally not a lot less than O(1), but could
      // approach zero.  This splits the ratio of "data/expected" into a
      // difference of logs so that it avoids denormalization problems
      // (i.e. big/small).
      stat = newmc - dataVal + dataVal*(TMath::Log(dataVal)-TMath::Log(newmc));
    }
    else {
      // The mc predicted value is "zero", and the data value is not zero.
      // Inconceivable!  Protect against actual zero expectation, which means
      // data events have a "zero" probability results in an infinite
      // likelihood.  Minimizers don't like it when they recieve an infinite
      // function value since that introduces a discontinuity in the
      // function. This has the practical effect of capping the likelihood per
      // bin, and prevents discontinuities in the likelihood.
      LogErrorIf(verboseLevel>=1)
        << "Data and predicted value give infinite statistical LLH / "
        << "Data: " << dataVal
        << " / Barlow Beeston adjusted MC: " << newmc
        << std::endl;
      const double mc = std::numeric_limits<double>::min();
      stat = mc - dataVal + dataVal*(TMath::Log(dataVal) - TMath::Log(mc));
    }

    // Build the chisq value based on previous calculations.
    double chisq =  2.0*stat;
    // Apply the Barlow-Beeston penalty.
    if (not usePoissonLikelihood) chisq += 2.0 * penalty;

    // Warn when the expected value for a bin is going to zero.
    if (predVal == 0.0
        and dataVal < std::numeric_limits<double>::epsilon()) [[unlikely]] {
      if( allowZeroMcWhenZeroData ) {
        // Need to warn the user something is wrong with the binning
        // definition.  This might indicate that more MC stat would be needed,
        // or the binning needs to be reconsidered..
        LogErrorOnce
          << "Sample bin with no events in the data and MC bin."
          << "This is an ill conditioned problem. Please check your inputs."
          << std::endl;
      }
      else {
        LogWarningIf(verboseLevel > 0)
          << "Infinite statistical LLH --"
          << " Data: " << dataVal
          << " / MC: " << predVal
          << " Adjusted MC: " << newmc
          << " / Stat: " << stat
          << " Penalty: " << penalty
          << " Truncated ChiSq: " << chisq
          << std::endl;
      }
    }

    if (not std::isfinite(chisq)) [[unlikely]] {
      LogWarning << "Non finite chi2: " << chisq << std::endl
                 << " bin " << bin_ << std::endl
                 << GET_VAR_NAME_VALUE(predVal) << std::endl
                 << GET_VAR_NAME_VALUE(dataVal) << std::endl
                 << GET_VAR_NAME_VALUE(newmc) << std::endl
                 << GET_VAR_NAME_VALUE(stat) << std::endl
                 << GET_VAR_NAME_VALUE(penalty) << std::endl
                 << GET_VAR_NAME_VALUE(mcuncert) << std::endl
                 << GET_VAR_NAME_VALUE(nomMC->GetBinContent(bin_)) << std::endl
                 << GET_VAR_NAME_VALUE(nomMC->GetBinError(bin_)) << std::endl
                 << GET_VAR_NAME_VALUE(predMC->GetBinError(bin_)) << std::endl
                 << GET_VAR_NAME_VALUE(predMC->GetBinContent(bin_)) << std::endl;
    }

    if(verboseLevel>=3){
      LogTrace << "Bin #" << bin_ << ": chisq(" << chisq << ") / predVal(" << predVal << ") / dataVal(" << dataVal << ")" << std::endl;
    }

    // The chi-squared value must not be NaN.  If it is, then the calculation
    // should stop since the likelihood and it's derivatives are broken.  This
    // can only happen with invalid inputs.
    LogThrowIf(std::isnan(chisq),"Chi2 is NaN");

    return chisq;
  }

  // BarlowLLH_BANFF_OA2020
  double BarlowLLH_BANFF_OA2020::eval(const Sample& sample_, int bin_){
    // From BANFF: origin/OA2020 branch -> BANFFBinnedSample::CalcLLRContrib()

    //Loop over all the bins one by one using their unique bin index.
    //Use the stored nBins value and bins array so avoid trying to calculate
    //over underflow or overflow bins.
    double chisq{0};

    double dataVal = sample_.getDataContainer().histogram->GetBinContent(bin_);
    double predVal = sample_.getMcContainer().histogram->GetBinContent(bin_);
    double mcuncert = sample_.getMcContainer().histogram->GetBinError(bin_);

    //implementing Barlow-Beeston correction for LH calculation the
    //following comments are inspired/copied from Clarence's comments in the
    //MaCh3 implementation of the same feature

    // The MC used in the likeliihood calculation
    // Is allowed to be changed by Barlow Beeston beta parameters
    double newmc = predVal;
    // Not full Barlow-Beeston or what is referred to as "light": we're not
    // introducing any more parameters Assume the MC has a Gaussian
    // distribution around generated As in https://arxiv.org/abs/1103.0354 eq
    // 10, 11

    // The penalty from MC statistics
    double penalty = 0;
    // Barlow-Beeston uses fractional uncertainty on MC, so sqrt(sum[w^2])/mc
    double fractional = std::sqrt(mcuncert)/predVal;
    // -b/2a in quadratic equation
    double temp = predVal*fractional*fractional-1;
    // b^2 - 4ac in quadratic equation
    double temp2 = temp*temp + 4*dataVal*fractional*fractional;
    LogThrowIf(temp2 < 0, "Negative square root in Barlow Beeston coefficient calculation!");

    // Solve for the positive beta
    double beta = (-1*temp+std::sqrt(temp2))/2.;
    newmc = predVal*beta;
    // And penalise the movement in beta relative the mc uncertainty
    penalty = (beta-1)*(beta-1)/(2*fractional*fractional);

    //    LogThrowIf(
    //        std::isnan(penalty),
    //        GET_VAR_NAME_VALUE(fractional)
    //        << " / " << GET_VAR_NAME_VALUE(mcuncert)
    //        << " / " << GET_VAR_NAME_VALUE(predVal)
    //        << " / " << GET_VAR_NAME_VALUE(bin_)
    //        );

    // And calculate the new Poisson likelihood
    // For Barlow-Beeston newmc is modified, so can only calculate Poisson likelihood after Barlow-Beeston
    double stat = 0;
    if (dataVal == 0) stat = newmc;
    else if (newmc > 0) stat = newmc-dataVal+dataVal*TMath::Log(dataVal/newmc);

    if( std::isnan(penalty) ){ penalty = (1+stat)*1E32; } // huge penalty

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
               << sample_.getMcContainer().histogram->GetBinError(bin_) << " "
               << sample_.getMcContainer().histogram->GetBinContent(bin_) << std::endl;
    }

    LogThrowIf(std::isnan(chisq), "NaN chi2 " << predVal << " " << dataVal
               << sample_.getMcContainer().histogram->GetBinError(bin_) << " "
               << sample_.getMcContainer().histogram->GetBinContent(bin_));

    return chisq;
  }

  // Chi2
  double Chi2::eval(const Sample& sample_, int bin_){
    double predVal = sample_.getMcContainer().histogram->GetBinContent(bin_);
    double dataVal = sample_.getDataContainer().histogram->GetBinContent(bin_);
    if( predVal == 0 ){
      // should not be the case right?
      LogAlert << "Zero MC events in bin " << bin_ << ". predVal = " << predVal << ", dataVal = " << dataVal
               << ". Setting llh = +inf for this bin." << std::endl;
      return std::numeric_limits<double>::infinity();
    }
    return TMath::Sq(predVal - dataVal)/predVal;
  }

  // PoissonLLH
  double PoissonLLH::eval(const Sample& sample_, int bin_){
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

  // LeastSquaresLLH
  void LeastSquaresLLH::readConfigImpl(){
    LogWarning << "Using LeastSquaresLLH: NOT A REAL LIKELIHOOD" << std::endl;
    lsqPoissonianApproximation = GenericToolbox::Json::fetchValue(_config_, "lsqPoissonianApproximation", lsqPoissonianApproximation);
    LogWarning << "Using Least Squares Poissonian Approximation" << std::endl;
  }
  double LeastSquaresLLH::eval(const Sample& sample_, int bin_){
    double predVal = sample_.getMcContainer().histogram->GetBinContent(bin_);
    double dataVal = sample_.getDataContainer().histogram->GetBinContent(bin_);
    double v = dataVal - predVal;
    v = v*v;
    if (lsqPoissonianApproximation && dataVal > 1.0) v /= 0.5*dataVal;
    return v;
  }

  // BarlowLLH
  double BarlowLLH::eval(const Sample& sample_, int bin_){
    rel_var = sample_.getMcContainer().histogram->GetBinError(bin_) / TMath::Sq(sample_.getMcContainer().histogram->GetBinContent(bin_));
    b       = (sample_.getMcContainer().histogram->GetBinContent(bin_) * rel_var) - 1;
    c       = 4 * sample_.getDataContainer().histogram->GetBinContent(bin_) * rel_var;

    beta   = (-b + std::sqrt(b * b + c)) / 2.0;
    mc_hat = sample_.getMcContainer().histogram->GetBinContent(bin_) * beta;

    // Calculate the following LLH:
    //-2lnL = 2 * beta*mc - data + data * ln(data / (beta*mc)) + (beta-1)^2 / sigma^2
    // where sigma^2 is the same as above.
    chi2 = 0.0;
    if(sample_.getDataContainer().histogram->GetBinContent(bin_) <= 0.0) {
      chi2 = 2 * mc_hat;
      chi2 += (beta - 1) * (beta - 1) / rel_var;
    }
    else{
      chi2 = 2 * (mc_hat - sample_.getDataContainer().histogram->GetBinContent(bin_));
      if(sample_.getDataContainer().histogram->GetBinContent(bin_) > 0.0) {
        chi2 += 2 * sample_.getDataContainer().histogram->GetBinContent(bin_) *
          std::log(sample_.getDataContainer().histogram->GetBinContent(bin_) / mc_hat);
      }
      chi2 += (beta - 1) * (beta - 1) / rel_var;
    }
    return chi2;
  }

  double BarlowLLH_BANFF_OA2021_SFGD::eval(const Sample& sample_, int bin_){

    double dataVal = sample_.getDataContainer().histogram->GetBinContent(bin_);
    double predVal = sample_.getMcContainer().histogram->GetBinContent(bin_);
    double mcuncert = sample_.getMcContainer().histogram->GetBinError(bin_);

    double chisq = 0.0;

    bool usePoissonLikelihood = false;

    double newmc = predVal;

    // The penalty from MC statistics
    double penalty = 0;

    // SFGD detector uncertainty
    double sfgd_det_uncert = 0.;
    if (sample_.getName().find("SFGD") != std::string::npos){
      // to be applied on SFGD samples only
      sfgd_det_uncert = 0.;
      if (sample_.getName().find("FHC") != std::string::npos){
        if (sample_.getName().find("0p") != std::string::npos){
          sfgd_det_uncert = 0.02;
        }
        else if (sample_.getName().find("Np") != std::string::npos){
          sfgd_det_uncert = 0.04;
        }
      }
      else if (sample_.getName().find("RHC") != std::string::npos){
        if (sample_.getName().find("0n") != std::string::npos){
          sfgd_det_uncert = 0.025;
        }
        else if (sample_.getName().find("Nn") != std::string::npos){
          sfgd_det_uncert = 0.05;
        }
      }
    }

    // DETECTOR UNCERTAINTY FOR WAGASCI
    double wg_det_uncert = 0.;
    if (sample_.getName().find("WAGASCI") != std::string::npos){
      wg_det_uncert = 0.;
      if(sample_.getName().find("#0pi") != std::string::npos) {
        if(sample_.getName().find("PM-BM") != std::string::npos) wg_det_uncert = 0.05;
        if(sample_.getName().find("PM-WMRD") != std::string::npos) wg_det_uncert = 0.1;
        if(sample_.getName().find("DWG-BM") != std::string::npos) wg_det_uncert = 0.1;
        if(sample_.getName().find("UWG-BM") != std::string::npos) wg_det_uncert = 0.12;
        if(sample_.getName().find("UWG-WMRD") != std::string::npos) wg_det_uncert = 0.1;
      }
      if(sample_.getName().find("#1pi") != std::string::npos)  {
        if(sample_.getName().find("PM") != std::string::npos) wg_det_uncert = 0.1;
        if(sample_.getName().find("WG") != std::string::npos) wg_det_uncert = 0.08;
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
    double beta = (-1 * temp + std::sqrt(temp2)) / 2.;
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
               << sample_.getMcContainer().histogram->GetBinError(bin_) << " "
               << sample_.getMcContainer().histogram->GetBinContent(bin_) << std::endl;
    }

    return chisq;
  }

}

//  A Lesser GNU Public License

//  Copyright (C) 2023 GUNDAM DEVELOPERS

//  This library is free software; you can redistribute it and/or
//  modify it under the terms of the GNU Lesser General Public
//  License as published by the Free Software Foundation; either
//  version 2.1 of the License, or (at your option) any later version.

//  This library is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
//  Lesser General Public License for more details.

//  You should have received a copy of the GNU Lesser General Public
//  License along with this library; if not, write to the
//
//  Free Software Foundation, Inc.
//  51 Franklin Street, Fifth Floor,
//  Boston, MA  02110-1301  USA

// Local Variables:
// mode:c++
// c-basic-offset:2
// End:
