//
// Created by Adrien BLANCHET on 23/06/2022.
//

#include "JointProbability.h"

#include "Logger.h"
#include "GenericToolbox.h"
#include "JsonUtils.h"

#include "TMath.h"


LoggerInit([]{
  Logger::setUserHeaderStr("[JointProbability]");
});

namespace JointProbability{

  // JointProbabilityPlugin
  void JointProbabilityPlugin::readConfigImpl() {
    llhPluginSrc = JsonUtils::fetchValue<std::string>(_config_, "llhPluginSrc");
    llhSharedLib = JsonUtils::fetchValue<std::string>(_config_, "llhSharedLib");
  }
  void JointProbabilityPlugin::initializeImpl(){
    if( not llhSharedLib.empty() ) this->load();
    else if( not llhPluginSrc.empty() ){ this->compile(); this->load(); }
    else{ LogThrow("Can't initialize JointProbabilityPlugin without llhSharedLib nor llhPluginSrc."); }
  }
  double JointProbabilityPlugin::eval(const FitSample& sample_, int bin_) {
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
    system ( ss.str().c_str() );
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
    usePoissonLikelihood = JsonUtils::fetchValue(_config_, "usePoissonLikelihood", usePoissonLikelihood);
    BBNoUpdateWeights = JsonUtils::fetchValue(_config_, "BBNoUpdateWeights", BBNoUpdateWeights);

    LogInfo << "Using BarlowLLH_BANFF_OA2021 parameters:" << std::endl;
    LogInfo << GET_VAR_NAME_VALUE(usePoissonLikelihood) << std::endl;
    LogInfo << GET_VAR_NAME_VALUE(BBNoUpdateWeights) << std::endl;
  }
  double BarlowLLH_BANFF_OA2021::eval(const FitSample& sample_, int bin_){

    TH1D* data = sample_.getDataContainer().histogram.get();
    TH1D* predMC = sample_.getMcContainer().histogram.get();
    TH1D* nomMC = sample_.getMcContainer().histogramNominal.get();

    // From OA2021_Eb branch -> BANFFBinnedSample::CalcLLRContrib
    // https://github.com/t2k-software/BANFF/blob/OA2021_Eb/src/BANFFSample/BANFFBinnedSample.cxx

    double chisq = 0.0;
    double dataVal, predVal;

//    bool usePoissonLikelihood = (bool)ND::params().GetParameterI("BANFF.UsePoissonLikelihood");
//    bool BBNoUpdateWeights = (bool)ND::params().GetParameterI("BANFF.BarlowBeestonNoUpdateWeights");

    //Loop over all the bins one by one using their unique bin index.
    //Use the stored nBins value and bins array so avoid trying to calculate
    //over underflow or overflow bins.
//    for (int i = 0; i < nBins; i++)
//    {

    dataVal = data->GetBinContent(bin_);
    predVal = predMC->GetBinContent(bin_);
    double mcuncert;

    // Why SQUARE?? -> GetBinError is returning the sqrt(Sum^2) but the BANFF let the BBH make the sqrt
    // https://github.com/t2k-software/BANFF/blob/9140ec11bd74606c10ab4af9ec525352de119c06/src/BANFFSample/BANFFBinnedSample.cxx#L374
    if (BBNoUpdateWeights){
      mcuncert = nomMC->GetBinError(bin_) * nomMC->GetBinError(bin_);
    }
    else{
      mcuncert = predMC->GetBinError(bin_) * predMC->GetBinError(bin_);
    }

    //implementing Barlow-Beeston correction for LH calculation
    //the following comments are inspired/copied from Clarence's comments in the MaCh3
    //implementation of the same feature

    // The MC used in the likelihood calculation
    // Is allowed to be changed by Barlow Beeston beta parameters
    double newmc = predVal;
    // Not full Barlow-Beeston or what is referred to as "light": we're not introducing any more parameters
    // Assume the MC has a Gaussian distribution around generated
    // As in https://arxiv.org/abs/1103.0354 eq 10, 11

    // The penalty from MC statistics
    double penalty = 0;
    // Barlow-Beeston uses fractional uncertainty on MC, so sqrt(sum[w^2])/mc
    double fractional = sqrt(mcuncert) / predVal;
    // -b/2a in quadratic equation
    double temp = predVal * fractional * fractional - 1;
    // b^2 - 4ac in quadratic equation
    double temp2 = temp * temp + 4 * dataVal * fractional * fractional;
    if (temp2 < 0)
    {
      std::cerr << "Negative square root in Barlow Beeston coefficient calculation!" << std::endl;
      std::cerr << __FILE__ << ":" << __LINE__ << std::endl;
      throw;
    }
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

    if (std::isnan(chisq))
    {
      std::cout << "Infinite chi2 " << predVal << " " << dataVal
                << " " << nomMC->GetBinContent(bin_) << " "
                << predMC->GetBinError(bin_) << " "
                << predMC->GetBinContent(bin_) << std::endl;
    }
//    }

    return chisq;
  }

  // BarlowLLH_BANFF_OA2020
  double BarlowLLH_BANFF_OA2020::eval(const FitSample& sample_, int bin_){
    // From BANFF: origin/OA2020 branch -> BANFFBinnedSample::CalcLLRContrib()

    //Loop over all the bins one by one using their unique bin index.
    //Use the stored nBins value and bins array so avoid trying to calculate
    //over underflow or overflow bins.
    double chisq{0};

    double dataVal = sample_.getDataContainer().histogram->GetBinContent(bin_);
    double predVal = sample_.getMcContainer().histogram->GetBinContent(bin_);
    double mcuncert = sample_.getMcContainer().histogram->GetBinError(bin_);

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

  // PoissonLLH
  double PoissonLLH::eval(const FitSample& sample_, int bin_){
    double predVal = sample_.getMcContainer().histogram->GetBinContent(bin_);
    double dataVal = sample_.getDataContainer().histogram->GetBinContent(bin_);
    if(predVal <= 0){
      LogAlert << "Zero MC events in bin " << bin_ << ". predVal = " << predVal << ", dataVal = " << dataVal
               << ". Setting chi2_stat = 0 for this bin." << std::endl;
      return 0;
    }
    return 2.0 * (predVal - dataVal + dataVal * TMath::Log(dataVal / predVal));
  }

  // BarlowLLH
  double BarlowLLH::eval(const FitSample& sample_, int bin_){
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

  double BarlowLLH_BANFF_OA2021_SFGD::eval(const FitSample& sample_, int bin_){

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
      double fractional = sqrt((mcuncert / predVal)*(mcuncert / predVal) + sfgd_det_uncert*sfgd_det_uncert + wg_det_uncert*wg_det_uncert); // Add SFGD detector uncertainty 
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
              << sample_.getMcContainer().histogram->GetBinError(bin_) << " "
              << sample_.getMcContainer().histogram->GetBinContent(bin_) << std::endl;
      }

      return chisq;
  }

}
