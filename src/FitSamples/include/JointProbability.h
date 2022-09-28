//
// Created by Adrien BLANCHET on 23/06/2022.
//

#ifndef GUNDAM_JOINTPROBABILITY_H
#define GUNDAM_JOINTPROBABILITY_H

#include "FitSample.h"

#include "string"
#include "sstream"
#include "memory"
#include <dlfcn.h>



namespace JointProbability{

  class JointProbability {

  public:
    JointProbability() = default;
    virtual ~JointProbability() = default;

    // two choices -> either override bin by bin llh or global eval function
    virtual double eval(const FitSample& sample_, int bin_){ return 0; }
    virtual double eval(const FitSample& sample_){
      double out{0};
      int nBins = int(sample_.getBinning().getBinsList().size());
      for( int iBin = 1 ; iBin <= nBins ; iBin++ ){ out += this->eval(sample_, iBin); }
      return out;
    }
  };


  class JointProbabilityPlugin : public JointProbability{

  private:
    void* fLib{nullptr};
    void* evalFcn{nullptr};

  public:
    void compile(const std::string& srcPath_){
      LogInfo << "Compiling: " << srcPath_ << std::endl;
      std::string outLib = srcPath_.substr(0, srcPath_.find_last_of('.')) + ".so";

      // create library
      std::stringstream ss;
      LogThrowIf( getenv("CXX") == nullptr, "CXX env is not set. Can't compile." );
      ss << "$CXX -std=c++11 -shared " << srcPath_ << " -o " << outLib;
      system ( ss.str().c_str() );
      this->load(outLib);
    }
    void load(const std::string& sharedLibPath_){
      LogInfo << "Loading shared lib: " << sharedLibPath_ << std::endl;
      fLib = dlopen( sharedLibPath_.c_str(), RTLD_LAZY );
      LogThrowIf(fLib == nullptr, "Cannot open library: " << dlerror());
      evalFcn = (dlsym(fLib, "evalFct"));
      LogThrowIf(evalFcn == nullptr, "Cannot open evalFcn");
    }

    double eval(const FitSample& sample_, int bin_) override {
      LogThrowIf(evalFcn == nullptr, "Library not loaded properly.");
      return reinterpret_cast<double(*)(double, double, double)>(evalFcn)(
          sample_.getDataContainer().histogram->GetBinContent(bin_),
          sample_.getMcContainer().histogram->GetBinContent(bin_),
          sample_.getMcContainer().histogram->GetBinError(bin_)
          );
    }

  };

  class PoissonLLH : public JointProbability{
    double eval(const FitSample& sample_, int bin_) override;
  };

  class BarlowLLH : public JointProbability{
    double eval(const FitSample& sample_, int bin_) override;
  private:
    double rel_var, b, c, beta, mc_hat, chi2;
  };

  class BarlowLLH_BANFF_OA2020 : public JointProbability{
    double eval(const FitSample& sample_, int bin_) override;
  };
  class BarlowLLH_BANFF_OA2021 : public JointProbability{
    double eval(const FitSample& sample_, int bin_) override;
  };

}






#endif //GUNDAM_JOINTPROBABILITY_H
