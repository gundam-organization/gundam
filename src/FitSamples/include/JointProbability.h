//
// Created by Adrien BLANCHET on 23/06/2022.
//

#ifndef GUNDAM_JOINTPROBABILITY_H
#define GUNDAM_JOINTPROBABILITY_H

#include "FitSample.h"
#include "JsonBaseClass.h"

#include "GenericToolbox.h"

#include "nlohmann/json.hpp"

#include "string"
#include "sstream"
#include "memory"
#include <dlfcn.h>



namespace JointProbability {

  class JointProbability : public JsonBaseClass {
  public:
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

  public:
    double eval(const FitSample& sample_, int bin_) override;

    std::string llhPluginSrc;
    std::string llhSharedLib;

  protected:
    void readConfigImpl() override;
    void initializeImpl() override;
    void compile();
    void load();

  private:
    void* fLib{nullptr};
    void* evalFcn{nullptr};

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

  protected:
    void readConfigImpl() override;

  public:
    double eval(const FitSample& sample_, int bin_) override;

    bool usePoissonLikelihood{false};
    bool BBNoUpdateWeights{false};
  };

  class BarlowLLH_BANFF_OA2021_SFGD : public JointProbability{
    double eval(const FitSample& sample_, int bin_) override;
  };

}






#endif //GUNDAM_JOINTPROBABILITY_H
