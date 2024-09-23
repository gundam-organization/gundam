//
// Created by Adrien BLANCHET on 23/06/2022.
//

#ifndef GUNDAM_JOINTPROBABILITY_H
#define GUNDAM_JOINTPROBABILITY_H

#include "Sample.h"
#include "JsonBaseClass.h"

#include "GenericToolbox.h"

#include "nlohmann/json.hpp"

#include <dlfcn.h>
#include <sstream>
#include <string>
#include <memory>

namespace JointProbability {

  class JointProbability : public JsonBaseClass {
  public:
    // two choices -> either override bin by bin llh or global eval function
    virtual double eval(const Sample& sample_, int bin_){ return 0; }
    virtual double eval(const Sample& sample_){
      double out{0};
      int nBins = int(sample_.getBinning().getBinList().size());
      for( int iBin = 1 ; iBin <= nBins ; iBin++ ){ out += this->eval(sample_, iBin); }
      return out;
    }
  };

  class JointProbabilityPlugin : public JointProbability{

  public:
    double eval(const Sample& sample_, int bin_) override;

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


  class Chi2 : public JointProbability{
    double eval(const Sample& sample_, int bin_) override;
  };

  class PoissonLLH : public JointProbability{
    double eval(const Sample& sample_, int bin_) override;
  };

  /// Evaluate the Least Squares difference between the expected and observed.
  /// This is NOT a real LLH function, but is good for debugging since it has
  /// minimal numeric problems (doesn't use any functions like Log or Sqrt).
  class LeastSquaresLLH : public JointProbability{
  protected:
    void readConfigImpl() override;

  public:
    double eval(const Sample& sample_, int bin_) override;

    /// If true the use Poissonian approximation with the variance equal to
    /// the observed value (i.e. the data).
    bool lsqPoissonianApproximation{false};
  };

  class BarlowLLH : public JointProbability{
    double eval(const Sample& sample_, int bin_) override;
  private:
    double rel_var, b, c, beta, mc_hat, chi2;
  };

  class BarlowLLH_BANFF_OA2020 : public JointProbability{
    double eval(const Sample& sample_, int bin_) override;
  };

  class BarlowLLH_BANFF_OA2021 : public JointProbability{

  protected:
    void readConfigImpl() override;

  public:
    double eval(const Sample& sample_, int bin_) override;

    int verboseLevel{0};
    bool allowZeroMcWhenZeroData{true};
    bool usePoissonLikelihood{false};
    // OA 2021 bug reimplementation
    bool BBNoUpdateWeights{false};
    // OA2021 bug reimplmentation (set to numeric_limits::min() to reproduce
    // the bug).
    double expectedValueMinimum{-1.0};
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
  };

  class BarlowLLH_BANFF_OA2021_SFGD : public JointProbability{
    double eval(const Sample& sample_, int bin_) override;
  };

}
#endif //GUNDAM_JOINTPROBABILITY_H
