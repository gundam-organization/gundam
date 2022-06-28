//
// Created by Adrien BLANCHET on 23/06/2022.
//

#ifndef GUNDAM_JOINTPROBABILITY_H
#define GUNDAM_JOINTPROBABILITY_H

#include "FitSample.h"



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
