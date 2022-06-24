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

    virtual double eval(const FitSample& sample_) = 0;
  };

  class PoissonLLH : public JointProbability{
    double eval(const FitSample& sample_) override;
  };

  class BarlowLLH : public JointProbability{
    double eval(const FitSample& sample_) override;
  };

  class BarlowLLH_BANFF_OA2020 : public JointProbability{
    double eval(const FitSample& sample_) override;
  };

}






#endif //GUNDAM_JOINTPROBABILITY_H
