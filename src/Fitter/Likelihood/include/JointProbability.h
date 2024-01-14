//
// Created by Nadrino on 23/06/2022.
//

#ifndef GUNDAM_JOINT_PROBABILITY_H
#define GUNDAM_JOINT_PROBABILITY_H

#include "Sample.h"
#include "JsonBaseClass.h"

#include <string>


namespace JointProbability {

#define ENUM_NAME JointProbabilityType
#define ENUM_FIELDS \
  ENUM_FIELD( PoissonLLH, 0 ) \
  ENUM_FIELD( LeastSquaresLLH ) \
  ENUM_FIELD( BarlowLLH ) \
  ENUM_FIELD( BarlowLLH_BANFF_OA2020 ) \
  ENUM_FIELD( BarlowLLH_BANFF_OA2021 ) \
  ENUM_FIELD( BarlowLLH_BANFF_OA2021_SFGD ) \
  ENUM_FIELD( Chi2 ) \
  ENUM_FIELD( Plugin )
#include "GenericToolbox.MakeEnum.h"

  class JointProbability : public JsonBaseClass {
  public:
    [[nodiscard]] virtual JointProbabilityType getType() const = 0;

    // two choices -> either override bin by bin llh or global eval function
    [[nodiscard]] virtual double eval(const Sample& sample_, int bin_) const { return 0; }
    [[nodiscard]] virtual double eval(const Sample& sample_) const {
      double out{0};
      int nBins = int(sample_.getBinning().getBinList().size());
      for( int iBin = 1 ; iBin <= nBins ; iBin++ ){ out += this->eval(sample_, iBin); }
      return out;
    }

  };
  class PoissonLLH : public JointProbability{
  public:
    [[nodiscard]] JointProbabilityType getType() const override { return JointProbabilityType::PoissonLLH; }
    [[nodiscard]] double eval(const Sample& sample_, int bin_) const override;
  };
  class LeastSquaresLLH : public JointProbability{
    /// Evaluate the Least Squares difference between the expected and observed.
    /// This is NOT a real LLH function, but is good for debugging since it has
    /// minimal numeric problems (doesn't use any functions like Log or Sqrt).

  protected:
    void readConfigImpl() override;

  public:
    [[nodiscard]] JointProbabilityType getType() const override { return JointProbabilityType::LeastSquaresLLH; }
    [[nodiscard]] double eval(const Sample& sample_, int bin_) const override;

    /// If true the use Poissonian approximation with the variance equal to
    /// the observed value (i.e. the data).
    bool lsqPoissonianApproximation{false};
  };
  class BarlowLLH : public JointProbability{
  public:
    [[nodiscard]] JointProbabilityType getType() const override { return JointProbabilityType::BarlowLLH; }
    [[nodiscard]] double eval(const Sample& sample_, int bin_) const override;
  private:

    struct Buffer{
      double rel_var, b, c, beta, mc_hat, chi2;
    };
    mutable Buffer _buf_{};
  };
  class BarlowLLH_BANFF_OA2020 : public JointProbability{
  public:
    [[nodiscard]] JointProbabilityType getType() const override { return JointProbabilityType::BarlowLLH_BANFF_OA2020; }
    [[nodiscard]] double eval(const Sample& sample_, int bin_) const override;
  };
  class BarlowLLH_BANFF_OA2021 : public JointProbability{

  protected:
    void readConfigImpl() override;

  public:
    [[nodiscard]] JointProbabilityType getType() const override { return JointProbabilityType::BarlowLLH_BANFF_OA2021; }
    [[nodiscard]] double eval(const Sample& sample_, int bin_) const override;

    int verboseLevel{0};
    bool throwIfInfLlh{false};
    bool allowZeroMcWhenZeroData{true};
    bool usePoissonLikelihood{false};
    bool BBNoUpdateWeights{false}; // OA 2021 bug reimplementation
  };
  class BarlowLLH_BANFF_OA2021_SFGD : public JointProbability{
  public:
    [[nodiscard]] JointProbabilityType getType() const override { return JointProbabilityType::BarlowLLH_BANFF_OA2021_SFGD; }
    [[nodiscard]] double eval(const Sample& sample_, int bin_) const override;
  };
  class Chi2 : public JointProbability{
  public:
    [[nodiscard]] JointProbabilityType getType() const override { return JointProbabilityType::Chi2; }
    [[nodiscard]] double eval(const Sample& sample_, int bin_) const override;
  };
  class Plugin : public JointProbability{

  public:
    [[nodiscard]] JointProbabilityType getType() const override { return JointProbabilityType::Plugin; }
    [[nodiscard]] double eval(const Sample& sample_, int bin_) const override;

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

  JointProbability* makeJointProbability(const std::string& type_);
  JointProbability* makeJointProbability(JointProbabilityType type_);

}






#endif //GUNDAM_JOINT_PROBABILITY_H
