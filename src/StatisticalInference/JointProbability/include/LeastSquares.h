//
// Created by Nadrino on 22/02/2024.
//

#ifndef GUNDAM_LEAST_SQUARES_H
#define GUNDAM_LEAST_SQUARES_H

#include "JointProbabilityBase.h"


namespace JointProbability{

  class LeastSquares : public JointProbabilityBase {

  /// Evaluate the Least Squares difference between the expected and observed.
  /// This is NOT a real LLH function, but is good for debugging since it has
  /// minimal numeric problems (doesn't use any functions like Log or Sqrt).

  protected:
    void configureImpl() override;

  public:
    [[nodiscard]] std::string getType() const override { return "PluginJointProbability"; }
    [[nodiscard]] double eval(double data_, double pred_, double err_, int bin_) const override;

    /// If true the use Poissonian approximation with the variance equal to
    /// the observed value (i.e. the data).
    bool lsqPoissonianApproximation{false};

  };

  void LeastSquares::configureImpl(){
    LogWarning << "Using LeastSquaresLLH: NOT A REAL LIKELIHOOD" << std::endl;
    GenericToolbox::Json::fillValue(_config_, lsqPoissonianApproximation, "lsqPoissonianApproximation");
    LogWarning << "Using Least Squares Poissonian Approximation" << std::endl;
  }

  double LeastSquares::eval(double data_, double pred_, double err_, int bin_) const {
    double predVal = pred_;
    double dataVal = data_;
    double v = dataVal - predVal;
    v = v*v;
    if (lsqPoissonianApproximation && dataVal > 1.0) v /= 0.5*dataVal;
    return v;
  }

}

#endif // GUNDAM_LEAST_SQUARES_H
