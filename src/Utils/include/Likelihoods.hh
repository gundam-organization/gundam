#ifndef LIKELIHOODS_HH
#define LIKELIHOODS_HH

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "TMath.h"

class CalcLLHFunc
{
public:
  virtual ~CalcLLHFunc() {};
  virtual double operator()(double mc, double w2, double data)
  {
    return 0.0;
  }
};

class PoissonLLH : public CalcLLHFunc
{
public:
  // Compute the standard Poisson likelihood (chi2_stat contribution from a certain sample and bin) based on the number of MC predicted (mc) and the number of data events (data):
  double operator()(double mc, double w2, double data) override
  {
    // generateFormula chi2 variable which will be updated below and then returned:
    double chi2 = 0.0;

    // If number of MC predicted events is greater than zero, we calculate the statistical chi2 according to 2 * (N_pred - N_data + N_data * log(N_data/N_pred)):
    if(mc > 0.0)
    {
      chi2 = 2 * (mc - data);
      if(data > 0.0)
        chi2 += 2 * data * TMath::Log(data / mc);
    }

    // If chi2 is greater or equal to zero, we return the value calculated above, otherwise zero is returned:
    return (chi2 >= 0.0) ? chi2 : 0.0;
  }
};

class PoissonLLH2 : public CalcLLHFunc
{
public:
  double operator()(double mc, double w2, double data) override {
    if( mc <= 0 ) return 0;
    return -TMath::Log(TMath::Poisson(data, mc));
  }
};

class PoissonLLH3 : public CalcLLHFunc
{
public:
  double operator()(double mc, double w2, double data) override {
    if( mc <= 0 ) return 0;
    return mc + std::lgamma(data) - data*std::log(mc);
  }
};

class EffLLH : public CalcLLHFunc
{
public:
  double operator()(double mc, double w2, double data) override
  {
    // Effective LLH based on Tianlu's paper.
    if(mc <= 0.0)
      return 0.0;

    const double b = mc / w2;
    const double a = (mc * b) + 1.0;
    const double k = data;

    return -2 * (a * std::log(b) + std::lgamma(k + a) - std::lgamma(k + 1)
                 - ((k + a) * std::log1p(b)) - std::lgamma(a));
  }
};

class BarlowLLH : public CalcLLHFunc {
public:
  double rel_var, b, c, beta, mc_hat, chi2;

  double operator()(double mc_, double w2_, double data_) override {
//    if(mc_ == data_ ) return 0;
    // Solving for the quadratic equation,
    // beta^2 + (mu * sigma^2 - 1)beta - data * sigma^2) = 0
    // where sigma^2 is the relative variance.
    rel_var = w2_ / (mc_ * mc_);
    b       = (mc_ * rel_var) - 1;
    c       = 4 * data_ * rel_var;

    beta   = (-b + std::sqrt(b * b + c)) / 2.0;
    mc_hat = mc_ * beta;

    // Calculate the following LLH:
    //-2lnL = 2 * beta*mc - data + data * ln(data / (beta*mc)) + (beta-1)^2 / sigma^2
    // where sigma^2 is the same as above.
    chi2 = 0.0;
    //if(data <= 0.0)
    //{
    //    chi2 = 2 * mc_hat;
    //    chi2 += (beta - 1) * (beta - 1) / rel_var;
    //}
    if(mc_hat > 0.0) {
      chi2 = 2 * (mc_hat - data_);
      if(data_ > 0.0)
        chi2 += 2 * data_ * std::log(data_ / mc_hat);

      chi2 += (beta - 1) * (beta - 1) / rel_var;
    }

    return (chi2 >= 0.0) ? chi2 : 0.0;
  }
};

class BarlowOA2020BugLLH : public CalcLLHFunc {
public:
  double rel_var, b, c, beta, mc_hat, chi2;

  double operator()(double mc_, double w2_, double data_) override {
//    if(mc_ == data_ ) return 0;
    // Solving for the quadratic equation,
    // beta^2 + (mu * sigma^2 - 1)beta - data * sigma^2) = 0
    // where sigma^2 is the relative variance.
    rel_var = std::sqrt(w2_) / (mc_ * mc_);
    b       = (mc_ * rel_var) - 1;
    c       = 4 * data_ * rel_var;

    beta   = (-b + std::sqrt(b * b + c)) / 2.0;
    mc_hat = mc_ * beta;

    // Calculate the following LLH:
    //-2lnL = 2 * beta*mc - data + data * ln(data / (beta*mc)) + (beta-1)^2 / sigma^2
    // where sigma^2 is the same as above.
    chi2 = 0.0;
    //if(data <= 0.0)
    //{
    //    chi2 = 2 * mc_hat;
    //    chi2 += (beta - 1) * (beta - 1) / rel_var;
    //}
    if(mc_hat > 0.0) {
      chi2 = 2 * (mc_hat - data_);
      if(data_ > 0.0)
        chi2 += 2 * data_ * std::log(data_ / mc_hat);

      chi2 += (beta - 1) * (beta - 1) / rel_var;
    }

    return (chi2 >= 0.0) ? chi2 : 0.0;
  }
};

class BarlowBeestonLLH : public CalcLLHFunc
{
public:
  double operator()(double mc, double w2, double data) override
  {
    // Solving for the quadratic equation,
    // beta^2 + (mu * sigma^2 - 1)beta - data * sigma^2) = 0
    // where sigma^2 is the relative variance.
    double rel_var = w2 / (mc * mc);
    double b       = (mc * rel_var) - 1;
    double c       = 4 * data * rel_var;

    double beta   = (-b + std::sqrt(b * b + c)) / 2.0;
    double mc_hat = mc * beta;

    // Calculate the following LLH:
    //-2lnL = 2 * beta*mc - data + data * ln(data / (beta*mc)) + (beta-1)^2 / sigma^2
    // where sigma^2 is the same as above.
    double chi2 = 0.0;
    //if(data <= 0.0)
    //{
    //    chi2 = 2 * mc_hat;
    //    chi2 += (beta - 1) * (beta - 1) / rel_var;
    //}
    if(mc_hat > 0.0)
    {
      chi2 = 2 * (mc_hat - data);
      if(data > 0.0)
        chi2 += 2 * data * std::log(data / mc_hat);

      chi2 += (beta - 1) * (beta - 1) / rel_var;
    }

    return (chi2 >= 0.0) ? chi2 : 0.0;
  }
};

#endif
