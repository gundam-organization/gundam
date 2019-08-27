#ifndef LIKELIHOODS_HH
#define LIKELIHOODS_HH

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

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
    double operator()(double mc, double w2, double data)
    {
        // Standard Poisson LLH.
        double chi2 = 0.0;
        if(mc > 0.0)
        {
            chi2 = 2 * (mc - data);
            if(data > 0.0)
                chi2 += 2 * data * TMath::Log(data / mc);
        }

        return (chi2 >= 0.0) ? chi2 : 0.0;
    }
};

class EffLLH : public CalcLLHFunc
{
public:
    double operator()(double mc, double w2, double data)
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

class BarlowLLH : public CalcLLHFunc
{
public:
    double operator()(double mc, double w2, double data)
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
