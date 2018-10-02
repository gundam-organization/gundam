#ifndef __AnaFitParameters_hh__
#define __AnaFitParameters_hh__

#include <iostream>
#include <string>
#include <vector>

#include <TMatrixTSym.h>
using TMatrixDSym = TMatrixTSym<double>;

#include "AnaSample.hh"

// some error codes
const int PASSEVENT = -1;
const int BADBIN    = -2;

class AnaFitParameters
{
public:
    AnaFitParameters();
    virtual ~AnaFitParameters();

    // pure virtual functions
    // InitEventMap -- defines a mapping between events
    // and some unique id (e.g., bin number in True Enu)
    virtual void InitParameters()                                        = 0;
    virtual void InitEventMap(std::vector<AnaSample*>& sample, int mode) = 0;
    // ReWeights a single event based on m_evmap obtained in InitEventMap
    virtual void ReWeight(AnaEvent* event, const std::string& det, int nsample, int nevent,
                          std::vector<double>& params)
        = 0;

    double GetChi2(const std::vector<double>& params);
    void SetCovarianceMatrix(const TMatrixDSym& covmat);
    std::string GetName() { return m_name; }
    TMatrixDSym* GetCovMat() { return covariance; }
    bool HasCovMat() { return covariance != nullptr; }

    void GetParNames(std::vector<std::string>& vec) { vec = pars_name; }
    void GetParPriors(std::vector<double>& vec) { vec = pars_prior; }
    double GetParPrior(int i) { return pars_prior.at(i); }
    void GetParSteps(std::vector<double>& vec) { vec = pars_step; }
    void GetParLimits(std::vector<double>& vec1, std::vector<double>& vec2)
    {
        vec1 = pars_limlow;
        vec2 = pars_limhigh;
    }

    void SetParNames(std::vector<std::string>& vec) { pars_name = vec; }
    void SetParPriors(std::vector<double>& vec) { pars_prior = vec; }
    void SetParSteps(std::vector<double>& vec) { pars_step = vec; }
    void SetParLimits(std::vector<double>& vec1, std::vector<double>& vec2)
    {
        pars_limlow  = vec1;
        pars_limhigh = vec2;
    }

    int GetNpar() { return Npar; }

protected:
    bool CheckDims(const std::vector<double>& params);

    std::size_t Npar;
    std::string m_name;
    std::vector<std::string> pars_name;
    std::vector<double> pars_prior; // prior values of param
    std::vector<double> pars_throw; // vector with param throws
    std::vector<double> pars_step;
    std::vector<double> pars_limlow;
    std::vector<double> pars_limhigh;

    // map for events in each sample
    std::vector<std::vector<int>> m_evmap;
    bool m_rng_priors;

    TMatrixDSym* covariance;
    TMatrixDSym* covarianceI;
};

#endif
