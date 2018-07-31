//////////////////////////////////////////////////////////
//
//  Fit parameters -- abstract base class for
//  parameters in the cross-section fit
//
//
//  Created: Thu Jun  6 14:02:44 CEST 2013
//  Modified:
//
//////////////////////////////////////////////////////////
#ifndef __AnaFitParameters_hh__
#define __AnaFitParameters_hh__

#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <TH1F.h>
#include <TMatrixDSym.h>

#include "AnaSample.hh"
#include "ThrowParms.hh"

using namespace std;

// some error codes
const int PASSEVENT = -1;
const int BADBIN    = -2;

struct PairCompare
{
    template<typename T, typename U>
    bool operator()(const std::pair<T, U>& lhs, const std::pair<T, U>& rhs)
    {
        return lhs.second < rhs.second;
    }
};

class AnaFitParameters
{
public:
    AnaFitParameters();
    virtual ~AnaFitParameters();

    // pure virtual functions
    // InitEventMap -- defines a mapping between events
    // and some unique id (e.g., bin number in True Enu)
    virtual void InitEventMap(std::vector<AnaSample*>& sample, int mode) = 0;
    // EventWeights calculates weights for all the samples
    virtual void EventWeights(std::vector<AnaSample*>& sample, std::vector<double>& params);
    // ReWeights a single event based on m_evmap obtained
    // in InitEventMap
    virtual void ReWeight(AnaEvent* event, const std::string& det, int nsample, int nevent,
                          std::vector<double>& params)
        = 0;

    virtual void InitParameters() = 0;
    virtual void AddDetector(const std::string& det, const std::vector<double>& bins, int offset);
    virtual double GetChi2(const std::vector<double>& params);

    virtual void SetCovarianceMatrix(const TMatrixDSym& covmat);
    virtual void SetRegCovarianceMatrix(TMatrixDSym* covmat);
    virtual void InitThrows();
    virtual void DoThrow(std::vector<double>& pars, int mode);
    std::string GetName() { return m_name; }
    TMatrixDSym* GetCovarMat() { return covariance; }
    TMatrixDSym* GetCovarMat_reg() { return regcovariance; }
    bool HasCovMat() { return hasCovMat; }
    bool HasRegCovMat() { return hasRegCovMat; }
    void SetFluxHisto(TH1F* h_flux);

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
    void SetNpar(int n)
    {
        std::cout << "[WARNING]: Overriding number of parameters." << std::endl;
        Npar = n;
    }

protected:
    bool CheckDims(const std::vector<double>& params);

    std::size_t Npar;
    std::string m_name;
    std::map<std::string, int> m_det_offset;
    std::map<std::string, std::vector<double>> m_det_bins;
    std::vector<std::string> pars_name;
    std::vector<double> pars_prior; // prior values of param
    std::vector<double> pars_throw; // vector with param throws
    std::vector<double> pars_step;
    std::vector<double> pars_limlow;
    std::vector<double> pars_limhigh;

    // map for events in each sample
    std::vector<std::vector<int>> m_evmap;
    bool m_rng_priors;
    bool hasCovMat, hasRegCovMat;

    TMatrixDSym* covariance; // cov matrix
    TMatrixDSym* covarianceI; // inverse of cov matrix
    TMatrixDSym* regcovariance; // cov matrix
    TMatrixDSym* regcovarianceI; // inverse of cov matrix
    ThrowParms* throwParms;

    TH1F* flux;
    TH1F* flux_mod;
};

#endif
