#ifndef __AnaFitParameters_hh__
#define __AnaFitParameters_hh__

#include <algorithm>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include <TDecompLU.h>
#include <TMatrixTSym.h>
using TMatrixDSym = TMatrixTSym<double>;

#include "AnaSample.hh"
#include "ColorOutput.hh"
#include "EigenDecomp.hh"
#include "ToyThrower.hh"

// some error codes
const int PASSEVENT = -1;
const int BADBIN    = -2;

enum RegMethod
{
    kL1Reg,
    kL2Reg
};

class AnaFitParameters
{
public:
    AnaFitParameters();
    virtual ~AnaFitParameters();

    virtual void InitParameters()                                        = 0;
    virtual void InitEventMap(std::vector<AnaSample*>& sample, int mode) = 0;
    virtual void ReWeight(AnaEvent* event, const std::string& det, int nsample, int nevent,
                          std::vector<double>& params) = 0;

    virtual double CalcRegularisation(const std::vector<double>& params) const
    { return 0.0; }
    virtual double CalcRegularisation(const std::vector<double>& params, double strength,
                                      RegMethod flag) const
    { return 0.0; }

    virtual void Print();
    virtual void PrintParameterInfo(int iPar_);

    void ThrowPar(std::vector<double>& param, const int seed = 0) const;
    double GetChi2(const std::vector<double>& params) const;
    void SetCovarianceMatrix(const TMatrixDSym& covmat, bool decompose = false);
    std::string GetName() const { return m_name; }
    TMatrixDSym* GetCovMat() const { return covariance; }
    TMatrixDSym* GetOriginalCovMat() const { return original_cov; }
    TMatrixDSym GetOriginalCovMat(const TMatrixDSym& cov, unsigned int start_idx) const
    {
        return eigen_decomp->GetOriginalCovMat(cov, start_idx);
    }
    std::vector<double> GetOriginalParameters(const std::vector<double>& param) const
    {
        return eigen_decomp->GetOriginalParameters(param);
    }
    std::vector<double> GetOriginalParameters(const std::vector<double>& param,
                                              unsigned int start_idx) const
    {
        return eigen_decomp->GetOriginalParameters(param, start_idx);
    }

    void GetParNames(std::vector<std::string>& vec) const { vec = pars_name; }
    std::vector<std::string>& GetParNames() { return pars_name; }
    void GetParPriors(std::vector<double>& vec) const { vec = pars_prior; }
    void GetParOriginal(std::vector<double>& vec) const { vec = pars_original; }
    double GetParOriginal(int i) const { return pars_original.at(i); }
    double GetParPrior(int i) const { return pars_prior.at(i); }
    void GetParSteps(std::vector<double>& vec) const { vec = pars_step; }
    void GetParFixed(std::vector<bool>& vec) const { vec = pars_fixed; }
    void GetParLimits(std::vector<double>& vec1, std::vector<double>& vec2) const
    {
        vec1 = pars_limlow;
        vec2 = pars_limhigh;
    }

    virtual void SetParNames(std::vector<std::string>& vec);
    void SetParPriors(std::vector<double>& vec) { pars_prior = vec; }
    void SetParSteps(std::vector<double>& vec) { pars_step = vec; }
    void SetParLimits(std::vector<double>& vec1, std::vector<double>& vec2)
    {
        pars_limlow  = vec1;
        pars_limhigh = vec2;
    }

    int GetNpar() const { return Npar; }
    void SetInfoFrac(double frac) { m_info_frac = frac; }
    double GetInfoFrac() const { return m_info_frac; }
    void SetRegularisation(double strength, const std::string method);
    void SetRegularisation(double strength, RegMethod flag = kL2Reg);
    bool HasCovMat() const { return covariance != nullptr; }
    bool IsRegularised() const { return m_regularised; }
    bool IsDecomposed() const { return m_decompose; }
    bool DoThrow() const { return m_do_throw; }
    bool DoRNGstart() const { return m_rng_start; }
    void SetThrow(bool flag = true) { m_do_throw = flag; }
    void SetRNGstart(bool flag = true) { m_rng_start = flag; }
    void SetWeightCap(double cap, bool flag = true) { m_do_cap_weights = flag; m_weight_cap = cap; }

protected:
    bool CheckDims(const std::vector<double>& params) const;

    std::size_t Npar;
    std::string m_name;
    std::vector<std::string> pars_name;
    std::vector<double> pars_original;
    std::vector<double> pars_prior;
    std::vector<double> pars_step;
    std::vector<double> pars_limlow;
    std::vector<double> pars_limhigh;
    std::vector<bool> pars_fixed;

    std::vector<std::vector<int>> m_evmap;
    bool m_rng_start;
    bool m_do_throw;
    bool m_decompose;
    bool m_regularised;
    bool m_do_cap_weights;
    int m_nb_dropped_dof;
    double m_weight_cap;
    double m_info_frac;
    double m_regstrength;
    RegMethod m_regmethod;

    EigenDecomp* eigen_decomp;
    TMatrixDSym* covariance;
    TMatrixDSym* covarianceI;
    TMatrixDSym* original_cov;

    const std::string TAG = color::GREEN_STR + "[AnaFitParameters]: " + color::RESET_STR;
    const std::string ERR = color::RED_STR + color::BOLD_STR + "[ERROR]: " + color::RESET_STR;
    const std::string WAR = color::RED_STR + "[WARNING]: " + color::RESET_STR;
};

#endif
