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
#include <string>
#include <vector>

#include <TMatrixDSym.h>
#include <TH1F.h>

//#include "AnaEvent.hh"
#include "AnaSample.hh"
#include "ThrowParms.hh"

//some error codes
#define PASSEVENT -1
#define BADBIN    -2

using namespace std;

class AnaFitParameters
{
    public:
        AnaFitParameters();
        virtual ~AnaFitParameters();

        // pure virtual functions
        // InitEventMap -- defines a mapping between events
        // and some unique id (e.g., bin number in True Enu)
        virtual void InitEventMap(std::vector<AnaSample*> &sample, int mode)=0;
        // EventWeights calculates weights for all the samples
        virtual void EventWeights(std::vector<AnaSample*> &sample,
                std::vector<double> &params) = 0;
        // ReWeights a single event based on m_evmap obtained
        // in InitEventMap
        virtual void ReWeight(AnaEvent *event, int nsample, int nevent,
                std::vector<double> &params) = 0;
        virtual void ReWeightIngrid(AnaEvent *event, int nsample, int nevent,
                std::vector<double> &params) = 0;

        virtual double GetChi2(std::vector<double> &params);

        virtual void SetCovarianceMatrix(TMatrixDSym *covmat);
        virtual void SetRegCovarianceMatrix(TMatrixDSym *covmat);
        virtual void InitThrows();
        virtual void DoThrow(std::vector<double> &pars, int mode);
        std::string GetName(){ return m_name; }
        TMatrixDSym* GetCovarMat(){ return covariance; }
        TMatrixDSym* GetCovarMat_reg(){ return regcovariance; }
        bool HasCovMat(){ return hasCovMat; }
        bool HasRegCovMat(){ return hasRegCovMat; }

        void GetParNames(std::vector<std::string> &vec)
        { vec = pars_name; }
        void GetParPriors(std::vector<double> &vec)
        { vec = pars_prior; }
        double GetParPrior(int i)
        { return pars_prior[i]; }
        void GetParSteps(std::vector<double> &vec)
        { vec = pars_step; }
        void GetParLimits(std::vector<double> &vec1,
                std::vector<double> &vec2)
        {
            vec1 = pars_limlow;
            vec2 = pars_limhigh;
        }

        void SetParNames(std::vector<std::string> &vec)
        { pars_name = vec; }
        void SetParPriors(std::vector<double> &vec)
        { pars_prior = vec; }
        void SetParSteps(std::vector<double> &vec)
        { pars_step = vec; }
        void SetParLimits(std::vector<double> &vec1,
                std::vector<double> &vec2)
        {
            pars_limlow = vec1;
            pars_limhigh = vec2;
        }

        void SetFluxHisto(TH1F* h_flux);

        void SetNpar(int n)
        {
            std::cout<<"WARNING: overriding number of parameters" << std::endl;
            Npar = n;
        }

    public:
        size_t Npar; //number of parameters

    protected:
        //check all dimensions
        void CheckDims(std::vector<double> &params);

        std::string m_name;
        std::vector<std::string> pars_name;
        std::vector<double> pars_prior; //prior values of param
        std::vector<double> pars_throw; //vector with param throws
        std::vector<double> pars_step;
        std::vector<double> pars_limlow;
        std::vector<double> pars_limhigh;

        //map for events in each sample
        std::vector< std::vector<int> > m_evmap;
        bool hasCovMat, checkDims, hasRegCovMat;
        //
        TMatrixDSym *covariance;  //cov matrix
        TMatrixDSym *covarianceI; //inverse of cov matrix
        TMatrixDSym *regcovariance;  //cov matrix
        TMatrixDSym *regcovarianceI; //inverse of cov matrix
        ThrowParms  *throwParms;

        TH1F* flux;
        TH1F* flux_mod;
};

#endif
