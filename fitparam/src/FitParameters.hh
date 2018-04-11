//////////////////////////////////////////////////////////
//
//  CCQE cross-section parameters
//
//
//
//  Created: Thu Jun 13 11:46:03 CEST 2013
//  Modified:
//
//////////////////////////////////////////////////////////

#ifndef __FitParameters_hh__
#define __FitParameters_hh__

#include "AnaFitParameters.hh"

struct FitBin
{
    double D1low, D1high;
    double D2low, D2high;

    FitBin() : D1low(0), D1high(0), D2low(0), D2high(0) {}
    FitBin(const double& D1_L, const double& D1_H, 
           const double& D2_L, const double& D2_H)
          : D1low(D1_L), D1high(D1_H),
            D2low(D2_L), D2high(D2_H)
          {}
};

class FitParameters : public AnaFitParameters
{
    public:
        FitParameters(const std::string& file_name, const std::string& par_name, bool altPriorsTest = false);
        ~FitParameters();

        void InitEventMap(std::vector<AnaSample*> &sample, int mode);
        void EventWeights(std::vector<AnaSample*> &sample, std::vector<double> &params);
        void ReWeight(AnaEvent *event, int nsample, int nevent,
                std::vector<double> &params);
        void ReWeightIngrid(AnaEvent *event, int nsample, int nevent,
                std::vector<double> &params);
        void SetBinning(const std::string& file_name);

    private:
        //binnig function
        int GetBinIndex(double D1, double D2);
        int ccqe_recode;              //reaction code - defunct as it stands, signal is defined manually in FitParameters.cc
        std::vector<FitBin> m_bins; //binning
};

#endif
