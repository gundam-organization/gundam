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

#include <fstream>
#include <iomanip>
#include <sstream>

#include <TRandom3.h>

#include "AnaFitParameters.hh"
#include "FitStructs.hh"

class FitParameters : public AnaFitParameters
{
    public:
        FitParameters(const std::string& par_name, bool random_priors = false);
        ~FitParameters();

        void InitParameters();
        void InitEventMap(std::vector<AnaSample*> &sample, int mode);
        void ReWeight(AnaEvent *event, const std::string& det, int nsample, int nevent,
                      std::vector<double> &params);
        bool SetBinning(const std::string& file_name, std::vector<xsllh::FitBin>& bins);
        void AddDetector(const std::string& det, const std::string& f_binning);

    private:
        int GetBinIndex(const std::string& det, double D1, double D2) const;
        std::map<std::string, std::vector<xsllh::FitBin> > m_fit_bins;
        std::vector<std::string> v_detectors;
};

#endif
