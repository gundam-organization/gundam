//////////////////////////////////////////////////////////
//
//  Xsec modeling parameters
//
//
//
//  Created: Oct 2013
//  Modified:
//
//////////////////////////////////////////////////////////

#ifndef __XsecParameters_hh__
#define __XsecParameters_hh__

#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>

#include <TFile.h>
#include <TGraph.h>

#include "AnaFitParameters.hh"
#include "XsecDial.hh"
// Hard coding of sample and raction types

// Sample types - should match whats in ccqefit.cc
enum SampleTypes
{
    muTPC       = 0,
    mupTPC      = 1,
    mupFGD      = 2,
    muFGDpTPC   = 3,
    muFGD       = 4,
    crCC1pi     = 5,
    crDIS       = 6,
    muTPCnp     = 7,
    muFGDpTPCnp = 8,
    muFGDnp     = 9
};

// Reaction and backgrounds
// the indices should match with trees
// case below is mectopology from HL2 v1r15 cc0pi
enum ReactionTypes
{
    cc0pi0p   = 0,
    ReCC0pi1p = 1,
    ReCC0pinp = 2,
    ReCC1pi   = 3,
    ReCCOther = 4,
    ReBkg     = 5,
    ReNULL    = 6,
    OutFGD    = 7
};

struct XsecBin
{
    double recoD1low, recoD1high;
    double trueD1low, trueD1high;
    double recoD2low, recoD2high;
    double trueD2low, trueD2high;
    SampleTypes topology;
    ReactionTypes reaction;
    std::vector<TGraph*> respfuncs;
};

class XsecParameters : public AnaFitParameters
{
    public:
        XsecParameters(const std::string& name = "par_xsec");
        ~XsecParameters();

        void StoreResponseFunctions(std::vector<TFile*> respfuncs,
                std::vector<std::pair<double, double>> v_D1edges,
                std::vector<std::pair<double, double>> v_D2edges);
        void InitEventMap(std::vector<AnaSample*>& sample, int mode);
        void EventWeights(std::vector<AnaSample*>& sample, std::vector<double>& params);
        void ReWeight(AnaEvent* event, int nsample, int nevent, std::vector<double>& params);
        void AddDetector(const std::string& det, const std::string& config);

    private:
        int GetBinIndex(SampleTypes sampletype, ReactionTypes reactype, double recoP, double trueP,
                double recoD2, double trueD2);
        std::vector<XsecBin> m_bins;
        std::map<std::string, std::vector<XsecDial>> m_dials;
};

#endif
