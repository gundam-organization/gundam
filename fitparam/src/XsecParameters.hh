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

#include "json.hpp"
using json = nlohmann::json;

class XsecParameters : public AnaFitParameters
{
    public:
        XsecParameters(const std::string& name = "par_xsec");
        ~XsecParameters();

        void InitEventMap(std::vector<AnaSample*>& sample, int mode);
        void InitParameters();
        void ReWeight(AnaEvent* event, const std::string& det, int nsample, int nevent, std::vector<double>& params);
        void AddDetector(const std::string& det, const std::string& config);

    private:
        std::vector<std::vector<std::vector<int>>> m_dial_evtmap;
        std::vector<std::string> v_detectors;
        std::map<std::string, std::vector<XsecDial>> m_dials;
        std::map<std::string, int> m_offset;
};

#endif
