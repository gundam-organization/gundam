#ifndef __NuclearParameters_hh__
#define __NuclearParameters_hh__

#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>

#include <TFile.h>
#include <TGraph.h>

#include "AnaFitParameters.hh"
#include "NuclearDial.hh"

#include "json.hpp"
using json = nlohmann::json;

class NuclearParameters : public AnaFitParameters
{
    public:
        NuclearParameters(const std::string& name = "par_Nuclear");
        ~NuclearParameters();

        void InitEventMap(std::vector<AnaSample*>& sample, int mode);
        void InitParameters();
        void ReWeight(AnaEvent* event, const std::string& det, int nsample, int nevent, std::vector<double>& params);
        void AddDetector(const std::string& det, const std::string& config);

    private:
        std::vector<std::vector<std::vector<int>>> m_dial_evtmap;
        std::vector<std::string> v_detectors;
        std::map<std::string, std::vector<NuclearDial>> m_dials;
        std::map<std::string, int> m_offset;

        const std::string TAG = color::GREEN_STR + "[NuclearParameters]: " + color::RESET_STR;
};

#endif
