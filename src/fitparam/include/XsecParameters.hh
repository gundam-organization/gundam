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
#include "../../utils/include/XsecDial.hh"

#include "json.hpp"
using json = nlohmann::json;

class XsecParameters : public AnaFitParameters
{
public:
    explicit XsecParameters(const std::string& name = "par_xsec");
    ~XsecParameters() override;

    void Print() override;
    void PrintParameterInfo(int iPar_) override;
    void PrintParameterInfo(int iDetector_, int iDial_);

    void InitEventMap(std::vector<AnaSample*>& samplesList_, int mode) override;
    void InitParameters() override;
    void ReWeight(AnaEvent* event, const std::string& detectorName, int nsample, int nevent, std::vector<double>& params) override;
    void AddDetector(const std::string& detectorName_, const std::string& configFilePath_);

    void SetEnableZeroWeightFenceGate(bool enableZeroWeightFenceGate_);

    int GetDetectorIndex(const std::string& detectorName_);
    std::vector<XsecDial> GetDetectorDials(const std::string& detectorName_);


private:

    std::vector<std::vector<std::vector<int>>> _eventDialSplineIndexList_; // [iSample][iEvent][iDial] = Spline index
    std::vector<std::vector<XsecDial>> _dialsList_;
    std::vector<std::string> v_detectors;
    std::vector<int> v_offsets;

    typedef struct{
        bool isBeingEdited;
        double fitParameterValue;
        double cachedWeight;
    } DialCache;

    std::vector<DialCache> _dialCacheWeight_;

    bool _enableZeroWeightFenceGate_;

};

#endif
