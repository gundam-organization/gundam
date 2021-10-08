#ifndef XSLLHFITTER_DETPARAMETERS_HH
#define XSLLHFITTER_DETPARAMETERS_HH

#include <iomanip>
#include <iostream>
#include <string>

#include "AnaFitParameters.hh"
#include "FitStructs.hh"
#include "GeneralizedFitBin.h"

class DetParameters : public AnaFitParameters
{
public:
    explicit DetParameters(const std::string& name);
    ~DetParameters() override;

    void InitParameters() override;
    void InitEventMap(std::vector<AnaSample*>& samplesList, int mode) override;
    int GetBinIndex(int sampleIndex_, const std::vector<double>& eventVarList_) const;
    void ReWeight(AnaEvent* event, const std::string& det, int nsample, int nevent, std::vector<double>& params) override;
    bool SetBinning(AnaSample* sample_, std::vector<GeneralizedFitBin>& bins);
    void AddDetector(const std::string& det, std::vector<AnaSample*>& v_sample, bool match_bins);

private:
    std::map<int, std::vector<GeneralizedFitBin>> m_sample_bins;
    std::map<int, int> m_sample_offset;
    std::vector<int> v_samples;

    const std::string TAG = color::GREEN_STR + "[DetParameters]: " + color::RESET_STR;
};

#endif // XSLLHFITTER_DETPARAMETERS_HH
