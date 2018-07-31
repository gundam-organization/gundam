//////////////////////////////////////////////////////////
//
//  Flux parameters
//
//
//
//  Created: Thu Jun 13 14:51:21 CEST 2013
//  Modified:
//
//////////////////////////////////////////////////////////

#ifndef __FluxParameters_hh__
#define __FluxParameters_hh__

#include "AnaFitParameters.hh"

class FluxParameters : public AnaFitParameters
{
public:
    FluxParameters(const std::string& name = "par_flux");
    ~FluxParameters();

    void InitParameters();
    void InitEventMap(std::vector<AnaSample*>& sample, int mode);
    void ReWeight(AnaEvent* event, const std::string& det, int nsample, int nevent,
                  std::vector<double>& params);

private:
    int GetBinIndex(const std::string& det, double enu);
    std::vector<double> m_enubins;
};

#endif
