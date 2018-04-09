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

#ifndef __FluxParameters_norm_hh__
#define __FluxParameters_norm_hh__

#include "AnaFitParameters.hh"

class FluxParameters_norm : public AnaFitParameters
{
public:
  FluxParameters_norm(const char *name = "par_flux");
  ~FluxParameters_norm();
  
  //void DoThrow(std::vector<double> &pars);
  void InitEventMap(std::vector<AnaSample*> &sample, int mode);
  void EventWeights(std::vector<AnaSample*> &sample, 
                    std::vector<double> &params);
  void ReWeight(AnaEvent *event, int nsample, int nevent,
                std::vector<double> &params);
  void ReWeightIngrid(AnaEvent *event, int nsample, int nevent,
                std::vector<double> &params);
};

#endif
