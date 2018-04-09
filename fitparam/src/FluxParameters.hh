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
  FluxParameters(std::vector<double> &enubins,
		 const char *name = "par_flux", bool addIngrid=false);
  ~FluxParameters();
  
  void InitEventMap(std::vector<AnaSample*> &sample, int mode);
  void EventWeights(std::vector<AnaSample*> &sample, 
		                std::vector<double> &params);
  void ReWeight(AnaEvent *event, int nsample, int nevent,
		            std::vector<double> &params);
  void ReWeightIngrid(AnaEvent *event, int nsample, int nevent,
                std::vector<double> &params);

private:
  int GetBinIndex(double enu); //binning function
  std::vector<double> m_enubins;
  int numu_flux;
};

#endif
