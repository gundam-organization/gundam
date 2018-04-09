//////////////////////////////////////////////////////////
//
//  NuclFSI modeling parameters
//
//
//
//  Created: Oct 2013
//  Modified:
//
//////////////////////////////////////////////////////////

#ifndef __NuclFSIParameters_hh__
#define __NuclFSIParameters_hh__

#include "AnaFitParameters.hh"
#include "XsecParameters.hh"
#include <TFile.h>
#include <TGraph.h>

struct NuclFSIBin
{
  double recoD1low, recoD1high;
  double trueD1low, trueD1high;
  double recoD2low, recoD2high;
  double trueD2low, trueD2high;
  SampleTypes topology;
  ReactionTypes reaction;
  std::vector<TGraph*> respfuncs;
};

class NuclFSIParameters : public AnaFitParameters
{
public:
  NuclFSIParameters(const char *name = "par_NuclFSI");
  ~NuclFSIParameters();
  
  void StoreResponseFunctions(std::vector<TFile*> respfuncs,
			      std::vector<std::pair <double,double> > v_D1edges, 
			      std::vector<std::pair <double,double> > v_D2edges);
  void InitEventMap(std::vector<AnaSample*> &sample, int mode); 
  void EventWeights(std::vector<AnaSample*> &sample, 
		    std::vector<double> &params);
  void ReWeight(AnaEvent *event, int nsample, int nevent,
		std::vector<double> &params);
  void ReWeightIngrid(AnaEvent *event, int nsample, int nevent,
                std::vector<double> &params);
private:
  int GetBinIndex(SampleTypes sampletype, ReactionTypes reactype, 
		  double recoP, double trueP, double recoD2, double trueD2);
  std::vector<NuclFSIBin> m_bins;
};

#endif
