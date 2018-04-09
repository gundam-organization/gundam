//////////////////////////////////////////////////////////
//
//  FSI modeling parameters
//
//
//
//  Created: Oct 2013
//  Modified:
//
//////////////////////////////////////////////////////////

#ifndef __FSIParameters_hh__
#define __FSIParameters_hh__

#include "AnaFitParameters.hh"
#include "XsecParameters.hh"
#include <TFile.h>
#include <TGraph.h>

struct FSIBin
{
  double recoD1low, recoD1high;
  double trueD1low, trueD1high;
  double recoD2low, recoD2high;
  double trueD2low, trueD2high;
  SampleTypes topology;
  ReactionTypes reaction;
  std::vector<TGraph*> respfuncs;
};

class FSIParameters : public AnaFitParameters
{
public:
  FSIParameters(const char *name = "par_PionFSI");
  ~FSIParameters();
  
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
  std::vector<FSIBin> m_bins;
};

#endif
