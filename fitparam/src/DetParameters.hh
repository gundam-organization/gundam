//////////////////////////////////////////////////////////
//
//  Det parameters
//
//
//
//  Created: Oct 2013
//  Modified:
//
//////////////////////////////////////////////////////////

#ifndef __DetParameters_hh__
#define __DetParameters_hh__

#include <TAxis.h>
#include "AnaFitParameters.hh"

struct DetBin
{
  double D1low, D1high;
  double D2low, D2high;
  int sample; //from 0 to 5 for the 6 regions
};

class DetParameters : public AnaFitParameters
{
public:
  DetParameters(const char *fname, 
                TVectorD* detweights, std::vector<AnaSample*> samples,
                const char *name = "par_det");
  ~DetParameters();
  
  void InitEventMap(std::vector<AnaSample*> &sample, int mode);
  void EventWeights(std::vector<AnaSample*> &sample, 
                    std::vector<double> &params);
  void ReWeight(AnaEvent *event, int nsample, int nevent,
                std::vector<double> &params);
  void SetBinning(const char *fname, std::vector<AnaSample*> &sample);

private:
  int GetBinIndex(double p, double D2, int sample_id); //binning function
                 std::vector<DetBin> m_bins;
};

#endif
