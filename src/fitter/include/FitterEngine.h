//
// Created by Adrien BLANCHET on 11/06/2021.
//

#ifndef XSLLHFITTER_FITTERENGINE_H
#define XSLLHFITTER_FITTERENGINE_H

#include "string"
#include "vector"
#include "memory"

#include "TDirectory.h"
#include "Math/Functor.h"
#include "Math/Minimizer.h"
#include "json.hpp"

#include "Propagator.h"

class FitterEngine {

public:
  FitterEngine();
  virtual ~FitterEngine();

  // Reset
  void reset();

  // Setters
  void setSaveDir(TDirectory *saveDir);
  void setConfig(const json &config_);

  // Init
  void initialize();

  // Core
  void generateSamplePlots(const std::string& saveDir_ = "");
  void generateOneSigmaPlots(const std::string& saveDir_ = "");

  void scanParameters(int nbSteps_, const std::string& saveDir_ = "");
  void scanParameter(int iPar, int nbSteps_, const std::string& saveDir_ = "");

  void fit();
  void updateChi2Cache();
  double evalFit(const double* parArray_);

protected:
  void initializePropagator();
  void initializeMinimizer();

private:
  // Parameters
  TDirectory* _saveDir_{nullptr};
  nlohmann::json _config_;

  // Internals
  int _nbFitCalls_{0};
  Propagator _propagator_;
  int _nbFitParameters_;
  std::shared_ptr<ROOT::Math::Minimizer> _minimizer_{nullptr};
  std::shared_ptr<ROOT::Math::Functor> _functor_{nullptr};

  // Buffers
  double _chi2Buffer_;
  double _chi2StatBuffer_;
  double _chi2PullsBuffer_;
  double _chi2RegBuffer_;

};


#endif //XSLLHFITTER_FITTERENGINE_H
