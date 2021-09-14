//
// Created by Nadrino on 11/06/2021.
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

#include "GenericToolbox.VariablesMonitor.h"
#include "GenericToolbox.CycleTimer.h"

#include "Propagator.h"

class FitterEngine {

public:
  FitterEngine();
  virtual ~FitterEngine();

  // Reset
  void reset();

  // Setters
  void setSaveDir(TDirectory *saveDir);
  void setConfig(const nlohmann::json &config_);


  // Init
  void initialize();

  // Getters
  bool isFitHasConverged() const;
  double getChi2Buffer() const;
  double getChi2StatBuffer() const;

  // Core
  void generateSamplePlots(const std::string& saveDir_ = "");
  void generateOneSigmaPlots(const std::string& saveDir_ = "");

  void fixGhostParameters();
  void scanParameters(int nbSteps_, const std::string& saveDir_ = "");
  void scanParameter(int iPar, int nbSteps_, const std::string& saveDir_ = "");
  void throwParameters(double gain_ = 1);

  void fit();
  void updateChi2Cache();
  double evalFit(const double* parArray_);

  void writePostFitData();


protected:
  void rescaleParametersStepSize();
  void initializeMinimizer(bool doReleaseFixed_ = false);



private:
  // Parameters
  TDirectory* _saveDir_{nullptr};
  nlohmann::json _config_{};

  // Internals
  bool _fitIsDone_{false};
  bool _fitUnderGoing_{false};
  bool _fitHasConverged_{false};

  int _nbFitCalls_{0};
  int _nbFitParameters_{0};
  int _nbParameters_{0};
  Propagator _propagator_{};
  std::shared_ptr<ROOT::Math::Minimizer> _minimizer_{nullptr};
  std::shared_ptr<ROOT::Math::Functor> _functor_{nullptr};
  TRandom3 _prng_;

  // Buffers
  double _chi2Buffer_{0};
  double _chi2StatBuffer_{0};
  double _chi2PullsBuffer_{0};
  double _chi2RegBuffer_{0};

  TTree* _chi2HistoryTree_{nullptr};
//  std::map<std::string, std::vector<double>> _chi2History_;

  GenericToolbox::VariablesMonitor _convergenceMonitor_;
  GenericToolbox::CycleTimer _evalFitAvgTimer_;

};


#endif //XSLLHFITTER_FITTERENGINE_H
