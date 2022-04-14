//
// Created by Nadrino on 11/06/2021.
//

#ifndef GUNDAM_FITTERENGINE_H
#define GUNDAM_FITTERENGINE_H


#include "Propagator.h"
//#include "MinimizerInterface.h"
#include "ScanConfig.h"

#include "GenericToolbox.VariablesMonitor.h"
#include "GenericToolbox.CycleTimer.h"

#include "TDirectory.h"
#include "Math/Functor.h"
#include "Math/Minimizer.h"
#include "json.hpp"

#include "string"
#include "vector"
#include "memory"


class FitterEngine {

public:
  FitterEngine();
  virtual ~FitterEngine();

  // Reset
  void reset();

  // Setters
  void setSaveDir(TDirectory *saveDir);
  void setConfig(const nlohmann::json &config_);
  void setNbScanSteps(int nbScanSteps);
  void setEnablePostFitScan(bool enablePostFitScan);

  // Init
  void initialize();

  // Getters
  bool isFitHasConverged() const;
  double getChi2Buffer() const;
  double getChi2StatBuffer() const;
  const Propagator& getPropagator() const;
  Propagator& getPropagator();

  // Core
  void generateSamplePlots(const std::string& savePath_ = "");
  void generateOneSigmaPlots(const std::string& savePath_ = "");

  void fixGhostFitParameters();
  void scanParameters(int nbSteps_ = -1, const std::string& saveDir_ = "");
  void scanParameter(int iPar, int nbSteps_ = -1, const std::string& saveDir_ = "");

  void fit();
  void updateChi2Cache();
  double evalFit(const double* parArray_);

  void writePostFitData(TDirectory* saveDir_);

protected:
  void rescaleParametersStepSize();
  void initializeMinimizer(bool doReleaseFixed_ = false);



private:
  // Parameters
  TDirectory* _saveDir_{nullptr};
  nlohmann::json _config_{};
  nlohmann::json _minimizerConfig_{};
  int _nbScanSteps_{100};
  bool _enablePostFitScan_{false};
  bool _useNormalizedFitSpace_{false};

  // Internals
  bool _fitIsDone_{false};
  bool _enableFitMonitor_{false};
  bool _fitHasConverged_{false};
  bool _isBadCovMat_{false};

  int _nbFitCalls_{0};
  int _nbFitParameters_{0};
  int _nbParameters_{0};
  std::vector<FitParameter*> _minimizerFitParameterPtr_;
  std::vector<FitParameterSet*> _minimizerFitParameterSetPtr_;

  Propagator _propagator_{};
//  MinimizerInterface _minimizerInterface_{};

  std::string _minimizerType_{};
  std::string _minimizerAlgo_{};
  std::shared_ptr<ROOT::Math::Minimizer> _minimizer_{nullptr};
  std::shared_ptr<ROOT::Math::Functor> _functor_{nullptr};
  TRandom3 _prng_;

  ScanConfig _scanConfig_;

  // Buffers
  double _chi2Buffer_{0};
  double _chi2StatBuffer_{0};
  double _chi2PullsBuffer_{0};
  double _chi2RegBuffer_{0};
  double _parStepGain_{0.1};

  TTree* _chi2HistoryTree_{nullptr};
//  std::map<std::string, std::vector<double>> _chi2History_;

  GenericToolbox::VariablesMonitor _convergenceMonitor_;
  GenericToolbox::CycleTimer _evalFitAvgTimer_;
  GenericToolbox::CycleTimer _outEvalFitAvgTimer_;
  GenericToolbox::CycleTimer _itSpeed_;

  const std::map<int, std::string> minuitStatusCodeStr{
      { 0 , "status = 0    : OK" },
      { 1 , "status = 1    : Covariance was mad  epos defined"},
      { 2 , "status = 2    : Hesse is invalid"},
      { 3 , "status = 3    : Edm is above max"},
      { 4 , "status = 4    : Reached call limit"},
      { 5 , "status = 5    : Any other failure"},
      { -1, "status = -1   : Unknown error?"}
  };
  const std::map<int, std::string> hesseStatusCodeStr{
      { 0, "status = 0    : OK" },
      { 1, "status = 1    : Hesse failed"},
      { 2, "status = 2    : Matrix inversion failed"},
      { 3, "status = 3    : Matrix is not pos defined"},
      { -1, "status = -1    : Minimize wasn't called before"}
  };
  const std::map<int, std::string> minosStatusCodeStr{
      { 0, "status = 0    : last MINOS run was succesfull" },
      { 1, "status = 1    : Maximum number of function calls exceeded when running for lower error"},
      { 2, "status = 2    : maximum number of function calls exceeded when running for upper error"},
      { 3, "status = 3    : new minimum found when running for lower error"},
      { 4, "status = 4    : new minimum found when running for upper error"},
      { 5, "status = 5    : any other failure"},
      { -1, "status = -1   : Minos is not run"}
  };

  // 0 not calculated 1 approximated 2 made pos def , 3 accurate
  const std::map<int, std::string> covMatrixStatusCodeStr{
      { 0, "status = 0    : not calculated" },
      { 1, "status = 1    : approximated"},
      { 2, "status = 2    : made pos def"},
      { 3, "status = 3    : accurate"}
  };


  struct ScanData{
    std::string folder{};
    std::string title{};
    std::string yTitle{};
    std::vector<double> yPoints{};
    std::function<double()> evalY{};
  };
  std::vector<ScanData> scanDataDict;

};


#endif //GUNDAM_FITTERENGINE_H
