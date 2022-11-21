//
// Created by Adrien BLANCHET on 16/12/2021.
//

#ifndef GUNDAM_MINIMIZERINTERFACE_H
#define GUNDAM_MINIMIZERINTERFACE_H


#include "FitParameterSet.h"
#include "JsonBaseClass.h"

#include "GenericToolbox.VariablesMonitor.h"
#include "GenericToolbox.CycleTimer.h"

#include "Math/Minimizer.h"
#include "Math/Functor.h"
#include "TDirectory.h"
#include "nlohmann/json.hpp"

#include "memory"
#include "vector"


class FitterEngine;

class MinimizerInterface : public JsonBaseClass {

public:
  explicit MinimizerInterface(FitterEngine* owner_);

  void setOwner(FitterEngine* owner_);
  void setEnablePostFitErrorEval(bool enablePostFitErrorEval_);
  void setMonitorRefreshRateInMs(int monitorRefreshRateInMs_);

  bool isFitHasConverged() const;
  bool isEnablePostFitErrorEval() const;
  GenericToolbox::VariablesMonitor &getConvergenceMonitor();
  std::vector<FitParameter *> &getMinimizerFitParameterPtr();
  const std::unique_ptr<ROOT::Math::Minimizer> &getMinimizer() const;

  void minimize();
  void calcErrors();

protected:
  void readConfigImpl() override;
  void initializeImpl() override;

  double evalFit(const double* parArray_);
  void writePostFitData(TDirectory* saveDir_);

  void enableFitMonitor(){ _enableFitMonitor_ = true; }
  void disableFitMonitor(){ _enableFitMonitor_ = false; }

private:
  // Parameters
  bool _useNormalizedFitSpace_{true};
  bool _enableSimplexBeforeMinimize_{false};
  bool _enablePostFitErrorEval_{true};
  bool _restoreStepSizeBeforeHesse_{false};
  bool _generatedPostFitParBreakdown_{false};
  bool _generatedPostFitEigenBreakdown_{false};
  bool _showParametersOnFitMonitor_{false};
  int _strategy_{1};
  int _printLevel_{2};
  int _simplexStrategy_{1};
  int _monitorRefreshRateInMs_{5000};
  int _monitorBashModeRefreshRateInS_{30};
  int _maxNbParametersPerLineOnMonitor_{15};
  double _tolerance_{1E-4};
  double _simplexToleranceLoose_{1000.};
  unsigned int _maxIterations_{500};
  unsigned int _maxFcnCalls_{1000000000};
  unsigned int _simplexMaxFcnCalls_{1000};
  std::string _minimizerType_{"Minuit2"};
  std::string _minimizerAlgo_{};
  std::string _errorAlgo_{"Hesse"};

  // internals
  bool _fitHasConverged_{false};
  bool _isBadCovMat_{false};
  bool _enableFitMonitor_{false};
  int _nbFitParameters_{-1};
  int _nbFitBins_{0};
  int _nbFreePars_{0};
  int _nbFitCalls_{0};
  FitterEngine* _owner_{nullptr};
  std::unique_ptr<ROOT::Math::Minimizer> _minimizer_{nullptr};
  std::unique_ptr<ROOT::Math::Functor> _functor_{nullptr};
  std::vector<FitParameter*> _minimizerFitParameterPtr_{};
  std::unique_ptr<TTree> _chi2HistoryTree_{nullptr};

  // monitors
  GenericToolbox::VariablesMonitor _convergenceMonitor_;
  GenericToolbox::CycleTimer _evalFitAvgTimer_;
  GenericToolbox::CycleTimer _outEvalFitAvgTimer_;
  GenericToolbox::CycleTimer _itSpeed_;

  // dict
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

};


#endif //GUNDAM_MINIMIZERINTERFACE_H
