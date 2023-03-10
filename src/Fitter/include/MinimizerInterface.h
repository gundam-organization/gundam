//
// Created by Adrien BLANCHET on 16/12/2021.
//

#ifndef GUNDAM_MINIMIZERINTERFACE_H
#define GUNDAM_MINIMIZERINTERFACE_H


#include "FitParameterSet.h"
#include "MinimizerBase.h"
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

class MinimizerInterface : public MinimizerBase {

public:
  explicit MinimizerInterface(FitterEngine* owner_);

  [[nodiscard]] bool isFitHasConverged() const override;
  [[nodiscard]] const std::unique_ptr<ROOT::Math::Minimizer> &getMinimizer() const;

  void minimize() override;
  void calcErrors() override;
  void scanParameters(TDirectory* saveDir_) override;

  double getTargetEdm() const;

protected:
  void readConfigImpl() override;
  void initializeImpl() override;

  void writePostFitData(TDirectory* saveDir_);
  void updateCacheToBestfitPoint();

private:

  // Parameters
  bool _enableSimplexBeforeMinimize_{false};
  // bool _enablePostFitErrorEval_{true};
  bool _restoreStepSizeBeforeHesse_{false};
  bool _generatedPostFitParBreakdown_{false};
  bool _generatedPostFitEigenBreakdown_{false};
  int _strategy_{1};
  int _printLevel_{2};
  int _simplexStrategy_{1};
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

  std::unique_ptr<ROOT::Math::Minimizer> _minimizer_{nullptr};

  // dict
  const std::map<int, std::string> minuitStatusCodeStr{
      { 0 , "status = 0    : OK" },
      { 1 , "status = 1    : Covariance was mad! Thus made pos defined"},
      { 2 , "status = 2    : Hesse is invalid"},
      { 3 , "status = 3    : Edm is above max"},
      { 4 , "status = 4    : Reached call limit"},
      { 5 , "status = 5    : Any other failure"},
      { -1, "status = -1   : We don't even know what happened"}
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
