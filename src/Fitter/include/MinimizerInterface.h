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

#include <memory>
#include <vector>

class FitterEngine;

class MinimizerInterface : public MinimizerBase {

public:
  explicit MinimizerInterface(FitterEngine* owner_);

  // setters
  void setEnableSimplexBeforeMinimize(bool enableSimplexBeforeMinimize_){ _enableSimplexBeforeMinimize_ = enableSimplexBeforeMinimize_; }

  // getters
  [[nodiscard]] std::string getMinimizerTypeName() const override { return "MinimizerInterface"; };
  [[nodiscard]] bool isFitHasConverged() const override;
  [[nodiscard]] double getTargetEdm() const;
  [[nodiscard]] const std::unique_ptr<ROOT::Math::Minimizer> &getMinimizer() const;

  void minimize() override;
  void calcErrors() override;
  void scanParameters(TDirectory* saveDir_) override;

  void saveMinimizerSettings(TDirectory* saveDir_) const {
    LogInfo << "Saving minimizer settings..." << std::endl;

    GenericToolbox::writeInTFile( saveDir_, TNamed("minimizerType", _minimizerType_.c_str()) );
    GenericToolbox::writeInTFile( saveDir_, TNamed("minimizerAlgo", _minimizerAlgo_.c_str()) );
    GenericToolbox::writeInTFile( saveDir_, TNamed("strategy", std::to_string(_strategy_).c_str()) );
    GenericToolbox::writeInTFile( saveDir_, TNamed("printLevel", std::to_string(_printLevel_).c_str()) );
    GenericToolbox::writeInTFile( saveDir_, TNamed("targetEDM", std::to_string(this->getTargetEdm()).c_str()) );
    GenericToolbox::writeInTFile( saveDir_, TNamed("maxIterations", std::to_string(_maxIterations_).c_str()) );
    GenericToolbox::writeInTFile( saveDir_, TNamed("maxFcnCalls", std::to_string(_maxFcnCalls_).c_str()) );
    GenericToolbox::writeInTFile( saveDir_, TNamed("tolerance", std::to_string(_tolerance_).c_str()) );
    GenericToolbox::writeInTFile( saveDir_, TNamed("stepSizeScaling", std::to_string(_stepSizeScaling_).c_str()) );
    GenericToolbox::writeInTFile( saveDir_, TNamed("useNormalizedFitSpace", std::to_string(getLikelihood().getUseNormalizedFitSpace()).c_str()) );

    if( _enableSimplexBeforeMinimize_ ){
      GenericToolbox::writeInTFile( saveDir_, TNamed("enableSimplexBeforeMinimize", std::to_string(_enableSimplexBeforeMinimize_).c_str()) );
      GenericToolbox::writeInTFile( saveDir_, TNamed("simplexMaxFcnCalls", std::to_string(_simplexMaxFcnCalls_).c_str()) );
      GenericToolbox::writeInTFile( saveDir_, TNamed("simplexToleranceLoose", std::to_string(_simplexToleranceLoose_).c_str()) );
      GenericToolbox::writeInTFile( saveDir_, TNamed("simplexStrategy", std::to_string(_simplexStrategy_).c_str()) );
    }

    if( this->isEnablePostFitErrorEval() ){
      GenericToolbox::writeInTFile( saveDir_, TNamed("enablePostFitErrorFit", std::to_string(this->isEnablePostFitErrorEval()).c_str()) );
      GenericToolbox::writeInTFile( saveDir_, TNamed("errorAlgo", _errorAlgo_.c_str()) );
    }
  }

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
  double _stepSizeScaling_{1};
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

};
#endif //GUNDAM_MINIMIZERINTERFACE_H
