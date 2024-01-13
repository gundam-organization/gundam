//
// Created by Adrien BLANCHET on 16/12/2021.
//

#ifndef GUNDAM_ROOTFACTORYINTERFACE_H
#define GUNDAM_ROOTFACTORYINTERFACE_H


#include "ParameterSet.h"
#include "MinimizerBase.h"
#include "JsonBaseClass.h"

#include "GenericToolbox.Utils.h"
#include "GenericToolbox.Time.h"

#include "Math/Minimizer.h"
#include "Math/Functor.h"
#include "TDirectory.h"
#include "nlohmann/json.hpp"

#include <memory>
#include <vector>


class RootFactoryInterface : public MinimizerBase {

protected:
  void readConfigImpl() override;
  void initializeImpl() override;

public:
  explicit RootFactoryInterface(FitterEngine* owner_): MinimizerBase(owner_) {}

  // setters
  void setEnableSimplexBeforeMinimize(bool enableSimplexBeforeMinimize_){ _preFitWithSimplex_ = enableSimplexBeforeMinimize_; }

  // overridden getters
  [[nodiscard]] bool isFitHasConverged() const override{ return _fitHasConverged_; }
  [[nodiscard]] std::string getMinimizerTypeName() const override { return "MinimizerInterface"; };

  // getters
  [[nodiscard]] double getTargetEdm() const;
  [[nodiscard]] const std::unique_ptr<ROOT::Math::Minimizer> &getMinimizer() const{ return _minimizer_; }

  // core overrides
  void minimize() override;

  void calcErrors() override;

  void scanParameters(TDirectory* saveDir_);

  // misc
  void saveMinimizerSettings(TDirectory* saveDir_) const;

protected:
  void findMinimumLikelihood();

  void writePostFitData(TDirectory* saveDir_);
  void updateCacheToBestfitPoint();

private:

  // Parameters
  bool _preFitWithSimplex_{false};
  // bool _enablePostFitErrorEval_{true};
  bool _restoreStepSizeBeforeHesse_{false};
  bool _generatedPostFitParBreakdown_{false};
  bool _generatedPostFitEigenBreakdown_{false};

  int _strategy_{1};
  int _printLevel_{2};
  int _simplexStrategy_{1};

  double _tolerance_{1E-4};
  double _stepSizeScaling_{1};
  double _simplexToleranceLoose_{1000.};

  unsigned int _maxIterations_{500};
  unsigned int _maxFcnCalls_{1000000000};
  unsigned int _simplexMaxFcnCalls_{1000};

  std::string _minimizerType_{"Minuit2"};
  std::string _minimizerAlgo_{"Migrad"};
  std::string _errorAlgo_{"Hesse"};

  // internals
  bool _fitHasConverged_{false};

  std::unique_ptr<ROOT::Math::Minimizer> _minimizer_{nullptr};

};
#endif //GUNDAM_ROOTFACTORYINTERFACE_H
