//
// Created by Nadrino on 16/12/2021.
//

#ifndef GUNDAM_ROOT_MINIMIZER_H
#define GUNDAM_ROOT_MINIMIZER_H


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


class RootMinimizer : public MinimizerBase {

protected:
  void readConfigImpl() override;
  void initializeImpl() override;

public:
  // overrides
  void minimize() override;
  void calcErrors() override;
  void scanParameters( TDirectory* saveDir_ ) override;
  bool isErrorCalcEnabled() const override { return not disableCalcError(); }

  // c-tor
  explicit RootMinimizer(FitterEngine* owner_): MinimizerBase(owner_) {}

  // setters
  void setEnableSimplexBeforeMinimize(bool enableSimplexBeforeMinimize_){ _preFitWithSimplex_ = enableSimplexBeforeMinimize_; }

  // const getters
  [[nodiscard]] double getTargetEdm() const;
  [[nodiscard]] const std::unique_ptr<ROOT::Math::Minimizer> &getMinimizer() const{ return _rootMinimizer_; }

  // core

  void saveMinimizerSettings(TDirectory* saveDir_) const;

protected:
  void writePostFitData(TDirectory* saveDir_);
  void updateCacheToBestfitPoint();
  void saveGradientSteps();

private:

  // Parameters
  bool _preFitWithSimplex_{false};
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

  /// A functor that can be called by Minuit or anybody else.  This wraps
  /// evalFit.
  ROOT::Math::Functor _functor_{};
  std::unique_ptr<ROOT::Math::Minimizer> _rootMinimizer_{nullptr};

};
#endif //GUNDAM_ROOT_MINIMIZER_H
