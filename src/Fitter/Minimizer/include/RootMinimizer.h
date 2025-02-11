//
// Created by Nadrino on 16/12/2021.
//

#ifndef GUNDAM_ROOT_MINIMIZER_H
#define GUNDAM_ROOT_MINIMIZER_H


#include "ParameterSet.h"
#include "MinimizerBase.h"

#include "GenericToolbox.Utils.h"
#include "GenericToolbox.Time.h"

#include "Math/Minimizer.h"
#include "Math/Functor.h"
#include "TDirectory.h"

#include <memory>
#include <vector>


class RootMinimizer : public MinimizerBase {

protected:
  void configureImpl() override;
  void initializeImpl() override;

public:
  // overrides
  void minimize() override;
  void calcErrors() override;
  void scanParameters( TDirectory* saveDir_ ) override;
  double evalFit( const double* parArray_ ) override;
  [[nodiscard]] bool isErrorCalcEnabled() const override { return not disableCalcError(); }

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

  // Dump the Math::Minimizer table of parameter settings.  This is mostly
  // useful for debugging.
  void dumpFitParameterSettings();

  // Dump the ROOT::Minuit2::MnUserParameterState.  This is mostly useful
  // for debugging.
  void dumpMinuit2State();

  // Parameters
  bool _preFitWithSimplex_{false};
  bool _restoreStepSizeBeforeHesse_{false};
  bool _generatedPostFitParBreakdown_{false};
  bool _generatedPostFitEigenBreakdown_{false};

  int _strategy_{1};
  int _printLevel_{2};
  int _simplexStrategy_{1};

  double _tolerance_{1E-4};
  double _tolerancePerDegreeOfFreedom_{std::nan("unset")};
  double _stepSizeScaling_{1};
  double _simplexToleranceLoose_{1000.};

  unsigned int _maxIterations_{500};
  unsigned int _maxFcnCalls_{1000000000};
  unsigned int _simplexMaxFcnCalls_{1000};

  std::string _minimizerType_{"Minuit2"};
  std::string _minimizerAlgo_{"Migrad"};
  std::string _errorAlgo_{"Hesse"};

  // internals
  bool _minimizeDone_{false};
  bool _fitHasConverged_{false};

  /// A functor that can be called by Minuit or anybody else.  This wraps
  /// evalFit.
  ROOT::Math::Functor _functor_{};
  std::unique_ptr<ROOT::Math::Minimizer> _rootMinimizer_{nullptr};

  struct GradientDescentMonitor{
    bool isEnabled{false};

    struct ValueDefinition{
      std::string name{};
      std::function<double(const RootMinimizer* this_)> getValueFct{};

      ValueDefinition(const std::string& name_, const std::function<double(const RootMinimizer* this_)>& getValueFct_){
        name = name_;
        getValueFct = getValueFct_;
      }
    };
    std::vector<ValueDefinition> valueDefinitionList{};

    struct StepPoint{
      JsonType parState;
      double fitCallNb{0};
      std::vector<double> valueMonitorList{}; // .size() = valueDefinitionList.size()
    };
    std::vector<StepPoint> stepPointList{};

    void addStep(const RootMinimizer* this_){
      stepPointList.emplace_back();
      stepPointList.back().valueMonitorList.reserve( valueDefinitionList.size() );
      fillLastStep(this_);
    }
    void fillLastStep(const RootMinimizer* this_){
      stepPointList.back().parState = this_->getModelPropagator().getParametersManager().exportParameterInjectorConfig();
      stepPointList.back().fitCallNb = this_->getMonitor().nbEvalLikelihoodCalls;
      for( auto& valueDefinition : valueDefinitionList ){
        stepPointList.back().valueMonitorList.emplace_back( valueDefinition.getValueFct(this_) );
      }
    }

    [[nodiscard]] int getValueIndex(const std::string& name_) const {
      int idx = GenericToolbox::findElementIndex(name_, valueDefinitionList, [](const ValueDefinition& elm){ return elm.name; });
      LogThrowIf(idx == -1, "Could not find element " << name_);
      return idx;
    }
    [[nodiscard]] double getLastStepValue(const std::string& name_) const {
      LogThrowIf(stepPointList.empty());
      return stepPointList.back().valueMonitorList[getValueIndex(name_)];
    }
    [[nodiscard]] double getLastStepDeltaValue(const std::string& name_) const {
      if( stepPointList.size() < 2 ){ return 0; }
      auto idx = getValueIndex(name_);
      return stepPointList[stepPointList.size()-2].valueMonitorList[idx] - stepPointList.back().valueMonitorList[idx];
    }

  };
  GradientDescentMonitor gradientDescentMonitor{};

};
#endif //GUNDAM_ROOT_MINIMIZER_H
