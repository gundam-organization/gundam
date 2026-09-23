//
// Created by Haowei Zheng on 19/08/2026.
//

#include "GridSearch.h"

#include "GundamGlobals.h"
#include "Logger.h"

#include "Math/Factory.h"

#include <cmath>


void GridSearch::configureImpl(){
  LogDebugIf(GundamGlobals::isDebug()) << "Configuring GridSearch..." << std::endl;

  // read general parameters first
  this->MinimizerBase::configureImpl();

  _config_.defineFields({
    {"minimizer"},
    {"algorithm"},
    {"strategy"},
    {"print_level"},
    {"tolerance"},
    {"maxIterations", {"max_iter"}},
    {"maxFcnCalls", {"max_fcn"}},
  });
  _config_.checkConfiguration();

  _config_.fillValue(_minimizerType_, "minimizer");
  _config_.fillValue(_minimizerAlgo_, "algorithm");

  _config_.fillValue(_strategy_, "strategy");
  _config_.fillValue(_printLevel_, "print_level");
  _config_.fillValue(_tolerance_, "tolerance");
  _config_.fillValue(_maxIterations_, "maxIterations");
  _config_.fillValue(_maxFcnCalls_, "maxFcnCalls");
}

void GridSearch::initializeImpl(){
  this->MinimizerBase::initializeImpl();

  LogInfo << "Initializing GridSearch..." << std::endl;

  LogInfo << "Tolerance is set to: " << _tolerance_ << std::endl;

  LogInfo << "Defining minimizer as: " << _minimizerType_ << "/" << _minimizerAlgo_ << std::endl;
  _rootMinimizer_ = std::unique_ptr<ROOT::Math::Minimizer>(
      ROOT::Math::Factory::CreateMinimizer(_minimizerType_, _minimizerAlgo_)
  );
  LogThrowIf(_rootMinimizer_ == nullptr, "Could not create minimizer: " << _minimizerType_ << "/" << _minimizerAlgo_);

  if( _minimizerAlgo_.empty() ){
    _minimizerAlgo_ = _rootMinimizer_->Options().MinimizerAlgorithm();
    LogWarning << "Using default minimizer algo: " << _minimizerAlgo_ << std::endl;
  }

  _functor_ = ROOT::Math::Functor(this, &GridSearch::evalFit, getMinimizerFitParameterPtr().size());
  _rootMinimizer_->SetFunction( _functor_ );
  _rootMinimizer_->SetStrategy(_strategy_);
  _rootMinimizer_->SetPrintLevel(_printLevel_);
  _rootMinimizer_->SetTolerance(_tolerance_);
  _rootMinimizer_->SetMaxIterations(_maxIterations_);
  _rootMinimizer_->SetMaxFunctionCalls(_maxFcnCalls_);

  // no Hesse: the grid search never needs the error matrix
  _rootMinimizer_->SetValidError(false);

  for( std::size_t iFitPar = 0 ; iFitPar < getMinimizerFitParameterPtr().size() ; iFitPar++ ){
    auto& fitPar = *(getMinimizerFitParameterPtr()[iFitPar]);

    LogThrowIf(std::isnan(fitPar.getStepSize()), "No step size provided for: " << fitPar.getFullTitle());

    if( not useNormalizedFitSpace() ){
      _rootMinimizer_->SetVariable(iFitPar, fitPar.getFullTitle(), fitPar.getParameterValue(), fitPar.getStepSize());

      // strange ROOT parameter setting...
      if( fitPar.getParameterLimits().hasBothBounds() ){
        _rootMinimizer_->SetVariableLimits(iFitPar, fitPar.getParameterLimits().min, fitPar.getParameterLimits().max);
      }
      else if( fitPar.getParameterLimits().hasLowerBound() ){
        _rootMinimizer_->SetVariableLowerLimit(iFitPar, fitPar.getParameterLimits().min);
      }
      else if( fitPar.getParameterLimits().hasUpperBound() ){
        _rootMinimizer_->SetVariableUpperLimit(iFitPar, fitPar.getParameterLimits().max);
      }
    }
    else{
      _rootMinimizer_->SetVariable(iFitPar, fitPar.getFullTitle(),
                                   ParameterSet::toNormalizedParValue(fitPar.getParameterValue(), fitPar),
                                   ParameterSet::toNormalizedParRange(fitPar.getStepSize(), fitPar)
      );
      // strange ROOT parameter setting...
      if( fitPar.getParameterLimits().hasBothBounds() ) {
        _rootMinimizer_->SetVariableLimits(
          iFitPar,
          ParameterSet::toNormalizedParValue(fitPar.getParameterLimits().min, fitPar),
          ParameterSet::toNormalizedParValue(fitPar.getParameterLimits().max, fitPar));
      }
      else if( fitPar.getParameterLimits().hasLowerBound() ){
        _rootMinimizer_->SetVariableLowerLimit(iFitPar, ParameterSet::toNormalizedParValue(fitPar.getParameterLimits().min, fitPar));
      }
      else if( fitPar.getParameterLimits().hasUpperBound() ){
        _rootMinimizer_->SetVariableUpperLimit(iFitPar, ParameterSet::toNormalizedParValue(fitPar.getParameterLimits().max, fitPar));
      }
    }
  }

  LogInfo << "GridSearch initialized." << std::endl;
}

void GridSearch::minimize(){
  // calling the common routine
  this->MinimizerBase::minimize();

  LogAlert << "GridSearch::minimize() is not implemented yet. Parameters are left unchanged." << std::endl;

  setMinimizerStatus( 0 );
}
