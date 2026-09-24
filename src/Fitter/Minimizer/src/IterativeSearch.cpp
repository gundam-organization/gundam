//
// Created by Haowei Zheng on 19/08/2026.
//

#include "IterativeSearch.h"

#include "GundamGlobals.h"
#include "Logger.h"

#include "Math/Factory.h"

#include <algorithm>
#include <cmath>


void IterativeSearch::configureImpl(){
  LogDebugIf(GundamGlobals::isDebug()) << "Configuring IterativeSearch..." << std::endl;

  // read general parameters first
  this->MinimizerBase::configureImpl();

  _config_.defineFields({
    {FieldFlag::MANDATORY, "searchParameters"},
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

  for( auto& parConfig : _config_.loop("searchParameters") ){
    parConfig.defineFields({
      {FieldFlag::MANDATORY, "parameterSetName"},
      {FieldFlag::MANDATORY, "parameterName"},
      {"min"},
      {"max"},
      {"nSteps"},
      {"values"},
    });
    parConfig.checkConfiguration();

    _searchParameterList_.emplace_back();
    auto& searchPar = _searchParameterList_.back();

    parConfig.fillValue(searchPar.parameterSetName, "parameterSetName");
    parConfig.fillValue(searchPar.parameterName, "parameterName");

    if( parConfig.hasField("values") ){
      parConfig.fillValue(searchPar.values, "values");
    }
    else{
      LogThrowIf(not parConfig.hasField("min") or not parConfig.hasField("max") or not parConfig.hasField("nSteps"),
                 searchPar.parameterName << ": needs either \"values\", or all of \"min\", \"max\" and \"nSteps\".");

      double min{parConfig.fetchValue<double>("min")};
      double max{parConfig.fetchValue<double>("max")};
      int nSteps{parConfig.fetchValue<int>("nSteps")};

      LogThrowIf(nSteps < 2, searchPar.parameterName << ": nSteps is " << nSteps << ", needs at least 2.");
      LogThrowIf(max <= min, searchPar.parameterName << ": max (" << max << ") is not above min (" << min << ").");

      // both ends are on the grid
      searchPar.values.reserve(nSteps);
      for( int iStep = 0 ; iStep < nSteps ; iStep++ ){
        searchPar.values.emplace_back( min + (max - min) * iStep / (nSteps - 1) );
      }
    }

    LogThrowIf(searchPar.values.empty(), searchPar.parameterName << ": empty value list.");
  }

  LogThrowIf(_searchParameterList_.empty(), "No search parameter defined.");
}

void IterativeSearch::initializeImpl(){

  // find the search parameters and flag them as fixed, so the base class does not count them as free
  this->resolveSearchParameters();

  this->MinimizerBase::initializeImpl();

  // the search parameters must not be handed to the minimizer
  this->stripSearchParametersFromList();

  LogInfo << "Initializing IterativeSearch..." << std::endl;

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

  _functor_ = ROOT::Math::Functor(this, &IterativeSearch::evalFit, getMinimizerFitParameterPtr().size());
  _rootMinimizer_->SetFunction( _functor_ );
  _rootMinimizer_->SetStrategy(_strategy_);
  _rootMinimizer_->SetPrintLevel(_printLevel_);
  _rootMinimizer_->SetTolerance(_tolerance_);
  _rootMinimizer_->SetMaxIterations(_maxIterations_);
  _rootMinimizer_->SetMaxFunctionCalls(_maxFcnCalls_);

  // no Hesse: the search never needs the error matrix
  _rootMinimizer_->SetValidError(false);

  // snapshot of the prefit point in fit space
  _prefitValues_.clear();
  _prefitSteps_.clear();
  _prefitValues_.reserve( getMinimizerFitParameterPtr().size() );
  _prefitSteps_.reserve( getMinimizerFitParameterPtr().size() );
  for( auto* parPtr : getMinimizerFitParameterPtr() ){
    LogThrowIf(std::isnan(parPtr->getStepSize()), "No step size provided for: " << parPtr->getFullTitle());

    if( useNormalizedFitSpace() ){
      _prefitValues_.emplace_back( ParameterSet::toNormalizedParValue(parPtr->getParameterValue(), *parPtr) );
      _prefitSteps_.emplace_back( ParameterSet::toNormalizedParRange(parPtr->getStepSize(), *parPtr) );
    }
    else{
      _prefitValues_.emplace_back( parPtr->getParameterValue() );
      _prefitSteps_.emplace_back( parPtr->getStepSize() );
    }
  }

  this->resetMinimizer( _prefitValues_ );

  LogInfo << _searchParameterList_.size() << " search parameters:" << std::endl;
  for( auto& searchPar : _searchParameterList_ ){
    LogInfo << "  " << searchPar.parPtr->getFullTitle() << ": " << searchPar.values.size() << " values from "
            << searchPar.values.front() << " to " << searchPar.values.back() << std::endl;
  }
  LogInfo << "Minimizer parameters: " << getMinimizerFitParameterPtr().size()
          << " (search parameters removed)" << std::endl;

  LogInfo << "IterativeSearch initialized." << std::endl;
}

void IterativeSearch::resolveSearchParameters(){

  for( auto& searchPar : _searchParameterList_ ){
    auto* parSetPtr = getModelPropagator().getParametersManager().getFitParameterSetPtr( searchPar.parameterSetName );
    LogThrowIf(parSetPtr == nullptr, "Could not find parameter set: " << searchPar.parameterSetName);

    // the search moves a physical parameter, which an eigen decomposed set does not expose
    LogThrowIf(parSetPtr->isEnableEigenDecomp(),
               searchPar.parameterSetName << " uses eigen decomposition, so " << searchPar.parameterName << " cannot be searched.");

    searchPar.parPtr = parSetPtr->getParameterPtr( searchPar.parameterName );
    LogThrowIf(searchPar.parPtr == nullptr, "Could not find parameter: " << searchPar.parameterName << " in " << searchPar.parameterSetName);

    LogThrowIf(not searchPar.parPtr->isEnabled(), searchPar.parPtr->getFullTitle() << " is disabled.");

    // parameters with the penalty disabled never reach the minimizer list (see MinimizerBase::initializeImpl)
    LogThrowIf(searchPar.parPtr->isPenaltyDisabled(),
               searchPar.parPtr->getFullTitle() << " has its penalty disabled. A search parameter has to be a minimizer parameter.");

    for( double value : searchPar.values ){
      LogThrowIf(not searchPar.parPtr->isInDomain(value),
                 searchPar.parPtr->getFullTitle() << ": value " << value << " is outside " << searchPar.parPtr->getParameterLimits());
    }

    // the search drives this parameter, not the minimizer
    searchPar.parPtr->setIsFixed( true );
  }
}

void IterativeSearch::stripSearchParametersFromList(){

  auto& parList = getMinimizerFitParameterPtr();

  for( auto& searchPar : _searchParameterList_ ){
    auto it = std::find( parList.begin(), parList.end(), searchPar.parPtr );
    LogThrowIf(it == parList.end(), searchPar.parPtr->getFullTitle() << " is not a minimizer parameter.");
    parList.erase( it );
  }
}

void IterativeSearch::resetMinimizer(const std::vector<double>& startValues_){

  // drops the whole Minuit2 state
  _rootMinimizer_->Clear();

  for( std::size_t iFitPar = 0 ; iFitPar < getMinimizerFitParameterPtr().size() ; iFitPar++ ){
    auto& fitPar = *(getMinimizerFitParameterPtr()[iFitPar]);

    _rootMinimizer_->SetVariable(iFitPar, fitPar.getFullTitle(), startValues_[iFitPar], _prefitSteps_[iFitPar]);

    // strange ROOT parameter setting...
    auto& limits = fitPar.getParameterLimits();
    if( useNormalizedFitSpace() ){
      if     ( limits.hasBothBounds() ){
        _rootMinimizer_->SetVariableLimits(iFitPar,
                                           ParameterSet::toNormalizedParValue(limits.min, fitPar),
                                           ParameterSet::toNormalizedParValue(limits.max, fitPar));
      }
      else if( limits.hasLowerBound() ){
        _rootMinimizer_->SetVariableLowerLimit(iFitPar, ParameterSet::toNormalizedParValue(limits.min, fitPar));
      }
      else if( limits.hasUpperBound() ){
        _rootMinimizer_->SetVariableUpperLimit(iFitPar, ParameterSet::toNormalizedParValue(limits.max, fitPar));
      }
    }
    else{
      if     ( limits.hasBothBounds() ){ _rootMinimizer_->SetVariableLimits(iFitPar, limits.min, limits.max); }
      else if( limits.hasLowerBound() ){ _rootMinimizer_->SetVariableLowerLimit(iFitPar, limits.min); }
      else if( limits.hasUpperBound() ){ _rootMinimizer_->SetVariableUpperLimit(iFitPar, limits.max); }
    }

    // fixed by the user, not by the search (search parameters are not in this list)
    if( fitPar.isFixed() ){ _rootMinimizer_->FixVariable(iFitPar); }
  }
}

void IterativeSearch::minimize(){
  // calling the common routine
  this->MinimizerBase::minimize();

  LogAlert << "IterativeSearch::minimize() is not implemented yet. Parameters are left unchanged." << std::endl;

  setMinimizerStatus( 0 );
}
