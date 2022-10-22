//
// Created by Adrien BLANCHET on 16/12/2021.
//

#include "Minimizer.h"
#include "FitterEngine.h"
#include "JsonUtils.h"

#include "Logger.h"

#include "Math/Factory.h"


LoggerInit([]{
  Logger::setUserHeaderStr("[MinimizerInterface]");
});


Minimizer::Minimizer() {}
Minimizer::~Minimizer() {}

void Minimizer::reset(){
  _isInitialized_ = false;
  _minimizer_.reset();
}

void Minimizer::setConfig(const nlohmann::json &config) {
  _config_ = config;
  JsonUtils::forwardConfig(_config_);
}
void Minimizer::setFitterEnginePtr(void *fitterEnginePtr) {
  _fitterEnginePtr_ = fitterEnginePtr;
}

void Minimizer::initialize(){
  LogThrowIf(_isInitialized_, "Already initialized.")
  LogThrowIf(_fitterEnginePtr_== nullptr, "_fitterEnginePtr_ not set.")

  _parSetListPtr_ = &( static_cast<FitterEngine*>(_fitterEnginePtr_)->getPropagator().getParameterSetsList() );

  _minimizerType_ = JsonUtils::fetchValue(_config_, "minimizer", "Minuit2");
  _minimizerAlgo_ = JsonUtils::fetchValue(_config_, "algorithm", "");

  _useNormalizedFitSpace_ = JsonUtils::fetchValue(_config_, "useNormalizedFitSpace", true);

  _minimizer_ = std::shared_ptr<ROOT::Math::Minimizer>(
      ROOT::Math::Factory::CreateMinimizer(_minimizerType_, _minimizerAlgo_)
  );
  LogThrowIf(_minimizer_ == nullptr, "Could not create minimizer: " << _minimizerType_ << "/" << _minimizerAlgo_)
  if( _minimizerAlgo_.empty() ) _minimizerAlgo_ = _minimizer_->Options().MinimizerAlgorithm();

  this->fillNbFitParameters();

  _llhEvalFunction_ = ROOT::Math::Functor(_fitterEnginePtr_, &FitterEngine::evalFit, _nbFitParameters_);

  _minimizer_->SetFunction(_llhEvalFunction_);
  _minimizer_->SetStrategy(JsonUtils::fetchValue(_config_, "strategy", 1));
  _minimizer_->SetPrintLevel(JsonUtils::fetchValue(_config_, "print_level", 2));
  _minimizer_->SetTolerance(JsonUtils::fetchValue(_config_, "tolerance", 1E-4));
  _minimizer_->SetMaxIterations(JsonUtils::fetchValue(_config_, "max_iter", (unsigned int)(500) ));
  _minimizer_->SetMaxFunctionCalls(JsonUtils::fetchValue(_config_, "max_fcn", (unsigned int)(1E9)));

  this->defineFitParameters();

  _isInitialized_ = true;
}

bool Minimizer::isMinimizeSucceeded() const {
  return _minimizeSucceeded_;
}
bool Minimizer::isIsInitialized() const {
  return _isInitialized_;
}

void Minimizer::minimize() {
  LogWarning << std::endl << GenericToolbox::addUpDownBars("Calling minimize...") << std::endl;
  LogInfo << "Number of defined parameters: " << _minimizer_->NDim() << std::endl
          << "Number of free parameters   : " << _minimizer_->NFree() << std::endl
          << "Number of fixed parameters  : " << _minimizer_->NDim() - _minimizer_->NFree()
          << std::endl;

  _minimizeSucceeded_ = _minimizer_->Minimize();
}

void Minimizer::fillNbFitParameters(){
  LogThrowIf(_fitterEnginePtr_== nullptr, "_fitterEnginePtr_ not set.")

  _nbFitParameters_ = 0;
  for( auto& parSet : *_parSetListPtr_ ){
    if( parSet.isUseEigenDecompInFit() ){
      for( int iEigen = 0 ; iEigen < parSet.getNbEnabledEigenParameters() ; iEigen++ ){
        if( not parSet.isEigenParFixed(iEigen) ) _nbFitParameters_++;
      }
    }
    else{
      for( auto& par : parSet.getParameterList() ){
        if( par.isEnabled() and not par.isFixed() ) _nbFitParameters_++;
      }
    }
  }
  LogInfo << "Found " << _nbFitParameters_ << " fit parameters." << std::endl;
}
void Minimizer::defineFitParameters(){

  for( int iFitPar = 0 ; iFitPar < _nbFitParameters_ ; iFitPar++ ){

  }

  int iPar = -1;
  auto* parSetListPtr = &( static_cast<FitterEngine*>(_fitterEnginePtr_)->getPropagator().getParameterSetsList() );
  for( auto& parSet : *parSetListPtr ){

    if( not parSet.isUseEigenDecompInFit() ){
      for( auto& par : parSet.getParameterList()  ){
        iPar++;
        if(not _useNormalizedFitSpace_){
          _minimizer_->SetVariable( iPar,parSet.getName() + "/" + par.getTitle(), par.getParameterValue(),par.getStepSize() );
          if(par.getMinValue() == par.getMinValue()){ _minimizer_->SetVariableLowerLimit(iPar, par.getMinValue()); }
          if(par.getMaxValue() == par.getMaxValue()){ _minimizer_->SetVariableUpperLimit(iPar, par.getMaxValue()); }
          // Changing the boundaries, change the value/step size?
          _minimizer_->SetVariableValue(iPar, par.getParameterValue());
          _minimizer_->SetVariableStepSize(iPar, par.getStepSize());
        }
        else{
          _minimizer_->SetVariable( iPar,parSet.getName() + "/" + par.getTitle(),
                                    FitParameterSet::toNormalizedParValue(par.getParameterValue(), par),
                                    FitParameterSet::toNormalizedParRange(par.getStepSize(), par)
          );
          if(par.getMinValue() == par.getMinValue()){ _minimizer_->SetVariableLowerLimit(iPar, FitParameterSet::toNormalizedParValue(par.getMinValue(), par)); }
          if(par.getMaxValue() == par.getMaxValue()){ _minimizer_->SetVariableUpperLimit(iPar, FitParameterSet::toNormalizedParValue(par.getMaxValue(), par)); }
          // Changing the boundaries, change the value/step size?
          _minimizer_->SetVariableValue(iPar, FitParameterSet::toNormalizedParValue(par.getParameterValue(), par));
          _minimizer_->SetVariableStepSize(iPar, FitParameterSet::toNormalizedParRange(par.getStepSize(), par));
        }


        if( not JsonUtils::fetchValue(parSet.getConfig(), "releaseFixedParametersOnHesse", true) ){
          if( not par.isEnabled() or par.isFixed() ) _minimizer_->FixVariable(iPar);
        }
      } // par
    }
    else{
      for( int iEigen = 0 ; iEigen < parSet.getNbEnabledEigenParameters() ; iEigen++ ){
        iPar++;
        if(not _useNormalizedFitSpace_){
          _minimizer_->SetVariable( iPar,parSet.getName() + "/eigen_#" + std::to_string(iEigen),
                                    parSet.getEigenParameterValue(iEigen),
                                    parSet.getEigenParStepSize(iEigen)
          );
        }
        else{
          _minimizer_->SetVariable( iPar,parSet.getName() + "/eigen_#" + std::to_string(iEigen),
                                    parSet.toNormalizedEigenParValue(parSet.getEigenParameterValue(iEigen),iEigen),
                                    parSet.toNormalizedEigenParRange(parSet.getEigenParStepSize(iEigen), iEigen)
          );
        }

        if( not JsonUtils::fetchValue(parSet.getConfig(), "releaseFixedParametersOnHesse", true) ){
          if( parSet.isEigenParFixed(iEigen) ) {
            _minimizer_->FixVariable(iPar);
          }
        }
      }
    }

  } // parSet

}




