//
// Created by Adrien Blanchet on 13/10/2023.
//

#include "ParametersManager.h"
#include "ConfigUtils.h"
#include "GundamGlobals.h"

#include "GenericToolbox.Utils.h"
#include "GenericToolbox.Json.h"
#include "Logger.h"

#include <sstream>


#ifndef DISABLE_USER_HEADER
LoggerInit([]{ Logger::setUserHeaderStr("[ParameterManager]"); });
#endif


// logger
void ParametersManager::muteLogger(){ Logger::setIsMuted( true ); }
void ParametersManager::unmuteLogger(){ Logger::setIsMuted( false ); }

// config
void ParametersManager::readConfigImpl(){

  _throwToyParametersWithGlobalCov_ = GenericToolbox::Json::fetchValue(_config_, "throwToyParametersWithGlobalCov", _throwToyParametersWithGlobalCov_);
  _reThrowParSetIfOutOfPhysical_ = GenericToolbox::Json::fetchValue(_config_, {{"reThrowParSetIfOutOfBounds"},{"reThrowParSetIfOutOfPhysical"}}, _reThrowParSetIfOutOfPhysical_);

  _parameterSetListConfig_ = GenericToolbox::Json::fetchValue(_config_, "parameterSetList", _parameterSetListConfig_);

  _parameterSetList_.clear(); // make sure there nothing in case readConfig is called more than once
  _parameterSetList_.reserve( _parameterSetListConfig_.size() );
  for( const auto& parameterSetConfig : _parameterSetListConfig_ ){
    _parameterSetList_.emplace_back();
    _parameterSetList_.back().readConfig( parameterSetConfig );

    // clear the parameter sets that have been disabled
    if( not _parameterSetList_.back().isEnabled() ){
      LogDebugIf(GundamGlobals::isDebugConfig()) << "Removing disabled parSet: " << _parameterSetList_.back().getName() << std::endl;
      _parameterSetList_.pop_back();
    }
  }

}
void ParametersManager::initializeImpl(){
  int nEnabledPars = 0;
  for( auto& parSet : _parameterSetList_ ){
    parSet.initialize();

    int nPars{0};
    for( auto& par : parSet.getParameterList() ){
      if( par.isEnabled() ){ nPars++; }
    }

    nEnabledPars += nPars;
    LogInfo << nPars << " enabled parameters in " << parSet.getName() << std::endl;
  }
  LogInfo << "Total number of parameters: " << nEnabledPars << std::endl;

  LogInfo << "Building global covariance matrix (" << nEnabledPars << "x" << nEnabledPars << ")" << std::endl;
  _globalCovarianceMatrix_ = std::make_shared<TMatrixD>(nEnabledPars, nEnabledPars );
  int parSetOffset = 0;
  for( auto& parSet : _parameterSetList_ ){
    if( parSet.getPriorCovarianceMatrix() != nullptr ){
      int iGlobalOffset{-1};
      bool hasZero{false};
      for(int iCov = 0 ; iCov < parSet.getPriorCovarianceMatrix()->GetNrows() ; iCov++ ){
        if( not parSet.getParameterList()[iCov].isEnabled() ){ continue; }
        iGlobalOffset++;
        _globalCovParList_.emplace_back( &parSet.getParameterList()[iCov] );
        int jGlobalOffset{-1};
        for(int jCov = 0 ; jCov < parSet.getPriorCovarianceMatrix()->GetNcols() ; jCov++ ){
          if( not parSet.getParameterList()[jCov].isEnabled() ){ continue; }
          jGlobalOffset++;
          (*_globalCovarianceMatrix_)[parSetOffset + iGlobalOffset][parSetOffset + jGlobalOffset] = (*parSet.getPriorCovarianceMatrix())[iCov][jCov];
        }
      }
      parSetOffset += (iGlobalOffset+1);
    }
    else{
      // diagonal
      for( auto& par : parSet.getParameterList() ){
        if( not par.isEnabled() ){ continue; }
        _globalCovParList_.emplace_back(&par);
        if( par.isFree() ){
          (*_globalCovarianceMatrix_)[parSetOffset][parSetOffset] = 0;
        }
        else{
          (*_globalCovarianceMatrix_)[parSetOffset][parSetOffset] = par.getStdDevValue() * par.getStdDevValue();
        }
        parSetOffset++;
      }
    }
  }

}

// const core
std::string ParametersManager::getParametersSummary(bool showEigen_ ) const{
  std::stringstream ss;
  for( auto &parSet: getParameterSetsList() ){
    if( not parSet.isEnabled() ){ continue; }
    if( not ss.str().empty() ) ss << std::endl;
    ss << parSet.getName();
    for( auto &par: parSet.getParameterList() ){
      if( not par.isEnabled() ){ continue; }
      ss << std::endl << "  " << par.getTitle() << ": " << par.getParameterValue();
    }
  }
  return ss.str();
}
JsonType ParametersManager::exportParameterInjectorConfig() const{
  JsonType out;

  std::vector<JsonType> parSetConfig;
  parSetConfig.reserve( _parameterSetList_.size() );
  for( auto& parSet : _parameterSetList_ ){
    if( not parSet.isEnabled() ){ continue; }
    parSetConfig.emplace_back( parSet.exportInjectorConfig() );
  }

  out["parameterSetList"] = parSetConfig;

  out = GenericToolbox::Json::readConfigJsonStr(
      // conversion: json -> str -> json obj (some broken JSON version)
      GenericToolbox::Json::toReadableString(
          out
      )
  );

  return out;
}
const ParameterSet* ParametersManager::getFitParameterSetPtr(const std::string& name_) const{
  for( auto& parSet : _parameterSetList_ ){
    if( parSet.getName() == name_ ) return &parSet;
  }
  std::vector<std::string> parSetNames{};
  parSetNames.reserve( _parameterSetList_.size() );
  for( auto& parSet : _parameterSetList_ ){ parSetNames.emplace_back(parSet.getName()); }
  LogThrow("Could not find fit parameter set named \"" << name_ << "\" among defined: " << GenericToolbox::toString(parSetNames));
  return nullptr;
}

// core
void ParametersManager::throwParameters(){

  if( _throwToyParametersWithGlobalCov_ ){
    LogInfo << "Throwing parameter using global covariance matrix..." << std::endl;
    this->throwParametersFromGlobalCovariance(false);
  }
  else{
    LogInfo << "Throwing parameter using parSet covariance matrices..." << std::endl;
    this->throwParametersFromParSetCovariance();
  }

}
void ParametersManager::throwParametersFromParSetCovariance(){
  LogInfo << "Throwing parameter using each parameter sets..." << std::endl;
  for( auto& parSet : _parameterSetList_ ){
    if( not parSet.isEnabled() ) continue;

    LogContinueIf( not parSet.isEnabledThrowToyParameters(), "Toy throw is disabled for " << parSet.getName() );

    if( parSet.getPriorCovarianceMatrix() != nullptr ){
      LogWarning << parSet.getName() << ": throwing correlated parameters..." << std::endl;
      LogScopeIndent;
      parSet.throwParameters(_reThrowParSetIfOutOfPhysical_);
    } // throw?
    else{
      LogAlert << "No correlation matrix defined for " << parSet.getName() << ". NOT THROWING. (dev: could throw only with sigmas?)" << std::endl;
    }
  } // parSet
}
void ParametersManager::initializeStrippedGlobalCov(){
  LogInfo << "Creating stripped global covariance matrix..." << std::endl;
  LogThrowIf( _globalCovarianceMatrix_ == nullptr, "Global covariance matrix not set." );

  _strippedParameterList_.clear();
  for( int iGlobPar = 0 ; iGlobPar < _globalCovarianceMatrix_->GetNrows() ; iGlobPar++ ){
    if( _globalCovParList_[iGlobPar]->isFixed() ){ continue; }
    if( _globalCovParList_[iGlobPar]->isFree() and (*_globalCovarianceMatrix_)[iGlobPar][iGlobPar] == 0 ){ continue; }
    _strippedParameterList_.emplace_back( _globalCovParList_[iGlobPar] );
  }

  int nStripped{int(_strippedParameterList_.size())};
  _strippedCovarianceMatrix_ = std::make_shared<TMatrixD>(nStripped, nStripped);

  for( int iStrippedPar = 0 ; iStrippedPar < nStripped ; iStrippedPar++ ){
    int iGlobPar{GenericToolbox::findElementIndex(_strippedParameterList_[iStrippedPar], _globalCovParList_)};
    for( int jStrippedPar = 0 ; jStrippedPar < nStripped ; jStrippedPar++ ){
      int jGlobPar{GenericToolbox::findElementIndex(_strippedParameterList_[jStrippedPar], _globalCovParList_)};
      (*_strippedCovarianceMatrix_)[iStrippedPar][jStrippedPar] = (*_globalCovarianceMatrix_)[iGlobPar][jGlobPar];
    }
  }
}
void ParametersManager::throwParametersFromGlobalCovariance(bool quietVerbose_){

  if( _strippedCovarianceMatrix_ == nullptr ){
    initializeStrippedGlobalCov();
  }

  bool isLoggerAlreadyMuted{Logger::isMuted()};
  GenericToolbox::ScopedGuard g{
      [&](){ if(quietVerbose_ and not isLoggerAlreadyMuted) Logger::setIsMuted(true); },
      [&](){ if(quietVerbose_ and not isLoggerAlreadyMuted) Logger::setIsMuted(false); }
  };

  if(quietVerbose_){
    Logger::setIsMuted(quietVerbose_);
  }

  if( _choleskyMatrix_ == nullptr ){
    LogInfo << "Generating global cholesky matrix" << std::endl;
    _choleskyMatrix_ = std::shared_ptr<TMatrixD>(
        GenericToolbox::getCholeskyMatrix(_strippedCovarianceMatrix_.get())
    );
  }

  int throwNb{0};
  while( true ) {
    throwNb++;
    bool rethrow{false};
    auto throws = GenericToolbox::throwCorrelatedParameters(_choleskyMatrix_.get());
    for( int iPar = 0 ; iPar < _choleskyMatrix_->GetNrows() ; iPar++ ){
      auto* parPtr = _strippedParameterList_[iPar];
      parPtr->setThrowValue(parPtr->getPriorValue() + throws[iPar]);
      if ( not std::isnan(parPtr->getMinValue()) and parPtr->getThrowValue() < parPtr->getMinValue()) {
        LogAlert << "Thrown value lower than min bound -> " << parPtr->getThrowValue() << " < min(" << parPtr->getMinValue() << ") " << parPtr->getFullTitle() << std::endl;
        rethrow = true;
        break;
      }
      if ( not std::isnan(parPtr->getMaxValue()) and parPtr->getThrowValue() > parPtr->getMaxValue()) {
        LogAlert << "Thrown value greater than max bound -> " << parPtr->getThrowValue() << " > max(" << parPtr->getMaxValue() << ") " << parPtr->getFullTitle() << std::endl;
        rethrow = true;
        break;
      }
      parPtr->setParameterValue( parPtr->getThrowValue() );
      if( not _reThrowParSetIfOutOfPhysical_ ) continue;
      if( not std::isnan(parPtr->getMinPhysical()) and parPtr->getParameterValue() < parPtr->getMinPhysical() ){
        rethrow = true;
        LogAlert << "thrown value lower than physical min bound -> "
                 << parPtr->getSummary(true) << std::endl;
        break;
      }
      if( not std::isnan(parPtr->getMaxPhysical()) and parPtr->getParameterValue() > parPtr->getMaxPhysical() ){
        rethrow = true;
        LogAlert << "thrown value higher than physical max bound -> "
                 << parPtr->getSummary(true) << std::endl;
        break;
      }
    }

    // Making sure eigen decomposed parameters get the conversion done
    for( auto& parSet : _parameterSetList_ ) {
      if (rethrow) break;  // short circuit if we are already rethrowing.
      if( not parSet.isEnabled() ) continue;
      if( not parSet.isEnableEigenDecomp() ) continue;
      parSet.propagateOriginalToEigen();
      // also check the bounds of real parameter space
      for( auto& par : parSet.getEigenParameterList() ){
        if( not par.isEnabled() ) continue;
        if( par.isValueWithinBounds() ) continue;
        // re-do the throwing
        rethrow = true;
        break;
      }
    }

    if( rethrow ) {
      LogThrowIf( throwNb > 100000, "Too many throw attempts")
      // wrap back to the while loop
      LogWarning << "Rethrowing after attempt #" << throwNb << std::endl;
      continue;
    }

    for( auto& parSet : _parameterSetList_ ){
      if( not parSet.isEnabled() ){ continue; }
      LogInfo << parSet.getName() << ":" << std::endl;
      for( auto& par : parSet.getParameterList() ){
        if( not par.isEnabled() ){ continue; }
        LogScopeIndent;
        par.setThrowValue( par.getParameterValue() );
        LogInfo << "Thrown par " << par.getFullTitle() << ": " << par.getPriorValue();
        LogInfo << " becomes " << par.getParameterValue() << std::endl;
      }
      if( not parSet.isEnableEigenDecomp() ) continue;
      LogInfo << "Translated to eigen space:" << std::endl;
      for( auto& eigenPar : parSet.getEigenParameterList() ){
        if( not eigenPar.isEnabled() ){ continue; }
        LogScopeIndent;
        eigenPar.setThrowValue( eigenPar.getParameterValue() );
        LogInfo << "Eigen par " << eigenPar.getFullTitle() << ": " << eigenPar.getPriorValue();
        LogInfo << " becomes " << eigenPar.getParameterValue() << std::endl;
      }
    }

    // reached this point: all parameters are within bounds
    break;
  }
}

void ParametersManager::moveParametersToPrior(){
  for( auto& parSet : _parameterSetList_ ){
    if( not parSet.isEnabled() ){ continue; }
    parSet.moveParametersToPrior();
  }
}
void ParametersManager::convertEigenToOrig(){
  for( auto& parSet : _parameterSetList_ ){
    if( not parSet.isEnabled() ){ continue; }
    if( parSet.isEnableEigenDecomp() ){ parSet.propagateEigenToOriginal(); }
  }
}
void ParametersManager::injectParameterValues(const JsonType &config_) {
  LogWarning << "Injecting parameters..." << std::endl;

  if( not GenericToolbox::Json::doKeyExist(config_, "parameterSetList") ){
    LogError << "Bad parameter injector config: missing \"parameterSetList\" entry" << std::endl;
    LogError << GenericToolbox::Json::toReadableString( config_ ) << std::endl;
    return;
  }

  for( auto& entryParSet : GenericToolbox::Json::fetchValue<JsonType>( config_, "parameterSetList" ) ){
    auto parSetName = GenericToolbox::Json::fetchValue<std::string>(entryParSet, "name");
    LogInfo << "Reading injection parameters for parSet: " << parSetName << std::endl;

    auto* selectedParSet = this->getFitParameterSetPtr(parSetName );
    LogThrowIf( selectedParSet == nullptr, "Could not find parSet: " << parSetName );

    selectedParSet->injectParameterValues(entryParSet);
  }
}
ParameterSet* ParametersManager::getFitParameterSetPtr(const std::string& name_){
  return const_cast<ParameterSet*>(const_cast<const ParametersManager*>(this)->getFitParameterSetPtr(name_));
}
bool ParametersManager::hasValidParameterSets() const {
  for (const ParameterSet& parSet: getParameterSetsList()) {
    if (not parSet.isEnabled()) continue;
    if (not parSet.isValid()) return false;
  }
  return true;
}
void ParametersManager::printConfiguration() const {

  LogInfo << GET_VAR_NAME_VALUE(_throwToyParametersWithGlobalCov_) << std::endl;
  LogInfo << GET_VAR_NAME_VALUE(_reThrowParSetIfOutOfPhysical_) << std::endl;

  LogInfo << _parameterSetList_.size() << " parameter sets defined." << std::endl;
  for( auto& parSet : _parameterSetList_ ){ parSet.printConfiguration(); }

}

void ParametersManager::setParameterValidity(const std::string& v) {
  for (ParameterSet& parSet: getParameterSetsList()) {
    parSet.setValidity(v);
  }
}
