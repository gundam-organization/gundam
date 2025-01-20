//
// Created by Nadrino on 13/10/2023.
//

#include "ParametersManager.h"
#include "ConfigUtils.h"

#include "GenericToolbox.Utils.h"
#include "GenericToolbox.Json.h"
#include "Logger.h"
#include "GundamUtils.h"

#include <sstream>


LoggerInit([]{
  Logger::setUserHeaderStr("[ParameterManager]");
});


// logger
void ParametersManager::muteLogger(){ Logger::setIsMuted( true ); }
void ParametersManager::unmuteLogger(){ Logger::setIsMuted( false ); }

// config
void ParametersManager::readConfigImpl(){

  _parameterSetListConfig_ = GenericToolbox::Json::fetchValue(_config_, "parameterSetList", _parameterSetListConfig_);

  _reThrowParSetIfOutOfBounds_ = GenericToolbox::Json::fetchValue(_config_, "reThrowParSetIfOutOfBounds", _reThrowParSetIfOutOfBounds_);
  _throwToyParametersWithGlobalCov_ = GenericToolbox::Json::fetchValue(_config_, "throwToyParametersWithGlobalCov", _throwToyParametersWithGlobalCov_);

  LogInfo << "Reading parameter configuration..." << std::endl;
  _parameterSetList_.clear(); // make sure there nothing in case readConfig is called more than once
  _parameterSetList_.reserve( _parameterSetListConfig_.size() );
  for( const auto& parameterSetConfig : _parameterSetListConfig_ ){
    _parameterSetList_.emplace_back();
    _parameterSetList_.back().readConfig( parameterSetConfig );
    LogInfo << _parameterSetList_.back().getSummary() << std::endl;
  }
  LogInfo << _parameterSetList_.size() << " parameter sets defined." << std::endl;

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
      parSet.throwParameters(_reThrowParSetIfOutOfBounds_);
    } // throw?
    else{
      LogAlert << "No correlation matrix defined for " << parSet.getName() << ". NOT THROWING. (dev: could throw only with sigmas?)" << std::endl;
    }
  } // parSet
}
void ParametersManager::throwParametersFromGlobalCovariance(bool quietVerbose_){

  if( _strippedCovarianceMatrix_ == nullptr ){
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

  bool keepThrowing{true};
  int throwNb{0};

  while( keepThrowing ){
    throwNb++;
    bool rethrow{false};
    auto throws = GundamUtils::throwCorrelatedParameters(_choleskyMatrix_.get());
    for( int iPar = 0 ; iPar < _choleskyMatrix_->GetNrows() ; iPar++ ){
      auto* parPtr = _strippedParameterList_[iPar];
      parPtr->setParameterValue( parPtr->getPriorValue() + throws[iPar] );
      if( _reThrowParSetIfOutOfBounds_ ){
        if      ( not std::isnan(parPtr->getMinValue()) and parPtr->getParameterValue() < parPtr->getMinValue() ){
          rethrow = true;
          LogAlert << GenericToolbox::ColorCodes::redLightText << "thrown value lower than min bound -> " << GenericToolbox::ColorCodes::resetColor
                   << parPtr->getSummary(true) << std::endl;
        }
        else if( not std::isnan(parPtr->getMaxValue()) and parPtr->getParameterValue() > parPtr->getMaxValue() ){
          rethrow = true;
          LogAlert << GenericToolbox::ColorCodes::redLightText <<"thrown value higher than max bound -> " << GenericToolbox::ColorCodes::resetColor
                   << parPtr->getSummary(true) << std::endl;
        }
      }
    }

    // Making sure eigen decomposed parameters get the conversion done
    for( auto& parSet : _parameterSetList_ ){
      if( not parSet.isEnabled() ) continue;
      if( parSet.isEnableEigenDecomp() ){
        parSet.propagateOriginalToEigen();

        // also check the bounds of real parameter space
        if( _reThrowParSetIfOutOfBounds_ ){
          for( auto& par : parSet.getEigenParameterList() ){
            if( not par.isEnabled() ) continue;
            if( not par.isValueWithinBounds() ){
              // re-do the throwing
              rethrow = true;
              break;
            }
          }
        }
      }
    }


    if( rethrow ){
      // wrap back to the while loop
      LogWarning << "Re-throwing attempt #" << throwNb << std::endl;
      continue;
    }
    else{
      for( auto& parSet : _parameterSetList_ ){
        LogInfo << parSet.getName() << ":" << std::endl;
        for( auto& par : parSet.getParameterList() ){
          LogScopeIndent;
          if( ParameterSet::isValidCorrelatedParameter(par) ){
            par.setThrowValue( par.getParameterValue() );
            LogInfo << "Thrown par " << par.getFullTitle() << ": " << par.getPriorValue();
            LogInfo << " → " << par.getParameterValue() << std::endl;
          }
        }
        if( parSet.isEnableEigenDecomp() ){
          LogInfo << "Translated to eigen space:" << std::endl;
          for( auto& eigenPar : parSet.getEigenParameterList() ){
            LogScopeIndent;
            eigenPar.setThrowValue( eigenPar.getParameterValue() );
            LogInfo << "Eigen par " << eigenPar.getFullTitle() << ": " << eigenPar.getPriorValue();
            LogInfo << " → " << eigenPar.getParameterValue() << std::endl;
          }
        }
      }

    }

    // reached this point: all parameters are within bounds
    keepThrowing = false;
  }
}

void ParametersManager::throwParametersFromGlobalCovariance(std::vector<double> &weightsChiSquare){
    throwParametersFromGlobalCovariance(weightsChiSquare,0,0,0);
}// end of function

void ParametersManager::throwParametersFromGlobalCovariance(std::vector<double> &weightsChiSquare,
                                                            double pedestalEntity, double pedestalLeftEdge, double pedestalRightEdge
                                                            ){

    // check that weightsChiSquare is an empty vector
    LogThrowIf( not weightsChiSquare.empty(), "ERROR: argument weightsChiSquare is not empty" );

    if( _strippedCovarianceMatrix_ == nullptr ){
        LogInfo << "Creating stripped global covariance matrix..." << std::endl;
        LogThrowIf( _globalCovarianceMatrix_ == nullptr, "Global covariance matrix not set." );
        int nStripped{0};
        for( int iDiag = 0 ; iDiag < _globalCovarianceMatrix_->GetNrows() ; iDiag++ ){
            if( (*_globalCovarianceMatrix_)[iDiag][iDiag] != 0 ){ nStripped++; }
        }

        LogInfo << "Stripped global covariance matrix is " << nStripped << "x" << nStripped << std::endl;
        _strippedCovarianceMatrix_ = std::make_shared<TMatrixD>(nStripped, nStripped);
        int iStrippedBin{-1};
        for( int iBin = 0 ; iBin < _globalCovarianceMatrix_->GetNrows() ; iBin++ ){
            if( (*_globalCovarianceMatrix_)[iBin][iBin] == 0 ){ continue; }
            iStrippedBin++;
            int jStrippedBin{-1};
            for( int jBin = 0 ; jBin < _globalCovarianceMatrix_->GetNrows() ; jBin++ ){
                if( (*_globalCovarianceMatrix_)[jBin][jBin] == 0 ){ continue; }
                jStrippedBin++;
                (*_strippedCovarianceMatrix_)[iStrippedBin][jStrippedBin] = (*_globalCovarianceMatrix_)[iBin][jBin];
            }
        }

        _strippedParameterList_.reserve( nStripped );
        for( auto& parSet : _parameterSetList_ ){
            if( not parSet.isEnabled() ) continue;
            for( auto& par : parSet.getParameterList() ){
                if( not par.isEnabled() ) continue;
                _strippedParameterList_.emplace_back(&par);
            }
        }
        LogThrowIf( _strippedParameterList_.size() != nStripped, "Enabled parameters list don't correspond to the matrix" );
    }

    if( _choleskyMatrix_ == nullptr ){
        LogInfo << "Generating global cholesky matrix" << std::endl;
        _choleskyMatrix_ = std::shared_ptr<TMatrixD>(
                GenericToolbox::getCholeskyMatrix(_strippedCovarianceMatrix_.get())
        );
    }

    bool keepThrowing{true};
//  int throwNb{0};

    while( keepThrowing ){
//    throwNb++;
        bool rethrow{false};
        std::vector<double> throws;
        std::vector<double> weights;
        if(pedestalEntity==0){
            GundamUtils::throwCorrelatedParameters(_choleskyMatrix_.get(),throws, weights);
            for(int i=0;i<weights.size();i++){
              LogInfo<<"{ParametersManager::throwParsFromGlobalCov}} weights["<<i<<"] = "<<weights[i]<<std::endl;
            }
        }else{
            GundamUtils::throwCorrelatedParameters(_choleskyMatrix_.get(),throws, weights,
                                                      pedestalEntity,pedestalLeftEdge,pedestalRightEdge);
        }
        if(throws.size() != weights.size()){
            LogInfo<<"{ParametersManager}ERROR: throws.size() != weights.size() "<< throws.size()<<" "<<weights.size()<<std::endl;
        }
        if(weights.size() != _choleskyMatrix_->GetNrows()){
            LogInfo<<"{ParametersManager}ERROR: throws.size() != _choleskyMatrix_->GetNrows() "<< throws.size()<<" "<<_choleskyMatrix_->GetNrows()<<std::endl;
        }
        for( int iPar = 0 ; iPar < _choleskyMatrix_->GetNrows() ; iPar++ ){
            _strippedParameterList_[iPar]->setParameterValue(
                    _strippedParameterList_[iPar]->getPriorValue()
                    + throws[iPar]
            );
            weightsChiSquare.push_back(weights[iPar]);

            if( _reThrowParSetIfOutOfBounds_ ){
                if( not _strippedParameterList_[iPar]->isValueWithinBounds() ){
                    // re-do the throwing
//          LogDebug << "Not within bounds: " << _strippedParameterList_[iPar]->getSummary() << std::endl;
                    rethrow = true;
                }
            }
        }

        // Making sure eigen decomposed parameters get the conversion done
        for( auto& parSet : _parameterSetList_ ){
            if( not parSet.isEnabled() ) continue;
            if( parSet.isUseEigenDecompInFit() ){
                parSet.propagateOriginalToEigen();

                // also check the bounds of real parameter space
                if( _reThrowParSetIfOutOfBounds_ ){
                    for( auto& par : parSet.getEigenParameterList() ){
                        if( not par.isEnabled() ) continue;
                        if( not par.isValueWithinBounds() ){
                            // re-do the throwing
                            rethrow = true;
                            break;
                        }
                    }
                }
            }
        }


        if( rethrow ){
            // wrap back to the while loop
//      LogDebug << "RE-THROW #" << throwNb << std::endl;
            continue;
        }

        // reached this point: all parameters are within bounds
        keepThrowing = false;
    }
    for{int i=0;i<weightsChiSquare.size();i++){
        LogInfo<<"{ParametersManager::throwParsFromGlobalCov output} weightsChiSquare["<<i<<"] = "<<weightsChiSquare[i]<<std::endl;

}// end of function

void ParametersManager::throwParametersFromTStudent(std::vector<double> &weightsChiSquare,double nu_){
    // check that weightsChiSquare is an empty vector
    LogThrowIf( not weightsChiSquare.empty(), "ERROR: argument weightsChiSquare is not empty" );

    if( _strippedCovarianceMatrix_ == nullptr ){
        LogInfo << "Creating stripped global covariance matrix..." << std::endl;
        LogThrowIf( _globalCovarianceMatrix_ == nullptr, "Global covariance matrix not set." );
        int nStripped{0};
        for( int iDiag = 0 ; iDiag < _globalCovarianceMatrix_->GetNrows() ; iDiag++ ){
            if( (*_globalCovarianceMatrix_)[iDiag][iDiag] != 0 ){ nStripped++; }
        }

        LogInfo << "Stripped global covariance matrix is " << nStripped << "x" << nStripped << std::endl;
        _strippedCovarianceMatrix_ = std::make_shared<TMatrixD>(nStripped, nStripped);
        int iStrippedBin{-1};
        for( int iBin = 0 ; iBin < _globalCovarianceMatrix_->GetNrows() ; iBin++ ){
            if( (*_globalCovarianceMatrix_)[iBin][iBin] == 0 ){ continue; }
            iStrippedBin++;
            int jStrippedBin{-1};
            for( int jBin = 0 ; jBin < _globalCovarianceMatrix_->GetNrows() ; jBin++ ){
                if( (*_globalCovarianceMatrix_)[jBin][jBin] == 0 ){ continue; }
                jStrippedBin++;
                (*_strippedCovarianceMatrix_)[iStrippedBin][jStrippedBin] = (*_globalCovarianceMatrix_)[iBin][jBin];
            }
        }

        _strippedParameterList_.reserve( nStripped );
        for( auto& parSet : _parameterSetList_ ){
            if( not parSet.isEnabled() ) continue;
            for( auto& par : parSet.getParameterList() ){
                if( not par.isEnabled() ) continue;
                _strippedParameterList_.emplace_back(&par);
            }
        }
        LogThrowIf( _strippedParameterList_.size() != nStripped, "Enabled parameters list don't correspond to the matrix" );
    }

    if( _choleskyMatrix_ == nullptr ){
        LogInfo << "Generating global cholesky matrix" << std::endl;
        _choleskyMatrix_ = std::shared_ptr<TMatrixD>(
                GenericToolbox::getCholeskyMatrix(_strippedCovarianceMatrix_.get())
        );
    }

    bool keepThrowing{true};
//  int throwNb{0};

    while( keepThrowing ){
//    throwNb++;
        bool rethrow{false};
        std::vector<double> throws,weights;
        // calling Toolbox function to throw random parameters
        GundamUtils::throwTStudentParameters(_choleskyMatrix_.get(),nu_,throws, weights);
        ///////
        if(throws.size() != weights.size()){
            LogInfo<<"WARNING: throws.size() != weights.size() "<< throws.size()<<weights.size()<<std::endl;
        }
        for( int iPar = 0 ; iPar < _choleskyMatrix_->GetNrows() ; iPar++ ){
            _strippedParameterList_[iPar]->setParameterValue(
                    _strippedParameterList_[iPar]->getPriorValue()
                    + throws[iPar]
            );
            weightsChiSquare.push_back(weights[iPar]);

            if( _reThrowParSetIfOutOfBounds_ ){
                if( not _strippedParameterList_[iPar]->isValueWithinBounds() ){
                    // re-do the throwing
//          LogDebug << "Not within bounds: " << _strippedParameterList_[iPar]->getSummary() << std::endl;
                    rethrow = true;
                }
            }
        }

        // Making sure eigen decomposed parameters get the conversion done
        for( auto& parSet : _parameterSetList_ ){
            if( not parSet.isEnabled() ) continue;
            if( parSet.isUseEigenDecompInFit() ){
                parSet.propagateOriginalToEigen();

                // also check the bounds of real parameter space
                if( _reThrowParSetIfOutOfBounds_ ){
                    for( auto& par : parSet.getEigenParameterList() ){
                        if( not par.isEnabled() ) continue;
                        if( not par.isValueWithinBounds() ){
                            // re-do the throwing
                            rethrow = true;
                            break;
                        }
                    }
                }
            }
        }


        if( rethrow ){
            // wrap back to the while loop
//      LogDebug << "RE-THROW #" << throwNb << std::endl;
            continue;
        }

        // reached this point: all parameters are within bounds
        keepThrowing = false;
    }
}


void ParametersManager::moveParametersToPrior(){
  for( auto& parSet : _parameterSetList_ ){
    if( not parSet.isEnabled() ){ continue; }
    parSet.moveParametersToPrior();
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

