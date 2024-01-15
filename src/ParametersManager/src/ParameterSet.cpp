//
// Created by Nadrino on 21/05/2021.
//

#include "ParameterSet.h"
#include "DataBinSet.h"

#include "GundamGlobals.h"
#include "ParameterThrowerMarkHarz.h"
#include "ConfigUtils.h"

#include "GenericToolbox.Root.h"
#include "GenericToolbox.Json.h"
#include "GenericToolbox.Utils.h"
#include "Logger.h"

#include <memory>

LoggerInit([]{
  Logger::setUserHeaderStr("[ParameterSet]");
} );


void ParameterSet::readConfigImpl(){
  LogThrowIf(_config_.empty(), "FitParameterSet config not set.");

  _name_ = GenericToolbox::Json::fetchValue<std::string>(_config_, "name");
  LogInfo << std::endl << "Initializing parameter set: " << _name_ << std::endl;

  _isEnabled_ = GenericToolbox::Json::fetchValue<bool>(_config_, "isEnabled");
  LogReturnIf(not _isEnabled_, _name_ << " parameters are disabled.");

  _nbParameterDefinition_ = GenericToolbox::Json::fetchValue(_config_, "numberOfParameters", _nbParameterDefinition_);
  _nominalStepSize_ = GenericToolbox::Json::fetchValue(_config_, "nominalStepSize", _nominalStepSize_);

  _useOnlyOneParameterPerEvent_ = GenericToolbox::Json::fetchValue<bool>(_config_, "useOnlyOneParameterPerEvent", false);
  _printDialSetsSummary_ = GenericToolbox::Json::fetchValue<bool>(_config_, "printDialSetsSummary", _printDialSetsSummary_);
  _printParametersSummary_ = GenericToolbox::Json::fetchValue<bool>(_config_, "printParametersSummary", _printDialSetsSummary_);

  if( GenericToolbox::Json::doKeyExist(_config_, "parameterLimits") ){
    auto parLimits = GenericToolbox::Json::fetchValue(_config_, "parameterLimits", JsonType());
    _globalParameterMinValue_ = GenericToolbox::Json::fetchValue(parLimits, "minValue", _globalParameterMinValue_);
    _globalParameterMaxValue_ = GenericToolbox::Json::fetchValue(parLimits, "maxValue", _globalParameterMaxValue_);
  }

  _useEigenDecomposition_ = GenericToolbox::Json::fetchValue(_config_ , {{"useEigenDecomposition"}, {"useEigenDecompInFit"}}, _useEigenDecomposition_);
  if( _useEigenDecomposition_ ){
    LogWarning << "Using eigen decomposition in fit." << std::endl;
    LogScopeIndent;

    _maxNbEigenParameters_ = GenericToolbox::Json::fetchValue(_config_ , "maxNbEigenParameters", -1);
    if( _maxNbEigenParameters_ != -1 ){
      LogInfo << "Maximum nb of eigen parameters is set to " << _maxNbEigenParameters_ << std::endl;
    }
    _maxEigenFraction_ = GenericToolbox::Json::fetchValue(_config_ , "maxEigenFraction", double(1.));
    if( _maxEigenFraction_ != 1 ){
      LogInfo << "Max eigen fraction set to: " << _maxEigenFraction_*100 << "%" << std::endl;
    }

    if( GenericToolbox::Json::doKeyExist(_config_, "eigenParBounds") ){
      auto eigenLimits = GenericToolbox::Json::fetchValue(_config_, "eigenParBounds", JsonType());
      _eigenParBounds_.first = GenericToolbox::Json::fetchValue(eigenLimits, "minValue", _eigenParBounds_.first);
      _eigenParBounds_.second = GenericToolbox::Json::fetchValue(eigenLimits, "maxValue", _eigenParBounds_.second);
      LogInfo << "Using eigen parameter limits: [ " << _eigenParBounds_.first << ", " << _eigenParBounds_.first << "]" << std::endl;
    }

    _devUseParLimitsOnEigen_ = GenericToolbox::Json::fetchValue(_config_, "devUseParLimitsOnEigen", _devUseParLimitsOnEigen_);
    if( _devUseParLimitsOnEigen_ ){ LogAlert << "USING DEV OPTION: _devUseParLimitsOnEigen_ = true" << std::endl; }

  }

  _enablePca_ = GenericToolbox::Json::fetchValue(_config_, std::vector<std::string>{"allowPca", "runPcaCheck", "enablePca"}, _enablePca_);
  _enabledThrowToyParameters_ = GenericToolbox::Json::fetchValue(_config_, "enabledThrowToyParameters", _enabledThrowToyParameters_);
  _maskForToyGeneration_ = GenericToolbox::Json::fetchValue(_config_, "maskForToyGeneration", _maskForToyGeneration_);
  _customFitParThrow_ = GenericToolbox::Json::fetchValue(_config_, "customFitParThrow", std::vector<JsonType>());
  _releaseFixedParametersOnHesse_ = GenericToolbox::Json::fetchValue(_config_, "releaseFixedParametersOnHesse", _releaseFixedParametersOnHesse_);

  _parameterDefinitionFilePath_ = GenericToolbox::Json::fetchValue( _config_,
    {{"parameterDefinitionFilePath"}, {"covarianceMatrixFilePath"} }, _parameterDefinitionFilePath_
  );
  _covarianceMatrixTMatrixD_ = GenericToolbox::Json::fetchValue(_config_, "covarianceMatrixTMatrixD", _covarianceMatrixTMatrixD_);
  _parameterPriorTVectorD_ = GenericToolbox::Json::fetchValue(_config_, "parameterPriorTVectorD", _parameterPriorTVectorD_);
  _parameterNameTObjArray_ = GenericToolbox::Json::fetchValue(_config_, "parameterNameTObjArray", _parameterNameTObjArray_);
  _parameterLowerBoundsTVectorD_ = GenericToolbox::Json::fetchValue(_config_, "parameterLowerBoundsTVectorD", _parameterLowerBoundsTVectorD_);
  _parameterUpperBoundsTVectorD_ = GenericToolbox::Json::fetchValue(_config_, "parameterUpperBoundsTVectorD", _parameterUpperBoundsTVectorD_);
  _throwEnabledListPath_ = GenericToolbox::Json::fetchValue(_config_, "throwEnabledList", _throwEnabledListPath_);

  _parameterDefinitionConfig_ = GenericToolbox::Json::fetchValue(_config_, "parameterDefinitions", _parameterDefinitionConfig_);
  _dialSetDefinitions_ = GenericToolbox::Json::fetchValue(_config_, "dialSetDefinitions", _dialSetDefinitions_);
  _enableOnlyParameters_ = GenericToolbox::Json::fetchValue(_config_, "enableOnlyParameters", _enableOnlyParameters_);
  _disableParameters_ = GenericToolbox::Json::fetchValue(_config_, "disableParameters", _disableParameters_);


  // MISC / DEV
  _useMarkGenerator_ = GenericToolbox::Json::fetchValue(_config_, "useMarkGenerator", _useMarkGenerator_);
  _useEigenDecompForThrows_ = GenericToolbox::Json::fetchValue(_config_, "useEigenDecompForThrows", _useEigenDecompForThrows_);

  this->readParameterDefinitionFile();

  if( _nbParameterDefinition_ == -1 ){
    LogWarning << "No number of parameter provided. Looking for alternative definitions..." << std::endl;

    if( not _dialSetDefinitions_.empty() ){
      for( auto& dialSetDef : _dialSetDefinitions_.get<std::vector<JsonType>>() ){
        if( GenericToolbox::Json::doKeyExist(dialSetDef, "parametersBinningPath") ){
          LogInfo << "Found parameter binning within dialSetDefinition. Defining parameters number..." << std::endl;
          DataBinSet b;
          b.readBinningDefinition( GenericToolbox::Json::fetchValue<std::string>(dialSetDef, "parametersBinningPath") );
          // DON'T SORT THE BINNING -> tide to the cov matrix
          _nbParameterDefinition_ = int(b.getBinList().size());
          break;
        }
      }
    }

    if( _nbParameterDefinition_ == -1 and not _parameterDefinitionConfig_.empty() ){
      LogInfo << "Using parameter definition config list to determine the number of parameters..." << std::endl;
      _nbParameterDefinition_ = int(_parameterDefinitionConfig_.get<std::vector<JsonType>>().size());
    }

    LogThrowIf(_nbParameterDefinition_==-1, "Could not figure out the number of parameters to be defined for the set: " << _name_ );
  }

  this->defineParameters();
}
void ParameterSet::initializeImpl() {
  LogInfo << "Initializing \"" << this->getName() << "\"" << std::endl;

  LogReturnIf(not _isEnabled_, this->getName() << " is not enabled. Skipping.");

  for( auto& par : _parameterList_ ){
    par.initialize();
    if( _printParametersSummary_ and par.isEnabled() ){
      LogInfo << par.getSummary(not _printDialSetsSummary_) << std::endl;
    }
  }

  // Make the matrix inversion
  this->processCovarianceMatrix();
}

// statics in src dependent
void ParameterSet::muteLogger(){ Logger::setIsMuted(true ); }
void ParameterSet::unmuteLogger(){ Logger::setIsMuted(false ); }

// Post-init
void ParameterSet::processCovarianceMatrix(){

  if( _priorCovarianceMatrix_ == nullptr ){ return; } // nothing to do

  LogInfo << "Stripping the matrix from fixed/disabled parameters..." << std::endl;
  int nbFitParameters{0};
  for( const auto& par : _parameterList_ ){
    if( ParameterSet::isValidCorrelatedParameter(par) ) nbFitParameters++;
  }
  LogInfo << nbFitParameters << " effective parameters were defined in set: " << getName() << std::endl;

  _strippedCovarianceMatrix_ = std::make_shared<TMatrixDSym>(nbFitParameters);
  int iStrippedPar = -1;
  for( int iPar = 0 ; iPar < int(_parameterList_.size()) ; iPar++ ){
    if( not ParameterSet::isValidCorrelatedParameter(_parameterList_[iPar]) ) continue;
    iStrippedPar++;
    int jStrippedPar = -1;
    for( int jPar = 0 ; jPar < int(_parameterList_.size()) ; jPar++ ){
      if( not ParameterSet::isValidCorrelatedParameter(_parameterList_[jPar]) ) continue;
      jStrippedPar++;
      (*_strippedCovarianceMatrix_)[iStrippedPar][jStrippedPar] = (*_priorCovarianceMatrix_)[iPar][jPar];
    }
  }
  _deltaVectorPtr_ = std::make_shared<TVectorD>(_strippedCovarianceMatrix_->GetNrows());

  LogThrowIf(not _strippedCovarianceMatrix_->IsSymmetric(), "Covariance matrix is not symmetric");

  if( not _useEigenDecomposition_ ){
    LogWarning << "Computing inverse of the stripped covariance matrix: "
               << _strippedCovarianceMatrix_->GetNcols() << "x"
               << _strippedCovarianceMatrix_->GetNrows() << std::endl;
    _inverseStrippedCovarianceMatrix_ = std::shared_ptr<TMatrixD>((TMatrixD*)(_strippedCovarianceMatrix_->Clone()));
    _inverseStrippedCovarianceMatrix_->Invert();
  }
  else {
    LogWarning << "Decomposing the stripped covariance matrix..." << std::endl;
    _eigenParameterList_.resize(_strippedCovarianceMatrix_->GetNrows(), Parameter(this));

    _eigenDecomp_     = std::make_shared<TMatrixDSymEigen>(*_strippedCovarianceMatrix_);

    // Used for base swapping
    _eigenValues_     = std::shared_ptr<TVectorD>( (TVectorD*) _eigenDecomp_->GetEigenValues().Clone() );
    _eigenValuesInv_  = std::shared_ptr<TVectorD>( (TVectorD*) _eigenDecomp_->GetEigenValues().Clone() );
    _eigenVectors_    = std::shared_ptr<TMatrixD>( (TMatrixD*) _eigenDecomp_->GetEigenVectors().Clone() );
    _eigenVectorsInv_ = std::make_shared<TMatrixD>(TMatrixD::kTransposed, *_eigenVectors_ );

    double eigenCumulative = 0;
    _nbEnabledEigen_ = 0;
    double eigenTotal = _eigenValues_->Sum();

    LogInfo << "Covariance eigen values are between " << _eigenValues_->Min() << " and " << _eigenValues_->Max() << std::endl;
    LogThrowIf(_eigenValues_->Min() < 0, "Input covariance matrix is not positive definite.");

    _inverseStrippedCovarianceMatrix_ = std::make_shared<TMatrixD>(_strippedCovarianceMatrix_->GetNrows(), _strippedCovarianceMatrix_->GetNrows());
    _projectorMatrix_                 = std::make_shared<TMatrixD>(_strippedCovarianceMatrix_->GetNrows(), _strippedCovarianceMatrix_->GetNrows());

    auto* eigenState = new TVectorD(_eigenValues_->GetNrows());

    for (int iEigen = 0; iEigen < _eigenValues_->GetNrows(); iEigen++) {

      _eigenParameterList_[iEigen].setIsEigen(true);
      _eigenParameterList_[iEigen].setIsEnabled(true);
      _eigenParameterList_[iEigen].setIsFixed(false);
      _eigenParameterList_[iEigen].setParameterIndex(iEigen);
      _eigenParameterList_[iEigen].setStdDevValue(TMath::Sqrt((*_eigenValues_)[iEigen]));
      _eigenParameterList_[iEigen].setStepSize(TMath::Sqrt((*_eigenValues_)[iEigen]));
      _eigenParameterList_[iEigen].setName("eigen");

      (*_eigenValuesInv_)[iEigen] = 1./(*_eigenValues_)[iEigen];
      (*eigenState)[iEigen] = 1.;

      eigenCumulative += (*_eigenValues_)[iEigen];
      if(    ( _maxNbEigenParameters_ != -1 and iEigen >= _maxNbEigenParameters_ )
          or ( _maxEigenFraction_ != 1      and eigenCumulative / eigenTotal > _maxEigenFraction_ ) ){
        eigenCumulative -= (*_eigenValues_)[iEigen]; // not included
        (*_eigenValues_)[iEigen] = 0;
        (*_eigenValuesInv_)[iEigen] = 0;
        (*eigenState)[iEigen] = 0;

        _eigenParameterList_[iEigen].setIsFixed(true);

        break;
      }
      _nbEnabledEigen_++;

    } // iEigen

    TMatrixD* eigenStateMatrix    = GenericToolbox::makeDiagonalMatrix(eigenState);
    TMatrixD* diagInvMatrix = GenericToolbox::makeDiagonalMatrix(_eigenValuesInv_.get());

    (*_projectorMatrix_) =  (*_eigenVectors_);
    (*_projectorMatrix_) *= (*eigenStateMatrix);
    (*_projectorMatrix_) *= (*_eigenVectorsInv_);

    (*_inverseStrippedCovarianceMatrix_) =  (*_eigenVectors_);
    (*_inverseStrippedCovarianceMatrix_) *= (*diagInvMatrix);
    (*_inverseStrippedCovarianceMatrix_) *= (*_eigenVectorsInv_);

    delete eigenState;
    delete eigenStateMatrix;
    delete diagInvMatrix;

    LogWarning << "Eigen decomposition with " << _nbEnabledEigen_ << " / " << _eigenValues_->GetNrows() << " vectors" << std::endl;
    if(_nbEnabledEigen_ != _eigenValues_->GetNrows() ){
      LogInfo << "Max eigen fraction set to " << _maxEigenFraction_*100 << "%" << std::endl;
      LogInfo << "Fraction taken: " << eigenCumulative / eigenTotal*100 << "%" << std::endl;
    }

    _originalParBuffer_ = std::make_shared<TVectorD>(_strippedCovarianceMatrix_->GetNrows() );
    _eigenParBuffer_    = std::make_shared<TVectorD>(_strippedCovarianceMatrix_->GetNrows() );

//    LogAlert << "Disabling par/dial limits" << std::endl;
//    for( auto& par : _parameterList_ ){
//      par.setMinValue(std::nan(""));
//      par.setMaxValue(std::nan(""));
//      for( auto& dialSet : par.getDialSetList() ){
//        dialSet.setMinDialResponse(std::nan(""));
//        dialSet.setMaxDialResponse(std::nan(""));
//      }
//    }

    // Put original parameters to the prior
    for( auto& par : _parameterList_ ){
      par.setValueAtPrior();
    }

    // Original parameter values are already set -> need to propagate to Eigen parameter list
    this->propagateOriginalToEigen();

    // Tag the prior
    for( auto& eigenPar : _eigenParameterList_ ){
      eigenPar.setCurrentValueAsPrior();

      if( _devUseParLimitsOnEigen_ ){
        eigenPar.setMinValue( _parameterList_[eigenPar.getParameterIndex()].getMinValue() );
        eigenPar.setMaxValue( _parameterList_[eigenPar.getParameterIndex()].getMaxValue() );

        LogThrowIf( not std::isnan(eigenPar.getMinValue()) and eigenPar.getPriorValue() < eigenPar.getMinValue(), "PRIOR IS BELLOW MIN: " << eigenPar.getSummary(true) );
        LogThrowIf( not std::isnan(eigenPar.getMaxValue()) and eigenPar.getPriorValue() > eigenPar.getMaxValue(), "PRIOR IS ABOVE MAX: " << eigenPar.getSummary(true) );
      }
      else{
        eigenPar.setMinValue( _eigenParBounds_.first );
        eigenPar.setMaxValue( _eigenParBounds_.second );

        LogThrowIf( not std::isnan(eigenPar.getMinValue()) and eigenPar.getPriorValue() < eigenPar.getMinValue(), "Prior value is bellow min: " << eigenPar.getSummary(true) );
        LogThrowIf( not std::isnan(eigenPar.getMaxValue()) and eigenPar.getPriorValue() > eigenPar.getMaxValue(), "Prior value is above max: " << eigenPar.getSummary(true) );
      }
    }

  }

}

// Getters
const std::vector<Parameter>& ParameterSet::getEffectiveParameterList() const{
  if( _useEigenDecomposition_ ) return _eigenParameterList_;
  return _parameterList_;
}

// non const getters
std::vector<Parameter>& ParameterSet::getEffectiveParameterList(){
  if( _useEigenDecomposition_ ) return _eigenParameterList_;
  return _parameterList_;
}

// Parameter throw
void ParameterSet::updateDeltaVector() const {
  int iFit{0};
  for( const auto& par : _parameterList_ ){
    if( ParameterSet::isValidCorrelatedParameter(par) ){
      (*_deltaVectorPtr_)[iFit++] = par.getParameterValue() - par.getPriorValue();
    }
  }
}

void ParameterSet::moveFitParametersToPrior(){
  LogInfo << "Moving back fit parameters to their prior value in set: " << getName() << std::endl;

  if( not _useEigenDecomposition_ ){
    for( auto& par : _parameterList_ ){
      if( par.isFixed() or not par.isEnabled() ){ continue; }
      par.setParameterValue(par.getPriorValue());
    }
  }
  else{
    for( auto& eigenPar : _eigenParameterList_ ){
      if( eigenPar.isFixed() or not eigenPar.isEnabled() ) continue;
      eigenPar.setParameterValue(eigenPar.getPriorValue());
    }
    this->propagateEigenToOriginal();
  }

}
void ParameterSet::throwFitParameters(bool rethrowIfNotInbounds_, double gain_){

  LogThrowIf(_strippedCovarianceMatrix_==nullptr, "No covariance matrix provided");


  TVectorD throwsList{_strippedCovarianceMatrix_->GetNrows()};

  // generic function to handle multiple throws
  std::function<void(std::function<void()>)> throwParsFct =
      [&](const std::function<void()>& throwFct_){

        int nTries{0};
        bool throwIsValid{false};
        while( not throwIsValid ){

          throwFct_();

          // throws with this function are always done in real space.
          int iFit{-1};
          for( auto& par : this->getParameterList() ){
            if( ParameterSet::isValidCorrelatedParameter(par) ){
              iFit++;
              par.setThrowValue( par.getPriorValue() + gain_ * throwsList[iFit] );
              par.setParameterValue( par.getThrowValue() );
            }
          }
          if( _useEigenDecomposition_ ){
            this->propagateOriginalToEigen();
            for( auto& eigenPar : _eigenParameterList_ ){
              eigenPar.setThrowValue( eigenPar.getParameterValue() );
            }
          }

          throwIsValid = true; // default case
          if( rethrowIfNotInbounds_ ){
            LogInfo << "Checking if the thrown parameters of the set are within bounds..." << std::endl;

            for( auto& par : this->getEffectiveParameterList() ){
              if      ( not std::isnan(par.getMinValue()) and par.getParameterValue() < par.getMinValue() ){
                throwIsValid = false;
                LogAlert << GenericToolbox::ColorCodes::redLightText << "thrown value lower than min bound -> " << GenericToolbox::ColorCodes::resetColor
                         << par.getSummary(true) << std::endl;
              }
              else if( not std::isnan(par.getMaxValue()) and par.getParameterValue() > par.getMaxValue() ){
                throwIsValid = false;
                LogAlert << GenericToolbox::ColorCodes::redLightText <<"thrown value higher than max bound -> " << GenericToolbox::ColorCodes::resetColor
                         << par.getSummary(true) << std::endl;
              }
            }

            if( not throwIsValid ){
              LogAlert << "Rethrowing \"" << this->getName() << "\"... try #" << nTries+1 << std::endl;
              nTries++;
              continue;
            }
            else{
              LogInfo << "Keeping throw after " << nTries+1 << " attempt(s)." << std::endl;
            }
          } // check bounds?

          // alright at this point it's fine, print them
          for( auto& par : _parameterList_ ){
            if( ParameterSet::isValidCorrelatedParameter(par) ){
              LogInfo << "Thrown par " << par.getTitle() << ": " << par.getPriorValue();
              LogInfo << " → " << par.getParameterValue() << std::endl;
            }
          }
          if( _useEigenDecomposition_ ){
            LogInfo << "Translated to eigen space:" << std::endl;
            for( auto& eigenPar : _eigenParameterList_ ){
              LogInfo << "Eigen par " << eigenPar.getTitle() << ": " << eigenPar.getPriorValue();
              LogInfo << " → " << eigenPar.getParameterValue() << std::endl;
            }
          }
        }

  };

  if( _useMarkGenerator_ ){
    int iPar{0};
    for( auto& par : _parameterList_ ){
      if( ParameterSet::isValidCorrelatedParameter(par) ){ throwsList[iPar++] = par.getPriorValue(); }
    }

    if( _markHartzGen_ == nullptr ){
      LogInfo << "Generating Cholesky matrix in set: " << getName() << std::endl;
      _markHartzGen_ = std::make_shared<ParameterThrowerMarkHarz>(throwsList, *_strippedCovarianceMatrix_);
    }
    TVectorD throws(_strippedCovarianceMatrix_->GetNrows());

    std::vector<double> throwPars(_strippedCovarianceMatrix_->GetNrows());
    std::function<void()> markScottThrowFct = [&](){
      _markHartzGen_->ThrowSet(throwPars);
      // THROWS ARE CENTERED AROUND 1!!

      // convert to TVectorD
      int iPar{0};
      for( auto& thrownPar : throwPars ){
        throwsList[iPar++] = thrownPar;
      }
    };

    throwParsFct( markScottThrowFct );
  }
  else{
    if( _useEigenDecompForThrows_ and _useEigenDecomposition_ ){
      LogInfo << "Throwing eigen parameters for " << _name_ << std::endl;

      int nTries{0};
      bool throwIsValid{false};
      while( not throwIsValid ){
        for( auto& eigenPar : _eigenParameterList_ ){
          eigenPar.setThrowValue(eigenPar.getPriorValue() + gain_ * gRandom->Gaus(0, eigenPar.getStdDevValue()));
          eigenPar.setParameterValue( eigenPar.getThrowValue() );
        }
        this->propagateEigenToOriginal();

        throwIsValid = true;
        if( rethrowIfNotInbounds_ ){
          LogInfo << "Checking if the thrown parameters of the set are within bounds..." << std::endl;

          for( auto& par : this->getEffectiveParameterList() ){
            if( not std::isnan(par.getMinValue()) and par.getParameterValue() < par.getMinValue() ){
              throwIsValid = false;
              LogAlert << GenericToolbox::ColorCodes::redLightText << "thrown value lower than min bound -> " << GenericToolbox::ColorCodes::resetColor
                       << par.getSummary(true) << std::endl;
            }
            else if( not std::isnan(par.getMaxValue()) and par.getParameterValue() > par.getMaxValue() ){
              throwIsValid = false;
              LogAlert << GenericToolbox::ColorCodes::redLightText <<"thrown value higher than max bound -> " << GenericToolbox::ColorCodes::resetColor
                       << par.getSummary(true) << std::endl;
            }
          }

          if( not throwIsValid ){
            LogAlert << "Rethrowing \"" << this->getName() << "\"... try #" << nTries+1 << std::endl;
            nTries++;
            continue;
          }
          else{
            LogInfo << "Keeping throw after " << nTries << " attempt(s)." << std::endl;
          }
        } // check bounds?

        for( auto& par : _parameterList_ ){
          LogInfo << "Thrown par (through eigen decomp) " << par.getTitle() << ": " << par.getPriorValue();
          par.setThrowValue(par.getParameterValue());
          LogInfo << " → " << par.getParameterValue() << std::endl;
        }
      }


    }
    else{
      LogInfo << "Throwing parameters for " << _name_ << " using Cholesky matrix" << std::endl;

      if( not _correlatedVariableThrower_.isInitialized() ){
        _correlatedVariableThrower_.setCovarianceMatrixPtr(_strippedCovarianceMatrix_.get());
        _correlatedVariableThrower_.initialize();
      }

      std::function<void()> gundamThrowFct = [&](){
        _correlatedVariableThrower_.throwCorrelatedVariables(throwsList);
      };

      throwParsFct( gundamThrowFct );

    }
  }

}

void ParameterSet::propagateOriginalToEigen(){
  // First propagate to the buffer
  int iParOffSet{0};
  for( const auto& par : _parameterList_ ){
    if( par.isFixed() or not par.isEnabled() ) continue;
    (*_originalParBuffer_)[iParOffSet++] = par.getParameterValue();
  }

  // Base swap: ORIG -> EIGEN
  (*_eigenParBuffer_) =  (*_originalParBuffer_);
  (*_eigenParBuffer_) *= (*_eigenVectorsInv_);

  // Propagate back to eigen parameters
  for( int iEigen = 0 ; iEigen < _eigenParBuffer_->GetNrows() ; iEigen++ ){
    _eigenParameterList_[iEigen].setParameterValue((*_eigenParBuffer_)[iEigen]);
  }
}
void ParameterSet::propagateEigenToOriginal(){
  // First propagate to the buffer
  for( int iEigen = 0 ; iEigen < _eigenParBuffer_->GetNrows() ; iEigen++ ){
    (*_eigenParBuffer_)[iEigen] = _eigenParameterList_[iEigen].getParameterValue();
  }

  // Base swap: EIGEN -> ORIG
  (*_originalParBuffer_) = (*_eigenParBuffer_);
  (*_originalParBuffer_) *= (*_eigenVectors_);

  // Propagate back to the real parameters
  int iParOffSet{0};
  for( auto& par : _parameterList_ ){
    if( par.isFixed() or not par.isEnabled() ) continue;
    par.setParameterValue((*_originalParBuffer_)[iParOffSet++]);
  }
}


// Misc
std::string ParameterSet::getSummary() const {
  std::stringstream ss;

  ss << "FitParameterSet summary: " << _name_ << " -> enabled=" << _isEnabled_;

  if(_isEnabled_){

    ss << ", nbParameters: " << _parameterList_.size();

    if( not _parameterList_.empty() ){

      GenericToolbox::TablePrinter t;
      t.setColTitles({ {"Title"}, {"Value"}, {"Prior"}, {"StdDev"}, {"Min"}, {"Max"}, {"Status"} });


      for( const auto& par : _parameterList_ ){
        std::string colorStr;
        std::string statusStr;

        if( not par.isEnabled() ) { statusStr = "Disabled"; colorStr = GenericToolbox::ColorCodes::yellowBackground; }
        else if( par.isFixed() )  { statusStr = "Fixed";    colorStr = GenericToolbox::ColorCodes::redBackground; }
        else if( par.isFree() )   { statusStr = "Free";     colorStr = GenericToolbox::ColorCodes::blueBackground; }
        else                      { statusStr = "Fit"; }

#ifdef NOCOLOR
        colorStr = "";
#endif

        t.addTableLine({
                           par.getTitle(),
                           std::to_string( par.getParameterValue() ),
                           std::to_string( par.getPriorValue() ),
                           std::to_string( par.getStdDevValue() ),
                           std::to_string( par.getMinValue() ),
                           std::to_string( par.getMaxValue() ),
                           statusStr
                       }, colorStr);
      }

      t.printTable();
    }
  }

  return ss.str();
}
JsonType ParameterSet::exportInjectorConfig() const{
  JsonType out;

  out["name"] = this->getName();

  std::vector<JsonType> parJsonList{};
  parJsonList.reserve( _parameterList_.size() );

  for( auto& par : _parameterList_ ){
    if( not par.isEnabled() ){ continue; }
    parJsonList.emplace_back();

    if( par.getName().empty() ){
      // will be identified by their index. For instance: '#52'
      parJsonList.back()["title"] = par.getTitle();
    }
    else{
      parJsonList.back()["name"] = par.getName();
    }

    parJsonList.back()["value"] = par.getParameterValue();
  }

  out["parameterValues"] = parJsonList;

  return out;
}
void ParameterSet::injectParameterValues(const JsonType& config_){
  LogWarning << "Importing parameters from config for \"" << this->getName() << "\"" << std::endl;

  auto config = ConfigUtils::getForwardedConfig(config_);
  LogThrowIf( config.empty(), "Invalid injector config" << std::endl << config_ );
  LogThrowIf( not GenericToolbox::Json::doKeyExist(config, "name"), "No parameter set name provided in" << std::endl << config_ );
  LogThrowIf( GenericToolbox::Json::fetchValue<std::string>(config, "name") != this->getName(),
              "Mismatching between parSet name (" << this->getName() << ") and injector config ("
              << GenericToolbox::Json::fetchValue<std::string>(config, "name") << ")" );

  auto parValues = GenericToolbox::Json::fetchValue( config, "parameterValues", JsonType() );
  if     ( parValues.empty() ) {
    LogThrow( "No parameterValues provided." );
  }
  else if( parValues.is_string() ){
    //
    LogInfo << "Reading parameter values from file: " << parValues.get<std::string>() << std::endl;
    auto parList = GenericToolbox::dumpFileAsVectorString( parValues.get<std::string>(), true );
    LogThrowIf( parList.size() != this->getNbParameters()  ,
                parList.size() << " parameters provided for " << this->getName() << ", expecting " << this->getNbParameters()
    );

    for( size_t iPar = 0 ; iPar < this->getNbParameters() ; iPar++ ) {

      if( not this->getParameterList()[iPar].isEnabled() ){
        LogAlert << "NOT injecting \"" << this->getParameterList()[iPar].getFullTitle() << "\" as it is disabled." << std::endl;
        continue;
      }

      LogScopeIndent;
      LogInfo << "Injecting \"" << this->getParameterList()[iPar].getFullTitle() << "\": " << parList[iPar] << std::endl;
      this->getParameterList()[iPar].setParameterValue( std::stod(parList[iPar]) );
    }
  }
  else{
    LogScopeIndent;
    for( auto& parValueEntry : parValues ){
      if     ( GenericToolbox::Json::doKeyExist(parValueEntry, "name") ) {
        auto parName = GenericToolbox::Json::fetchValue<std::string>(parValueEntry, "name");
        auto* parPtr = this->getParameterPtr(parName);
        LogThrowIf(parPtr == nullptr, "Could not find " << parName << " among the defined parameters in " << this->getName());


        if( not parPtr->isEnabled() ){
          LogAlert << "NOT injecting \"" << parPtr->getFullTitle() << "\" as it is disabled." << std::endl;
          continue;
        }

        LogInfo << "Injecting \"" << parPtr->getFullTitle() << "\": " << GenericToolbox::Json::fetchValue<double>(parValueEntry, "value") << std::endl;
        parPtr->setParameterValue( GenericToolbox::Json::fetchValue<double>(parValueEntry, "value") );
      }
      else if( GenericToolbox::Json::doKeyExist(parValueEntry, "title") ){
        auto parTitle = GenericToolbox::Json::fetchValue<std::string>(parValueEntry, "title");
        auto* parPtr = this->getParameterPtrWithTitle(parTitle);
        LogThrowIf(parPtr == nullptr, "Could not find " << parTitle << " among the defined parameters in " << this->getName());


        if( not parPtr->isEnabled() ){
          LogAlert << "NOT injecting \"" << parPtr->getFullTitle() << "\" as it is disabled." << std::endl;
          continue;
        }

        LogInfo << "Injecting \"" << parPtr->getFullTitle() << "\": " << GenericToolbox::Json::fetchValue<double>(parValueEntry, "value") << std::endl;
        parPtr->setParameterValue( GenericToolbox::Json::fetchValue<double>(parValueEntry, "value") );
      }
      else if( GenericToolbox::Json::doKeyExist(parValueEntry, "index") ){
        auto parIndex = GenericToolbox::Json::fetchValue<int>(parValueEntry, "index");
        LogThrowIf( parIndex < 0 or parIndex >= this->getParameterList().size(),
                    "invalid parameter index (" << parIndex << ") for injection in parSet: " << this->getName() );

        auto* parPtr = &this->getParameterList()[parIndex];
        if( not parPtr->isEnabled() ){
          LogAlert << "NOT injecting \"" << parPtr->getFullTitle() << "\" as it is disabled." << std::endl;
          continue;
        }

        LogInfo << "Injecting \"" << parPtr->getFullTitle() << "\": " << GenericToolbox::Json::fetchValue<double>(parValueEntry, "value") << std::endl;
        parPtr->setParameterValue( GenericToolbox::Json::fetchValue<double>(parValueEntry, "value") );
      }
      else {
        LogThrow("Unsupported: " << parValueEntry);
      }
    }
  }

  if( this->useEigenDecomposition() ){
    LogInfo << "Propagating back to the eigen decomposed parameters for parSet: " << this->getName() << std::endl;
    this->propagateOriginalToEigen();
  }

}
Parameter* ParameterSet::getParameterPtr(const std::string& parName_){
  if( not parName_.empty() ){
    for( auto& par : _parameterList_ ){
      if( par.getName() == parName_ ){ return &par; }
    }
  }
  return nullptr;
}
Parameter* ParameterSet::getParameterPtrWithTitle(const std::string& parTitle_){
  if( not parTitle_.empty() ){
    for( auto& par : _parameterList_ ){
      if( par.getTitle() == parTitle_ ){ return &par; }
    }
  }
  return nullptr;
}

// Static
double ParameterSet::toNormalizedParRange(double parRange, const Parameter& par){
  return (parRange)/par.getStdDevValue();
}
double ParameterSet::toNormalizedParValue(double parValue, const Parameter& par) {
  return ParameterSet::toNormalizedParRange(parValue - par.getPriorValue(), par);
}
double ParameterSet::toRealParRange(double normParRange, const Parameter& par){
  return normParRange*par.getStdDevValue();
}
double ParameterSet::toRealParValue(double normParValue, const Parameter& par) {
  return normParValue*par.getStdDevValue() + par.getPriorValue();
}
bool ParameterSet::isValidCorrelatedParameter(const Parameter& par_){
  return ( par_.isEnabled() and not par_.isFixed() and not par_.isFree() );
}


// Protected
void ParameterSet::readParameterDefinitionFile(){

  if( _parameterDefinitionFilePath_.empty() ) return;

  std::string path = GenericToolbox::expandEnvironmentVariables(_parameterDefinitionFilePath_);
  if (path != _parameterDefinitionFilePath_) {
    LogInfo << "Using parameter definition file " << path
            << std::endl;
  }

  std::unique_ptr<TFile> parDefFile(TFile::Open(path.c_str()));
  LogThrowIf(parDefFile == nullptr or not parDefFile->IsOpen(), "Could not open: " << path);

  TObject* objBuffer{nullptr};

  if( _covarianceMatrixTMatrixD_.empty() ){
    LogWarning << "No covariance matrix provided. Free parameter definition expected." << std::endl;
  }
  else{
    objBuffer = parDefFile->Get(_covarianceMatrixTMatrixD_.c_str());
    LogThrowIf(objBuffer == nullptr, "Can't find \"" << _covarianceMatrixTMatrixD_ << "\" in " << parDefFile->GetPath())
    _priorCovarianceMatrix_ = std::shared_ptr<TMatrixDSym>((TMatrixDSym*) objBuffer->Clone());
    _priorCorrelationMatrix_ = std::shared_ptr<TMatrixDSym>((TMatrixDSym*) GenericToolbox::convertToCorrelationMatrix((TMatrixD*)_priorCovarianceMatrix_.get()));
    LogThrowIf(_nbParameterDefinition_ != -1, "Nb of parameter was manually defined but the covariance matrix");
    _nbParameterDefinition_ = _priorCovarianceMatrix_->GetNrows();
  }

  // parameterPriorTVectorD
  if(not _parameterPriorTVectorD_.empty()){
    LogInfo << "Reading provided parameterPriorTVectorD: \"" << _parameterPriorTVectorD_ << "\"" << std::endl;

    objBuffer = parDefFile->Get(_parameterPriorTVectorD_.c_str());
    LogThrowIf(objBuffer == nullptr, "Can't find \"" << _parameterPriorTVectorD_ << "\" in " << parDefFile->GetPath())
    _parameterPriorList_ = std::shared_ptr<TVectorD>((TVectorD*) objBuffer->Clone());

    LogThrowIf(_parameterPriorList_->GetNrows() != _nbParameterDefinition_,
      "Parameter prior list don't have the same size("
      << _parameterPriorList_->GetNrows()
      << ") as cov matrix(" << _nbParameterDefinition_ << ")"
    );
  }

  // parameterNameTObjArray
  if(not _parameterNameTObjArray_.empty()){
    LogInfo << "Reading provided parameterNameTObjArray: \"" << _parameterNameTObjArray_ << "\"" << std::endl;

    objBuffer = parDefFile->Get(_parameterNameTObjArray_.c_str());
    LogThrowIf(objBuffer == nullptr, "Can't find \"" << _parameterNameTObjArray_ << "\" in " << parDefFile->GetPath())
    _parameterNamesList_ = std::shared_ptr<TObjArray>((TObjArray*) objBuffer->Clone());
  }

  // parameterLowerBoundsTVectorD
  if( not _parameterLowerBoundsTVectorD_.empty() ){
    LogInfo << "Reading provided parameterLowerBoundsTVectorD: \"" << _parameterLowerBoundsTVectorD_ << "\"" << std::endl;

    objBuffer = parDefFile->Get(_parameterLowerBoundsTVectorD_.c_str());
    LogThrowIf(objBuffer == nullptr, "Can't find \"" << _parameterLowerBoundsTVectorD_ << "\" in " << parDefFile->GetPath())
    _parameterLowerBoundsList_ = std::shared_ptr<TVectorD>((TVectorD*) objBuffer->Clone());

    LogThrowIf(_parameterLowerBoundsList_->GetNrows() != _nbParameterDefinition_,
                "Parameter prior list don't have the same size(" << _parameterLowerBoundsList_->GetNrows()
                                                                 << ") as cov matrix(" << _nbParameterDefinition_ << ")" );
  }

  // parameterUpperBoundsTVectorD
  if( not _parameterUpperBoundsTVectorD_.empty() ){
    LogInfo << "Reading provided parameterUpperBoundsTVectorD: \"" << _parameterUpperBoundsTVectorD_ << "\"" << std::endl;

    objBuffer = parDefFile->Get(_parameterUpperBoundsTVectorD_.c_str());
    LogThrowIf(objBuffer == nullptr, "Can't find \"" << _parameterUpperBoundsTVectorD_ << "\" in " << parDefFile->GetPath())
    _parameterUpperBoundsList_ = std::shared_ptr<TVectorD>((TVectorD*) objBuffer->Clone());
    LogThrowIf(_parameterUpperBoundsList_->GetNrows() != _nbParameterDefinition_,
                "Parameter prior list don't have the same size(" << _parameterUpperBoundsList_->GetNrows()
                                                                 << ") as cov matrix(" << _nbParameterDefinition_ << ")" );
  }

  if( not _throwEnabledListPath_.empty() ){
    LogInfo << "Reading provided throwEnabledList: \"" << _throwEnabledListPath_ << "\"" << std::endl;

    objBuffer = parDefFile->Get(_throwEnabledListPath_.c_str());
    LogThrowIf(objBuffer == nullptr, "Can't find \"" << _throwEnabledListPath_ << "\" in " << parDefFile->GetPath())
    _throwEnabledList_ = std::shared_ptr<TVectorD>((TVectorD*) objBuffer->Clone());
  }

  parDefFile->Close();
}
void ParameterSet::defineParameters(){
  LogInfo << "Defining " << _nbParameterDefinition_ << " parameters for the set: " << getName() << std::endl;
  _parameterList_.resize(_nbParameterDefinition_, Parameter(this));
  int parIndex{0};

  for( auto& par : _parameterList_ ){
    par.setParameterIndex(parIndex++);

    if( _priorCovarianceMatrix_ != nullptr ){
      par.setStdDevValue(TMath::Sqrt((*_priorCovarianceMatrix_)[par.getParameterIndex()][par.getParameterIndex()]));
      par.setStepSize(TMath::Sqrt((*_priorCovarianceMatrix_)[par.getParameterIndex()][par.getParameterIndex()]));
    }
    else{
      LogThrowIf(std::isnan(_nominalStepSize_), "Can't define free parameter without a \"nominalStepSize\"");
      par.setStdDevValue(_nominalStepSize_); // stdDev will only be used for display purpose
      par.setStepSize(_nominalStepSize_);
      par.setPriorType(Parameter::PriorType::Flat);
      par.setIsFree(true);
    }

    if( _parameterNamesList_ != nullptr ){ par.setName(_parameterNamesList_->At(par.getParameterIndex())->GetName()); }

    // par is now fully identifiable.
    if( not _enableOnlyParameters_.empty() ){
      bool isEnabled = false;
      for( auto& enableEntry : _enableOnlyParameters_ ){
        if( GenericToolbox::Json::doKeyExist(enableEntry, "name")
            and par.getName() == GenericToolbox::Json::fetchValue<std::string>(enableEntry, "name") ){
          isEnabled = true;
          break;
        }
      }

      if( not isEnabled ){
        // set it of
        par.setIsEnabled( false );
      }
      else{
        LogInfo << "Enabling parameter \"" << par.getFullTitle() << "\" as it is set in enableOnlyParameters" << std::endl;
      }
    }
    if( not _disableParameters_.empty() ){
      bool isEnabled = true;
      for( auto& disableEntry : _disableParameters_ ){
        if( GenericToolbox::Json::doKeyExist(disableEntry, "name")
            and par.getName() == GenericToolbox::Json::fetchValue<std::string>(disableEntry, "name") ){
          isEnabled = false;
          break;
        }
      }

      if( not isEnabled ){
        LogWarning << "Skipping parameter \"" << par.getFullTitle() << "\" as it is set in disableParameters" << std::endl;
        par.setIsEnabled( false );
        continue;
      }
    }



    if( _parameterPriorList_ != nullptr ){ par.setPriorValue((*_parameterPriorList_)[par.getParameterIndex()]); }
    else{ par.setPriorValue(1); }

    par.setParameterValue(par.getPriorValue());

    if( not std::isnan(_globalParameterMinValue_) ){ par.setMinValue(_globalParameterMinValue_); }
    if( not std::isnan(_globalParameterMaxValue_) ){ par.setMaxValue(_globalParameterMaxValue_); }

    if( _parameterLowerBoundsList_ != nullptr ){ par.setMinValue((*_parameterLowerBoundsList_)[par.getParameterIndex()]); }
    if( _parameterUpperBoundsList_ != nullptr ){ par.setMaxValue((*_parameterUpperBoundsList_)[par.getParameterIndex()]); }

    LogThrowIf( not std::isnan(par.getMinValue()) and par.getPriorValue() < par.getMinValue(), "PRIOR IS BELLOW MIN: " << par.getSummary(true) );
    LogThrowIf( not std::isnan(par.getMaxValue()) and par.getPriorValue() > par.getMaxValue(), "PRIOR IS ABOVE MAX: " << par.getSummary(true) );

    if( not _parameterDefinitionConfig_.empty() ){
      // Alternative 1: define dials then parameters
      ConfigUtils::forwardConfig(_parameterDefinitionConfig_);
      if (_parameterNamesList_ != nullptr) {
        // Find the parameter using the name from the vector of names for
        // the covariance.
        auto parConfig = GenericToolbox::Json::fetchMatchingEntry(_parameterDefinitionConfig_, "name", std::string(_parameterNamesList_->At(par.getParameterIndex())->GetName()));
        if( parConfig.empty() ) parConfig = GenericToolbox::Json::fetchMatchingEntry(_parameterDefinitionConfig_, "parameterName", std::string(_parameterNamesList_->At(par.getParameterIndex())->GetName()));
        if( parConfig.empty() ){
            // try with par index
          parConfig = GenericToolbox::Json::fetchMatchingEntry(_parameterDefinitionConfig_, "parameterIndex", par.getParameterIndex());
        }
        par.setParameterDefinitionConfig(parConfig);
      }
      else {
        // No covariance provided, so find the name based on the order in
        // the parameter set.
        auto configVector = _parameterDefinitionConfig_.get<std::vector<JsonType>>();
        LogThrowIf(configVector.size() <= par.getParameterIndex());
        auto parConfig = configVector.at(par.getParameterIndex());
        auto parName = GenericToolbox::Json::fetchValue<std::string>(parConfig, {{"name"}, {"parameterName"}});
        if (not parName.empty()) par.setName(parName);
        par.setParameterDefinitionConfig(parConfig);
        LogWarning << "Parameter #" << par.getParameterIndex()
                   << " (name \"" << par.getName() << "\")"
                   << " not defined by covariance matrix file"
                   << std::endl;
      }
    }
    else if( not _dialSetDefinitions_.empty() ){
      // Alternative 2: define dials then parameters
      par.setDialSetConfig( _dialSetDefinitions_ );
    }
    par.readConfig();
  }
}



