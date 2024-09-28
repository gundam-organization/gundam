//
// Created by Nadrino on 21/05/2021.
//

#include "ParameterSet.h"
#include "BinSet.h"

#include "GundamGlobals.h"
#include "ParameterThrowerMarkHarz.h"
#include "ConfigUtils.h"

#include "GenericToolbox.Root.h"

#include "GenericToolbox.Utils.h"
#include "Logger.h"

#include <memory>

#ifndef DISABLE_USER_HEADER
LoggerInit([]{ Logger::setUserHeaderStr("[ParameterSet]"); });
#endif


void ParameterSet::configureImpl(){

  GenericToolbox::Json::fillValue(_config_, _name_, "name");
  LogThrowIf(_name_.empty(), "ParameterSet have no name.");
  LogDebugIf(GundamGlobals::isDebugConfig()) << "Reading config for parameter set: " << _name_ << std::endl;

  GenericToolbox::Json::fillValue(_config_, _isEnabled_, "isEnabled");
  if( not _isEnabled_ ){
    LogDebugIf(GundamGlobals::isDebugConfig()) << " -> marked as disabled." << std::endl;
    return; // don't go any further
  }

  GenericToolbox::Json::fillValue(_config_, _nbParameterDefinition_, "numberOfParameters");
  GenericToolbox::Json::fillValue(_config_, _nominalStepSize_, "nominalStepSize");

  GenericToolbox::Json::fillValue(_config_, _useOnlyOneParameterPerEvent_, "useOnlyOneParameterPerEvent");
  GenericToolbox::Json::fillValue(_config_, _printDialSetsSummary_, "printDialSetsSummary");
  GenericToolbox::Json::fillValue(_config_, _printParametersSummary_, "printParametersSummary");

  GenericToolbox::Json::fillValue(_config_, _globalParRange_.min, "parameterLimits/minValue");
  GenericToolbox::Json::fillValue(_config_, _globalParRange_.max, "parameterLimits/maxValue");

  GenericToolbox::Json::fillValue(_config_, _enablePca_, {{"enablePca"},{"allowPca"},{"fixGhostFitParameters"}});
  GenericToolbox::Json::fillValue(_config_, _enabledThrowToyParameters_, "enabledThrowToyParameters");
  GenericToolbox::Json::fillValue(_config_, _customParThrow_, {{"customParThrow"},{"customFitParThrow"}});
  GenericToolbox::Json::fillValue(_config_, _releaseFixedParametersOnHesse_, "releaseFixedParametersOnHesse");

  GenericToolbox::Json::fillValue(_config_, _parameterDefinitionFilePath_, {{"parameterDefinitionFilePath"},{"covarianceMatrixFilePath"}});
  GenericToolbox::Json::fillValue(_config_, _covarianceMatrixPath_, {{"covarianceMatrix"},{"covarianceMatrixTMatrixD"}});
  GenericToolbox::Json::fillValue(_config_, _parameterNameListPath_, {{"parameterNameList"},{"parameterNameTObjArray"}});
  GenericToolbox::Json::fillValue(_config_, _parameterPriorValueListPath_, {{"parameterPriorValueList"},{"parameterPriorTVectorD"}});

  GenericToolbox::Json::fillValue(_config_, _parameterLowerBoundsTVectorD_, "parameterLowerBoundsTVectorD");
  GenericToolbox::Json::fillValue(_config_, _parameterUpperBoundsTVectorD_, "parameterUpperBoundsTVectorD");
  GenericToolbox::Json::fillValue(_config_, _throwEnabledListPath_, "throwEnabledList");

  GenericToolbox::Json::fillValue(_config_, _parameterDefinitionConfig_, "parameterDefinitions");
  GenericToolbox::Json::fillValue(_config_, _dialSetDefinitions_, "dialSetDefinitions");
  GenericToolbox::Json::fillValue(_config_, _enableOnlyParameters_, "enableOnlyParameters");
  GenericToolbox::Json::fillValue(_config_, _disableParameters_, "disableParameters");

  // throws options
  GenericToolbox::Json::fillValue(_config_, _useMarkGenerator_, "useMarkGenerator");
  GenericToolbox::Json::fillValue(_config_, _useEigenDecompForThrows_, "useEigenDecompForThrows");

  // eigen related parameters
  GenericToolbox::Json::fillValue(_config_, _enableEigenDecomp_, {{"enableEigenDecomp"},{"useEigenDecompInFit"}});
  GenericToolbox::Json::fillValue(_config_, _allowEigenDecompWithBounds_, "allowEigenDecompWithBounds");
  GenericToolbox::Json::fillValue(_config_, _maxNbEigenParameters_, "maxNbEigenParameters");
  GenericToolbox::Json::fillValue(_config_, _maxEigenFraction_, "maxEigenFraction");
  GenericToolbox::Json::fillValue(_config_, _eigenSvdThreshold_, "eigenSvdThreshold");

  GenericToolbox::Json::fillValue(_config_, _eigenParRange_.min, "eigenParBounds/minValue");
  GenericToolbox::Json::fillValue(_config_, _eigenParRange_.max, "eigenParBounds/maxValue");

  // legacy
  GenericToolbox::Json::fillValue(_config_, _maskForToyGeneration_, "maskForToyGeneration");

  // dev option -> was used for validation
  GenericToolbox::Json::fillValue(_config_, _devUseParLimitsOnEigen_, "devUseParLimitsOnEigen");


  // individual parameter definitions:
  if( not _parameterDefinitionFilePath_.empty() ){ readParameterDefinitionFile(); }

  if( _nbParameterDefinition_ == -1 ){
    // no number of parameter provided -> parameters were not defined
    // looking for alternative/legacy definitions...

    if( not _dialSetDefinitions_.empty() ){
      for( auto& dialSetDef : _dialSetDefinitions_ ){

        JsonType parameterBinning{};
        GenericToolbox::Json::fetchValue<JsonType>(dialSetDef, {{"binning"}, {"parametersBinningPath"}}, parameterBinning);

        if( parameterBinning.empty() ){ continue; }

        LogInfo << "Found parameter binning within dialSetDefinition. Defining parameters number..." << std::endl;
        BinSet b;
        b.configure( parameterBinning );
        // DON'T SORT THE BINNING -> tide to the cov matrix
        _nbParameterDefinition_ = int(b.getBinList().size());

        // don't fetch other dataset as they should always have the same assumption
        break;

      }
    }

    if( _nbParameterDefinition_ == -1 and not _parameterDefinitionConfig_.empty() ){
      LogDebugIf(GundamGlobals::isDebugConfig()) << "Using parameter definition config list to determine the number of parameters..." << std::endl;
      _nbParameterDefinition_ = int(_parameterDefinitionConfig_.get<std::vector<JsonType>>().size());
    }

    LogThrowIf(_nbParameterDefinition_==-1, "Could not figure out the number of parameters to be defined for the set: " << _name_ );
  }

  this->defineParameters();
}
void ParameterSet::initializeImpl() {
  for( auto& par : _parameterList_ ){
    par.initialize();
//    if( _printParametersSummary_ and par.isEnabled() ){
//      LogInfo << par.getSummary(not _printDialSetsSummary_) << std::endl;
//    }
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
  int nbParameters{0};
  int configWarnings{0};
  for( const auto& par : _parameterList_ ){
    if( not ParameterSet::isValidCorrelatedParameter(par) ) continue;
    nbParameters++;
    if( not isEnableEigenDecomp() ) continue;
    // Warn if using eigen decomposition with bounded parameters.
    if ( std::isnan(par.getMinValue()) and std::isnan(par.getMaxValue())) continue;
    LogAlert << "Undefined behavior: Eigen-decomposition of a bounded parameter: "
               << par.getFullTitle()
               << std::endl;
    ++configWarnings;
  }
  LogInfo << nbParameters << " effective parameters were defined in set: " << getName() << std::endl;

  if( nbParameters == 0 ){
    LogAlert << "No parameter is enabled. Disabling the parameter set." << std::endl;
    _isEnabled_ = false;
    return;
  }

  if (configWarnings > 0) {
    LogError << "Undefined behavior: Using bounded parameters with eigendecomposition"
             << std::endl;
    if ( not _allowEigenDecompWithBounds_ ) {
      LogError << "Eigendecomposition not allowed with parameter bounds"
               << std::endl;
      LogError << "Add 'allowEigenDecompWithBounds' to config file to enable"
               << std::endl;
      std::exit(1);
    }
  }
  LogInfo << nbParameters << " effective parameters were defined in set: " << getName() << std::endl;

  _strippedCovarianceMatrix_ = std::make_shared<TMatrixDSym>(nbParameters);
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

  if( not isEnableEigenDecomp() ){
    LogWarning << "Computing inverse of the stripped covariance matrix: "
               << _strippedCovarianceMatrix_->GetNcols() << "x"
               << _strippedCovarianceMatrix_->GetNrows() << std::endl;
    _inverseStrippedCovarianceMatrix_ = std::shared_ptr<TMatrixD>((TMatrixD*)(_strippedCovarianceMatrix_->Clone()));
    _inverseStrippedCovarianceMatrix_->Invert();
  }
  else {
    LogWarning << "Decomposing the stripped covariance matrix..." << std::endl;
    _eigenParameterList_.resize(_strippedCovarianceMatrix_->GetNrows(), Parameter(this));

    LogAlertIf(_strippedCovarianceMatrix_->GetNrows() > 1000) << "Decomposing matrix with " << _strippedCovarianceMatrix_->GetNrows() << " dim might take a while..." << std::endl;
    _eigenDecomp_     = std::make_shared<TMatrixDSymEigen>(*_strippedCovarianceMatrix_);

    // Used for base swapping
    _eigenValues_     = std::shared_ptr<TVectorD>( (TVectorD*) _eigenDecomp_->GetEigenValues().Clone() );
    _eigenValuesInv_  = std::shared_ptr<TVectorD>( (TVectorD*) _eigenDecomp_->GetEigenValues().Clone() );
    _eigenVectors_    = std::shared_ptr<TMatrixD>( (TMatrixD*) _eigenDecomp_->GetEigenVectors().Clone() );
    _eigenVectorsInv_ = std::make_shared<TMatrixD>(TMatrixD::kTransposed, *_eigenVectors_ );

    _nbEnabledEigen_ = 0;
    double eigenTotal = _eigenValues_->Sum();

    LogInfo << "Covariance eigen values are between " << _eigenValues_->Min() << " and " << _eigenValues_->Max() << std::endl;
    if( std::isnan(_eigenSvdThreshold_) ){
      LogThrowIf(_eigenValues_->Min() < 0, "Input covariance matrix is not positive definite.");
    }
    else if( _eigenValues_->Min()/_eigenValues_->Max() < _eigenSvdThreshold_ ){
      LogAlert << "Eigen values bellow the threshold(" << _eigenSvdThreshold_ << "). Using SVD..." << std::endl;
    }

    _inverseStrippedCovarianceMatrix_ = std::make_shared<TMatrixD>(_strippedCovarianceMatrix_->GetNrows(), _strippedCovarianceMatrix_->GetNrows());
    _projectorMatrix_                 = std::make_shared<TMatrixD>(_strippedCovarianceMatrix_->GetNrows(), _strippedCovarianceMatrix_->GetNrows());

    auto* eigenState = new TVectorD(_eigenValues_->GetNrows());

    for (int iEigen = 0; iEigen < _eigenValues_->GetNrows(); iEigen++) {

      _eigenParameterList_[iEigen].setParameterIndex( iEigen );
      _eigenParameterList_[iEigen].setIsEnabled(true);
      _eigenParameterList_[iEigen].setIsEigen(true);
      _eigenParameterList_[iEigen].setStdDevValue(TMath::Sqrt((*_eigenValues_)[iEigen]));
      _eigenParameterList_[iEigen].setStepSize(TMath::Sqrt((*_eigenValues_)[iEigen]));
      _eigenParameterList_[iEigen].setName("eigen");

      // fixing all of them by default
      _eigenParameterList_[iEigen].setIsFixed(true);
      (*eigenState)[iEigen] = 0;
      (*_eigenValuesInv_)[iEigen] = 1./(*_eigenValues_)[iEigen];

    }


    double eigenCumulative{0};

    // this loop assumes all eigen values are stored in decreasing order
    for (int iEigen = 0; iEigen < _eigenValues_->GetNrows(); iEigen++) {

      if( not std::isnan( _eigenSvdThreshold_ ) ){
        // check the current matrix conditioning
        if( (*_eigenValues_)[iEigen]/_eigenValues_->Max() < _eigenSvdThreshold_ ){
          LogAlert << "Keeping " << iEigen << " eigen values with SVD." << std::endl;
          break; // decreasing order
        }
      }


      if(    ( _maxNbEigenParameters_ != -1 and iEigen >= _maxNbEigenParameters_ )
             or ( _maxEigenFraction_ != 1 and (eigenCumulative + (*_eigenValues_)[iEigen]) / eigenTotal > _maxEigenFraction_ ) ){
        break; // decreasing order
      }

      // if we reach this point, the eigen value is accepted
      _eigenParameterList_[iEigen].setIsFixed( false );
      (*eigenState)[iEigen] = 1.;
      eigenCumulative += (*_eigenValues_)[iEigen];
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

        LogThrowIf( not std::isnan(eigenPar.getMinValue()) and eigenPar.getPriorValue() < eigenPar.getMinValue(), "PRIOR IS BELLOW MIN: " << eigenPar.getSummary() );
        LogThrowIf( not std::isnan(eigenPar.getMaxValue()) and eigenPar.getPriorValue() > eigenPar.getMaxValue(), "PRIOR IS ABOVE MAX: " << eigenPar.getSummary() );
      }
      else{
        eigenPar.setMinValue( _eigenParRange_.min );
        eigenPar.setMaxValue( _eigenParRange_.max );

        LogThrowIf( not std::isnan(eigenPar.getMinValue()) and eigenPar.getPriorValue() < eigenPar.getMinValue(), "Prior value is bellow min: " << eigenPar.getSummary() );
        LogThrowIf( not std::isnan(eigenPar.getMaxValue()) and eigenPar.getPriorValue() > eigenPar.getMaxValue(), "Prior value is above max: " << eigenPar.getSummary() );
      }
    }

  }

}

// Getters
const std::vector<Parameter>& ParameterSet::getEffectiveParameterList() const{
  if( isEnableEigenDecomp() ) return getEigenParameterList();
  return getParameterList();
}

bool ParameterSet::isValid() const {
  for (const Parameter& par : getParameterList()) {
    if (not par.isEnabled()) continue;
    if (not par.isValidValue(par.getParameterValue())) return false;
  }
  return true;
}

// non const getters
std::vector<Parameter>& ParameterSet::getEffectiveParameterList(){
  if( isEnableEigenDecomp() ) return getEigenParameterList();
  return getParameterList();
}

// Core
void ParameterSet::updateDeltaVector() const{
  int iFit{0};
  for( const auto& par : _parameterList_ ){
    if( ParameterSet::isValidCorrelatedParameter(par) ){
      (*_deltaVectorPtr_)[iFit++] = par.getParameterValue() - par.getPriorValue();
    }
  }
}

void ParameterSet::setValidity(const std::string& validity) {
  for (Parameter& par : getParameterList()) {
    par.setValidity(validity);
  }
  LogWarning << "Set parameter set validity to " << validity << std::endl;
}

// Parameter throw
void ParameterSet::moveParametersToPrior(){
  if( not isEnableEigenDecomp() ){
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
void ParameterSet::throwParameters(bool rethrowIfNotInPhysical_, double gain_){

  LogThrowIf(_strippedCovarianceMatrix_==nullptr, "No covariance matrix provided");

  TVectorD throwsList{_strippedCovarianceMatrix_->GetNrows()};

  // generic function to handle multiple throws
  std::function<void(std::function<void()>)> throwParsFct =
      [&](const std::function<void()>& throwFct_){

        int nTries{0};
        while( true ){
          ++nTries;
          throwFct_();

          // throws with this function are always done in real space.
          int iFit{-1};
          for( auto& par : this->getParameterList() ){
            if( ParameterSet::isValidCorrelatedParameter(par) ){
              iFit++;
              par.setThrowValue( par.getPriorValue() + gain_*throwsList[iFit] );
            }
          }

          bool throwIsValid = true; // default case
          LogInfo << "Check that thrown parameters are within bounds..."
                  << std::endl;

          for( auto& par : this->getParameterList() ){
            if( not ParameterSet::isValidCorrelatedParameter(par) ) continue;
            if( not std::isnan(par.getMinValue())
                and par.getThrowValue() < par.getMinValue() ){
              throwIsValid = false;
              LogAlert << "thrown value lower than min bound -> "
                       << par.getSummary() << std::endl;
              break;
            }
            if( not std::isnan(par.getMaxValue())
                and par.getThrowValue() > par.getMaxValue() ){
              throwIsValid = false;
              LogAlert <<"thrown value higher than max bound -> "
                       << par.getSummary() << std::endl;
              break;
            }

            if ( not rethrowIfNotInPhysical_ ) continue;
            if( not std::isnan(par.getMinPhysical())
                and par.getThrowValue() < par.getMinPhysical() ){
              throwIsValid = false;
              LogAlert << "thrown value lower than min physical bound -> "
                       << par.getSummary() << std::endl;
              break;
            }
            if( not std::isnan(par.getMaxPhysical())
                and par.getThrowValue() > par.getMaxPhysical() ){
              throwIsValid = false;
              LogAlert <<"thrown value higher than max physical bound -> "
                       << par.getSummary() << std::endl;
              break;
            }
          }

          if (not throwIsValid) {
            LogAlert << "Rethrowing \"" << this->getName() << "\"... try #" << nTries+1 << std::endl;
            LogThrowIf(nTries > 10000, "Failed to find valid throw");
            continue;
          }

          // Throw looks OK, so set the parameter values.
          for( auto& par : this->getParameterList() ){
            if( ParameterSet::isValidCorrelatedParameter(par) ){
              par.setParameterValue(par.getThrowValue());
            }
          }

          // Copy the throws to the eigen space and set the throw value in
          // eigenspace too.
          if( isEnableEigenDecomp() ){
            this->propagateOriginalToEigen();
            for( auto& eigenPar : _eigenParameterList_ ){
              eigenPar.setThrowValue( eigenPar.getParameterValue() );
            }
          }

          // alright at this point it's fine, print them
          for( auto& par : _parameterList_ ){
            if( ParameterSet::isValidCorrelatedParameter(par) ){
              LogInfo << "Thrown par " << par.getTitle() << ": " << par.getPriorValue();
              LogInfo << " becomes " << par.getParameterValue() << std::endl;
            }
          }
          if( isEnableEigenDecomp() ){
            LogInfo << "Translated to eigen space:" << std::endl;
            for( auto& eigenPar : _eigenParameterList_ ){
              LogInfo << "Eigen par " << eigenPar.getTitle() << ": " << eigenPar.getPriorValue();
              LogInfo << " becomes " << eigenPar.getParameterValue() << std::endl;
            }
          }
          break;
        }
      }; // End of generic function handling multiple throws

  if( _useMarkGenerator_ ){
    // Throw using an alternative method that was copied from BANFF
    LogAlert << "Alternative toy generator used: Mark Hartz Generator"
             << std::endl;
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
  else if( _useEigenDecompForThrows_ and isEnableEigenDecomp() ){
    // Throw using a deprecated alternative method.  Do not use.
    LogAlert << "Alternative toy generator used: Eigen Decomposition Generator"
             << std::endl;
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
      if( true ){
        LogInfo << "Checking if the thrown parameters of the set are within bounds..." << std::endl;

        for( auto& par : this->getEffectiveParameterList() ){
          if( not std::isnan(par.getMinValue()) and par.getParameterValue() < par.getMinValue() ){
            throwIsValid = false;
            LogAlert << GenericToolbox::ColorCodes::redLightText << "thrown value lower than min bound -> " << GenericToolbox::ColorCodes::resetColor
                     << par.getSummary() << std::endl;
          }
          else if( not std::isnan(par.getMaxValue()) and par.getParameterValue() > par.getMaxValue() ){
            throwIsValid = false;
            LogAlert << GenericToolbox::ColorCodes::redLightText <<"thrown value higher than max bound -> " << GenericToolbox::ColorCodes::resetColor
                     << par.getSummary() << std::endl;
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
  else {
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
    _eigenParameterList_[iEigen].setParameterValue((*_eigenParBuffer_)[iEigen], true);
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
    par.setParameterValue((*_originalParBuffer_)[iParOffSet++], true);
  }
}


// Misc
std::string ParameterSet::getSummary() const {
  std::stringstream ss;

  ss << "ParameterSet summary: " << _name_ << " -> enabled=" << _isEnabled_;

  if(_isEnabled_){

    ss << ", nbParameters: " << _parameterList_.size();

    if( _printParametersSummary_ ){
      if( not _parameterList_.empty() ){

        GenericToolbox::TablePrinter t;
        t.setColTitles({ {"Title"}, {"Value"}, {"Prior"}, {"StdDev"}, {"Min"}, {"Max"}, {"Status"} });


        for( const auto& par : _parameterList_ ){
          std::string colorStr;
          std::string statusStr;

          if( not par.isEnabled() ) { continue; }
          else if( par.isFixed() )  { statusStr = "Fixed (prior applied)";    colorStr = GenericToolbox::ColorCodes::yellowLightText; }
          else if( par.isFree() )   { statusStr = "Free";     colorStr = GenericToolbox::ColorCodes::blueLightText; }
          else                      { statusStr = "Constrained"; }

#ifdef NOCOLOR
          colorStr = "";
#endif

          t.addTableLine({
                             par.getTitle(),
                             std::to_string( par.isValueWithinBounds() ?
                                             par.getParameterValue()
                                             : std::nan("Invalid") ),
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

  if( this->isEnableEigenDecomp() ){
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

void ParameterSet::nullify(){
  // TODO: reimplement toy disabling
  std::string name{_name_};
  (*this) = ParameterSet();
  this->setName(name);
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

void ParameterSet::printConfiguration() const {
  LogInfo << "name(" << _name_ << ")";
  LogInfo << ", nPars(" << _nbParameterDefinition_ << ")";
  LogInfo << std::endl;

  for( auto& par : _parameterList_ ){ par.printConfiguration(); }
}


// Protected
void ParameterSet::readParameterDefinitionFile(){

  TObject* objBuffer{nullptr};

  std::string path = GenericToolbox::expandEnvironmentVariables(_parameterDefinitionFilePath_);
  std::unique_ptr<TFile> parDefFile(TFile::Open(path.c_str()));
  LogThrowIf(parDefFile == nullptr or not parDefFile->IsOpen(), "Could not open: " << path);


  // define with the covariance matrix size
  if( not _covarianceMatrixPath_.empty() ){
    objBuffer = parDefFile->Get(_covarianceMatrixPath_.c_str());
    LogThrowIf(objBuffer == nullptr, "Can't find \"" << _covarianceMatrixPath_ << "\" in " << parDefFile->GetPath())
    _priorCovarianceMatrix_ = std::shared_ptr<TMatrixDSym>((TMatrixDSym*) objBuffer->Clone());
    _priorCorrelationMatrix_ = std::shared_ptr<TMatrixDSym>((TMatrixDSym*) GenericToolbox::convertToCorrelationMatrix((TMatrixD*)_priorCovarianceMatrix_.get()));
    LogThrowIf(_nbParameterDefinition_ != -1, "Nb of parameter was manually defined but the covariance matrix");
    _nbParameterDefinition_ = _priorCovarianceMatrix_->GetNrows();
  }

  // parameterPriorTVectorD
  if(not _parameterPriorValueListPath_.empty()){
    objBuffer = parDefFile->Get(_parameterPriorValueListPath_.c_str());
    LogThrowIf(objBuffer == nullptr, "Can't find \"" << _parameterPriorValueListPath_ << "\" in " << parDefFile->GetPath())
    _parameterPriorList_ = std::shared_ptr<TVectorD>((TVectorD*) objBuffer->Clone());
    LogThrowIf(_parameterPriorList_->GetNrows() != _nbParameterDefinition_,
      "Parameter prior list don't have the same size("
      << _parameterPriorList_->GetNrows()
      << ") as cov matrix(" << _nbParameterDefinition_ << ")"
    );
  }

  // parameterNameTObjArray
  if(not _parameterNameListPath_.empty()){
    objBuffer = parDefFile->Get(_parameterNameListPath_.c_str());
    LogThrowIf(objBuffer == nullptr, "Can't find \"" << _parameterNameListPath_ << "\" in " << parDefFile->GetPath())
    _parameterNamesList_ = std::shared_ptr<TObjArray>((TObjArray*) objBuffer->Clone());
  }

  // parameterLowerBoundsTVectorD
  if( not _parameterLowerBoundsTVectorD_.empty() ){
    objBuffer = parDefFile->Get(_parameterLowerBoundsTVectorD_.c_str());
    LogThrowIf(objBuffer == nullptr, "Can't find \"" << _parameterLowerBoundsTVectorD_ << "\" in " << parDefFile->GetPath())
    _parameterLowerBoundsList_ = std::shared_ptr<TVectorD>((TVectorD*) objBuffer->Clone());

    LogThrowIf(_parameterLowerBoundsList_->GetNrows() != _nbParameterDefinition_,
                "Parameter prior list don't have the same size(" << _parameterLowerBoundsList_->GetNrows()
                                                                 << ") as cov matrix(" << _nbParameterDefinition_ << ")" );
  }

  // parameterUpperBoundsTVectorD
  if( not _parameterUpperBoundsTVectorD_.empty() ){
    objBuffer = parDefFile->Get(_parameterUpperBoundsTVectorD_.c_str());
    LogThrowIf(objBuffer == nullptr, "Can't find \"" << _parameterUpperBoundsTVectorD_ << "\" in " << parDefFile->GetPath())
    _parameterUpperBoundsList_ = std::shared_ptr<TVectorD>((TVectorD*) objBuffer->Clone());
    LogThrowIf(_parameterUpperBoundsList_->GetNrows() != _nbParameterDefinition_,
                "Parameter prior list don't have the same size(" << _parameterUpperBoundsList_->GetNrows()
                                                                 << ") as cov matrix(" << _nbParameterDefinition_ << ")" );
  }

  if( not _throwEnabledListPath_.empty() ){
    objBuffer = parDefFile->Get(_throwEnabledListPath_.c_str());
    LogThrowIf(objBuffer == nullptr, "Can't find \"" << _throwEnabledListPath_ << "\" in " << parDefFile->GetPath())
    _throwEnabledList_ = std::shared_ptr<TVectorD>((TVectorD*) objBuffer->Clone());
  }

  parDefFile->Close();
}
void ParameterSet::defineParameters(){
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
        LogDebugIf(GundamGlobals::isDebugConfig()) << "Enabling parameter \"" << par.getFullTitle() << "\" as it is set in enableOnlyParameters" << std::endl;
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
        LogDebugIf(GundamGlobals::isDebugConfig()) << "Skipping parameter \"" << par.getFullTitle() << "\" as it is set in disableParameters" << std::endl;
        par.setIsEnabled( false );
        continue;
      }
    }

    if( _parameterPriorList_ != nullptr ){ par.setPriorValue((*_parameterPriorList_)[par.getParameterIndex()]); }
    else{ par.setPriorValue(1); }

    par.setParameterValue(par.getPriorValue());

    if( not std::isnan(_globalParRange_.min) ){ par.setMinValue(_globalParRange_.min); }
    if( not std::isnan(_globalParRange_.max) ){ par.setMaxValue(_globalParRange_.max); }

    if( _parameterLowerBoundsList_ != nullptr ){ par.setMinValue((*_parameterLowerBoundsList_)[par.getParameterIndex()]); }
    if( _parameterUpperBoundsList_ != nullptr ){ par.setMaxValue((*_parameterUpperBoundsList_)[par.getParameterIndex()]); }

    LogThrowIf( not std::isnan(par.getMinValue()) and par.getPriorValue() < par.getMinValue(), "PRIOR IS BELLOW MIN: " << par.getSummary() );
    LogThrowIf( not std::isnan(par.getMaxValue()) and par.getPriorValue() > par.getMaxValue(), "PRIOR IS ABOVE MAX: " << par.getSummary() );

    if( not _parameterDefinitionConfig_.empty() ){
      // Alternative 1: define dials then parameters
      if (_parameterNamesList_ != nullptr) {
        // Find the parameter using the name from the vector of names for
        // the covariance.
        auto parConfig = GenericToolbox::Json::fetchMatchingEntry(_parameterDefinitionConfig_, "name", std::string(_parameterNamesList_->At(par.getParameterIndex())->GetName()));
        if( parConfig.empty() ) parConfig = GenericToolbox::Json::fetchMatchingEntry(_parameterDefinitionConfig_, "parameterName", std::string(_parameterNamesList_->At(par.getParameterIndex())->GetName()));
        if( parConfig.empty() ){
            // try with par index
          parConfig = GenericToolbox::Json::fetchMatchingEntry(_parameterDefinitionConfig_, "parameterIndex", par.getParameterIndex());
        }
        par.setConfig(parConfig);
      }
      else {
        // No covariance provided, so find the name based on the order in
        // the parameter set.
        auto configVector = _parameterDefinitionConfig_.get<std::vector<JsonType>>();
        LogThrowIf(configVector.size() <= par.getParameterIndex());
        auto parConfig = configVector.at(par.getParameterIndex());
        auto parName = GenericToolbox::Json::fetchValue<std::string>(parConfig, {{"name"}, {"parameterName"}});
        if (not parName.empty()) par.setName(parName);
        par.setConfig(parConfig);
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
    par.configure();
  }
}
