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


void ParameterSet::configureImpl(){

  _config_.defineFields(std::vector<ConfigReader::FieldDefinition>{
    {FieldFlag::MANDATORY, "name"},
    {"isEnabled"},
    {"isScanEnabled"},
    {"numberOfParameters"},
    {"nominalStepSize"},
    {"parametersRange", {"parameterLimits"}},
    {"enablePca", {"allowPca", "fixGhostFitParameters"}},
    {"enableThrowToyParameters",{"enabledThrowToyParameters"}},
    {"printDialSetSummary", {"printDialSetsSummary"}},
    {"customParThrow", {"customFitParThrow"}},
    {"printParametersSummary", {"printParameterSummary"}},
    {"parameterDefinitionFilePath", {"covarianceMatrixFilePath"}},
    {"covarianceMatrix", {"covarianceMatrixTMatrixD"}},
    {"parameterNameList", {"parameterNameTObjArray"}},
    {"parameterPriorValueList", {"parameterPriorTVectorD"}},
    {"parameterLowerBoundsList", {"parameterLowerBoundsTVectorD"}},
    {"parameterUpperBoundsList", {"parameterUpperBoundsTVectorD"}},
    {"throwEnabledList"},
    {"parameterDefinitions"},
    {"dialSetDefinitions"},
    {"enableOnlyParameters"},
    {"disableParameters"},
    {"useMarkGenerator"},
    {"useEigenDecompForThrows"},
    {"enableEigenDecomp",{"useEigenDecompInFit"}},
    {"allowEigenDecompWithBounds"},
    {"maxNbEigenParameters"},
    {"maxEigenFraction"},
    {"eigenValueThreshold",{"eigenSvdThreshold"}},
    {"eigenParBounds"},
    {"eigenParBounds/minValue"},
    {"eigenParBounds/maxValue"},
    {"maskForToyGeneration"},
    {"devUseParLimitsOnEigen"},
    {"releaseFixedParametersOnHesse"},
    {"skipVariedEventRates"},
    {"disableOneSigmaPlots"},
  });
  _config_.checkConfiguration();

  _config_.fillValue(_name_, "name");
  LogExitIf(_name_.empty(), "Config error -- parameter set without a name.");
  LogDebugIf(GundamGlobals::isDebug()) << "Reading config for parameter set: " << _name_ << std::endl;

  _config_.fillValue(_isEnabled_, "isEnabled");
  if( not _isEnabled_ ){
    LogDebugIf(GundamGlobals::isDebug()) << " -> marked as disabled." << std::endl;
    return; // don't go any further
  }

  _config_.fillValue(_isScanEnabled_, "isScanEnabled");
  _config_.fillValue(_skipVariedEventRates_, "skipVariedEventRates");
  _config_.fillValue(_disableOneSigmaPlots_, "disableOneSigmaPlots");

  _config_.fillValue(_nbParameterDefinition_, "numberOfParameters");
  _config_.fillValue(_nominalStepSize_, "nominalStepSize");

  _config_.fillValue(_printDialSetsSummary_, "printDialSetSummary");
  _config_.fillValue(_printParametersSummary_, "printParametersSummary");

  _config_.fillValue(_globalParRange_, "parametersRange");

  _config_.fillValue(_enablePca_, "enablePca");

  // throw related
  _config_.fillValue(_enabledThrowToyParameters_, "enableThrowToyParameters");
  _config_.fillValue(_customParThrow_, "customParThrow");
  _config_.fillValue(_releaseFixedParametersOnHesse_, "releaseFixedParametersOnHesse");

  _config_.fillValue(_parameterDefinitionFilePath_, "parameterDefinitionFilePath");
  _config_.fillValue(_covarianceMatrixPath_, "covarianceMatrix");
  _config_.fillValue(_parameterNameListPath_, "parameterNameList");
  _config_.fillValue(_parameterPriorValueListPath_, "parameterPriorValueList");

  _config_.fillValue(_parameterLowerBoundsTVectorD_, "parameterLowerBoundsList");
  _config_.fillValue(_parameterUpperBoundsTVectorD_, "parameterUpperBoundsList");
  _config_.fillValue(_throwEnabledListPath_, "throwEnabledList");

  _config_.fillValue(_parameterDefinitionConfig_, "parameterDefinitions");
  _config_.fillValue(_dialSetDefinitions_, "dialSetDefinitions");
  _config_.fillValue(_enableOnlyParameters_, "enableOnlyParameters");
  _config_.fillValue(_disableParameters_, "disableParameters");

  // throw options
  _config_.fillValue(_useMarkGenerator_, "useMarkGenerator");
  _config_.fillValue(_useEigenDecompForThrows_, "useEigenDecompForThrows");

  // eigen related parameters
  _config_.fillValue(_enableEigenDecomp_, "enableEigenDecomp");
  _config_.fillValue(_allowEigenDecompWithBounds_, "allowEigenDecompWithBounds");
  _config_.fillValue(_maxNbEigenParameters_, "maxNbEigenParameters");
  _config_.fillValue(_maxEigenFraction_, "maxEigenFraction");
  _config_.fillValue(_eigenSvdThreshold_, "eigenValueThreshold");

  if( _config_.hasField("eigenParBounds/minValue") or _config_.hasField("eigenParBounds/maxValue") ){
    // legacy
    _config_.fillValue(_eigenParRange_.min, "eigenParBounds/minValue");
    _config_.fillValue(_eigenParRange_.max, "eigenParBounds/maxValue");
  }
  else{
    // use range definition instead `eigenParBounds: [0, 1]`
    _config_.fillValue(_eigenParRange_, "eigenParBounds");
  }

  // legacy
  _config_.fillValue(_maskForToyGeneration_, "maskForToyGeneration");

  // dev option -> was used for validation
  _config_.fillValue(_devUseParLimitsOnEigen_, "devUseParLimitsOnEigen");

  // individual parameter definitions:
  if( not _parameterDefinitionFilePath_.empty() ){ readParameterDefinitionFile(); }

  if( _nbParameterDefinition_ == -1 ){
    // no number of parameters provided -> parameters were not defined
    // looking for alternative/legacy definitions...

    if( not _dialSetDefinitions_.empty() ){
      for( auto& dialSetDef : _dialSetDefinitions_.loop() ){

        // dial library is top level, so using simple field def here
        dialSetDef.defineFields({{"binning", {"parametersBinningPath"}}});
        if( not dialSetDef.hasField("binning") ){ continue; }

        auto parameterBinning = dialSetDef.fetchValue<ConfigReader>("binning");
        if( parameterBinning.empty() ){ continue; }

        LogInfo << "Found parameter binning within dialSetDefinition. Defining parameters number..." << std::endl;
        BinSet b;
        b.configure( parameterBinning );
        // DON'T SORT THE BINNING -> tide to the cov matrix
        _nbParameterDefinition_ = int(b.getBinList().size());

        // don't fetch another dataset as they should always have the same assumption
        break;

      }
    }

    if( _nbParameterDefinition_ == -1 and not _parameterDefinitionConfig_.empty() ){
      LogDebugIf(GundamGlobals::isDebug()) << "Using parameter definition config list to determine the number of parameters..." << std::endl;
      _nbParameterDefinition_ = int(_parameterDefinitionConfig_.getConfig().size());
      for(auto& parDef : _parameterDefinitionConfig_.loop() ){ Parameter::prepareConfig(parDef); }
    }

    LogExitIf(_nbParameterDefinition_==-1, "Could not figure out the number of parameters to be defined for the set: " << _name_ );
  }

  if (_nbParameterDefinition_ < 1){
    LogError << "CONFIG ERROR: Parameter set \"" << getName() << "\" without parameters." << std::endl;
  }

  this->defineParameters();
}
void ParameterSet::initializeImpl() {

  _config_.printUnusedKeys();

  for( auto& par : _parameterList_ ){
    par.initialize();
  }

  // Make the matrix inversion
  this->processCovarianceMatrix();
}

// statics in src dependent
void ParameterSet::muteLogger(){ Logger::setIsMuted(true ); }
void ParameterSet::unmuteLogger(){ Logger::setIsMuted(false ); }

// Post-init
void ParameterSet::processCovarianceMatrix(){

  if( _priorFullCovarianceMatrix_ == nullptr ){ return; } // nothing to do

  LogInfo << "Stripping the matrix from fixed/disabled parameters..." << std::endl;
  int nbParameters{0};
  int configWarnings{0};
  for( const auto& par : _parameterList_ ){
    if( not ParameterSet::isValidCorrelatedParameter(par) ) continue;
    nbParameters++;
    if( not isEnableEigenDecomp() ) continue;
    // Warn if using eigen decomposition with bounded parameters.
    if( par.getParameterLimits().isUnbounded() ){ continue; }
    LogAlert << "Undefined behavior: Eigen-decomposition of a bounded parameter: "
               << par.getFullTitle()
               << std::endl;
    ++configWarnings;
  }

  if( nbParameters == 0 ){
    LogAlert << "No parameter is enabled. Disabling the parameter set." << std::endl;
    _isEnabled_ = false;
    return;
  }

  if (configWarnings > 0) {
    LogWarning << "Undefined behavior: Using bounded parameters with eigendecomposition"
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

  _priorCovarianceMatrix_ = std::make_shared<TMatrixDSym>(nbParameters);
  int iStrippedPar = -1;
  for( int iPar = 0 ; iPar < int(_parameterList_.size()) ; iPar++ ){
    if( not ParameterSet::isValidCorrelatedParameter(_parameterList_[iPar]) ) continue;
    iStrippedPar++;
    int jStrippedPar = -1;
    for( int jPar = 0 ; jPar < int(_parameterList_.size()) ; jPar++ ){
      if( not ParameterSet::isValidCorrelatedParameter(_parameterList_[jPar]) ) continue;
      jStrippedPar++;
      (*_priorCovarianceMatrix_)[iStrippedPar][jStrippedPar] = (*_priorFullCovarianceMatrix_)[iPar][jPar];
    }
  }
  _deltaVectorPtr_ = std::make_shared<TVectorD>(_priorCovarianceMatrix_->GetNrows());

  LogExitIf(not _priorCovarianceMatrix_->IsSymmetric(),
            getName() << ":Covariance matrix is not symmetric");

  if( not isEnableEigenDecomp() ){
    LogInfo << "Computing inverse of the stripped covariance matrix: "
               << _priorCovarianceMatrix_->GetNcols() << "x"
               << _priorCovarianceMatrix_->GetNrows() << std::endl;
    _inverseCovarianceMatrix_ = std::shared_ptr<TMatrixD>((TMatrixD*)(_priorCovarianceMatrix_->Clone()));

    double det{-1};
    _inverseCovarianceMatrix_->Invert(&det);

    bool failed{false};
    if( det <= 0 ){
      _priorCovarianceMatrix_->Print();
      LogError << "Stripped covariance must be positive definite: " << det << std::endl;
      failed = true;
    }

    TVectorD eigenValues;

    if(_inverseCovarianceMatrix_->GetNrows() == 1) {
      eigenValues.ResizeTo(1);
      eigenValues[0] = (*_inverseCovarianceMatrix_)[0][0];
    }
    else {
      // https://root-forum.cern.ch/t/tmatrixt-get-eigenvalues/25924
      _inverseCovarianceMatrix_->EigenVectors(eigenValues);
    }

    if( eigenValues.Min() < 0 ){
      LogError << "Negative eigen values for prior cov matrix: " << eigenValues.Min() << std::endl;
      failed = true;
    }

    LogExitIf(failed, "Failed inverting prior covariance matrix of par set: " << getName() );

  }
  else {
    LogInfo << "Decomposing the stripped covariance matrix..." << std::endl;
    _eigenParameterList_.resize(_priorCovarianceMatrix_->GetNrows(), Parameter(this));

    LogAlertIf(_priorCovarianceMatrix_->GetNrows() > 1000) << "Decomposing matrix with " << _priorCovarianceMatrix_->GetNrows() << " dim might take a while..." << std::endl;
    _eigenDecomp_     = std::make_shared<TMatrixDSymEigen>(*_priorCovarianceMatrix_);

    // Used for base swapping
    _eigenValues_     = std::shared_ptr<TVectorD>( (TVectorD*) _eigenDecomp_->GetEigenValues().Clone() );
    _eigenValuesInv_  = std::shared_ptr<TVectorD>( (TVectorD*) _eigenDecomp_->GetEigenValues().Clone() );
    _eigenVectors_    = std::shared_ptr<TMatrixD>( (TMatrixD*) _eigenDecomp_->GetEigenVectors().Clone() );
    _eigenVectorsInv_ = std::make_shared<TMatrixD>(TMatrixD::kTransposed, *_eigenVectors_ );

    if( not std::isnan(_eigenSvdThreshold_)
        and _eigenValues_->Min()/_eigenValues_->Max() < _eigenSvdThreshold_ ){
      // zero the ruled out eigen values
      int removed{0};
      for( int iEigen = 0; iEigen < _eigenValues_->GetNrows(); iEigen++ ){
        double rat = (*_eigenValues_)[iEigen]/_eigenValues_->Max();
        if( rat < _eigenSvdThreshold_ ){
          LogAlert << "Eigenvalue " << iEigen
                   << " below eigenSvdThreshold:"
                   << " Eigenvalue: " << (*_eigenValues_)[iEigen]
                   << "/" << _eigenValues_->Max()
                   << " (" << rat << " < " << _eigenSvdThreshold_ << ")"
                   << std::endl;
          (*_eigenValues_)[iEigen] = 0;
          ++removed;
        }
      }
      if (removed > 0) {
        LogAlert << "Eigen values below the threshold(" << _eigenSvdThreshold_
                 << "): " << removed << " and set to zero."
                 << std::endl;
      }
    }

    // In any case the eigen values should have been cleaned up
    LogInfo << "Covariance eigen values are between " << _eigenValues_->Min() << " and " << _eigenValues_->Max() << std::endl;
    LogExitIf(_eigenValues_->Min() < 0, "Input covariance matrix is not positive definite.");

    _nbEnabledEigen_ = 0;
    double eigenTotal = _eigenValues_->Sum();

    _inverseCovarianceMatrix_ = std::make_shared<TMatrixD>(_priorCovarianceMatrix_->GetNrows(), _priorCovarianceMatrix_->GetNrows());
    _projectorMatrix_         = std::make_shared<TMatrixD>(_priorCovarianceMatrix_->GetNrows(), _priorCovarianceMatrix_->GetNrows());

    auto eigenState = std::make_unique<TVectorD>(_eigenValues_->GetNrows());

    for( int iEigen = 0; iEigen < _eigenValues_->GetNrows(); iEigen++ ){

      _eigenParameterList_[iEigen].setParameterIndex( iEigen );
      _eigenParameterList_[iEigen].setIsEnabled(true);
      _eigenParameterList_[iEigen].setIsEigen(true);
      _eigenParameterList_[iEigen].setStdDevValue(std::sqrt((*_eigenValues_)[iEigen]));
      _eigenParameterList_[iEigen].setStepSize(std::sqrt((*_eigenValues_)[iEigen]));
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
        // ruled out values have been set to 0
        if( (*_eigenValues_)[iEigen] <= 0 ){
          LogAlert << "Keeping " << iEigen << " positive eigen values."
                   << std::endl;
          break; // as they are in decreasing order
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

    auto eigenStateMatrix = std::unique_ptr<TMatrixD>(GenericToolbox::makeDiagonalMatrix(eigenState.get()));
    auto diagInvMatrix = std::unique_ptr<TMatrixD>(GenericToolbox::makeDiagonalMatrix(_eigenValuesInv_.get()));

    (*_projectorMatrix_) =  (*_eigenVectors_);
    (*_projectorMatrix_) *= (*eigenStateMatrix);
    (*_projectorMatrix_) *= (*_eigenVectorsInv_);

    (*_inverseCovarianceMatrix_) =  (*_eigenVectors_);
    (*_inverseCovarianceMatrix_) *= (*diagInvMatrix);
    (*_inverseCovarianceMatrix_) *= (*_eigenVectorsInv_);

    LogInfo << "Eigen decomposition with " << _nbEnabledEigen_ << " / " << _eigenValues_->GetNrows() << " vectors" << std::endl;
    if(_nbEnabledEigen_ != _eigenValues_->GetNrows() ){
      LogInfo << "Max eigen fraction set to " << _maxEigenFraction_*100 << "%" << std::endl;
      LogInfo << "Fraction taken: " << eigenCumulative / eigenTotal*100 << "%" << std::endl;
    }

    _originalParBuffer_ = std::make_shared<TVectorD>(_priorCovarianceMatrix_->GetNrows() );
    _eigenParBuffer_    = std::make_shared<TVectorD>(_priorCovarianceMatrix_->GetNrows() );

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
        eigenPar.setLimits( _parameterList_[eigenPar.getParameterIndex()].getParameterLimits() );
      }
      else{
        eigenPar.setLimits( _eigenParRange_ );
      }
      if ( not eigenPar.getParameterLimits().isInBounds(eigenPar.getPriorValue()) ) {
        LogError << "Prior for eigen parameter is out of bounds: "
                 << eigenPar.getSummary()
                 << std::endl;
        LogExit("Eigenparameter prior is out of bounds");
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
  LogInfo << "Set parameter set validity to " << validity << std::endl;
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

  TVectorD throwsList;
  if (_priorCovarianceMatrix_) {
    throwsList.ResizeTo(_priorCovarianceMatrix_->GetNrows());
  }

  // generic function to handle multiple throws
  std::function<void(std::function<void()>)> throwParsFct =
      [&](const std::function<void()>& throwFct_){

        LogInfoIf(gain_!=1) << "Throw gain is " << gain_ << std::endl;

        int nTries{0};
        while( true ){
          ++nTries;

          if( nTries > 1000 ){
            LogExit("Failed to find valid throw");
          }

          throwFct_();

          // assuming
          int nThrowsOutOfBounds{0};
          LogInfo << "Check that thrown parameters are within bounds..." << std::endl;

          // throws with this function are always done in real space.
          int iFit{-1};
          for( auto& par : this->getParameterList() ){
            if( ParameterSet::isValidCorrelatedParameter(par) ){
              iFit++;
              if (par.isThrown()) {
                  par.setThrowValue( par.getPriorValue() + gain_*throwsList[iFit] );
              }
            }
            else if (par.isThrown() and par.isFree()) {
              double aMin = par.getThrowLimits().min;
              double aMax = par.getThrowLimits().max;
              if (not std::isfinite(aMin)) {
                aMin = par.getPriorValue() - gain_*par.getStepSize();
                LogWarning << "Free parameter " << par.getName()
                         << " thrown without lower bound (new: " << aMin << ")"
                         << std::endl;
              }
              if (not std::isfinite(aMax)) {
                aMax = par.getPriorValue() + gain_*par.getStepSize();
                LogWarning << "Free parameter " << par.getName()
                         << " thrown without upper bound (new: " << aMax << ")"
                         << std::endl;
              }
              par.setThrowValue(gRandom->Uniform(aMin,aMax));
            }
            else if (par.isThrown()) {
              double stdDev = par.getStdDevValue();
              if (not std::isfinite(stdDev) and stdDev <= 0) {
                // No standard deviation, so use the step size.
                stdDev = par.getStepSize();
              }
              if (std::isfinite(stdDev) and stdDev > 0) {
                par.setThrowValue(par.getPriorValue()
                                  + gRandom->Gaus(0, gain_*stdDev));
              }
              else {
                LogError << "Thrown parameter " << par.getName()
                         << " without standard deviation"
                         << std::endl;
                LogExit("Bad thrown parameter");
              }
            }
            if( not par.getThrowLimits().isInBounds(par.getThrowValue()) ){
              nThrowsOutOfBounds++;
              LogWarning << par.getTitle() << " was thrown out of limits: "
                       << par.getThrowValue() << " -> " << par.getThrowLimits() << std::endl;
            }
          }

          if( nThrowsOutOfBounds != 0 ){
            LogAlert << nThrowsOutOfBounds << " parameter were found out of set limits."
            << " Rethrowing \"" << this->getName() << "\"... try #" << nTries+1 << std::endl;
            continue;
          }

          // Throw looks OK, so set the parameter values.
          LogInfoIf(nTries!=1) << "Keeping throw after " << nTries << " tries." << std::endl;
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
          GenericToolbox::TablePrinter t;
          t.setColTitles({{"Parameter"}, {"Prior"}, {"StdDev"}, {"ThrowLimits"}, {"Thrown"}});
          for( auto& par : _parameterList_ ){
            if(par.isThrown() and ParameterSet::isValidCorrelatedParameter(par) ){
              t.addTableLine({
                par.getTitle(),
                std::to_string(par.getPriorValue()),
                std::to_string(par.getStdDevValue()),
                par.getThrowLimits().toString(),
                std::to_string(par.getThrowValue())
              });
            }
            else if (par.isThrown()) {
              double sd = par.getStdDevValue();
              if (not std::isfinite(sd) or sd <= 0) sd = par.getStepSize();
              std::string tmp = std::to_string(sd);
              if (par.isFree()) tmp = "free";
              t.addTableLine({
                par.getTitle(),
                std::to_string(par.getPriorValue()),
                tmp,
                par.getThrowLimits().toString(),
                std::to_string(par.getThrowValue())
                });
            }
          }
          t.printTable();

          if( isEnableEigenDecomp() ){
            LogInfo << "Translated to eigen space:" << std::endl;
            t.reset();
            t.setColTitles({{"Eigen"}, {"Prior"}, {"StdDev"}, {"ThrowLimits"}, {"Thrown"}});
            for( auto& eigenPar : _eigenParameterList_ ){
              t.addTableLine({
                eigenPar.getTitle(),
                std::to_string(eigenPar.getPriorValue()),
                std::to_string(eigenPar.getStdDevValue()),
                eigenPar.getThrowLimits().toString(),
                std::to_string(eigenPar.getThrowValue())
              });
            }
            t.printTable();
          }
          break;
        }
      }; // End of generic function handling multiple throws

  if( _useMarkGenerator_ ){
    // Throw using an alternative method that was copied from BANFF
    LogAlert << "Throwing parameters for " << _name_
             << " using alternate generator: Mark Hartz Generator"
             << std::endl;

    int iPar{0};
    for( auto& par : _parameterList_ ){
      if( ParameterSet::isValidCorrelatedParameter(par) ){ throwsList[iPar++] = par.getPriorValue(); }
    }

    if( _markHartzGen_ == nullptr ){
      LogInfo << "Generating Cholesky matrix in set: " << getName() << std::endl;
      _markHartzGen_ = std::make_shared<ParameterThrowerMarkHarz>(throwsList, *_priorCovarianceMatrix_);
    }
    // TVectorD throws(_priorCovarianceMatrix_->GetNrows());

    std::vector<double> throwPars;
    std::function<void()> markScottThrowFct = [&](){
      if ( _markHartzGen_ == nullptr) return;

      throwPars.resize(_priorCovarianceMatrix_->GetNrows());

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
    // Throw using eigen value decomposition.  This will work with degenerate
    // "covariance matrices", even if they are not positive definite (ouch),
    // so it provides an alternative to Cholesky decomposition, but is not the
    // best throwing method.
    LogAlert << "Throwing parameters for " << _name_
             << " using alternate generator: Eigen Decomposition Generator"
             << std::endl;

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
          if( not par.getParameterLimits().isInBounds(par.getParameterValue()) ) {
            throwIsValid = false;
            LogAlert << "Thrown value not within limits -> "
                     << par.getParameterValue()
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

      if( not _correlatedVariableThrower_.isInitialized()
          and _priorCovarianceMatrix_ != nullptr){
        _correlatedVariableThrower_.setCovarianceMatrixPtr(_priorCovarianceMatrix_.get());
        _correlatedVariableThrower_.initialize();
        int iFit{-1};
        for( auto& par : this->getParameterList() ){
          if( ParameterSet::isValidCorrelatedParameter(par) ){
            iFit++;
            _correlatedVariableThrower_.getParLimitList().at(iFit) = par.getThrowLimits();
            // cancel the prior, thrown values are centered around 0
            _correlatedVariableThrower_.getParLimitList().at(iFit) -= par.getPriorValue();
          }
        }
        _correlatedVariableThrower_.setNbMaxTries(100000); // could be a user parameter
        _correlatedVariableThrower_.extractBlocks();
      }

      std::function<void()> gundamThrowFct = [&](){
        if ( _priorCovarianceMatrix_ == nullptr) return;
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
        t.setColTitles({ {"Title"}, {"Current"}, {"Prior"}, {"StdDev"}, {"Limits"}, {"Property"} });

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
                             par.getParameterLimits().toString(),
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
  LogInfo << "Importing parameters from config for \"" << this->getName() << "\"" << std::endl;

  auto config = ConfigUtils::getForwardedConfig(config_);
  LogExitIf( config.empty(), "Invalid injector config" << std::endl << config_ );
  LogExitIf( not GenericToolbox::Json::doKeyExist(config, "name"), "No parameter set name provided in" << std::endl << config_ );
  LogExitIf( GenericToolbox::Json::fetchValue<std::string>(config, "name") != this->getName(),
              "Mismatching between parSet name (" << this->getName() << ") and injector config ("
              << GenericToolbox::Json::fetchValue<std::string>(config, "name") << ")" );

  auto parValues = GenericToolbox::Json::fetchValue( config, "parameterValues", JsonType() );
  if     ( parValues.empty() ) {
    LogExit( "No parameter values provided." );
  }
  else if( parValues.is_string() ){
    //
    LogInfo << "Reading parameter values from file: " << parValues.get<std::string>() << std::endl;
    auto parList = GenericToolbox::dumpFileAsVectorString( parValues.get<std::string>(), true );
    LogExitIf( parList.size() != this->getNbParameters()  ,
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
        LogExitIf(parPtr == nullptr, "Could not find " << parName << " among the defined parameters in " << this->getName());


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
        LogExitIf(parPtr == nullptr, "Could not find " << parTitle << " among the defined parameters in " << this->getName());


        if( not parPtr->isEnabled() ){
          LogAlert << "NOT injecting \"" << parPtr->getFullTitle() << "\" as it is disabled." << std::endl;
          continue;
        }

        LogInfo << "Injecting \"" << parPtr->getFullTitle() << "\": " << GenericToolbox::Json::fetchValue<double>(parValueEntry, "value") << std::endl;
        parPtr->setParameterValue( GenericToolbox::Json::fetchValue<double>(parValueEntry, "value") );
      }
      else if( GenericToolbox::Json::doKeyExist(parValueEntry, "index") ){
        auto parIndex = GenericToolbox::Json::fetchValue<int>(parValueEntry, "index");
        LogExitIf( parIndex < 0 or parIndex >= this->getParameterList().size(),
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
        LogExit("Unsupported: " << parValueEntry);
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

  GenericToolbox::TablePrinter t;
  t << "Title" << GenericToolbox::TablePrinter::NextColumn;
  t << "Prior" << GenericToolbox::TablePrinter::NextColumn;
  t << "StdDev" << GenericToolbox::TablePrinter::NextColumn;
  t << "Limits" << GenericToolbox::TablePrinter::NextLine;

  int nPars{0};
  for( auto& par : _parameterList_ ){
    if( not par.isEnabled() ){ continue; }
    t << par.getTitle() << GenericToolbox::TablePrinter::NextColumn;
    t << par.getPriorValue() << GenericToolbox::TablePrinter::NextColumn;
    t << (par.getPriorType() == Parameter::PriorType::Flat ? "Free": std::to_string(par.getStdDevValue())) << GenericToolbox::TablePrinter::NextColumn;
    t << par.getParameterLimits() << GenericToolbox::TablePrinter::NextLine;
    nPars++;
  }

  LogInfo << getName() << " has " << nPars << " defined parameters:" << std::endl;
  t.printTable();

}


// Protected
void ParameterSet::readParameterDefinitionFile(){
  // new generalised way of defining parameters:
  // path can be set as: /path/to/rootfile.root:folder/in/tfile/object

  std::string parDefFilePathPrefix{};

  // legacy option
  if( not _parameterDefinitionFilePath_.empty() ){ parDefFilePathPrefix = _parameterDefinitionFilePath_ + ":"; }

  if(not _covarianceMatrixPath_.empty()){ GenericToolbox::fetchObject(parDefFilePathPrefix+_covarianceMatrixPath_, _priorFullCovarianceMatrix_); }
  if(not _parameterPriorValueListPath_.empty()){ GenericToolbox::fetchObject(parDefFilePathPrefix+_parameterPriorValueListPath_, _parameterPriorList_); }
  if(not _parameterNameListPath_.empty()){ GenericToolbox::fetchObject(parDefFilePathPrefix+_parameterNameListPath_, _parameterNamesList_); }
  if(not _parameterLowerBoundsTVectorD_.empty()){ GenericToolbox::fetchObject(parDefFilePathPrefix+_parameterLowerBoundsTVectorD_, _parameterLowerBoundsList_); }
  if(not _parameterUpperBoundsTVectorD_.empty()){ GenericToolbox::fetchObject(parDefFilePathPrefix+_parameterUpperBoundsTVectorD_, _parameterUpperBoundsList_); }
  if(not _throwEnabledListPath_.empty()){ GenericToolbox::fetchObject(parDefFilePathPrefix+_throwEnabledListPath_, _throwEnabledList_); }


  // setups
  if( _priorFullCovarianceMatrix_ != nullptr ){
    _nbParameterDefinition_ = _priorFullCovarianceMatrix_->GetNrows();
  }

  // sanity checks
  LogExitIf(_parameterPriorList_ != nullptr and _parameterPriorList_->GetNrows() != _nbParameterDefinition_,
             "Parameter prior list don't have the same size(" << _parameterPriorList_->GetNrows()
              << ") as cov matrix(" << _nbParameterDefinition_ << ")");
  LogExitIf(_parameterNamesList_ != nullptr and _parameterNamesList_->GetEntries() != _nbParameterDefinition_,
             "_parameterNamesList_ don't have the same size(" << _parameterNamesList_->GetEntries()
                                                              << ") as cov matrix(" << _nbParameterDefinition_ << ")" );
  LogExitIf(_parameterLowerBoundsList_ != nullptr and _parameterLowerBoundsList_->GetNrows() != _nbParameterDefinition_,
             "Parameter lower bound list don't have the same size(" << _parameterLowerBoundsList_->GetNrows()
                                                              << ") as cov matrix(" << _nbParameterDefinition_ << ")" );
  LogExitIf(_parameterUpperBoundsList_ != nullptr and _parameterUpperBoundsList_->GetNrows() != _nbParameterDefinition_,
             "Parameter upper bound list don't have the same size(" << _parameterUpperBoundsList_->GetNrows()
                                                              << ") as cov matrix(" << _nbParameterDefinition_ << ")" );
  LogExitIf(_throwEnabledList_ != nullptr and _throwEnabledList_->GetNrows() != _nbParameterDefinition_,
             "Throw enabled list don't have the same size(" << _throwEnabledList_->GetNrows()
                                                              << ") as cov matrix(" << _nbParameterDefinition_ << ")" );
}
void ParameterSet::defineParameters(){
  _parameterList_.resize(_nbParameterDefinition_, Parameter(this));
  int parIndex{0};

  if (_parameterList_.size() < 1) {
    LogError << "CONFIG ERROR: Parameter set \"" << getName() << "\"<< defined without any parameters" << std::endl;
  }

  for( auto& par : _parameterList_ ){
    par.setParameterIndex(parIndex++);

    if( _priorFullCovarianceMatrix_ != nullptr ){
      par.setStdDevValue(std::sqrt((*_priorFullCovarianceMatrix_)[par.getParameterIndex()][par.getParameterIndex()]));
      par.setStepSize(std::sqrt((*_priorFullCovarianceMatrix_)[par.getParameterIndex()][par.getParameterIndex()]));
    }
    else{
      // stdDev will only be used for display purpose
      par.setStdDevValue(_nominalStepSize_);
      par.setStepSize(_nominalStepSize_);
      par.setPriorType(Parameter::PriorType::Flat);
      par.setIsFree(true);
    }

    if( _parameterNamesList_ != nullptr ){ par.setName(_parameterNamesList_->At(par.getParameterIndex())->GetName()); }

    // par is now fully identifiable.
    if( not _enableOnlyParameters_.empty() ){
      bool isEnabled = false;
      for( auto& enableEntry : _enableOnlyParameters_ ){
        enableEntry.clearFields();
        enableEntry.defineFields({{"name"}});
        if( enableEntry.hasField("name")
            and par.getName() == enableEntry.fetchValue<std::string>("name") ){
          isEnabled = true;
          break;
        }
      }

      if( not isEnabled ){
        // set it off
        par.setIsEnabled( false );
      }
      else{
        LogDebugIf(GundamGlobals::isDebug()) << "Enabling parameter \"" << par.getFullTitle() << "\" as it is set in enableOnlyParameters" << std::endl;
      }
    }
    if( not _disableParameters_.empty() ){
      bool isEnabled = true;
      for( auto& disableEntry : _disableParameters_ ){
        disableEntry.clearFields();
        disableEntry.defineFields({{"name"}});
        if( disableEntry.hasField("name")
            and par.getName() == disableEntry.fetchValue<std::string>("name") ){
          isEnabled = false;
          break;
        }
      }

      if( not isEnabled ){
        LogDebugIf(GundamGlobals::isDebug()) << "Skipping parameter \"" << par.getFullTitle() << "\" as it is set in disableParameters" << std::endl;
        par.setIsEnabled( false );
        continue;
      }
    }

    if( _parameterPriorList_ != nullptr ){ par.setPriorValue((*_parameterPriorList_)[par.getParameterIndex()]); }
    else{ par.setPriorValue(1); }

    par.setParameterValue(par.getPriorValue());

    par.setLimits(_globalParRange_);

    GenericToolbox::Range rootBounds{};
    if( _parameterLowerBoundsList_ != nullptr ){ rootBounds.min = ((*_parameterLowerBoundsList_)[par.getParameterIndex()]); }
    if( _parameterUpperBoundsList_ != nullptr ){ rootBounds.max = ((*_parameterUpperBoundsList_)[par.getParameterIndex()]); }
    par.setLimits( rootBounds );

    LogExitIf( not par.getParameterLimits().isInBounds(par.getPriorValue()), "PRIOR IS NOT IN BOUNDS: " << par.getSummary() );

    if( not _parameterDefinitionConfig_.empty() ){
      // Alternative 1: define dials then parameters
      if (_parameterNamesList_ != nullptr) {
        // Find the parameter using the name from the vector of names for
        // the covariance.

        ConfigReader selectedParConfig;

        // search with name
        std::string parName = _parameterNamesList_->At(par.getParameterIndex())->GetName();
        for( auto& parConfig : _parameterDefinitionConfig_.loop() ){
          parConfig.defineFields({
            {"name", {"parameterName"}},
          });
          if( parConfig.hasField("name") ){
            if( parName == parConfig.fetchValue<std::string>("name") ){
              selectedParConfig = parConfig;
              break;
            }
          }
        }

        // not found? try with the index
        if( selectedParConfig.empty() ){
          for( auto& parConfig : _parameterDefinitionConfig_.loop() ){
            parConfig.defineFields({{"parameterIndex"}});
            if( parConfig.hasField("parameterIndex") ){
              if( par.getParameterIndex() == parConfig.fetchValue<int>("parameterIndex") ){
                selectedParConfig = parConfig;
                break;
              }
            }
          }
        }

        par.setConfig( selectedParConfig );
      }
      else{
        // No covariance provided, so find the name based on the order in
        // the parameter set.
        auto configVector = _parameterDefinitionConfig_.loop();
        LogThrowIf(configVector.size() <= par.getParameterIndex(),
                   "Parameter index out of range");
        auto& parConfig = configVector.at(par.getParameterIndex());
        par.setConfig( parConfig );

        LogWarning << "Parameter #" << par.getParameterIndex()
                   << " not defined by covariance matrix file"
                   << std::endl;
      }
    }
    else if( not _dialSetDefinitions_.empty() ){
      // Alternative 2: define dials then parameters
      par.setDialSetConfig( _dialSetDefinitions_.loop() );
    }
    par.configure();
  }
}
