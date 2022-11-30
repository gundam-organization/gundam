//
// Created by Nadrino on 21/05/2021.
//

#include "FitParameterSet.h"

#include <memory>
#include "JsonUtils.h"
#include "GlobalVariables.h"

#include "GenericToolbox.h"
#include "GenericToolbox.Root.h"
#include "GenericToolbox.TablePrinter.h"
#include "Logger.h"

LoggerInit([]{
  Logger::setUserHeaderStr("[FitParameterSet]");
} );


void FitParameterSet::readConfigImpl(){
  LogThrowIf(_config_.empty(), "FitParameterSet config not set.");

  _name_ = JsonUtils::fetchValue<std::string>(_config_, "name");
  LogInfo << "Initializing parameter set: " << _name_ << std::endl;

  _isEnabled_ = JsonUtils::fetchValue<bool>(_config_, "isEnabled");
  LogReturnIf(not _isEnabled_, _name_ << " parameters are disabled.");

  _nbParameterDefinition_ = JsonUtils::fetchValue(_config_, "numberOfParameters", _nbParameterDefinition_);
  _nominalStepSize_ = JsonUtils::fetchValue(_config_, "nominalStepSize", _nominalStepSize_);

  _useOnlyOneParameterPerEvent_ = JsonUtils::fetchValue<bool>(_config_, "useOnlyOneParameterPerEvent", false);
  _printDialSetsSummary_ = JsonUtils::fetchValue<bool>(_config_, "printDialSetsSummary", _printDialSetsSummary_);
  _printParametersSummary_ = JsonUtils::fetchValue<bool>(_config_, "printParametersSummary", _printDialSetsSummary_);

  if( JsonUtils::doKeyExist(_config_, "parameterLimits") ){
    auto parLimits = JsonUtils::fetchValue(_config_, "parameterLimits", nlohmann::json());
    _globalParameterMinValue_ = JsonUtils::fetchValue(parLimits, "minValue", std::nan("UNSET"));
    _globalParameterMaxValue_ = JsonUtils::fetchValue(parLimits, "maxValue", std::nan("UNSET"));
  }

  _useEigenDecompInFit_ = JsonUtils::fetchValue(_config_ , "useEigenDecompInFit", false);
  if( _useEigenDecompInFit_ ){
    LogWarning << "Using eigen decomposition in fit." << std::endl;
    _maxNbEigenParameters_ = JsonUtils::fetchValue(_config_ , "maxNbEigenParameters", -1);
    if( _maxNbEigenParameters_ != -1 ){
      LogInfo << "Maximum nb of eigen parameters is set to " << _maxNbEigenParameters_ << std::endl;
    }
    _maxEigenFraction_ = JsonUtils::fetchValue(_config_ , "maxEigenFraction", double(1.));
    if( _maxEigenFraction_ != 1 ){
      LogInfo << "Max eigen fraction set to: " << _maxEigenFraction_*100 << "%" << std::endl;
    }
  }

  _enablePca_ = JsonUtils::fetchValue(_config_, std::vector<std::string>{"fixGhostFitParameters", "enablePca"}, _enablePca_);
  _enabledThrowToyParameters_ = JsonUtils::fetchValue(_config_, "enabledThrowToyParameters", _enabledThrowToyParameters_);
  _customFitParThrow_ = JsonUtils::fetchValue(_config_, "customFitParThrow", std::vector<nlohmann::json>());
  _releaseFixedParametersOnHesse_ = JsonUtils::fetchValue(_config_, "releaseFixedParametersOnHesse", _releaseFixedParametersOnHesse_);

  _parameterDefinitionFilePath_ = JsonUtils::fetchValue( _config_,
    {{"parameterDefinitionFilePath"}, {"covarianceMatrixFilePath"} }, _parameterDefinitionFilePath_
  );
  _covarianceMatrixTMatrixD_ = JsonUtils::fetchValue(_config_, "covarianceMatrixTMatrixD", _covarianceMatrixTMatrixD_);
  _parameterPriorTVectorD_ = JsonUtils::fetchValue(_config_, "parameterPriorTVectorD", _parameterPriorTVectorD_);
  _parameterNameTObjArray_ = JsonUtils::fetchValue(_config_, "parameterNameTObjArray", _parameterNameTObjArray_);
  _parameterLowerBoundsTVectorD_ = JsonUtils::fetchValue(_config_, "parameterLowerBoundsTVectorD", _parameterLowerBoundsTVectorD_);
  _parameterUpperBoundsTVectorD_ = JsonUtils::fetchValue(_config_, "parameterUpperBoundsTVectorD", _parameterUpperBoundsTVectorD_);
  _throwEnabledListPath_ = JsonUtils::fetchValue(_config_, "throwEnabledList", _throwEnabledListPath_);

  _parameterDefinitionConfig_ = JsonUtils::fetchValue(_config_, "parameterDefinitions", _parameterDefinitionConfig_);
  _dialSetDefinitions_ = JsonUtils::fetchValue(_config_, "dialSetDefinitions", _dialSetDefinitions_);


  // MISC / DEV

  _devUseParLimitsOnEigen_ = JsonUtils::fetchValue(_config_, "devUseParLimitsOnEigen", _devUseParLimitsOnEigen_);
  if( _devUseParLimitsOnEigen_ ){
    LogAlert << "USING DEV OPTION: _devUseParLimitsOnEigen_ = true" << std::endl;
  }



  this->readParameterDefinitionFile();

  if( _nbParameterDefinition_ == -1 ){
    LogWarning << "No number of parameter provided. Looking for alternative definitions..." << std::endl;

    if( not _dialSetDefinitions_.empty() ){
      for( auto& dialSetDef : _dialSetDefinitions_.get<std::vector<nlohmann::json>>() ){
        if( JsonUtils::doKeyExist(dialSetDef, "parametersBinningPath") ){
          LogInfo << "Found parameter binning within dialSetDefinition. Defining parameters number..." << std::endl;
          DataBinSet b;
          b.readBinningDefinition( JsonUtils::fetchValue<std::string>(dialSetDef, "parametersBinningPath") );
          _nbParameterDefinition_ = int(b.getBinsList().size());
          break;
        }
      }
    }

    if( _nbParameterDefinition_ == -1 and not _parameterDefinitionConfig_.empty() ){
      LogInfo << "Using parameter definition config list to determine the number of parameters..." << std::endl;
      _nbParameterDefinition_ = int(_parameterDefinitionConfig_.get<std::vector<nlohmann::json>>().size());
    }

    LogThrowIf(_nbParameterDefinition_==-1, "Could not figure out the number of parameters to be defined for the set: " << _name_ );
  }

  this->defineParameters();
}
void FitParameterSet::initializeImpl() {
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

void FitParameterSet::setMaskedForPropagation(bool maskedForPropagation) {
  _maskedForPropagation_ = maskedForPropagation;
}

void FitParameterSet::processCovarianceMatrix(){

  if( _priorCovarianceMatrix_ == nullptr ){ return; } // nothing to do

  LogInfo << "Stripping the matrix from fixed/disabled parameters in set: " << getName() << std::endl;
  int nbFitParameters{0};
  for( const auto& par : _parameterList_ ){
    if( par.isEnabled() and not par.isFixed() and not par.isFree() ) nbFitParameters++;
  }
  LogInfo << "Effective nb parameters: " << nbFitParameters << std::endl;

  _strippedCovarianceMatrix_ = std::make_shared<TMatrixDSym>(nbFitParameters);
  int iStrippedPar = -1;
  for( int iPar = 0 ; iPar < int(_parameterList_.size()) ; iPar++ ){
    if( not _parameterList_[iPar].isEnabled() or _parameterList_[iPar].isFixed() or _parameterList_[iPar].isFree() ) continue;
    iStrippedPar++;
    int jStrippedPar = -1;
    for( int jPar = 0 ; jPar < int(_parameterList_.size()) ; jPar++ ){
      if( not _parameterList_[jPar].isEnabled() or _parameterList_[jPar].isFixed() or _parameterList_[jPar].isFree() ) continue;
      jStrippedPar++;
      (*_strippedCovarianceMatrix_)[iStrippedPar][jStrippedPar] = (*_priorCovarianceMatrix_)[iPar][jPar];
    }
  }
  _deltaParameterList_ = std::make_shared<TVectorD>(_strippedCovarianceMatrix_->GetNrows());

  if( not _useEigenDecompInFit_ ){
    LogWarning << "Computing inverse of the stripped covariance matrix: "
               << _strippedCovarianceMatrix_->GetNcols() << "x"
               << _strippedCovarianceMatrix_->GetNrows() << std::endl;
    _inverseStrippedCovarianceMatrix_ = std::shared_ptr<TMatrixD>((TMatrixD*)(_strippedCovarianceMatrix_->Clone()));
    _inverseStrippedCovarianceMatrix_->Invert();
  }
  else {
    LogWarning << "Decomposing the stripped covariance matrix in set: " << getName() << std::endl;
    _eigenParameterList_.resize(_strippedCovarianceMatrix_->GetNrows(), FitParameter(this));

    _eigenDecomp_     = std::make_shared<TMatrixDSymEigen>(*_strippedCovarianceMatrix_);

    // Used for base swapping
    _eigenValues_     = std::shared_ptr<TVectorD>( (TVectorD*) _eigenDecomp_->GetEigenValues().Clone() );
    _eigenValuesInv_  = std::shared_ptr<TVectorD>( (TVectorD*) _eigenDecomp_->GetEigenValues().Clone() );
    _eigenVectors_    = std::shared_ptr<TMatrixD>( (TMatrixD*) _eigenDecomp_->GetEigenVectors().Clone() );
    _eigenVectorsInv_ = std::make_shared<TMatrixD>(TMatrixD::kTransposed, *_eigenVectors_ );

    double eigenCumulative = 0;
    _nbEnabledEigen_ = 0;
    double eigenTotal = _eigenValues_->Sum();

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
    propagateOriginalToEigen();

    // Tag the prior
    for( auto& eigenPar : _eigenParameterList_ ){
      eigenPar.setCurrentValueAsPrior();

      if( _devUseParLimitsOnEigen_ ){
        eigenPar.setMinValue( _parameterList_[eigenPar.getParameterIndex()].getMinValue() );
        eigenPar.setMaxValue( _parameterList_[eigenPar.getParameterIndex()].getMaxValue() );

        LogThrowIf( not std::isnan(eigenPar.getMinValue()) and eigenPar.getPriorValue() < eigenPar.getMinValue(), "PRIOR IS BELLOW MIN: " << eigenPar.getSummary(true) );
        LogThrowIf( not std::isnan(eigenPar.getMaxValue()) and eigenPar.getPriorValue() > eigenPar.getMaxValue(), "PRIOR IS ABOVE MAX: " << eigenPar.getSummary(true) );
      }
    }

  }

}

// Getters
bool FitParameterSet::isEnabled() const {
  return _isEnabled_;
}
bool FitParameterSet::isEnablePca() const {
  return _enablePca_;
}
bool FitParameterSet::isEnabledThrowToyParameters() const {
  return _enabledThrowToyParameters_;
}
const std::string &FitParameterSet::getName() const {
  return _name_;
}
bool FitParameterSet::isMaskedForPropagation() const {
  return _maskedForPropagation_;
}
std::vector<FitParameter> &FitParameterSet::getParameterList() {
  return _parameterList_;
}
std::vector<FitParameter> &FitParameterSet::getEigenParameterList(){
  return _eigenParameterList_;
}
const std::vector<FitParameter> &FitParameterSet::getParameterList() const{
  return _parameterList_;
}
std::vector<FitParameter>& FitParameterSet::getEffectiveParameterList(){
  if( _useEigenDecompInFit_ ) return _eigenParameterList_;
  return _parameterList_;
}
const std::vector<FitParameter>& FitParameterSet::getEffectiveParameterList() const{
  if( _useEigenDecompInFit_ ) return _eigenParameterList_;
  return _parameterList_;
}
const nlohmann::json &FitParameterSet::getDialSetDefinitions() const {
  return _dialSetDefinitions_;
}

// Core
size_t FitParameterSet::getNbParameters() const {
  return _parameterList_.size();
}
double FitParameterSet::getPenaltyChi2() {

  if (not _isEnabled_) { return 0; }

  double chi2 = 0;

  if( _priorCovarianceMatrix_ != nullptr ){
    if( _useEigenDecompInFit_ ){
      for( const auto& eigenPar : _eigenParameterList_ ){
        if( eigenPar.isFixed() ) continue;
        chi2 += TMath::Sq( (eigenPar.getParameterValue() - eigenPar.getPriorValue()) / eigenPar.getStdDevValue() ) ;
      }
    }
    else{
      // make delta vector
      this->fillDeltaParameterList();

      // compute penalty term with covariance
      chi2 = (*_deltaParameterList_) * ( (*_inverseStrippedCovarianceMatrix_) * (*_deltaParameterList_) );
    }
  }

  return chi2;
}

// Parameter throw
void FitParameterSet::moveFitParametersToPrior(){
  LogInfo << "Moving back fit parameters to their prior value in set: " << getName() << std::endl;

  if( not _useEigenDecompInFit_ ){
    for( auto& par : _parameterList_ ){
      if( par.isFixed() ){ continue; }
      par.setParameterValue(par.getPriorValue());
    }
  }
  else{
    for( auto& eigenPar : _eigenParameterList_ ){
      if( eigenPar.isFixed() ) continue;
      eigenPar.setParameterValue(eigenPar.getPriorValue());
    }
    this->propagateEigenToOriginal();
  }

}
void FitParameterSet::throwFitParameters(double gain_){

  LogThrowIf(_strippedCovarianceMatrix_==nullptr, "No covariance matrix provided");

//  if( not _useEigenDecompInFit_ ){
    LogInfo << "Throwing parameters for " << _name_ << " using Cholesky matrix" << std::endl;

    if( _choleskyMatrix_ == nullptr ){
      LogInfo << "Generating Cholesky matrix in set: " << getName() << std::endl;
      _choleskyMatrix_ = std::shared_ptr<TMatrixD>(
          GenericToolbox::getCholeskyMatrix(_strippedCovarianceMatrix_.get())
      );
    }

    auto throws = GenericToolbox::throwCorrelatedParameters(_choleskyMatrix_.get());

    int iFit{-1};
    for( auto& par : _parameterList_ ){
      if( par.isEnabled() and not par.isFixed() and not par.isFree() ){
        iFit++;
        LogInfo << "Throwing par " << par.getTitle() << ": " << par.getParameterValue();
        par.setThrowValue(par.getPriorValue() + gain_ * throws[iFit]);
        par.setParameterValue( par.getThrowValue() );
        LogInfo << " → " << par.getParameterValue() << std::endl;
      }
      else{
        LogWarning << "Skipping parameter: " << par.getTitle() << std::endl;
      }
    }
//  }
//  else{
//    LogInfo << "Throwing eigen parameters for " << _name_ << std::endl;
//    for( auto& eigenPar : _eigenParameterList_ ){
//      if( eigenPar.isFixed() ){ LogWarning << "Eigen parameter #" << eigenPar.getParameterIndex() << " is fixed. Not throwing" << std::endl; continue; }
//      eigenPar.setThrowValue(eigenPar.getPriorValue() + gain_ * gRandom->Gaus(0, eigenPar.getStdDevValue()));
//      eigenPar.setParameterValue( eigenPar.getThrowValue() );
//    }
//    this->propagateEigenToOriginal();
//  }

  if( _useEigenDecompInFit_ ){
    this->propagateOriginalToEigen();
    for( auto& eigenPar : _eigenParameterList_ ){
      eigenPar.setThrowValue( eigenPar.getParameterValue() );
    }
  }

}
const std::vector<nlohmann::json>& FitParameterSet::getCustomFitParThrow() const{
  return _customFitParThrow_;
}

// Eigen
bool FitParameterSet::isUseEigenDecompInFit() const {
  return _useEigenDecompInFit_;
}
int FitParameterSet::getNbEnabledEigenParameters() const {
  return _nbEnabledEigen_;
}

const TMatrixD* FitParameterSet::getInvertedEigenVectors() const{
  return _eigenVectorsInv_.get();
}
const TMatrixD* FitParameterSet::getEigenVectors() const{
  return _eigenVectors_.get();
}
void FitParameterSet::propagateOriginalToEigen(){
  // First propagate to the buffer
  int iParOffSet{0};
  for( const auto& par : _parameterList_ ){
    if( par.isFixed() or not par.isEnabled() ) continue;
    (*_originalParBuffer_)[iParOffSet++] = par.getParameterValue();
  }

  // Base swap: ORIG -> EIGEN
  (*_eigenParBuffer_) = (*_originalParBuffer_);
  (*_eigenParBuffer_) *= (*_eigenVectorsInv_);

  // Propagate back to eigen parameters
  for( int iEigen = 0 ; iEigen < _eigenParBuffer_->GetNrows() ; iEigen++ ){
    _eigenParameterList_[iEigen].setParameterValue((*_eigenParBuffer_)[iEigen]);
  }
}
void FitParameterSet::propagateEigenToOriginal(){
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
std::string FitParameterSet::getSummary() const {
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

double FitParameterSet::toNormalizedParRange(double parRange, const FitParameter& par){
  return (parRange)/par.getStdDevValue();
}
double FitParameterSet::toNormalizedParValue(double parValue, const FitParameter& par) {
  return FitParameterSet::toNormalizedParRange(parValue - par.getPriorValue(), par);
}
double FitParameterSet::toRealParRange(double normParRange, const FitParameter& par){
  return normParRange*par.getStdDevValue();
}
double FitParameterSet::toRealParValue(double normParValue, const FitParameter& par) {
  return normParValue*par.getStdDevValue() + par.getPriorValue();
}


// Protected
void FitParameterSet::readParameterDefinitionFile(){

  if( _parameterDefinitionFilePath_.empty() ) return;

  std::unique_ptr<TFile> parDefFile(TFile::Open(_parameterDefinitionFilePath_.c_str()));
  LogThrowIf(parDefFile == nullptr or not parDefFile->IsOpen(), "Could not open: " << _parameterDefinitionFilePath_)

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
void FitParameterSet::defineParameters(){
  LogInfo << "Defining " << _nbParameterDefinition_ << " parameters for the set: " << getName() << std::endl;
  _parameterList_.resize(_nbParameterDefinition_, FitParameter(this));
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
      par.setPriorType(PriorType::Flat);
      par.setIsFree(true);
    }

    if( _parameterNamesList_ != nullptr ){ par.setName(_parameterNamesList_->At(par.getParameterIndex())->GetName()); }
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
      JsonUtils::forwardConfig(_parameterDefinitionConfig_);
      auto parConfig = JsonUtils::fetchMatchingEntry(_parameterDefinitionConfig_, "parameterName", std::string(_parameterNamesList_->At(par.getParameterIndex())->GetName()));
      if( parConfig.empty() ){
        // try with par index
        parConfig = JsonUtils::fetchMatchingEntry(_parameterDefinitionConfig_, "parameterIndex", par.getParameterIndex());
      }
      par.setParameterDefinitionConfig(parConfig);
    }
    else if( not _dialSetDefinitions_.empty() ){
      // Alternative 2: define dials then parameters
      par.setDialSetConfig( _dialSetDefinitions_ );
    }
    par.readConfig();
  }
}

void FitParameterSet::fillDeltaParameterList(){
  int iFit{0};
  for( const auto& par : _parameterList_ ){
    if( par.isEnabled() and not par.isFixed() and not par.isFree() ){
      (*_deltaParameterList_)[iFit++] = par.getParameterValue() - par.getPriorValue();
    }
  }
}

bool FitParameterSet::isUseOnlyOneParameterPerEvent() const {
  return _useOnlyOneParameterPerEvent_;
}

const std::shared_ptr<TMatrixDSym> &FitParameterSet::getPriorCorrelationMatrix() const {
  return _priorCorrelationMatrix_;
}
const std::shared_ptr<TMatrixDSym> &FitParameterSet::getPriorCovarianceMatrix() const {
  return _priorCovarianceMatrix_;
}


