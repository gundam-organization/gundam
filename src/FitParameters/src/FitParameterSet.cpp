//
// Created by Nadrino on 21/05/2021.
//

#include "FitParameterSet.h"
#include "JsonUtils.h"
#include "GlobalVariables.h"

#include "GenericToolbox.h"
#include "GenericToolbox.Root.h"
#include "Logger.h"

LoggerInit([](){
  Logger::setUserHeaderStr("[FitParameterSet]");
} )

FitParameterSet::FitParameterSet() {
  this->reset();
}
FitParameterSet::~FitParameterSet() {
  this->reset();
}

void FitParameterSet::reset() {

  _isInitialized_ = false;

  _config_ = nlohmann::json();

  _maxEigenFraction_ = 1;
  _parameterPriorList_ = nullptr;
  _parameterNamesList_ = nullptr;
  _choleskyMatrix_ = nullptr;

  _parameterList_.clear();

}

void FitParameterSet::setConfig(const nlohmann::json &config_) {
  _config_ = config_;
  while(_config_.is_string()){
    LogWarning << "Forwarding FitParameterSet config to: \"" << _config_.get<std::string>() << "\"..." << std::endl;
    _config_ = JsonUtils::readConfigFile(_config_.get<std::string>());
  }
}
void FitParameterSet::setSaveDir(TDirectory* saveDir_){
  _saveDir_ = saveDir_;
}

void FitParameterSet::initialize() {

  this->initializeFromConfig();
  if( not _isEnabled_ ) return;

  // Make the matrix inversion
  this->prepareFitParameters();

  _throwMcBeforeFit_ = JsonUtils::fetchValue(_config_, "throwMcBeforeFit", _throwMcBeforeFit_);

  _isInitialized_ = true;
}
void FitParameterSet::prepareFitParameters(){

  LogInfo << "Stripping the matrix from fixed/disabled parameters..." << std::endl;
  int nbFitParameters{0};
  for( const auto& par : _parameterList_ ){
    if( par.isEnabled() and not par.isFixed() ) nbFitParameters++;
  }
  LogInfo << "Effective nb parameters: " << nbFitParameters << std::endl;

  _strippedCovarianceMatrix_ = std::make_shared<TMatrixDSym>(nbFitParameters);
  int iStrippedPar = -1;
  for( int iPar = 0 ; iPar < int(_parameterList_.size()) ; iPar++ ){
    if( not _parameterList_[iPar].isEnabled() or _parameterList_[iPar].isFixed() ) continue;
    iStrippedPar++;
    int jStrippedPar = -1;
    for( int jPar = 0 ; jPar < int(_parameterList_.size()) ; jPar++ ){
      if( not _parameterList_[jPar].isEnabled() or _parameterList_[jPar].isFixed() ) continue;
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
    LogWarning << "Decomposing the stripped covariance matrix..." << std::endl;
    _eigenParameterList_.resize(_strippedCovarianceMatrix_->GetNrows());

    _eigenDecomp_     = std::shared_ptr<TMatrixDSymEigen>(new TMatrixDSymEigen(*_strippedCovarianceMatrix_));

    // Used for base swapping
    _eigenValues_     = std::shared_ptr<TVectorD>( (TVectorD*) _eigenDecomp_->GetEigenValues().Clone() );
    _eigenValuesInv_  = std::shared_ptr<TVectorD>( (TVectorD*) _eigenDecomp_->GetEigenValues().Clone() );
    _eigenVectors_    = std::shared_ptr<TMatrixD>( (TMatrixD*) _eigenDecomp_->GetEigenVectors().Clone() );
    _eigenVectorsInv_ = std::shared_ptr<TMatrixD>(new TMatrixD(TMatrixD::kTransposed, *_eigenVectors_) );

    double eigenCumulative = 0;
    _nbEnabledEigen_ = 0;
    double eigenTotal = _eigenValues_->Sum();

    _inverseStrippedCovarianceMatrix_ = std::shared_ptr<TMatrixD>(new TMatrixD(_strippedCovarianceMatrix_->GetNrows(), _strippedCovarianceMatrix_->GetNrows()));
    _projectorMatrix_                 = std::shared_ptr<TMatrixD>(new TMatrixD(_strippedCovarianceMatrix_->GetNrows(), _strippedCovarianceMatrix_->GetNrows()));

    auto* eigenState = new TVectorD(_eigenValues_->GetNrows());

    for (int iEigen = 0; iEigen < _eigenValues_->GetNrows(); iEigen++) {

      _eigenParameterList_[iEigen].setIsEnabled(true);
      _eigenParameterList_[iEigen].setIsFixed(false);
      _eigenParameterList_[iEigen].setParameterIndex(iEigen);
      _eigenParameterList_[iEigen].setStdDevValue((*_eigenValues_)[iEigen]);
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

    _originalParBuffer_ = std::shared_ptr<TVectorD>(new TVectorD(_strippedCovarianceMatrix_->GetNrows()) );
    _eigenParBuffer_    = std::shared_ptr<TVectorD>(new TVectorD(_strippedCovarianceMatrix_->GetNrows()) );

    // Put original parameters to the prior
    for( auto& par : _parameterList_ ){
      par.setValueAtPrior();
    }

    // Original parameter values are already set -> need to propagate to Eigen parameter list
    propagateOriginalToEigen();

    // Tag the prior
    for( auto& eigenPar : _eigenParameterList_ ){
      eigenPar.setCurrentValueAsPrior();
    }

  }

}

// Getters
bool FitParameterSet::isEnabled() const {
  return _isEnabled_;
}
bool FitParameterSet::isEnableThrowMcBeforeFit() const {
  return _throwMcBeforeFit_;
}
const std::string &FitParameterSet::getName() const {
  return _name_;
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
TMatrixDSym *FitParameterSet::getOriginalCovarianceMatrix() const {
  return _priorCovarianceMatrix_.get();
}
const nlohmann::json &FitParameterSet::getConfig() const {
  return _config_;
}

std::vector<FitParameter>& FitParameterSet::getEffectiveParameterList(){
  if( _useEigenDecompInFit_ ) return _eigenParameterList_;
  return _parameterList_;
}
const std::vector<FitParameter>& FitParameterSet::getEffectiveParameterList() const{
  if( _useEigenDecompInFit_ ) return _eigenParameterList_;
  return _parameterList_;
}

// Core
size_t FitParameterSet::getNbParameters() const {
  return _parameterList_.size();
}
double FitParameterSet::getChi2() {

  if (not _isEnabled_) { return 0; }

  LogThrowIf(_inverseStrippedCovarianceMatrix_==nullptr, GET_VAR_NAME_VALUE(_inverseStrippedCovarianceMatrix_))

  double chi2 = 0;

  if( _useEigenDecompInFit_ ){
    for( const auto& eigenPar : _eigenParameterList_ ){
      if( eigenPar.isFixed() ) continue;
      chi2 += TMath::Sq( (eigenPar.getParameterValue() - eigenPar.getPriorValue()) / eigenPar.getStdDevValue() ) ;
    }
  }
  else {
    // make delta vector
    this->fillDeltaParameterList();

    // compute penalty term with covariance
    chi2 = (*_deltaParameterList_) * ( (*_inverseStrippedCovarianceMatrix_) * (*_deltaParameterList_) );
  }

  return chi2;
}

// Parameter throw
void FitParameterSet::moveFitParametersToPrior(){
  LogInfo << "Moving back fit parameters to their prior value..." << std::endl;

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

  LogThrowIf(_strippedCovarianceMatrix_==nullptr, "No covariance matrix provided")

  if( not _useEigenDecompInFit_ ){
    LogInfo << "Throwing parameters for " << _name_ << " using Cholesky matrix" << std::endl;

    if( _choleskyMatrix_ == nullptr ){
      LogInfo << "Generating Cholesky matrix..." << std::endl;
      _choleskyMatrix_ = std::shared_ptr<TMatrixD>(
          GenericToolbox::getCholeskyMatrix(_strippedCovarianceMatrix_.get())
      );
    }

    auto throws = GenericToolbox::throwCorrelatedParameters(_choleskyMatrix_.get());
    int iPar{0};
    for( auto& par : _parameterList_ ){
      if( not par.isEnabled() ) LogWarning << "Parameter " << par.getTitle() << " is disabled. Not throwing" << std::endl; continue;
      if( par.isFixed() ){ LogWarning << "Parameter " << par.getTitle() << " is fixed. Not throwing" << std::endl; continue; }
      LogInfo << "Throwing par " << par.getTitle() << ": " << par.getParameterValue();
      par.setParameterValue( par.getPriorValue() + gain_ * throws[iPar++] );
      LogInfo << " → " << par.getParameterValue() << std::endl;
    }
  }
  else{
    LogInfo << "Throwing eigen parameters for " << _name_ << std::endl;
    for( auto& eigenPar : _eigenParameterList_ ){
      if( eigenPar.isFixed() ){ LogWarning << "Eigen parameter #" << eigenPar.getParameterIndex() << " is fixed. Not throwing" << std::endl; continue; }
      eigenPar.setParameterValue(
          eigenPar.getPriorValue() + gain_ * GlobalVariables::getPrng().Gaus(0, eigenPar.getStdDevValue())
          );
    }
    this->propagateEigenToOriginal();
  }

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

  ss << "FitParameterSet: " << _name_ << " -> initialized=" << _isInitialized_ << ", enabled=" << _isEnabled_;

  if(_isInitialized_ and _isEnabled_){
    ss << ", nbParameters: " << _parameterList_.size() << "(defined)/" << _strippedCovarianceMatrix_->GetNrows() << "(covariance)";
    if( not _parameterList_.empty() ){
      for( const auto& parameter : _parameterList_ ){
        ss << std::endl << GenericToolbox::indentString(parameter.getSummary(), 2);
      }
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
void FitParameterSet::passIfInitialized(const std::string &methodName_) const {
  if( not _isInitialized_ ){
    LogError << "Can't do \"" << methodName_ << "\" while not initialized." << std::endl;
    throw std::logic_error("class not initialized");
  }
}
void FitParameterSet::initializeFromConfig(){

  LogThrowIf(_config_.empty(), "FitParameterSet config not set.")

  _name_ = JsonUtils::fetchValue<std::string>(_config_, "name", "");
  LogInfo << "Initializing parameter set: " << _name_ << std::endl;

  if( _saveDir_ != nullptr ){
    _saveDir_ = GenericToolbox::mkdirTFile(_saveDir_, _name_);
  }

  _isEnabled_ = JsonUtils::fetchValue<bool>(_config_, "isEnabled");
  if( not _isEnabled_ ){
    LogWarning << _name_ << " parameters are disabled." << std::endl;
    return;
  }

  this->readInputCovarianceMatrix();
  this->readInputParameterOptions();
}

void FitParameterSet::readInputCovarianceMatrix(){

  TObject* objBuffer{nullptr};
  std::string strBuffer;

  strBuffer = JsonUtils::fetchValue<std::string>(_config_, "covarianceMatrixFilePath");
  std::shared_ptr<TFile> covMatrixFile(TFile::Open(strBuffer.c_str()));
  LogThrowIf(covMatrixFile == nullptr or not covMatrixFile->IsOpen(), "Could not open: " << strBuffer)

  strBuffer = JsonUtils::fetchValue<std::string>(_config_, "covarianceMatrixTMatrixD");
  objBuffer = covMatrixFile->Get(strBuffer.c_str());
  LogThrowIf(objBuffer == nullptr, "Can't find \"" << strBuffer << "\" in " << covMatrixFile->GetPath())
  _priorCovarianceMatrix_ = std::shared_ptr<TMatrixDSym>((TMatrixDSym*) objBuffer->Clone());
  _priorCorrelationMatrix_ = std::shared_ptr<TMatrixDSym>((TMatrixDSym*) GenericToolbox::convertToCorrelationMatrix((TMatrixD*)_priorCovarianceMatrix_.get()));


  if( _saveDir_ != nullptr ){
    GenericToolbox::mkdirTFile(_saveDir_, "inputs")->cd();

    ((TMatrixD*) _priorCovarianceMatrix_.get())->Write("CovarianceMatrix_TMatrixD");
    GenericToolbox::convertToTH2D(_priorCovarianceMatrix_.get())->Write("CovarianceMatrix_TH2D");

    auto* correlationMatrix = GenericToolbox::convertToCorrelationMatrix((TMatrixD*)_priorCovarianceMatrix_.get());
    correlationMatrix->Write("CorrelationMatrix_TMatrixD");
    GenericToolbox::convertToTH2D(correlationMatrix)->Write("CorrelationMatrix_TH2D");
  }

  // parameterPriorTVectorD
  strBuffer = JsonUtils::fetchValue(_config_, "parameterPriorTVectorD", "");
  if(not strBuffer.empty()){
    LogInfo << "Reading provided parameterPriorTVectorD: \"" << strBuffer << "\"" << std::endl;

    objBuffer = covMatrixFile->Get(strBuffer.c_str());
    LogThrowIf(objBuffer == nullptr, "Can't find \"" << strBuffer << "\" in " << covMatrixFile->GetPath())
    _parameterPriorList_ = std::shared_ptr<TVectorD>((TVectorD*) objBuffer->Clone());

    LogThrowIf(_parameterPriorList_->GetNrows() != _priorCovarianceMatrix_->GetNrows(),
                "Parameter prior list don't have the same size(" << _parameterPriorList_->GetNrows()
                                                                 << ") as cov matrix(" << _priorCovarianceMatrix_->GetNrows() << ")" );
  }
  else{
    LogWarning << "No parameterPriorTVectorD provided, all parameter prior are set to 1." << std::endl;
    _parameterPriorList_ = std::make_shared<TVectorD>(_priorCovarianceMatrix_->GetNrows());
    for( int iPar = 0 ; iPar < _parameterPriorList_->GetNrows() ; iPar++ ){ (*_parameterPriorList_)[iPar] = 1; }
  }

  // parameterNameTObjArray
  strBuffer = JsonUtils::fetchValue<std::string>(_config_, "parameterNameTObjArray", "");
  if(not strBuffer.empty()){
    LogInfo << "Reading provided parameterNameTObjArray: \"" << strBuffer << "\"" << std::endl;

    objBuffer = covMatrixFile->Get(strBuffer.c_str());
    LogThrowIf(objBuffer == nullptr, "Can't find \"" << strBuffer << "\" in " << covMatrixFile->GetPath())
    _parameterNamesList_ = std::shared_ptr<TObjArray>((TObjArray*) objBuffer->Clone());
  }
  else{
    LogInfo << "No parameterNameTObjArray provided, parameters will be referenced with their index." << std::endl;
    _parameterNamesList_ = std::make_shared<TObjArray>(_priorCovarianceMatrix_->GetNrows());
    for( int iPar = 0 ; iPar < _parameterPriorList_->GetNrows() ; iPar++ ){
      _parameterNamesList_->Add(new TNamed("", ""));
    }
  }

  // parameterLowerBoundsTVectorD
  strBuffer = JsonUtils::fetchValue<std::string>(_config_, "parameterLowerBoundsTVectorD", "");
  if( not strBuffer.empty() ){
    LogInfo << "Reading provided parameterLowerBoundsTVectorD: \"" << strBuffer << "\"" << std::endl;

    objBuffer = covMatrixFile->Get(strBuffer.c_str());
    LogThrowIf(objBuffer == nullptr, "Can't find \"" << strBuffer << "\" in " << covMatrixFile->GetPath())
    _parameterLowerBoundsList_ = std::shared_ptr<TVectorD>((TVectorD*) objBuffer->Clone());

    LogThrowIf(_parameterLowerBoundsList_->GetNrows() != _priorCovarianceMatrix_->GetNrows(),
                "Parameter prior list don't have the same size(" << _parameterLowerBoundsList_->GetNrows()
                                                                 << ") as cov matrix(" << _priorCovarianceMatrix_->GetNrows() << ")" );
  }

  // parameterUpperBoundsTVectorD
  strBuffer = JsonUtils::fetchValue<std::string>(_config_, "parameterUpperBoundsTVectorD", "");
  if( not strBuffer.empty() ){
    LogInfo << "Reading provided parameterUpperBoundsTVectorD: \"" << strBuffer << "\"" << std::endl;

    objBuffer = covMatrixFile->Get(strBuffer.c_str());
    LogThrowIf(objBuffer == nullptr, "Can't find \"" << strBuffer << "\" in " << covMatrixFile->GetPath())
    _parameterUpperBoundsList_ = std::shared_ptr<TVectorD>((TVectorD*) objBuffer->Clone());
    LogThrowIf(_parameterUpperBoundsList_->GetNrows() != _priorCovarianceMatrix_->GetNrows(),
                "Parameter prior list don't have the same size(" << _parameterUpperBoundsList_->GetNrows()
                                                                 << ") as cov matrix(" << _priorCovarianceMatrix_->GetNrows() << ")" );
  }

  covMatrixFile->Close();
}
void FitParameterSet::readInputParameterOptions(){

  _useOnlyOneParameterPerEvent_ = JsonUtils::fetchValue<bool>(_config_, "useOnlyOneParameterPerEvent", false);

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

  LogInfo << "Defining parameters..." << std::endl;
  _parameterList_.resize(_priorCovarianceMatrix_->GetNrows());
  for(int iParameter = 0 ; iParameter < _priorCovarianceMatrix_->GetNrows() ; iParameter++ ){

    _parameterList_[iParameter].setParSetRef(this);
    _parameterList_[iParameter].setParameterIndex(iParameter);

    _parameterList_[iParameter].setStdDevValue(TMath::Sqrt((*_priorCovarianceMatrix_)[iParameter][iParameter]));
    _parameterList_[iParameter].setStepSize(TMath::Sqrt((*_priorCovarianceMatrix_)[iParameter][iParameter]));

    _parameterList_[iParameter].setName(_parameterNamesList_->At(iParameter)->GetName());
    _parameterList_[iParameter].setParameterValue((*_parameterPriorList_)[iParameter]);
    _parameterList_[iParameter].setPriorValue((*_parameterPriorList_)[iParameter]);

    _parameterList_[iParameter].setDialsWorkingDirectory(JsonUtils::fetchValue<std::string>(_config_, "dialSetWorkingDirectory", "./"));

    if( JsonUtils::doKeyExist(_config_, "parameterDefinitions") ){
      // Alternative 1: define parameters then dials
      auto parsConfig = JsonUtils::fetchValue<nlohmann::json>(_config_, "parameterDefinitions");
      JsonUtils::forwardConfig(parsConfig);
      auto parConfig = JsonUtils::fetchMatchingEntry(parsConfig, "parameterName", std::string(_parameterNamesList_->At(iParameter)->GetName()));
      if( parConfig.empty() ){
        // try with par index
        parConfig = JsonUtils::fetchMatchingEntry(parsConfig, "parameterIndex", iParameter);
      }
      _parameterList_[iParameter].setParameterDefinitionConfig(parConfig);
    }
    else if( JsonUtils::doKeyExist(_config_, "dialSetDefinitions") ){
      // Alternative 2: define dials then parameters
      _parameterList_[iParameter].setDialSetConfig(JsonUtils::fetchValue<nlohmann::json>(_config_, "dialSetDefinitions"));
    }

    _parameterList_[iParameter].setEnableDialSetsSummary(JsonUtils::fetchValue<bool>(_config_, "printDialSetsSummary", false));

    if( _globalParameterMinValue_ == _globalParameterMinValue_ ){
      _parameterList_[iParameter].setMinValue(_globalParameterMinValue_);
    }
    if( _globalParameterMaxValue_ == _globalParameterMaxValue_ ){
      _parameterList_[iParameter].setMaxValue(_globalParameterMaxValue_);
    }

    if( _parameterLowerBoundsList_ != nullptr ){
      _parameterList_[iParameter].setMinValue((*_parameterLowerBoundsList_)[iParameter]);
    }
    if( _parameterUpperBoundsList_ != nullptr ){
      _parameterList_[iParameter].setMaxValue((*_parameterUpperBoundsList_)[iParameter]);
    }

    _parameterList_[iParameter].initialize();

  }

}

void FitParameterSet::fillDeltaParameterList(){
  int iFit{0};
  for( const auto& par : _parameterList_ ){
    if( par.isEnabled() and not par.isFixed() ){
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



