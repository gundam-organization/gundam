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
  if(_covarianceMatrixFile_ != nullptr){
    _covarianceMatrixFile_->Close(); // should delete every attached pointer
  }

  _maxEigenFraction_ = 1;
  _covarianceMatrixFile_ = nullptr;
  _covarianceMatrix_ = nullptr;
  _originalCorrelationMatrix_ = nullptr;
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

  if( _config_.empty() ){
    LogError << "Json config not set" << std::endl;
    throw std::logic_error("json config not set");
  }

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

  this->readCovarianceMatrix();

  _useOnlyOneParameterPerEvent_ = JsonUtils::fetchValue<bool>(_config_, "useOnlyOneParameterPerEvent", false);

  if( JsonUtils::doKeyExist(_config_, "parameterLimits") ){
    auto parLimits = JsonUtils::fetchValue(_config_, "parameterLimits", nlohmann::json());
    _globalParameterMinValue_ = JsonUtils::fetchValue(parLimits, "minValue", std::nan("UNSET"));
    _globalParameterMaxValue_ = JsonUtils::fetchValue(parLimits, "maxValue", std::nan("UNSET"));
  }

  // Optional parameters:
  std::string pathBuffer;

  // parameterPriorTVectorD
  pathBuffer = JsonUtils::fetchValue<std::string>(_config_, "parameterPriorTVectorD", "");
  if(not pathBuffer.empty()){
    LogDebug << "Reading provided parameterPriorTVectorD..." << std::endl;
    _parameterPriorList_ = (TVectorT<double>*) _covarianceMatrixFile_->Get(pathBuffer.c_str());
    // Sanity checks
    if( _parameterPriorList_ == nullptr ){
      LogError << "Could not find \"" << pathBuffer << "\" into \"" << _covarianceMatrixFile_->GetName() << "\"" << std::endl;
      throw std::runtime_error("TObject not found.");
    }
    else if(_parameterPriorList_->GetNrows() != _covarianceMatrix_->GetNrows() ){
      LogError << GET_VAR_NAME_VALUE(_parameterPriorList_->GetNrows() != _covarianceMatrix_->GetNrows()) << std::endl;
      throw std::runtime_error("TObject size mismatch.");
    }
  }
  else{
    LogDebug << "No parameterPriorTVectorD provided, all parameter prior are set to 1." << std::endl;
    _parameterPriorList_ = new TVectorT<double>(_covarianceMatrix_->GetNrows());
    for( int iPar = 0 ; iPar < _parameterPriorList_->GetNrows() ; iPar++ ){
      (*_parameterPriorList_)[iPar] = 1;
      if( _useEigenDecompInFit_ ) (*_originalParValues_)[iPar] = 1;
    }
  }

  pathBuffer = JsonUtils::fetchValue<std::string>(_config_, "parameterLowerBoundsTVectorD", "");
  if( not pathBuffer.empty() ){
    _parameterLowerBoundsList_ = (TVectorT<double>*) _covarianceMatrixFile_->Get(pathBuffer.c_str());
    LogThrowIf(_parameterLowerBoundsList_ == nullptr, "Could not fetch parameterLowerBoundsTVectorD: \"" << pathBuffer)
    LogThrowIf(_parameterLowerBoundsList_->GetNrows() != _covarianceMatrix_->GetNrows(),
               "parameterLowerBoundsTVectorD \"" << pathBuffer << "\" have not the right size.")
  }

  pathBuffer = JsonUtils::fetchValue<std::string>(_config_, "parameterUpperBoundsTVectorD", "");
  if( not pathBuffer.empty() ){
    _parameterUpperBoundsList_ = (TVectorT<double>*) _covarianceMatrixFile_->Get(pathBuffer.c_str());
    LogThrowIf(_parameterUpperBoundsList_ == nullptr, "Could not fetch parameterUpperBoundsTVectorD: \"" << pathBuffer)
    LogThrowIf(_parameterUpperBoundsList_->GetNrows() != _covarianceMatrix_->GetNrows(),
               "parameterUpperBoundsTVectorD \"" << pathBuffer << "\" have not the right size.")
  }

  // parameterNameTObjArray
  pathBuffer = JsonUtils::fetchValue<std::string>(_config_, "parameterNameTObjArray", "");
  if(not pathBuffer.empty()){
    LogInfo << "Reading provided parameterNameTObjArray..." << std::endl;
    _parameterNamesList_ = (TObjArray*) _covarianceMatrixFile_->Get(pathBuffer.c_str());
    // Sanity checks
    if( _parameterNamesList_ == nullptr ){
      LogError << "Could not find \"" << pathBuffer << "\" into \"" << _covarianceMatrixFile_->GetName() << "\"" << std::endl;
      throw std::runtime_error("TObject not found.");
    }
  }
  else{
    LogInfo << "No parameterNameTObjArray provided, parameters will be referenced with their index." << std::endl;
    _parameterNamesList_ = new TObjArray(_covarianceMatrix_->GetNrows());
    for( int iPar = 0 ; iPar < _parameterPriorList_->GetNrows() ; iPar++ ){
      _parameterNamesList_->Add(new TNamed("", ""));
    }
  }

  LogInfo << "Initializing fit parameters..." << std::endl;
  _parameterList_.reserve(_covarianceMatrix_->GetNcols()); // need to keep the memory at the same place -> FitParameter* will be used
  for(int iParameter = 0 ; iParameter < _covarianceMatrix_->GetNcols() ; iParameter++ ){
    _parameterList_.emplace_back();
    _parameterList_.back().setParSetRef(this);
    _parameterList_.back().setParameterIndex(iParameter);
    _parameterList_.back().setName(_parameterNamesList_->At(iParameter)->GetName());
    _parameterList_.back().setParameterValue((*_parameterPriorList_)[iParameter]);
    _parameterList_.back().setPriorValue((*_parameterPriorList_)[iParameter]);
    _parameterList_.back().setStdDevValue(TMath::Sqrt((*_covarianceMatrix_)[iParameter][iParameter]));
    _parameterList_.back().setStepSize(TMath::Sqrt((*_covarianceMatrix_)[iParameter][iParameter]));

    _parameterList_.back().setDialsWorkingDirectory(JsonUtils::fetchValue<std::string>(_config_, "dialSetWorkingDirectory", "./"));


    if( JsonUtils::doKeyExist(_config_, "parameterDefinitions") ){
      // Alternative 1: define parameters then dials
      auto parsConfig = JsonUtils::fetchValue<nlohmann::json>(_config_, "parameterDefinitions");
      JsonUtils::forwardConfig(parsConfig);
      auto parConfig = JsonUtils::fetchMatchingEntry(parsConfig, "parameterName", std::string(_parameterNamesList_->At(iParameter)->GetName()));
      if( parConfig.empty() ){
        // try with par index
        parConfig = JsonUtils::fetchMatchingEntry(parsConfig, "parameterIndex", iParameter);
      }
      _parameterList_.back().setParameterDefinitionConfig(parConfig);
    }
    else if( JsonUtils::doKeyExist(_config_, "dialSetDefinitions") ){
      // Alternative 2: define dials then parameters
      _parameterList_.back().setDialSetConfig(JsonUtils::fetchValue<nlohmann::json>(_config_, "dialSetDefinitions"));
    }

    _parameterList_.back().setEnableDialSetsSummary(JsonUtils::fetchValue<bool>(_config_, "printDialSetsSummary", false));

    if( _globalParameterMinValue_ == _globalParameterMinValue_ ){
      _parameterList_.back().setMinValue(_globalParameterMinValue_);
    }
    if( _globalParameterMaxValue_ == _globalParameterMaxValue_ ){
      _parameterList_.back().setMaxValue(_globalParameterMaxValue_);
    }

    if( _parameterLowerBoundsList_ != nullptr ){
      _parameterList_.back().setMinValue((*_parameterLowerBoundsList_)[iParameter]);
    }
    if( _parameterUpperBoundsList_ != nullptr ){
      _parameterList_.back().setMaxValue((*_parameterUpperBoundsList_)[iParameter]);
    }

    _parameterList_.back().initialize();
  }


  if( _useEigenDecompInFit_ ){
    LogDebug << "Initializing eigen objects..." << std::endl;
    _originalParValues_ = std::shared_ptr<TVectorD>(new TVectorD(_parameterPriorList_->GetNrows()));
    for( int iPar = 0 ; iPar < _parameterPriorList_->GetNrows() ; iPar++ ){
      (*_originalParValues_)[iPar] = _parameterList_.at(iPar).getParameterValue();
    }
    propagateOriginalToEigen();
    _eigenParPriorValues_ = std::shared_ptr<TVectorD>( (TVectorD*) _eigenParValues_->Clone() );
    _eigenParStepSize_ = std::shared_ptr<TVectorD>(new TVectorD(_eigenParValues_->GetNrows()));
    for( int iEigen = 0 ; iEigen < _eigenParValues_->GetNrows() ; iEigen++ ){
      (*_eigenParStepSize_)[iEigen] = this->getEigenSigma(iEigen);
    }
    _eigenParFixedList_.resize( _eigenParStepSize_->GetNrows(), false );
  }

  _throwMcBeforeFit_ = JsonUtils::fetchValue(_config_, "throwMcBeforeFit", _throwMcBeforeFit_);

  _isInitialized_ = true;
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
const std::vector<FitParameter> &FitParameterSet::getParameterList() const{
  return _parameterList_;
}
TMatrixDSym *FitParameterSet::getOriginalCovarianceMatrix() const {
  return _covarianceMatrix_;
}
const nlohmann::json &FitParameterSet::getConfig() const {
  return _config_;
}

// Core
size_t FitParameterSet::getNbParameters() const {
  return _parameterList_.size();
}
FitParameter& FitParameterSet::getFitParameter( size_t iPar_ ){
  return _parameterList_.at(iPar_);
}
double FitParameterSet::getChi2() const {

  if (not _isEnabled_) { return 0; }

  LogThrowIf(_inverseCovarianceMatrix_ == nullptr, GET_VAR_NAME_VALUE(_inverseCovarianceMatrix_))

  double chi2 = 0;

  // EIGEN DECOMP NOT VALID?? Why?
  if( _useEigenDecompInFit_ ){
    for( int iEigen = 0 ; iEigen < _nbEnabledEigen_ ; iEigen++ ){
      if( _eigenParFixedList_[iEigen] ) continue;
      chi2 += TMath::Sq((*_eigenParValues_)[iEigen] - (*_eigenParPriorValues_)[iEigen]) / (*_eigenValues_)[iEigen];
    }
  }
  else
  {
    double iDelta, jDelta;
    for (int iPar = 0; iPar < _inverseCovarianceMatrix_->GetNrows(); iPar++) {
      if( not _parameterList_[iPar].isEnabled() ) continue;
      iDelta = (_parameterList_[iPar].getParameterValue() - _parameterList_[iPar].getPriorValue());
      for (int jPar = 0; jPar < _inverseCovarianceMatrix_->GetNrows(); jPar++) {
        if( not _parameterList_[jPar].isEnabled() ) continue;
        jDelta = iDelta;
        jDelta *= (_parameterList_[jPar].getParameterValue() - _parameterList_[jPar].getPriorValue());
        jDelta *= (*_inverseCovarianceMatrix_)(iPar, jPar);
        chi2 += jDelta;
      }
    }
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
    for( int iEigen = 0 ; iEigen < _nbEnabledEigen_ ; iEigen++ ){
      if( _eigenParFixedList_[iEigen] ){ continue; }
      (*_eigenParValues_)[iEigen] = (*_eigenParPriorValues_)[iEigen];
    }
    this->propagateEigenToOriginal();
  }

}
void FitParameterSet::throwFitParameters(double gain_){

  if( _covarianceMatrix_ == nullptr ){
    LogAlert << "Can't throw parameters with \"" << _name_ << "\" since no covariance matrix has been provided." << std::endl;
    return;
  }

  if( not _useEigenDecompInFit_ ){
    LogInfo << "Throwing parameters for " << _name_ << " using Cholesky matrix" << std::endl;

    if( _choleskyMatrix_ == nullptr ){
      LogInfo << "Generating Cholesky matrix..." << std::endl;
      _choleskyMatrix_ = std::shared_ptr<TMatrixD>(
          GenericToolbox::getCholeskyMatrix(_covarianceMatrix_)
      );
    }

    auto throws = GenericToolbox::throwCorrelatedParameters(_choleskyMatrix_.get());
    int iPar{0};
    for( auto& par : _parameterList_ ){
      if( par.isFixed() ){ LogWarning << "Parameter " << par.getTitle() << " is fixed. Not throwing" << std::endl; continue; }
      LogInfo << "Throwing par " << par.getTitle() << ": " << par.getParameterValue();
      par.setParameterValue( par.getPriorValue() + gain_ * throws[iPar++] );
      LogInfo << " → " << par.getParameterValue() << std::endl;
    }
  }
  else{
    LogInfo << "Throwing eigen parameters for " << _name_ << std::endl;
    for( int iEigen = 0 ; iEigen < _nbEnabledEigen_ ; iEigen++ ){

      if( _eigenParFixedList_[iEigen] ){ LogWarning << "Eigen parameter #" << iEigen << " is fixed. Not throwing" << std::endl; continue; }

      LogInfo << "Throwing eigen #" << iEigen << ": " << (*_eigenParValues_)[iEigen];
      (*_eigenParValues_)[iEigen] = (*_eigenParPriorValues_)[iEigen] + gain_ * GlobalVariables::getPrng().Gaus(0, this->getEigenSigma(iEigen) );
      LogInfo << " → " << (*_eigenParValues_)[iEigen] << std::endl;
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
double FitParameterSet::getEigenParameterValue(int iPar_) const{
  return (*_eigenParValues_)[iPar_];
}
double FitParameterSet::getEigenValue(int iPar_) const{
  return (*_eigenValues_)[iPar_];
}
double FitParameterSet::getEigenSigma(int iPar_) const{
  return TMath::Sqrt((*_eigenValues_)[iPar_]);
}
void FitParameterSet::setEigenParameter( int iPar_, double value_ ){
  LogThrowIf(_eigenParValues_ == nullptr, "_eigenParValues_ not set.");
  (*_eigenParValues_)[iPar_] = value_;
}
void FitParameterSet::setEigenParStepSize( int iPar_, double step_ ){
  LogThrowIf(_eigenParStepSize_ == nullptr, "_eigenParStepSize_ not set.");
  (*_eigenParStepSize_)[iPar_] = step_;
}
void FitParameterSet::setEigenParIsFixed( int iPar_, bool isFixed_ ){
  _eigenParFixedList_[iPar_] = isFixed_;
}

bool FitParameterSet::isEigenParFixed( int iPar_ ) const{
  return _eigenParFixedList_[iPar_];
}
double FitParameterSet::getEigenParStepSize( int iPar_ ) const{
  return (*_eigenParStepSize_)[iPar_];
}

const TMatrixD* FitParameterSet::getInvertedEigenVectors() const{
  return _eigenVectorsInv_.get();
}
const TMatrixD* FitParameterSet::getEigenVectors() const{
  return _eigenVectors_.get();
}
void FitParameterSet::propagateEigenToOriginal(){
  (*_originalParValues_) = (*_eigenParValues_);
//  (*_originalParValues_) *= (*_eigenVectorsInv_);
  (*_originalParValues_) *= (*_eigenVectors_);
  for( int iOrig = 0 ; iOrig < _originalParValues_->GetNrows() ; iOrig++ ){
    _parameterList_.at(iOrig).setParameterValue((*_originalParValues_)[iOrig]);
  }
}
void FitParameterSet::propagateOriginalToEigen(){
  (*_eigenParValues_) = (*_originalParValues_);
//  (*_eigenParValues_) *= (*_eigenVectors_);
  (*_eigenParValues_) *= (*_eigenVectorsInv_);
}

// Misc
std::string FitParameterSet::getSummary() const {
  std::stringstream ss;

  ss << "FitParameterSet: " << _name_ << " -> initialized=" << _isInitialized_ << ", enabled=" << _isEnabled_;

  if(_isInitialized_ and _isEnabled_){
    ss << ", nbParameters: " << _parameterList_.size() << "(defined)/" << _covarianceMatrix_->GetNrows() << "(covariance)";
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


double FitParameterSet::toNormalizedEigenParRange(double parRange, int parIndex) const{
  return (parRange) / this->getEigenSigma(parIndex);
}
double FitParameterSet::toNormalizedEigenParValue(double parValue, int parIndex) const{
  return this->toNormalizedEigenParRange(parValue - (*_eigenParPriorValues_)[parIndex], parIndex);
}
double FitParameterSet::toRealEigenParRange(double normParRange, int parIndex) const{
  return normParRange * this->getEigenSigma(parIndex);
}
double FitParameterSet::toRealEigenParValue(double normParValue, int parIndex) const{
  return normParValue*this->getEigenSigma(parIndex) + (*_eigenParPriorValues_)[parIndex];
}


// Protected
void FitParameterSet::passIfInitialized(const std::string &methodName_) const {
  if( not _isInitialized_ ){
    LogError << "Can't do \"" << methodName_ << "\" while not initialized." << std::endl;
    throw std::logic_error("class not initialized");
  }
}
void FitParameterSet::readCovarianceMatrix(){

  _covarianceMatrixFile_ = std::shared_ptr<TFile>( TFile::Open(JsonUtils::fetchValue<std::string>(_config_, "covarianceMatrixFilePath").c_str()) );
  if( not _covarianceMatrixFile_->IsOpen() ){
    LogError << "Could not open: _covarianceMatrixFile_: " << _covarianceMatrixFile_->GetPath() << std::endl;
    throw std::runtime_error("Could not open: _covarianceMatrixFile_");
  }

  _covarianceMatrix_ = (TMatrixDSym*) _covarianceMatrixFile_->Get(
    JsonUtils::fetchValue<std::string>(_config_, "covarianceMatrixTMatrixD").c_str()
  );
  if(_covarianceMatrix_ == nullptr ){
    LogError << "Could not find: " << JsonUtils::fetchValue<std::string>(_config_, "covarianceMatrixTMatrixD")
             << " in " << _covarianceMatrixFile_->GetPath() << std::endl;
    throw std::runtime_error("Could not find: covarianceMatrixTObjectPath");
  }

  if( _saveDir_ != nullptr ){
    _saveDir_->cd();
    _covarianceMatrix_->Write("CovarianceMatrix_TMatrixDSym");
    GenericToolbox::convertToCorrelationMatrix((TMatrixD*)_covarianceMatrix_)->Write("CorrelationMatrix_TMatrixD");
  }

  _originalCorrelationMatrix_ = std::shared_ptr<TMatrixD>(
    GenericToolbox::convertToCorrelationMatrix((TMatrixD*) _covarianceMatrix_)
    );

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


  LogWarning << "Computing inverse of the covariance matrix: " << _covarianceMatrix_->GetNcols() << "x" << _covarianceMatrix_->GetNrows() << std::endl;
  if( not _useEigenDecompInFit_ ){
    LogDebug << "Using default matrix inversion..." << std::endl;
    _inverseCovarianceMatrix_ = std::shared_ptr<TMatrixD>((TMatrixD*)(_covarianceMatrix_->Clone()));
    _inverseCovarianceMatrix_->Invert();
  }
  else{
    LogInfo << "Decomposing covariance matrix..." << std::endl;

    _eigenDecomp_ = std::shared_ptr<TMatrixDSymEigen>(new TMatrixDSymEigen(*_covarianceMatrix_));
    _eigenValues_     = std::shared_ptr<TVectorD>( (TVectorD*) _eigenDecomp_->GetEigenValues().Clone() );
    _eigenValuesInv_  = std::shared_ptr<TVectorD>( (TVectorD*) _eigenDecomp_->GetEigenValues().Clone() );
    _eigenVectors_    = std::shared_ptr<TMatrixD>( (TMatrixD*) _eigenDecomp_->GetEigenVectors().Clone() );
    _eigenVectorsInv_ = std::shared_ptr<TMatrixD>(new TMatrixD(TMatrixD::kTransposed, *_eigenVectors_) );

    double eigenCumulative = 0;
    _nbEnabledEigen_ = 0;
    double eigenTotal = _eigenValues_->Sum();

    _inverseCovarianceMatrix_   = std::shared_ptr<TMatrixD>(new TMatrixD(_covarianceMatrix_->GetNrows(), _covarianceMatrix_->GetNrows()));
    _effectiveCovarianceMatrix_ = std::shared_ptr<TMatrixD>(new TMatrixD(_covarianceMatrix_->GetNrows(), _covarianceMatrix_->GetNrows()));
    _projectorMatrix_           = std::shared_ptr<TMatrixD>(new TMatrixD(_covarianceMatrix_->GetNrows(), _covarianceMatrix_->GetNrows()));

    auto* eigenState = new TVectorD(_eigenValues_->GetNrows());

    for (int iEigen = 0; iEigen < _eigenValues_->GetNrows(); iEigen++) {

      (*_eigenValuesInv_)[iEigen] = 1./(*_eigenValues_)[iEigen];
      (*eigenState)[iEigen] = 1.;

      eigenCumulative += (*_eigenValues_)[iEigen];
      if(    ( _maxNbEigenParameters_ != -1 and iEigen >= _maxNbEigenParameters_ )
          or ( _maxEigenFraction_ != 1      and eigenCumulative / eigenTotal > _maxEigenFraction_ ) ){
        eigenCumulative -= (*_eigenValues_)[iEigen]; // not included
        (*_eigenValues_)[iEigen] = 0;
        (*_eigenValuesInv_)[iEigen] = 0;
        (*eigenState)[iEigen] = 0;
        break;
      }
      _nbEnabledEigen_++;

    } // iEigen

    TMatrixD* eigenStateMatrix    = GenericToolbox::makeDiagonalMatrix(eigenState);
    TMatrixD* diagMatrix    = GenericToolbox::makeDiagonalMatrix(_eigenValues_.get());
    TMatrixD* diagInvMatrix = GenericToolbox::makeDiagonalMatrix(_eigenValuesInv_.get());

    (*_projectorMatrix_) =  (*_eigenVectors_);
    (*_projectorMatrix_) *= (*eigenStateMatrix);
    (*_projectorMatrix_) *= (*_eigenVectorsInv_);

    (*_inverseCovarianceMatrix_) =  (*_eigenVectors_);
    (*_inverseCovarianceMatrix_) *= (*diagInvMatrix);
    (*_inverseCovarianceMatrix_) *= (*_eigenVectorsInv_);

    (*_effectiveCovarianceMatrix_) =  (*_eigenVectors_);
    (*_effectiveCovarianceMatrix_) *= (*diagInvMatrix);
    (*_effectiveCovarianceMatrix_) *= (*_eigenVectorsInv_);

    delete eigenState;
    delete eigenStateMatrix;
    delete diagMatrix;
    delete diagInvMatrix;

    LogWarning << "Eigen decomposition with " << _nbEnabledEigen_ << " / " << _eigenValues_->GetNrows() << " vectors" << std::endl;
    if(_nbEnabledEigen_ != _eigenValues_->GetNrows() ){
      LogInfo << "Max eigen fraction set to " << _maxEigenFraction_*100 << "%" << std::endl;
      LogInfo << "Fraction taken: " << eigenCumulative / eigenTotal*100 << "%" << std::endl;
    }

    _originalParValues_ = std::shared_ptr<TVectorD>( new TVectorD(_covarianceMatrix_->GetNrows()) );
    _eigenParValues_    = std::shared_ptr<TVectorD>( new TVectorD(_covarianceMatrix_->GetNrows()) );

  }

  LogInfo << "Parameter set \"" << _name_ << "\" is handling " << _covarianceMatrix_->GetNcols() << " parameters." << std::endl;


}

bool FitParameterSet::isUseOnlyOneParameterPerEvent() const {
  return _useOnlyOneParameterPerEvent_;
}



