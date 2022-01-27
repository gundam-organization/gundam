//
// Created by Nadrino on 21/05/2021.
//

#include "FitParameterSet.h"
#include "JsonUtils.h"
#include "GlobalVariables.h"

#include "GenericToolbox.h"
#include "GenericToolbox.Root.h"
#include "GenericToolbox.TablePrinter.h"
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

  if( _priorCovarianceMatrix_ == nullptr ){ return; } // nothing to do

  LogInfo << "Stripping the matrix from fixed/disabled parameters..." << std::endl;
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

      _eigenParameterList_[iEigen].setIsEigen(true);
      _eigenParameterList_[iEigen].setIsEnabled(true);
      _eigenParameterList_[iEigen].setIsFixed(false);
      _eigenParameterList_[iEigen].setParSetRef(this);
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
TMatrixDSym *FitParameterSet::getPriorCovarianceMatrix() const {
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

    ss << ", nbParameters: " << _parameterList_.size();

    if( not _parameterList_.empty() ){

      std::vector<std::vector<std::string>> tableLines;
      tableLines.emplace_back(std::vector<std::string>{
        "Title",
        "Prior",
        "StdDev",
//        "StepSize",
        "Min",
        "Max",
        "Status"
      });


      for( const auto& par : _parameterList_ ){
        std::vector<std::string> lineValues(tableLines[0].size());
        lineValues[0] = par.getTitle();
        lineValues[1] = std::to_string( par.getPriorValue() );
        lineValues[2] = std::to_string( par.getStdDevValue() );
//        lineValues[3] = std::to_string( par.getStepSize() );

        lineValues[3] = std::to_string( par.getMinValue() );
        lineValues[4] = std::to_string( par.getMaxValue() );

        std::string colorStr;

        if( not par.isEnabled() ) { lineValues.back() = "Disabled"; colorStr = GenericToolbox::ColorCodes::yellowBackGround; }
        else if( par.isFixed() )  { lineValues.back() = "Fixed";    colorStr = GenericToolbox::ColorCodes::redBackGround; }
        else if( par.isFree() )   { lineValues.back() = "Free"; }
        else                      { lineValues.back() = "Fit"; }

        for( auto& line : lineValues ){
          if(not line.empty()) line = colorStr + line + GenericToolbox::ColorCodes::resetColor;
        }

        tableLines.emplace_back(lineValues);
      }

      GenericToolbox::TablePrinter t;
      t.fillTable(tableLines);
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

  if( _saveDir_ != nullptr ){ _saveDir_ = GenericToolbox::mkdirTFile(_saveDir_, _name_); }

  _isEnabled_ = JsonUtils::fetchValue<bool>(_config_, "isEnabled");
  if( not _isEnabled_ ){
    LogWarning << _name_ << " parameters are disabled." << std::endl;
    return;
  }

  this->readConfigOptions();
  this->defineParameters();
}

void FitParameterSet::readParameterDefinitionFile(){

  std::shared_ptr<TFile> parDefFile(TFile::Open(_parameterDefinitionFilePath_.c_str()));
  LogThrowIf(parDefFile == nullptr or not parDefFile->IsOpen(), "Could not open: " << _parameterDefinitionFilePath_)

  TObject* objBuffer{nullptr};
  std::string strBuffer;

  strBuffer = JsonUtils::fetchValue<std::string>(_config_, "covarianceMatrixTMatrixD", "");
  if( strBuffer.empty() ){
    LogWarning << "No covariance matrix provided. Free parameter definition expected." << std::endl;
  }
  else{
    objBuffer = parDefFile->Get(strBuffer.c_str());
    LogThrowIf(objBuffer == nullptr, "Can't find \"" << strBuffer << "\" in " << parDefFile->GetPath())
    _priorCovarianceMatrix_ = std::shared_ptr<TMatrixDSym>((TMatrixDSym*) objBuffer->Clone());
    _priorCorrelationMatrix_ = std::shared_ptr<TMatrixDSym>((TMatrixDSym*) GenericToolbox::convertToCorrelationMatrix((TMatrixD*)_priorCovarianceMatrix_.get()));

    if( _saveDir_ != nullptr ){
      // TODO: better writing of the inputs
      GenericToolbox::mkdirTFile(_saveDir_, "inputs")->cd();

      ((TMatrixD*) _priorCovarianceMatrix_.get())->Write("CovarianceMatrix_TMatrixD");
      GenericToolbox::convertTMatrixDtoTH2D((TMatrixD*) _priorCovarianceMatrix_.get())->Write("CovarianceMatrix_TH2D");

      auto* correlationMatrix = GenericToolbox::convertToCorrelationMatrix((TMatrixD*)_priorCovarianceMatrix_.get());
      correlationMatrix->Write("CorrelationMatrix_TMatrixD");
      GenericToolbox::convertTMatrixDtoTH2D(correlationMatrix)->Write("CorrelationMatrix_TH2D");
    }

    _nbParameterDefinition_ = _priorCovarianceMatrix_->GetNrows();
  }

  if( JsonUtils::doKeyExist(_config_, "enableParameterMask") ){
    // TODO: implement parameter mask
  }

  // parameterPriorTVectorD
  strBuffer = JsonUtils::fetchValue(_config_, "parameterPriorTVectorD", "");
  if(not strBuffer.empty()){
    LogInfo << "Reading provided parameterPriorTVectorD: \"" << strBuffer << "\"" << std::endl;

    objBuffer = parDefFile->Get(strBuffer.c_str());
    LogThrowIf(objBuffer == nullptr, "Can't find \"" << strBuffer << "\" in " << parDefFile->GetPath())
    _parameterPriorList_ = std::shared_ptr<TVectorD>((TVectorD*) objBuffer->Clone());

    LogThrowIf(_parameterPriorList_->GetNrows() != _nbParameterDefinition_,
                "Parameter prior list don't have the same size(" << _parameterPriorList_->GetNrows()
                                                                 << ") as cov matrix(" << _nbParameterDefinition_ << ")" );
  }

  // parameterNameTObjArray
  strBuffer = JsonUtils::fetchValue<std::string>(_config_, "parameterNameTObjArray", "");
  if(not strBuffer.empty()){
    LogInfo << "Reading provided parameterNameTObjArray: \"" << strBuffer << "\"" << std::endl;

    objBuffer = parDefFile->Get(strBuffer.c_str());
    LogThrowIf(objBuffer == nullptr, "Can't find \"" << strBuffer << "\" in " << parDefFile->GetPath())
    _parameterNamesList_ = std::shared_ptr<TObjArray>((TObjArray*) objBuffer->Clone());
  }

  // parameterLowerBoundsTVectorD
  strBuffer = JsonUtils::fetchValue<std::string>(_config_, "parameterLowerBoundsTVectorD", "");
  if( not strBuffer.empty() ){
    LogInfo << "Reading provided parameterLowerBoundsTVectorD: \"" << strBuffer << "\"" << std::endl;

    objBuffer = parDefFile->Get(strBuffer.c_str());
    LogThrowIf(objBuffer == nullptr, "Can't find \"" << strBuffer << "\" in " << parDefFile->GetPath())
    _parameterLowerBoundsList_ = std::shared_ptr<TVectorD>((TVectorD*) objBuffer->Clone());

    LogThrowIf(_parameterLowerBoundsList_->GetNrows() != _nbParameterDefinition_,
                "Parameter prior list don't have the same size(" << _parameterLowerBoundsList_->GetNrows()
                                                                 << ") as cov matrix(" << _nbParameterDefinition_ << ")" );
  }

  // parameterUpperBoundsTVectorD
  strBuffer = JsonUtils::fetchValue<std::string>(_config_, "parameterUpperBoundsTVectorD", "");
  if( not strBuffer.empty() ){
    LogInfo << "Reading provided parameterUpperBoundsTVectorD: \"" << strBuffer << "\"" << std::endl;

    objBuffer = parDefFile->Get(strBuffer.c_str());
    LogThrowIf(objBuffer == nullptr, "Can't find \"" << strBuffer << "\" in " << parDefFile->GetPath())
    _parameterUpperBoundsList_ = std::shared_ptr<TVectorD>((TVectorD*) objBuffer->Clone());
    LogThrowIf(_parameterUpperBoundsList_->GetNrows() != _nbParameterDefinition_,
                "Parameter prior list don't have the same size(" << _parameterUpperBoundsList_->GetNrows()
                                                                 << ") as cov matrix(" << _nbParameterDefinition_ << ")" );
  }

  parDefFile->Close();
}
void FitParameterSet::readConfigOptions(){
  LogInfo << __METHOD_NAME__ << std::endl;

  _nbParameterDefinition_ = JsonUtils::fetchValue(_config_, "numberOfParameters", _nbParameterDefinition_);
  _nominalStepSize_ = JsonUtils::fetchValue(_config_, "nominalStepSize", _nominalStepSize_);

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

  _parameterDefinitionFilePath_ = JsonUtils::fetchValue(
      _config_
      , std::vector<std::string>{
          "parameterDefinitionFilePath",
          "covarianceMatrixFilePath"}
      , ""
  );

  if( not _parameterDefinitionFilePath_.empty() ) this->readParameterDefinitionFile();

  if( _parameterPriorList_ == nullptr ){
    LogWarning << "No prior list provided, all parameter prior are set to 1." << std::endl;
    _parameterPriorList_ = std::make_shared<TVectorD>(_nbParameterDefinition_);
    for( int iPar = 0 ; iPar < _parameterPriorList_->GetNrows() ; iPar++ ){ (*_parameterPriorList_)[iPar] = 1; }
  }

}
void FitParameterSet::defineParameters(){
  LogInfo << "Defining parameters..." << std::endl;
  _parameterList_.resize(_nbParameterDefinition_);
  for(int iParameter = 0 ; iParameter < _nbParameterDefinition_ ; iParameter++ ){

    _parameterList_[iParameter].setParSetRef(this);
    _parameterList_[iParameter].setParameterIndex(iParameter);

    if( _priorCovarianceMatrix_ != nullptr ){
      _parameterList_[iParameter].setStdDevValue(TMath::Sqrt((*_priorCovarianceMatrix_)[iParameter][iParameter]));
      _parameterList_[iParameter].setStepSize(TMath::Sqrt((*_priorCovarianceMatrix_)[iParameter][iParameter]));
    }
    else{
      LogThrowIf(_nominalStepSize_==-1, "Can't define free parameter without a \"nominalStepSize\"")
      _parameterList_[iParameter].setStdDevValue(_nominalStepSize_); // stdDev will only be used for display purpose
      _parameterList_[iParameter].setStepSize(_nominalStepSize_);
      _parameterList_[iParameter].setPriorType(PriorType::Flat);
      _parameterList_[iParameter].setIsFree(true);
    }

    if(_parameterNamesList_ != nullptr) _parameterList_[iParameter].setName(_parameterNamesList_->At(iParameter)->GetName());
    _parameterList_[iParameter].setParameterValue((*_parameterPriorList_)[iParameter]);
    _parameterList_[iParameter].setPriorValue((*_parameterPriorList_)[iParameter]);

    _parameterList_[iParameter].setDialsWorkingDirectory(JsonUtils::fetchValue<std::string>(_config_, "dialSetWorkingDirectory", "./"));
    _parameterList_[iParameter].setEnableDialSetsSummary(JsonUtils::fetchValue<bool>(_config_, "printDialSetsSummary", false));

    if( _globalParameterMinValue_ == _globalParameterMinValue_ ){ _parameterList_[iParameter].setMinValue(_globalParameterMinValue_); }
    if( _globalParameterMaxValue_ == _globalParameterMaxValue_ ){ _parameterList_[iParameter].setMaxValue(_globalParameterMaxValue_); }

    if( _parameterLowerBoundsList_ != nullptr ){ _parameterList_[iParameter].setMinValue((*_parameterLowerBoundsList_)[iParameter]); }
    if( _parameterUpperBoundsList_ != nullptr ){ _parameterList_[iParameter].setMaxValue((*_parameterUpperBoundsList_)[iParameter]); }

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

    _parameterList_[iParameter].initialize();

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



