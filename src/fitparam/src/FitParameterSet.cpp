//
// Created by Nadrino on 21/05/2021.
//

#include "FitParameterSet.h"
#include "JsonUtils.h"

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

  _jsonConfig_ = nlohmann::json();
  if(_covarianceMatrixFile_ != nullptr){
    _covarianceMatrixFile_->Close(); // should delete every attached pointer
  }

  _covarianceMatrixFile_ = nullptr;
  _covarianceMatrix_ = nullptr;
  _correlationMatrix_ = nullptr;
  _parameterPriorList_ = nullptr;
  _parameterNamesList_ = nullptr;

  _parameterList_.clear();

}

void FitParameterSet::setJsonConfig(const nlohmann::json &jsonConfig) {
  _jsonConfig_ = jsonConfig;
  while(_jsonConfig_.is_string()){
    LogWarning << "Forwarding FitParameterSet config to: \"" << _jsonConfig_.get<std::string>() << "\"..." << std::endl;
    _jsonConfig_ = JsonUtils::readConfigFile(_jsonConfig_.get<std::string>());
  }
}

void FitParameterSet::initialize() {

  if( _jsonConfig_.empty() ){
    LogError << "Json config not set" << std::endl;
    throw std::logic_error("json config not set");
  }

  _name_ = JsonUtils::fetchValue<std::string>(_jsonConfig_, "name", "");
  LogInfo << "Initializing parameter set: " << _name_ << std::endl;

  _isEnabled_ = JsonUtils::fetchValue<bool>(_jsonConfig_, "isEnabled");
  if( not _isEnabled_ ){
    LogWarning << _name_ << " parameters are disabled." << std::endl;
    return;
  }

  _covarianceMatrixFile_ = TFile::Open(JsonUtils::fetchValue<std::string>(_jsonConfig_, "covarianceMatrixFilePath").c_str());
  if( not _covarianceMatrixFile_->IsOpen() ){
    LogError << "Could not open: _covarianceMatrixFile_: " << _covarianceMatrixFile_->GetPath() << std::endl;
    throw std::runtime_error("Could not open: _covarianceMatrixFile_");
  }

  _covarianceMatrix_ = (TMatrixDSym*) _covarianceMatrixFile_->Get(
    JsonUtils::fetchValue<std::string>(_jsonConfig_, "covarianceMatrixTMatrixD").c_str()
    );
  if( _covarianceMatrix_ == nullptr ){
    LogError << "Could not find: " << JsonUtils::fetchValue<std::string>(_jsonConfig_, "covarianceMatrixTObjectPath")
      << " in " << _covarianceMatrixFile_->GetPath() << std::endl;
    throw std::runtime_error("Could not find: covarianceMatrixTObjectPath");
  }
  LogWarning << "Computing inverse of the covariance matrix: " << _covarianceMatrix_->GetNcols() << "x" << _covarianceMatrix_->GetNrows() << std::endl;
  _inverseCovarianceMatrix_ = GenericToolbox::convertToSymmetricMatrix(
    GenericToolbox::invertMatrixSVD(
      (TMatrixD*) _covarianceMatrix_
    )["inverse_covariance_matrix"]
  );

  _correlationMatrix_ = GenericToolbox::convertToSymmetricMatrix(GenericToolbox::convertToCorrelationMatrix((TMatrixD*) _covarianceMatrix_));

  LogInfo << "Parameter set \"" << _name_ << "\" is handling " << _covarianceMatrix_->GetNcols() << " parameters." << std::endl;

  // Optional parameters:
  std::string pathBuffer;

  // parameterPriorTVectorD
  pathBuffer = JsonUtils::fetchValue<std::string>(_jsonConfig_, "parameterPriorTVectorD", "");
  if(not pathBuffer.empty()){
    LogDebug << "Reading provided parameterPriorTVectorD..." << std::endl;
    _parameterPriorList_ = (TVectorT<double>*) _covarianceMatrixFile_->Get(pathBuffer.c_str());
    // Sanity checks
    if( _parameterPriorList_ == nullptr ){
      LogError << "Could not find \"" << pathBuffer << "\" into \"" << _covarianceMatrixFile_->GetName() << "\"" << std::endl;
      throw std::runtime_error("TObject not found.");
    }
    else if( _parameterPriorList_->GetNrows() != _covarianceMatrix_->GetNrows() ){
      LogError << GET_VAR_NAME_VALUE(_parameterPriorList_->GetNrows() != _covarianceMatrix_->GetNrows()) << std::endl;
      throw std::runtime_error("TObject size mismatch.");
    }
  }
  else{
    LogDebug << "No parameterPriorTVectorD provided, all parameter prior are set to 1." << std::endl;
    _parameterPriorList_ = new TVectorT<double>(_covarianceMatrix_->GetNrows());
    for( int iPar = 0 ; iPar < _parameterPriorList_->GetNrows() ; iPar++ ){
      (*_parameterPriorList_)[iPar] = 1;
    }
  }

  // parameterNameTObjArray
  pathBuffer = JsonUtils::fetchValue<std::string>(_jsonConfig_, "parameterNameTObjArray", "");
  if(not pathBuffer.empty()){
    LogDebug << "Reading provided parameterNameTObjArray..." << std::endl;
    _parameterNamesList_ = (TObjArray*) _covarianceMatrixFile_->Get(pathBuffer.c_str());
    // Sanity checks
    if( _parameterNamesList_ == nullptr ){
      LogError << "Could not find \"" << pathBuffer << "\" into \"" << _covarianceMatrixFile_->GetName() << "\"" << std::endl;
      throw std::runtime_error("TObject not found.");
    }
    else if( _parameterNamesList_->GetSize() != _covarianceMatrix_->GetNrows() ){
      LogError << GET_VAR_NAME_VALUE(_parameterNamesList_->GetSize() != _covarianceMatrix_->GetNrows()) << std::endl;
      throw std::runtime_error("TObject size mismatch.");
    }
  }
  else{
    LogDebug << "No parameterNameTObjArray provided, parameters will be referenced with their index." << std::endl;
    _parameterNamesList_ = new TObjArray(_covarianceMatrix_->GetNrows());
    for( int iPar = 0 ; iPar < _parameterPriorList_->GetNrows() ; iPar++ ){
      _parameterNamesList_->Add(new TNamed("", ""));
    }
  }

  LogDebug << "Initializing parameters..." << std::endl;
  for( int iPararmeter = 0 ; iPararmeter < _covarianceMatrix_->GetNcols() ; iPararmeter++ ){
    _parameterList_.emplace_back();
    _parameterList_.back().setParameterIndex(iPararmeter);
    _parameterList_.back().setName(_parameterNamesList_->At(iPararmeter)->GetName());
    _parameterList_.back().setParameterValue((*_parameterPriorList_)[iPararmeter]);
    _parameterList_.back().setPriorValue((*_parameterPriorList_)[iPararmeter]);
    _parameterList_.back().setStdDevValue(TMath::Sqrt((*_covarianceMatrix_)[iPararmeter][iPararmeter]));
    _parameterList_.back().setDialSetConfig(JsonUtils::fetchValue<nlohmann::json>(_jsonConfig_, "dialSetDefinitions"));
    _parameterList_.back().setDialsWorkingDirectory(JsonUtils::fetchValue<std::string>(_jsonConfig_, "dialSetWorkingDirectory", "./"));
    _parameterList_.back().setEnableDialSetsSummary(JsonUtils::fetchValue<bool>(_jsonConfig_, "printDialSetsSummary", false));

    _parameterList_.back().initialize();
  }

  _isInitialized_ = true;
}


// Getters
bool FitParameterSet::isEnabled() const {
  return _isEnabled_;
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
TMatrixDSym *FitParameterSet::getCovarianceMatrix() const {
  return _covarianceMatrix_;
}

// Core
size_t FitParameterSet::getNbParameters() const {
  return _parameterList_.size();
}
FitParameter& FitParameterSet::getFitParameter( size_t iPar_ ){
  return _parameterList_.at(iPar_);
}
double FitParameterSet::getChi2() const{
  double chi2 = 0;

  if( not _isEnabled_ ){
    return chi2;
  }

  if( _inverseCovarianceMatrix_ == nullptr ){
    LogError << GET_VAR_NAME_VALUE(_inverseCovarianceMatrix_) << std::endl;
    throw std::runtime_error("inverse matrix not set");
  }

  for(int iPar = 0; iPar < _inverseCovarianceMatrix_->GetNrows(); iPar++)
  {
    if( _parameterList_.at(iPar).isFixed() ) continue;
    for(int jPar = 0; jPar < _inverseCovarianceMatrix_->GetNrows(); jPar++)
    {
      if( _parameterList_.at(jPar).isFixed() ) continue;
      chi2
        +=  (_parameterList_[iPar].getParameterValue() - _parameterList_[iPar].getPriorValue())
          * (_parameterList_[jPar].getParameterValue() - _parameterList_[jPar].getPriorValue())
          * (*_inverseCovarianceMatrix_)(iPar, jPar);
    }
  }

  return chi2;
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


// Protected
void FitParameterSet::passIfInitialized(const std::string &methodName_) const {
  if( not _isInitialized_ ){
    LogError << "Can't do \"" << methodName_ << "\" while not initialized." << std::endl;
    throw std::logic_error("class not initialized");
  }
}

