//
// Created by Adrien BLANCHET on 21/05/2021.
//

#include "FitParameterSet.h"
#include "JsonUtils.h"

#include "Logger.h"

FitParameterSet::FitParameterSet() {
  Logger::setUserHeaderStr("[FitParameterSet]");
  this->reset();
}
FitParameterSet::~FitParameterSet() {
  this->reset();
}

void FitParameterSet::reset() {

  _isInitialized_ = false;

  delete _covarianceMatrix_; _covarianceMatrix_ = nullptr;
  delete _covarianceMatrixFile_; _covarianceMatrixFile_ = nullptr;

  _jsonConfig_ = nlohmann::json();
  _parameterList_.clear();

}

void FitParameterSet::setJsonConfig(const nlohmann::json &jsonConfig) {
  _jsonConfig_ = jsonConfig;
}

void FitParameterSet::initialize() {

  if( _jsonConfig_.empty() ){
    LogError << "Json config not set" << std::endl;
    throw std::logic_error("json config not set");
  }

  _name_ = JsonUtils::fetchValue<std::string>(_jsonConfig_, "name");
  LogInfo << "Initializing parameter set: " << _name_ << std::endl;

  _isEnabled_ = JsonUtils::fetchValue<bool>(_jsonConfig_, "isEnabled");
  if( not _isEnabled_ ){
    LogWarning << _name_ << " parameters are disabled. Skipping config parsing..." << std::endl;
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
    LogDebug << "No parameterNameTObjArray provided, default values." << std::endl;
    _parameterNamesList_ = new TObjArray(_covarianceMatrix_->GetNrows());
    for( int iPar = 0 ; iPar < _parameterPriorList_->GetNrows() ; iPar++ ){
      _parameterNamesList_->Add(new TNamed("", ""));
    }
  }

  nlohmann::json dialSetDefinitions = JsonUtils::fetchValue<nlohmann::json>(_jsonConfig_, "dialSetDefinitions");
  for( int iPararmeter = 0 ; iPararmeter < _covarianceMatrix_->GetNcols() ; iPararmeter++ ){
    _parameterList_.emplace_back();
    _parameterList_.back().setParameterIndex(iPararmeter);
    _parameterList_.back().setName(_parameterNamesList_->At(iPararmeter)->GetName());
    _parameterList_.back().setParameterValue((*_parameterPriorList_)[iPararmeter]);
    _parameterList_.back().setParameterDefinitionsConfig(dialSetDefinitions);
    _parameterList_.back().initialize();
  }

  _isInitialized_ = true;
}



