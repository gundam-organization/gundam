#include "GaussianConstraint.h"
#include "ParametersManager.h"

#include "Logger.h"

#include <TFormula.h>

#include <cmath>
#include <limits>
#include <regex>
#include <set>

void GaussianConstraint::configureImpl(){
  _config_.clearFields();
  _config_.defineFields({{"name"}, {"isEnabled"}, {"parameters"}, {"formula"}, {"type"}});
  _config_.checkConfiguration();

  _name_ = _config_.fetchValue<std::string>("name");
  _isEnabled_ = _config_.fetchValue<bool>("isEnabled", true);
  _parameterReferenceList_.clear();
  if( not _isEnabled_ ){ return; }

  _formulaString_ = _config_.fetchValue<std::string>("formula");
  LogThrowIf(_formulaString_.empty(), "Empty formula in external constraint " << _name_);
  auto typeConfig = _config_.fetchValue<ConfigReader>("type");
  typeConfig.clearFields();
  typeConfig.defineFields({{"name"}, {"mean"}, {"stdDev"}});
  typeConfig.checkConfiguration();
  LogThrowIf(typeConfig.fetchValue<std::string>("name") != "gaussian",
             "Unsupported external constraint type in " << _name_ << ": expected gaussian");
  _mean_ = typeConfig.fetchValue<double>("mean");
  _stdDev_ = typeConfig.fetchValue<double>("stdDev");
  LogThrowIf(not std::isfinite(_mean_), "Non-finite mean in external constraint " << _name_);
  LogThrowIf(not std::isfinite(_stdDev_) or _stdDev_ <= 0,
             "stdDev must be positive and finite in external constraint " << _name_);
  typeConfig.printUnusedKeys();

  std::set<std::string> aliases;
  for( auto parameterConfig : _config_.loop("parameters") ){
    parameterConfig.clearFields();
    parameterConfig.defineFields({{"name"}, {"parameterSetName"}, {"parameterName"}, {"parameterIndex"}});
    parameterConfig.checkConfiguration();
    ParameterReference reference;
    reference.name = parameterConfig.fetchValue<std::string>("name");
    reference.parameterSetName = parameterConfig.fetchValue<std::string>("parameterSetName");
    LogThrowIf(not std::regex_match(reference.name, std::regex("[A-Za-z_][A-Za-z0-9_]*")),
               "Invalid parameter alias " << reference.name << " in external constraint " << _name_);
    LogThrowIf(not aliases.insert(reference.name).second,
               "Duplicate parameter alias " << reference.name << " in external constraint " << _name_);
    LogThrowIf(parameterConfig.hasField("parameterName") == parameterConfig.hasField("parameterIndex"),
               "Specify exactly one of parameterName and parameterIndex in external constraint " << _name_);
    if( parameterConfig.hasField("parameterName") ){
      reference.parameterName = parameterConfig.fetchValue<std::string>("parameterName");
      LogThrowIf(reference.parameterName.empty(), "Empty parameterName in external constraint " << _name_);
    }
    else{
      reference.parameterIndex = parameterConfig.fetchValue<int>("parameterIndex");
      LogThrowIf(reference.parameterIndex < 0, "Negative parameterIndex in external constraint " << _name_);
    }
    _parameterReferenceList_.emplace_back(reference);
    parameterConfig.printUnusedKeys();
  }
}

void GaussianConstraint::initializeImpl(){
  if( not _isEnabled_ ){ return; }
  // Keep formulas out of ROOT's global registry. Evaluation uses local parameter arrays.
  _formula_ = std::make_shared<TFormula>(_name_.c_str(), _formulaString_.c_str(), false);
  LogThrowIf(not _formula_->IsValid(), "Invalid formula in external constraint " << _name_ << ": " << _formulaString_);
  LogThrowIf(_formula_->GetNdim() != 0,
             "Use only declared [alias] inputs, not x/y/z/t variables, in external constraint " << _name_);
  LogThrowIf(_formula_->GetNpar() != int(_parameterReferenceList_.size()),
             "Formula inputs do not match declared aliases in external constraint " << _name_);
  for( auto& reference : _parameterReferenceList_ ){
    reference.formulaIndex = _formula_->GetParNumber(reference.name.c_str());
    LogThrowIf(reference.formulaIndex < 0,
               "Unused parameter alias " << reference.name << " in external constraint " << _name_);
  }
  _config_.printUnusedKeys();
}

const Parameter& GaussianConstraint::resolveParameter(const ParametersManager& parametersManager_, const ParameterReference& reference_) const{
  const Parameter* selectedParameter{nullptr};
  int matchingSets{0};
  for( const auto& parameterSet : parametersManager_.getParameterSetsList() ){
    if( parameterSet.getName() != reference_.parameterSetName ){ continue; }
    ++matchingSets;
    for( const auto& parameter : parameterSet.getParameterList() ){
      if( reference_.parameterIndex >= 0
          ? parameter.getParameterIndex() != reference_.parameterIndex
          : parameter.getName() != reference_.parameterName ){ continue; }
      LogThrowIf(selectedParameter != nullptr,
                 "Ambiguous parameter reference " << reference_.name << " in external constraint " << _name_);
      selectedParameter = &parameter;
    }
  }
  LogThrowIf(matchingSets != 1 or selectedParameter == nullptr,
             "Cannot uniquely resolve " << reference_.parameterSetName << "/" << reference_.name
             << " in external constraint " << _name_);
  return *selectedParameter;
}

void GaussianConstraint::validateParameters(const ParametersManager& parametersManager_) const{
  if( not _isEnabled_ ){ return; }
  for( const auto& reference : _parameterReferenceList_ ){
    resolveParameter(parametersManager_, reference);
  }
}

double GaussianConstraint::evalFormula(const ParametersManager& parametersManager_) const{
  LogThrowIf(_formula_ == nullptr or not _formula_->IsValid(), "External constraint has no initialized formula: " << _name_);
  std::vector<double> values(_parameterReferenceList_.size());
  for( const auto& reference : _parameterReferenceList_ ){
    LogThrowIf(reference.formulaIndex < 0, "External constraint is not initialized: " << _name_);
    values[reference.formulaIndex] = resolveParameter(parametersManager_, reference).getParameterValue();
  }
  return _formula_->EvalPar(nullptr, values.data());
}

double GaussianConstraint::evalResidual(const ParametersManager& parametersManager_) const{
  return (evalFormula(parametersManager_) - _mean_) / _stdDev_;
}

double GaussianConstraint::eval(const ParametersManager& parametersManager_) const{
  if( not _isEnabled_ ){ return 0; }
  double residual = evalResidual(parametersManager_);
  if( not std::isfinite(residual) ){ return std::numeric_limits<double>::infinity(); }
  return residual * residual;
}
