#include "ExternalConstraints.h"

#include "Logger.h"

#include <set>

void ExternalConstraints::configureImpl(){
  _config_.clearFields();
  _config_.defineFields({{"isEnabled"}, {"constraintsList"}});
  _config_.checkConfiguration();
  _isEnabled_ = _config_.fetchValue<bool>("isEnabled", false);
  _gaussianConstraintList_.clear();
  if( not _isEnabled_ ){ return; }

  std::set<std::string> names;
  if( _config_.hasField("constraintsList") ){
    for( const auto& constraintConfig : _config_.loop("constraintsList") ){
      _gaussianConstraintList_.emplace_back();
      _gaussianConstraintList_.back().configure(constraintConfig);
      const auto& name = _gaussianConstraintList_.back().getName();
      LogThrowIf(not names.insert(name).second, "Duplicate external constraint name: " << name);
    }
  }

  if( _gaussianConstraintList_.empty() ){
    _isEnabled_ = false;
  }

}

void ExternalConstraints::initializeImpl(){
  if( not _isEnabled_ ){ return; }
  for( auto& constraint : _gaussianConstraintList_ ){ constraint.initialize(); }
  _config_.printUnusedKeys();
}

void ExternalConstraints::validateParameters(const ParametersManager& parametersManager_) const{
  if( not _isEnabled_ ){ return; }
  for( const auto& constraint : _gaussianConstraintList_ ){ constraint.validateParameters(parametersManager_); }
}

double ExternalConstraints::eval(const ParametersManager& parametersManager_) const{
  if( not _isEnabled_ ){ return 0; }
  double penalty{0};
  for( const auto& constraint : _gaussianConstraintList_ ){ penalty += constraint.eval(parametersManager_); }
  return penalty;
}
