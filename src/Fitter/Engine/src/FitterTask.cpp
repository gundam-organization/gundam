//
// Created by Nadrino on 04/02/2025.
//

#include "FitterTask.h"
#include "FitterEngine.h"
#include "RootMinimizer.h"

#include "Logger.h"


void FitterTask::run(FitterEngine *owner_){

  LogInfo << std::endl;
  LogInfo << "Running fitter task: " << _name_ << std::endl;


  if( not _isEnabled_ ){
    LogWarning << "Task is disabled. Skipping..." << std::endl;
    return;
  }


  // ---------------------
  // First handle every request about the Likelihood interface


  // ---------------------
  // Then minimizer related settings
  // TODO: handling minimizer type swap ( ROOT minimizer <-> MCMC )
  // reconfigure by overriding parameters
  if( owner_->getMinimizerPtr() == nullptr ) {

  }
  else{
    ConfigUtils::ConfigHandler conf{owner_->getMinimizer().getConfig()};
    conf.override(_config_);
    owner_->getMinimizer().setConfig(_config_);
    owner_->getMinimizer().configure();
    owner_->getMinimizer().initialize();
  }


  // ---------------------
  // check additional actions
  std::string actionStr{};
  GenericToolbox::Json::fillValue(_config_, actionStr, "action");
  if     ( actionStr == "minimize" )  { owner_->getMinimizer().minimize(); }
  else if( actionStr == "calcErrors" ){ owner_->getMinimizer().calcErrors(); }

  // outputs -> file write?

}

void FitterTask::configureImpl(){

  GenericToolbox::Json::fillValue(_config_, _name_, "name");
  LogExitIf(_name_.empty());

  GenericToolbox::Json::fillValue(_config_, _isEnabled_, "isEnabled");
  if( not _isEnabled_ ){ return; }


}
