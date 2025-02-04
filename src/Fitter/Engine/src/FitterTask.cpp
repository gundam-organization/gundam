//
// Created by Nadrino on 04/02/2025.
//

#include "FitterTask.h"
#include "FitterEngine.h"
#include "RootMinimizer.h"


void FitterTask::run(FitterEngine *owner_){

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


}
