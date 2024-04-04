//
// Created by Nadrino on 22/02/2024.
//

#ifndef GUNDAM_PLUGIN_JOINT_PROBABILITY_H
#define GUNDAM_PLUGIN_JOINT_PROBABILITY_H

#include "JointProbabilityBase.h"

#include <string>
#include <dlfcn.h>


namespace JointProbability{

  class PluginJointProbability : public JointProbabilityBase {

  public:
    [[nodiscard]] std::string getType() const override { return "PluginJointProbability"; }

    [[nodiscard]] double eval(const Sample& sample_, int bin_) const override;

    std::string llhPluginSrc;
    std::string llhSharedLib;

  protected:
    void readConfigImpl() override;
    void initializeImpl() override;
    void compile();
    void load();

  private:
    void* fLib{nullptr};
    void* evalFcn{nullptr};

  };

}


namespace JointProbability{

  // JointProbabilityPlugin
  void PluginJointProbability::readConfigImpl(){
    llhPluginSrc = GenericToolbox::Json::fetchValue<std::string>(_config_, "llhPluginSrc");
    llhSharedLib = GenericToolbox::Json::fetchValue<std::string>(_config_, "llhSharedLib");
  }

  void PluginJointProbability::initializeImpl(){
    if( not llhSharedLib.empty()) this->load();
    else if( not llhPluginSrc.empty()){
      this->compile();
      this->load();
    }
    else{ LogThrow("Can't initialize JointProbabilityPlugin without llhSharedLib nor llhPluginSrc."); }
  }

  double PluginJointProbability::eval( const Sample &sample_, int bin_ ) const{
    LogThrowIf(evalFcn == nullptr, "Library not loaded properly.");
    return reinterpret_cast<double (*)( double, double, double )>(evalFcn)(
        sample_.getDataContainer().getHistogram().binList[bin_].content,
        sample_.getMcContainer().getHistogram().binList[bin_].content,
        sample_.getMcContainer().getHistogram().binList[bin_].error
    );
  }

  void PluginJointProbability::compile(){
    LogInfo << "Compiling: " << llhPluginSrc << std::endl;
    llhSharedLib = GenericToolbox::replaceExtension(llhPluginSrc, "so");

    // create library
    std::stringstream ss;
    LogThrowIf(getenv("CXX") == nullptr, "CXX env is not set. Can't compile.");
    ss << "$CXX -std=c++11 -shared " << llhPluginSrc << " -o " << llhSharedLib;
    LogThrowIf(system(ss.str().c_str()) != 0, "Compile command failed.");
  }

  void PluginJointProbability::load(){
    LogInfo << "Loading shared lib: " << llhSharedLib << std::endl;
    fLib = dlopen(llhSharedLib.c_str(), RTLD_LAZY);
    LogThrowIf(fLib == nullptr, "Cannot open library: " << dlerror());
    evalFcn = (dlsym(fLib, "evalFct"));
    LogThrowIf(evalFcn == nullptr, "Cannot open evalFcn");
  }

}

#endif // GUNDAM_PLUGIN_JOINT_PROBABILITY_H
