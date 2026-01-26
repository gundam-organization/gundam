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

    [[nodiscard]] double eval(double data_, double pred_, double err_, int bin_) const override;

    std::string llhPluginSrc;
    std::string llhSharedLib;

  protected:
    void configureImpl() override;
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
  inline void PluginJointProbability::configureImpl(){
    _config_.fillValue(llhPluginSrc, "llhPluginSrc");
    _config_.fillValue(llhSharedLib, "llhSharedLib");
  }

  inline void PluginJointProbability::initializeImpl(){
    if( not llhSharedLib.empty()) this->load();
    else if( not llhPluginSrc.empty()){
      this->compile();
      this->load();
    }
    else{ LogThrow("Can't initialize JointProbabilityPlugin without llhSharedLib nor llhPluginSrc."); }
  }

  inline double PluginJointProbability::eval(double data_, double pred_, double err_, int bin_) const {
    LogThrowIf(evalFcn == nullptr, "Library not loaded properly.");
    return reinterpret_cast<double (*)( double, double, double )>(evalFcn)(
        data_, pred_, err_
    );
  }

  inline void PluginJointProbability::compile(){
    LogInfo << "Compiling: " << llhPluginSrc << std::endl;
    llhSharedLib = GenericToolbox::replaceExtension(llhPluginSrc, "so");

    // create library
    std::stringstream ss;
    LogThrowIf(getenv("CXX") == nullptr, "CXX env is not set. Can't compile.");
    ss << "$CXX -std=c++11 -shared " << llhPluginSrc << " -o " << llhSharedLib;
    LogThrowIf(system(ss.str().c_str()) != 0, "Compile command failed.");
  }

  inline void PluginJointProbability::load(){
    LogInfo << "Loading shared lib: " << llhSharedLib << std::endl;
    fLib = dlopen(llhSharedLib.c_str(), RTLD_LAZY);
    LogThrowIf(fLib == nullptr, "Cannot open library: " << dlerror());
    evalFcn = (dlsym(fLib, "evalFct"));
    LogThrowIf(evalFcn == nullptr, "Cannot open evalFcn");
  }

}

#endif // GUNDAM_PLUGIN_JOINT_PROBABILITY_H
