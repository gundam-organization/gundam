#include "CliConfigOverride.h"

#include "CmdLineParser.h"
#include "Logger.h"

#include <cmath>

namespace {
  JsonType& getAliasedConfig(JsonType& config_, const char* key_, const char* alias_){
    if( config_.contains(alias_) ){
      if( not config_.contains(key_) ){ return config_.at(alias_); }
      if( config_.at(key_) == config_.at(alias_) ){ config_.erase(alias_); }
    }
    return config_[key_];
  }

  JsonType& getPropagatorConfig(JsonType& fitterConfig_){
    auto& likelihoodConfig = fitterConfig_["likelihoodInterfaceConfig"];
    if( not likelihoodConfig.contains("propagatorConfig") and fitterConfig_.contains("propagatorConfig") ){
      return fitterConfig_.at("propagatorConfig");
    }
    return likelihoodConfig["propagatorConfig"];
  }
}

namespace CliConfigOverride {
  void applyConfigOverride(const CmdLineParser& parser_, JsonType& config_){
    const auto isTriggered = [&](const char* option_){
      return parser_.isOptionDefined(option_) and parser_.isOptionTriggered(option_);
    };

    if( isTriggered("enablePca") ){
      auto& fitterConfig = config_["fitterEngineConfig"];
      fitterConfig["enablePca"] = true;
      const auto valueCount = parser_.getNbValueSet("enablePca");
      if( valueCount >= 1 ){ fitterConfig["pcaThreshold"] = parser_.getOptionVal<double>("enablePca", 0); }
      if( valueCount >= 2 ){ fitterConfig["pcaMethod"] = parser_.getOptionVal<std::string>("enablePca", 1); }
    }

    if( isTriggered("scanParameters") ){
      auto& fitterConfig = config_["fitterEngineConfig"];
      fitterConfig["enablePreFitScan"] = true;
      fitterConfig["enablePostFitScan"] = true;
      if( parser_.getNbValueSet("scanParameters") == 1 ){
        const auto nbPoints = parser_.getOptionVal<int>("scanParameters");
        if( not fitterConfig.contains("parameterScannerConfig") and not fitterConfig.contains("scanConfig") ){
          auto& propagatorConfig = getPropagatorConfig(fitterConfig);
          if( propagatorConfig.contains("scanConfig") ){
            propagatorConfig.at("scanConfig")["nbPoints"] = nbPoints;
          }
          else{ fitterConfig["parameterScannerConfig"]["nbPoints"] = nbPoints; }
        }
        else{ getAliasedConfig(fitterConfig, "parameterScannerConfig", "scanConfig")["nbPoints"] = nbPoints; }
      }
    }

    if( isTriggered("kickMc") ){
      const auto gain = parser_.getNbValueSet("kickMc") == 0 ? 0.1 : parser_.getOptionVal<double>("kickMc");
      LogAlert << "Fit starting point randomized by " << gain << " sigma around prior values." << std::endl;
      auto& fitterConfig = config_["fitterEngineConfig"];
      fitterConfig["throwMcBeforeFit"] = true;
      fitterConfig["throwMcBeforeFitGain"] = gain;
    }

    if( isTriggered("debugFracOfEntries") ){
      LogExitIf(parser_.getNbValueSet("debugFracOfEntries") != 1, "Dataset fraction not specified.");
      const auto fraction = parser_.getOptionVal<double>("debugFracOfEntries");
      LogExitIf(not std::isfinite(fraction) or fraction < 0. or fraction > 1.,
                "debugLoadDatasetFraction must be finite and between 0 and 1");
      config_["fitterEngineConfig"]["likelihoodInterfaceConfig"]["datasetManagerConfig"]["debugLoadDatasetFraction"] = fraction;
    }

    if( isTriggered("injectParameterConfig") ){
      const auto path = parser_.getOptionVal<std::string>("injectParameterConfig");
      if( not path.empty() ){
        LogWarning << "Inject parameter config: " << path << std::endl;
        getPropagatorConfig(config_["fitterEngineConfig"])["parameterInjection"] = ConfigUtils::readConfigFile(path);
      }
    }

    if( isTriggered("generateOneSigmaPlots") ){
      config_["fitterEngineConfig"]["generateOneSigmaPlots"] = true;
    }

    if( isTriggered("skipHesse") ){
      auto& minimizerConfig = getAliasedConfig(config_["fitterEngineConfig"], "minimizerConfig", "mcmcConfig");
      minimizerConfig["enablePostFitErrorFit"] = false;
    }
  }
}
