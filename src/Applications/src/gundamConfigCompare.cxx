//
// Created by Adrien BLANCHET on 16/02/2022.
//

#include "GundamGreetings.h"
#include "ConfigUtils.h"

#include "CmdLineParser.h"
#include "Logger.h"
#include "GenericToolbox.h"
#include "GenericToolbox.Root.h"
#include "GenericToolbox.Json.h"

#include "nlohmann/json.hpp"

#include "string"
#include "vector"
#include <cstdlib>

LoggerInit([]{
Logger::setUserHeaderStr("[gundamConfigCompare.cxx]");
Logger::setPrefixFormat("{TIME} {USER_HEADER}");
});

bool __showAllKeys__{false};
std::vector<std::string> __pathBuffer__;

void compareConfigStage(const nlohmann::json& subConfig1, const nlohmann::json& subConfig2);


int main( int argc, char** argv ){
  GundamGreetings g;
  g.setAppName("ConfigCompare");
  g.hello();

  CmdLineParser clp(argc, argv);
  clp.addOption("config-1", {"-c1"}, "Path to first config file.", 1);
  clp.addOption("config-2", {"-c2"}, "Path to second config file.", 1);
  clp.addOption("file-1", {"-f1"}, "Path to first output fit file.", 1);
  clp.addOption("file-2", {"-f2"}, "Path to second output fit file.", 1);
  clp.addOption("show-all-keys", {"-a"}, "Show all keys.", 0);

  clp.parseCmdLine();

  if( clp.isNoOptionTriggered()
    or not ( clp.isOptionTriggered("config-1") or clp.isOptionTriggered("file-1") )
    or not ( clp.isOptionTriggered("config-2") or clp.isOptionTriggered("file-2") )
  ){
    LogError << "Missing options. Reminding usage..." << std::endl;
    LogInfo << clp.getConfigSummary() << std::endl;
    exit(EXIT_FAILURE);
  }

  LogInfo << "Reading config..." << std::endl;
  std::string configPath1;
  std::string configPath2;

  if     ( clp.isOptionTriggered("config-1") ){ configPath1 = clp.getOptionVal<std::string>("config-1"); }
  else if( clp.isOptionTriggered("file-1")   ){ configPath1 = clp.getOptionVal<std::string>("file-1"); }
  if     ( clp.isOptionTriggered("config-2") ){ configPath2 = clp.getOptionVal<std::string>("config-2"); }
  else if( clp.isOptionTriggered("file-2")   ){ configPath2 = clp.getOptionVal<std::string>("file-2"); }

  if( clp.isOptionTriggered("show-all-keys") ){ __showAllKeys__ = true; }

  LogThrowIf(not GenericToolbox::doesPathIsFile(configPath1), configPath1 << " not found.");
  LogThrowIf(not GenericToolbox::doesPathIsFile(configPath2), configPath2 << " not found.");

  nlohmann::json config1;
  if     ( clp.isOptionTriggered("config-1") ){ config1 = ConfigUtils::readConfigFile(configPath1); }
  else if( clp.isOptionTriggered("file-1") ){
    LogInfo << "config-1 is within the output of a fit run." << std::endl;
    LogThrowIf(not GenericToolbox::doesTFileIsValid(configPath1, {"gundamFitter/unfoldedConfig_TNamed"}),
               "Could not find config in file " << configPath1
    );
    auto* f = TFile::Open(configPath1.c_str());
    auto* conf = f->Get<TNamed>("gundamFitter/unfoldedConfig_TNamed");
    auto* version = f->Get<TNamed>("gundamFitter/gundamVersion_TNamed");
    auto* cmdLine = f->Get<TNamed>("gundamFitter/commandLine_TNamed");
    LogInfo << "Was run under GUNDAM v" << version->GetTitle() << " with cmdLine: "<< cmdLine->GetTitle() << std::endl;
    config1 = GenericToolbox::Json::readConfigJsonStr(conf->GetTitle());
    delete f;
  }

  nlohmann::json config2;
  if     ( clp.isOptionTriggered("config-2") ){ config2 = ConfigUtils::readConfigFile(configPath2); }
  else if( clp.isOptionTriggered("file-2") ){
    LogInfo << "config-1 is within the output of a fit run." << std::endl;
    LogThrowIf(not GenericToolbox::doesTFileIsValid(configPath2, {"gundamFitter/unfoldedConfig_TNamed"}),
               "Could not find config in file " << configPath2
    );
    auto* f = TFile::Open(configPath2.c_str());
    auto* conf = f->Get<TNamed>("gundamFitter/unfoldedConfig_TNamed");
    auto* version = f->Get<TNamed>("gundamFitter/gundamVersion_TNamed");
    auto* cmdLine = f->Get<TNamed>("gundamFitter/commandLine_TNamed");
    LogInfo << "Was run under GUNDAM v" << version->GetTitle() << " with cmdLine: "<< cmdLine->GetTitle() << std::endl;
    config2 = GenericToolbox::Json::readConfigJsonStr(conf->GetTitle());
    delete f;
  }

  ConfigUtils::unfoldConfig(config1);
  ConfigUtils::unfoldConfig(config2);

  compareConfigStage(config1, config2);

  g.goodbye();

  return EXIT_SUCCESS;
}


void compareConfigStage(const nlohmann::json& subConfig1, const nlohmann::json& subConfig2){
  std::string path = GenericToolbox::joinVectorString(__pathBuffer__, "/");

  if( subConfig1.is_array() and subConfig2.is_array() ){

    if( subConfig1.size() != subConfig2.size() ){
      LogError << path << "Array size mismatch: " << subConfig1.size() << " <-> " << subConfig2.size() << std::endl;
    }
    else{
      for( int iEntry = 0 ; iEntry < subConfig1.size() ; iEntry++ ){
        __pathBuffer__.emplace_back("#"+std::to_string(iEntry));
        compareConfigStage(subConfig1[iEntry], subConfig2[iEntry]);
        __pathBuffer__.pop_back();
      }
    }

  }
  else if( subConfig1.is_structured() and subConfig2.is_structured() ){
    std::vector<std::string> keysToFetch{GenericToolbox::Json::ls(subConfig1)};
    for( auto& key2 : GenericToolbox::Json::ls(subConfig2) ){
      if( not GenericToolbox::doesElementIsInVector(key2, keysToFetch) ){ keysToFetch.emplace_back(key2); }
    }

    for( auto& key : keysToFetch ){
      if     ( not GenericToolbox::Json::doKeyExist(subConfig1, key) ){
        LogError << path <<  " -> missing key \"" << key << "\" in c1." << std::endl;
        continue;
      }
      else if( not GenericToolbox::Json::doKeyExist(subConfig2, key ) ){
        LogError << path << " -> missing key \"" << key << "\" in c2." << std::endl;
        continue;
      }

      // both have the key:
      auto content1 = GenericToolbox::Json::fetchValue<nlohmann::json>(subConfig1, key);
      auto content2 = GenericToolbox::Json::fetchValue<nlohmann::json>(subConfig2, key);

      __pathBuffer__.emplace_back(key);
      compareConfigStage(content1, content2);
      __pathBuffer__.pop_back();
    }
  }
  else{
    if( subConfig1 != subConfig2 ){
      LogWarning << path << ": " << subConfig1 << " <-> " << subConfig2 << std::endl;
    }
    else if( __showAllKeys__ ){
      LogInfo << path << ": " << subConfig1 << " <-> " << subConfig2 << std::endl;
    }
  }

}
