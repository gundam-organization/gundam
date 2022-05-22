//
// Created by Adrien BLANCHET on 16/02/2022.
//

#include "GundamGreetings.h"
#include "JsonUtils.h"

#include "CmdLineParser.h"
#include "Logger.h"
#include "GenericToolbox.h"

#include "string"
#include "vector"
#include <cstdlib>

LoggerInit([]{
Logger::setUserHeaderStr("[gundamConfigCompare.cxx]");
Logger::setPrefixFormat("{TIME} {USER_HEADER}");
})

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
  clp.addOption("show-all-keys", {"-a"}, "Show all keys.", 0);

  clp.parseCmdLine();

  if( clp.isNoOptionTriggered()
    or not clp.isOptionTriggered("config-1")
    or not clp.isOptionTriggered("config-2")
  ){
    LogError << "Missing options. Reminding usage..." << std::endl;
    LogInfo << clp.getConfigSummary() << std::endl;
    exit(EXIT_FAILURE);
  }

  LogInfo << "Reading config..." << std::endl;
  auto configPath1 = clp.getOptionVal<std::string>("config-1");
  auto configPath2 = clp.getOptionVal<std::string>("config-2");
  if( clp.isOptionTriggered("show-all-keys") ){ __showAllKeys__ = true; }

  LogThrowIf(not GenericToolbox::doesPathIsFile(configPath1), configPath1 << " not found.")
  LogThrowIf(not GenericToolbox::doesPathIsFile(configPath2), configPath2 << " not found.")

  auto config1 = JsonUtils::readConfigFile(configPath1);
  auto config2 = JsonUtils::readConfigFile(configPath2);

  JsonUtils::unfoldConfig(config1);
  JsonUtils::unfoldConfig(config2);

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
    std::vector<std::string> keysToFetch{JsonUtils::ls(subConfig1)};
    for( auto& key2 : JsonUtils::ls(subConfig2) ){
      if( not GenericToolbox::doesElementIsInVector(key2, keysToFetch) ){ keysToFetch.emplace_back(key2); }
    }

    for( auto& key : keysToFetch ){
      if     ( not JsonUtils::doKeyExist(subConfig1, key) ){
        LogError << path <<  " -> missing key \"" << key << "\" in c1." << std::endl;
        continue;
      }
      else if( not JsonUtils::doKeyExist(subConfig2, key ) ){
        LogError << path << " -> missing key \"" << key << "\" in c2." << std::endl;
        continue;
      }

      // both have the key:
      auto content1 = JsonUtils::fetchValue<nlohmann::json>(subConfig1, key);
      auto content2 = JsonUtils::fetchValue<nlohmann::json>(subConfig2, key);

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
