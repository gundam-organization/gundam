//
// Created by Nadrino on 21/04/2021.
//

#include "GenericToolbox.Json.h"
#include "ConfigUtils.h"
#include "GundamGreetings.h"

#include "Logger.h"
#include "CmdLineParser.h"
#include "GenericToolbox.h"
#include "GenericToolbox.Root.h"

#include "nlohmann/json.hpp"

#include "string"



LoggerInit([]{
  Logger::setUserHeaderStr("[gundamConfigForwarder.cxx]");
});

int main( int argc, char** argv ){

  GundamGreetings g;
  g.setAppName("ConfigForwarder");
  g.hello();

  CmdLineParser clp(argc, argv);
  clp.addOption("config-file", {"-c"}, "Provide YAML/Json configuration file.", 1);
  clp.addOption("fit-file", {"-f"}, "Provide ROOT fit file which contains the configuration.", 1);
  clp.addOption("output-file-path", {"-o"}, "Set output file name.", 1);
  clp.addOption("overrideFiles", {"--override-files"}, "Provide config files that will override keys", -1);

  LogInfo << "Available options: " << std::endl;
  LogInfo << clp.getConfigSummary() << std::endl;

  clp.parseCmdLine();

  LogWarning << "Command line options:" << std::endl;
  LogWarning << clp.getValueSummary() << std::endl;

  LogInfo << "Reading config..." << std::endl;
  nlohmann::json config;
  if( clp.isOptionTriggered("config-file") ){
    auto configFilePath = clp.getOptionVal<std::string>("config-file");
    LogInfo << "Reading configuration file..." << std::endl;
    config = ConfigUtils::readConfigFile(configFilePath);
  }
  else if( clp.isOptionTriggered("fit-file") ){
    auto fitFilePath = clp.getOptionVal<std::string>("fit-file");
    auto* fitFile = GenericToolbox::openExistingTFile(fitFilePath);
    auto* conf = fitFile->Get<TNamed>("gundamFitter/unfoldedConfig_TNamed");
    LogThrowIf(conf==nullptr, "no config in ROOT file " << fitFilePath);
    config = GenericToolbox::Json::readConfigJsonStr(conf->GetTitle());
    fitFile->Close();
    delete fitFile;
  }
  else{
    LogThrow("No input provided.");
  }

  LogInfo << "Unfolding configuration file..." << std::endl;
  ConfigUtils::unfoldConfig(config);

  for( auto& overrideFile: clp.getOptionValList<std::string>("overrideFiles") ){
    LogInfo << "Overriding config with \"" << overrideFile << "\"" << std::endl;

    LogThrowIf(not GenericToolbox::doesPathIsFile(overrideFile), "Could not find " << overrideFile);

    auto jsonOverride = ConfigUtils::readConfigFile( overrideFile );
    ConfigUtils::unfoldConfig( jsonOverride );

    ConfigUtils::applyOverrides(config, jsonOverride);
  }



  auto configStr = GenericToolbox::Json::toReadableString(config);

  if( clp.isOptionTriggered("output-file-path") ){
    auto fConfigFilePath = clp.getOptionVal<std::string>("output-file-path");
    if( not GenericToolbox::doesStringEndsWithSubstring(fConfigFilePath, ".json") ) fConfigFilePath += ".json";


    LogInfo << "Writing as: " << fConfigFilePath << std::endl;
    GenericToolbox::dumpStringInFile(fConfigFilePath, configStr);
    LogInfo << "Unfolded config written as: " << fConfigFilePath << std::endl;
  }
  else{
    std::cout << configStr << std::endl;
    LogWarning << ">> To dump config in file, use -o option." << std::endl;
  }

  g.goodbye();
  return EXIT_SUCCESS;
}
