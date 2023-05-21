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
  Logger::getUserHeader() << "[" << FILENAME << "]";
});

int main( int argc, char** argv ){

  GundamGreetings g;
  g.setAppName("config unfolder tool");
  g.hello();

  CmdLineParser clp(argc, argv);
  clp.addOption("config-file", {"-c"}, "Provide YAML/Json configuration file.", 1);
  clp.addOption("output-file-path", {"-o"}, "Set output file name.", 1);
  clp.addOption("overrideFiles", {"-of", "--override-files"}, "Provide config files that will override keys", -1);

  LogInfo << "Available options: " << std::endl;
  LogInfo << clp.getConfigSummary() << std::endl;

  clp.parseCmdLine();

  LogWarning << "Command line options:" << std::endl;
  LogWarning << clp.getValueSummary() << std::endl;

  LogInfo << "Reading config..." << std::endl;

  // Import
  ConfigUtils::ConfigHandler configHandler( clp.getOptionVal<std::string>("config-file") );

  // Edit
  configHandler.override( clp.getOptionValList<std::string>("overrideFiles") );

  // Export
  if( clp.isOptionTriggered("output-file-path") ){
    configHandler.exportToJsonFile( clp.getOptionVal<std::string>("output-file-path") );
  }
  else{
    std::cout << configHandler.toString() << std::endl;
    LogWarning << ">> To dump config in file, use -o option." << std::endl;
  }

  g.goodbye();
  return EXIT_SUCCESS;
}
