//
// Created by Nadrino on 21/04/2021.
//

#include "GenericToolbox.Json.h"
#include "ConfigUtils.h"
#include "GundamGreetings.h"
#include "GundamUtils.h"

#include "Logger.h"
#include "CmdLineParser.h"
#include "GenericToolbox.Root.h"

#include "nlohmann/json.hpp"

#include <string>





int main( int argc, char** argv ){

  GundamGreetings g;
  g.setAppName("config unfold tool");
  g.hello();

  CmdLineParser clp(argc, argv);
  clp.addOption("configFile", {"-c"}, "Provide YAML/Json configuration file.", 1);
  clp.addOption("overrideFiles", {"-of", "--override-files"}, "Provide config files that will override keys", -1);

  clp.addOption("output-file", {"-o"}, "Set output file name.", 1, true);
  clp.addOption("appendix", {"--appendix"}, "Add appendix to output file name", 1);

  LogInfo << "Available options: " << std::endl;
  LogInfo << clp.getConfigSummary() << std::endl;

  clp.parseCmdLine();

  LogWarning << "Command line options:" << std::endl;
  LogWarning << clp.getValueSummary() << std::endl;

  LogInfo << "Reading config..." << std::endl;

  // Import
  ConfigUtils::ConfigHandler configHandler( clp.getOptionVal<std::string>("configFile") );
  configHandler.override( clp.getOptionValList<std::string>("overrideFiles") );

  // Export
  if( clp.isOptionTriggered("output-file") ){

    // appendixDict["optionName"] = "Appendix"
    // this list insure all appendices will appear in the same order
    std::vector<std::pair<std::string, std::string>> appendixDict{
        {"configFile", ""},
        {"overrideFiles", "With"},
        {"appendix", ""},
    };

    std::string outPath{GundamUtils::generateFileName(clp, appendixDict) + ".json"};

    if( clp.getNbValueSet("output-file") != 0 ){
      outPath = clp.getOptionVal<std::string>("output-file");
    }

    // export now
    configHandler.exportToJsonFile( outPath );
  }
  else{
    std::cout << configHandler.toString() << std::endl;
    LogWarning << ">> To dump config in file, use -o option." << std::endl;
  }

  g.goodbye();
  return EXIT_SUCCESS;
}
