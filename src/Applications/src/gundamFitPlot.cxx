//
// Created by Nadrino on 02/04/2024.
//

#include "GundamGreetings.h"
#include "GundamApp.h"
#include "ConfigUtils.h"
#include "GundamUtils.h"

#include "Logger.h"
#include "CmdLineParser.h"
#include "GenericToolbox.Json.h"


LoggerInit([]{
  Logger::getUserHeader() << "[" << FILENAME << "]";
});


int main( int argc, char** argv ){
  GundamGreetings g;
  g.setAppName("fit compare tool");
  g.hello();

  CmdLineParser clp;

  clp.addDummyOption("Main options");
  clp.addOption("configFile", {"-c"}, "Specify config file.", 1);
  clp.addOption("overrideFiles", {"-of", "--override-files"}, "Provide config files that will override keys", -1);
  clp.addOption("outputFilePath", {"-o", "--out-file"}, "Specify the output file", 1);
  clp.addOption("appendix", {"--appendix"}, "Add appendix to the output file name", 1);

  clp.addDummyOption("");

  LogInfo << "Options list:" << std::endl;
  LogInfo << clp.getConfigSummary() << std::endl << std::endl;

  // read cli
  clp.parseCmdLine(argc, argv);

  // printout what's been fired
  LogInfo << "Fired options are: " << std::endl << clp.getValueSummary() << std::endl;

  if( not clp.isOptionTriggered("config") ){
    LogError << "No config file provided." << std::endl;
    exit(EXIT_FAILURE);
  }

  // Reading configuration
  auto configFilePath = clp.getOptionVal("configFile", "");
  LogThrowIf(configFilePath.empty(), "Config file not provided.");

  ConfigUtils::ConfigHandler configHandler(configFilePath);
  configHandler.override( clp.getOptionValList<std::string>("overrideFiles") );

  // Output file path
  std::string outFileName;
  if( clp.isOptionTriggered("outputFilePath") ){ outFileName = clp.getOptionVal("outputFilePath", outFileName + ".root"); }
  else{

    std::string outFolder{"./"};
    if     ( clp.isOptionTriggered("outputDir") ){ outFolder = clp.getOptionVal<std::string>("outputDir"); }
    else if( GenericToolbox::Json::doKeyExist(configHandler.getConfig(), "outputFolder") ){
      outFolder = GenericToolbox::Json::fetchValue<std::string>(configHandler.getConfig(), "outputFolder");
    }

    // appendixDict["optionName"] = "Appendix"
    // this list insure all appendices will appear in the same order
    std::vector<std::pair<std::string, std::string>> appendixDict{
        {"configFile", "%s"},
        {"overrideFiles", "With_%s"},
        {"appendix", "%s"}
    };

    outFileName = GenericToolbox::joinPath(
        outFolder,
        GundamUtils::generateFileName(clp, appendixDict)
    ) + ".root";
  }

  GundamApp app{"main fitter"};

  // to write cmdLine info
  app.setCmdLinePtr( &clp );

  // unfolded config
  app.setConfigString( GenericToolbox::Json::toReadableString(configHandler.getConfig()) );

  // Ok, we should run. Create the out file.
  app.openOutputFile(outFileName);
  app.writeAppInfo();



  g.goodbye();
  return EXIT_SUCCESS;
}
