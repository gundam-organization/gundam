//
// Created by Adrien Blanchet on 23/05/2023.
//

#include "GundamGreetings.h"
#include "GundamUtils.h"
#include "ConfigUtils.h"

#include "GenericToolbox.h"
#include "GenericToolbox.Json.h"
#include "CmdLineParser.h"
#include "Logger.h"

#include <string>
#include <vector>
#include <cstdlib>


LoggerInit([]{
  Logger::getUserHeader() << "[" << FILENAME << "]";
});


int main(int argc, char** argv) {

  // --------------------------
  // Greetings:
  // --------------------------
  GundamGreetings g;
  g.setAppName("input zipper tool");
  g.hello();

  // --------------------------
  // Read Command Line Args:
  // --------------------------
  CmdLineParser clParser;
  clParser.getDescription() << " > " << FILENAME << " is a program that reads in the config inputs and copy inputs into a unified ZIP file." << std::endl;

  LogInfo << clParser.getDescription().str() << std::endl;

  clParser.addDummyOption("Options");
  clParser.addOption("configFile",      {"-c", "--config-file"}, "Specify path to the fitter config file");
  clParser.addOption("overrideFiles",   {"-of", "--override-files"}, "Provide config files that will override keys", -1);
  clParser.addOption("outputFolder",    {"-o", "--out-folder"}, "Output folder name");
  clParser.addOption("maxFileSizeInMb", {"--max-size"}, "Set the maximum size (in MB) an input file can be to be copied locally");

  clParser.addDummyOption();

  clParser.addTriggerOption("zipOutFolder", {"-z", "--zip"}, "Zip the output folder");

  LogInfo << "Usage: " << std::endl;
  LogInfo << clParser.getConfigSummary() << std::endl << std::endl;

  clParser.parseCmdLine(argc, argv);

  LogThrowIf(clParser.isNoOptionTriggered(), "No option was provided.");

  LogInfo << "Provided arguments: " << std::endl;
  LogInfo << clParser.getValueSummary() << std::endl << std::endl;
  LogInfo << clParser.dumpConfigAsJsonStr() << std::endl;

  LogThrowIf( not clParser.isOptionTriggered("configFile") );

  ConfigUtils::ConfigHandler configHandler( clParser.getOptionVal<std::string>("configFile") );
  configHandler.override( clParser.getOptionValList<std::string>("overrideFiles") );

  std::string outFolder;
  if( not clParser.isOptionTriggered("outputFolder") ){
    // appendixDict["optionName"] = "Appendix"
    // this list insure all appendices will appear in the same order
    std::vector<std::pair<std::string, std::string>> appendixDict{
        {"configFile", "%s"},
        {"overrideFiles", "With_%s"},
        {"maxFileSizeInMb", "MaxInputSize_%sMB"},
    };

    outFolder = {GundamUtils::generateFileName(clParser, appendixDict)};
  }
  else{
    outFolder = clParser.getOptionVal<std::string>("outputFolder");
  }

  LogInfo << "Output files will be written in: " << outFolder << std::endl;
  GenericToolbox::mkdirPath( outFolder );

  LogInfo << "Now copying input src files..." << std::endl;
  std::string pathBuffer;
  std::vector<std::string> recursivePathBufferList;
  std::function<void(nlohmann::json&)> recursive = [&](nlohmann::json& config_){

    if( config_.is_string() ){
      std::string srcPath = GenericToolbox::expandEnvironmentVariables(config_.get<std::string>());
      if( GenericToolbox::doesPathIsFile( srcPath ) ){

        double fSize{double( GenericToolbox::getFileSizeInBytes(srcPath) )};
        LogInfo << "Copying local file and overriding entry: " << GenericToolbox::getFileNameFromFilePath(srcPath)
        << " ( " << GenericToolbox::parseSizeUnits(fSize) << " )" << std::endl;

        if( clParser.isOptionTriggered("maxFileSizeInMb") ){
          if( fSize/1E6 > clParser.getOptionVal<double>("maxFileSizeInMb") ){
            LogAlert << "File too big wrt the threshold ( "
            << clParser.getOptionVal<double>("maxFileSizeInMb")
            << " MB ). Skipping the copy." << std::endl;
            return;
          }
        }
        auto localFolder{GenericToolbox::joinPath(recursivePathBufferList)};
        auto localPath{GenericToolbox::joinPath(localFolder, GenericToolbox::getFileNameFromFilePath(srcPath))};

        GenericToolbox::mkdirPath( GenericToolbox::joinPath(outFolder, localFolder) );
        GenericToolbox::copyFile( srcPath, GenericToolbox::joinPath(outFolder, localPath) );

        config_ = localPath;
      }

    }

    if( config_.is_primitive() ){ return; }

    for( auto& confEntry : config_.items() ){

      if( confEntry.value().is_structured() and GenericToolbox::Json::doKeyExist(confEntry.value(), "name") ){
        recursivePathBufferList.emplace_back( GenericToolbox::Json::fetchValue<std::string>(confEntry.value(), "name") );
      }
      else{
        recursivePathBufferList.emplace_back( confEntry.key() );
      }

      recursive( confEntry.value() );
      recursivePathBufferList.pop_back();
    }
  };
  recursive( configHandler.getConfig() );


  pathBuffer = GenericToolbox::joinPath(outFolder, "config.json");
  LogInfo << "Writing config under: " << pathBuffer << std::endl;
  GenericToolbox::dumpStringInFile( pathBuffer, configHandler.toString() );

  if( clParser.isOptionTriggered("zipOutFolder") ){
    LogInfo << "Creating a .gz archive of \"" << outFolder << "\"" << std::endl;
    auto err = std::system( ("tar -czvf \"" + outFolder + ".tar.gz\" \"" + outFolder + "\"").c_str() );
    LogThrowIf( err != 0, "Error while running the tar command." );
    LogInfo << "Removing created temp folder..." << std::endl;
    err = std::system( ("rm -r \"" + outFolder + "\"").c_str() );
    LogThrowIf( err != 0, "Error while running the tar command." );
    LogInfo << "Archive writen under \"" << outFolder << ".tar.gz\"" << std::endl;
  }

  return EXIT_SUCCESS;
}
