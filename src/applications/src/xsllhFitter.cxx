//
// Created by Adrien BLANCHET on 01/06/2021.
//

#include "string"

#include "yaml-cpp/yaml.h"

#include "CmdLineParser.h"
#include "Logger.h"

#include "JsonUtils.h"
#include "ParameterPropagator.h"
#include "GlobalVariables.h"

LoggerInit([](){
  Logger::setUserHeaderStr("[xsllhFitter.cxx]");
} )

int main(int argc, char** argv){
  CmdLineParser clParser;

  clParser.addTriggerOption("dry-run", {"--dry-run", "-d"},"Perform the full sequence of initialization, but don't do the actual fit.");
  clParser.addOption("config-file", {"-c", "--config-file"}, "Specify path to the fitter config file");
  clParser.addOption("nb-threads", {"-t", "--nb-threads"}, "Specify nb of parallel threads");

  LogInfo << "Usage: " << std::endl;
  LogInfo << clParser.getConfigSummary() << std::endl << std::endl;

  clParser.parseCmdLine(argc, argv);

  LogInfo << "Provided arguments: " << std::endl;
  LogInfo << clParser.getValueSummary() << std::endl << std::endl;
  LogInfo << clParser.dumpConfigAsJsonStr() << std::endl;

  auto configFilePath = clParser.getOptionVal<std::string>("config-file");

  int nThreads = 1;
  if( clParser.isOptionTriggered("nb-threads") ) nThreads = clParser.getOptionVal<int>("nb-threads");
  GlobalVariables::setNbThreads(nThreads);

  LogInfo << "Reading config file: " << configFilePath << std::endl;
  auto jsonConfig = JsonUtils::readConfigFile(configFilePath); // works with yaml

  ParameterPropagator parProp;
  parProp.setParameterSetConfig(JsonUtils::fetchValue<nlohmann::json>(jsonConfig, "fitParameterSets"));
  parProp.setSamplesConfig(JsonUtils::fetchValue<nlohmann::json>(jsonConfig, "samples"));

  TFile* f = TFile::Open(JsonUtils::fetchValue<std::string>(jsonConfig, "mc_file").c_str(), "READ");
  parProp.setDataTree( f->Get<TTree>("selectedEvents") );

  parProp.setMcFilePath(JsonUtils::fetchValue<std::string>(jsonConfig, "mc_file"));

  parProp.initialize();

  for( const auto& parameterSet : parProp.getParameterSetsList() ){
    LogInfo << parameterSet.getSummary() << std::endl;
  }

}