
//#include "JsonUtils.h"
#include "GundamGreetings.h"

#include "Logger.h"
#include "CmdLineParser.h"
//#include "GenericToolbox.h"

//#include "string"

#include <TFile.h>
#include "TDirectory.h"
#include <TH1D.h>
#include <TH2D.h>
#include <TMatrixT.h>
#include <TMatrixTSym.h>
#include <TMatrixD.h>

#include "FitParameterSet.h"
#include "FitSample.h"
#include "FitSampleSet.h"



#include "FitterEngine.h"
#include "VersionConfig.h"
#include "JsonUtils.h"
#include "GlobalVariables.h"
#ifdef GUNDAM_USING_CACHE_MANAGER
#include "CacheManager.h"
#endif
#include "GenericToolbox.h"
#include "GenericToolbox.Root.h"


#include <string>


LoggerInit([]{
  Logger::setUserHeaderStr("[gundamCalcXsec.cxx]");
});


int main(int argc, char** argv){

  GundamGreetings g;
  g.setAppName("GundamCalcXsec");
  g.hello();


  // --------------------------
  // Read Command Line Args:
  // --------------------------
  CmdLineParser clParser;

  clParser.addOption("configFile", {"-c", "--config-file"}, "Specify path to the fitter config file");
  clParser.addOption("fitterOutputFile", {"-f"}, "Specify the fitter output file");
  clParser.addOption("outputFile", {"-o", "--out-file"}, "Specify the CalcXsec output file");
  clParser.addOption("nbThreads", {"-t", "--nb-threads"}, "Specify nb of parallel threads");
  clParser.addOption("nToys", {"-n"}, "Specify number of toys");
  clParser.addOption("randomSeed", {"-s", "--seed"}, "Set random seed");

  LogInfo << "Usage: " << std::endl;
  LogInfo << clParser.getConfigSummary() << std::endl << std::endl;

  clParser.parseCmdLine(argc, argv);

  LogThrowIf(clParser.isNoOptionTriggered(), "No option was provided.");

  LogInfo << "Provided arguments: " << std::endl;
  LogInfo << clParser.getValueSummary() << std::endl << std::endl;


  if( clParser.isOptionTriggered("randomSeed") ){
    LogAlert << "Using user-specified random seed: " << clParser.getOptionVal<ULong_t>("randomSeed") << std::endl;
    gRandom->SetSeed(clParser.getOptionVal<ULong_t>("randomSeed"));
  }
  else{
    ULong_t seed = time(nullptr);
    LogInfo << "Using \"time(nullptr)\" random seed: " << seed << std::endl;
    gRandom->SetSeed(seed);
  }

  auto configFilePath = clParser.getOptionVal("configFile", "");
  LogThrowIf(configFilePath.empty(), "Config file not provided.");

  GlobalVariables::setNbThreads(clParser.getOptionVal("nbThreads", 1));
  LogInfo << "Running the fitter with " << GlobalVariables::getNbThreads() << " parallel threads." << std::endl;

  // --------------------------
  // Initialize the fitter:
  // --------------------------
  LogInfo << "Reading config file: " << configFilePath << std::endl;
  auto jsonConfig = JsonUtils::readConfigFile(configFilePath); // works with yaml

  if( JsonUtils::doKeyExist(jsonConfig, "minGundamVersion") ){
    LogThrowIf(
      not g.isNewerOrEqualVersion(JsonUtils::fetchValue<std::string>(jsonConfig, "minGundamVersion")),
      "Version check FAILED: " << GundamVersionConfig::getVersionStr() << " < " << JsonUtils::fetchValue<std::string>(jsonConfig, "minGundamVersion")
    );
    LogInfo << "Version check passed: " << GundamVersionConfig::getVersionStr() << " >= " << JsonUtils::fetchValue<std::string>(jsonConfig, "minGundamVersion") << std::endl;
  }



  // Open output of the fitter
  // Get configuration of fitter (TNamed) -> parse to json config file



  std::string outFileName = configFilePath;

  outFileName = clParser.getOptionVal("outputFile", outFileName + ".root");
  if( JsonUtils::doKeyExist(jsonConfig, "outputFolder") ){
    GenericToolbox::mkdirPath(JsonUtils::fetchValue<std::string>(jsonConfig, "outputFolder"));
    outFileName.insert(0, JsonUtils::fetchValue<std::string>(jsonConfig, "outputFolder") + "/");
  }

  LogWarning << "Creating output file: \"" << outFileName << "\"..." << std::endl;
  TFile* out = TFile::Open(outFileName.c_str(), "RECREATE");


  LogInfo << "Writing runtime parameters in output file..." << std::endl;

  // Gundam version?
  TNamed gundamVersionString("gundamVersion", GundamVersionConfig::getVersionStr().c_str());
  GenericToolbox::writeInTFile(GenericToolbox::mkdirTFile(out, "gundamCalcXsec"), &gundamVersionString);

  // Command line?
  TNamed commandLineString("commandLine", clParser.getCommandLineString().c_str());
  GenericToolbox::writeInTFile(GenericToolbox::mkdirTFile(out, "gundamFitter"), &commandLineString);

  // Create a propagator object

  
















  // Get best fit parameter values and postfit covariance matrix



  // Set best fit parameters and reweight events
  std::vector<FitParameterSet> _parameterSetsList_ =  fitter.getPropagator().getParameterSetsList();
  int i = 0;
  for( auto& parSet : _parameterSetsList_ ){
    for( auto& par : parSet.getParameterList() ){
      par.setParameterValue( postfit_params->GetBinContent(i+1) );
      ++i;
    }
  }




  // TODO: Get number of selected true signal events in each truth bin (after best fit reweight)
  // TODO: Get number of   all    true signal events in each truth bin (after best fit reweight)




  // Cholesky decompose postfit covariance matrix
  TMatrixT<double>* choleskyMatrix = GenericToolbox::getCholeskyMatrix(postfit_cov);



  // Create toys which throw the fit parameters according to chol.-decomp. postfit cov matrix
  // get number of toys from cmlParser
  for (int iToy = 0; iToy < 1000; iToy++) {
    std::vector<double> throws = GenericToolbox::throwCorrelatedParameters(choleskyMatrix);
    for (int j = 0; j < throws.size(); j++){
      throws.at(j) += postfit_params->GetBinContent(j+1);
      if ( throws.at(j) < 0) {
        throws.at(j) = 0.01;
      }
    }

    // Reweight events based on toy params
    int i = 0;
    for( auto& parSet : _parameterSetsList_ ){
      for( auto& par : parSet.getParameterList() ){
        par.setParameterValue( throws.at(i) );
        ++i;
      }
    }

    // TODO: Get number of selected true signal events in each truth bin (after each toy reweight)
    // TODO: Get number of   all    true signal events in each truth bin (after each toy reweight)



    // TODO: Apply efficiency correction
    // TODO: Apply normalizations for number of targets, integrated flux and bin widths

  }


  // TODO: calculated covariance matrix between computed cross section values for the different toys



  postfit_file->Close();















  LogWarning << "Closing output file \"" << out->GetName() << "\"..." << std::endl;
  out->Close();
  LogInfo << "Closed." << std::endl;

  // --------------------------
  // Goodbye:
  // --------------------------
  g.goodbye();

  GlobalVariables::getParallelWorker().reset();
}

