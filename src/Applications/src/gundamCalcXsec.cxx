
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
  g.setAppName("GundamFitter");
  g.hello();


  // --------------------------
  // Read Command Line Args:
  // --------------------------
  CmdLineParser clParser;

  clParser.addTriggerOption("dry-run", {"--dry-run", "-d"},"Perform the full sequence of initialization, but don't do the actual fit.");
  clParser.addTriggerOption("generateOneSigmaPlots", {"--one-sigma"}, "Generate one sigma plots");
  clParser.addTriggerOption("asimov", {"-a", "--asimov"}, "Use MC dataset to fill the data histograms");

  clParser.addOption("cache", {"-C", "--cache-enabled"}, "Enable the event weight cache");
  clParser.addOption("configFile", {"-c", "--config-file"}, "Specify path to the fitter config file");
  clParser.addOption("nbThreads", {"-t", "--nb-threads"}, "Specify nb of parallel threads");
  clParser.addOption("inputFile", {"-i", "--in-file"}, "Specify the input file (fitter output file)");
  clParser.addOption("outputFile", {"-o", "--out-file"}, "Specify the output file");
  clParser.addOption("scanParameters", {"--scan"}, "Enable parameter scan before and after the fit");
  clParser.addOption("toyFit", {"--toy"}, "Run a toy fit");
  clParser.addOption("randomSeed", {"-s", "--seed"}, "Set random seed");

  clParser.getOptionPtr("scanParameters")->setAllowEmptyValue(true); // --scan can be followed or not by the number of steps
  clParser.getOptionPtr("toyFit")->setAllowEmptyValue(true); // --toy can be followed or not by the number of steps

  LogInfo << "Usage: " << std::endl;
  LogInfo << clParser.getConfigSummary() << std::endl << std::endl;

  clParser.parseCmdLine(argc, argv);

  LogThrowIf(clParser.isNoOptionTriggered(), "No option was provided.");

  LogInfo << "Provided arguments: " << std::endl;
  LogInfo << clParser.getValueSummary() << std::endl << std::endl;
  LogInfo << clParser.dumpConfigAsJsonStr() << std::endl;

  std::string cacheDefault = "off";
#ifdef GUNDAM_USING_CACHE_MANAGER
  if (Cache::Manager::HasCUDA()) cacheDefault = "on";
#endif
  std::string cacheEnabled = clParser.getOptionVal("cache",cacheDefault);
  if (cacheEnabled != "on") {
      LogInfo << "Cache::Manager disabled" << std::endl;
      GlobalVariables::setEnableCacheManager(false);
  }
  else {
      LogInfo << "Enabling Cache::Manager" << std::endl;
      GlobalVariables::setEnableCacheManager(true);
  }

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
  GenericToolbox::writeInTFile(GenericToolbox::mkdirTFile(out, "gundamFitter"), &gundamVersionString);

  // Command line?
  TNamed commandLineString("commandLine", clParser.getCommandLineString().c_str());
  GenericToolbox::writeInTFile(GenericToolbox::mkdirTFile(out, "gundamFitter"), &commandLineString);

  // Config unfolded ?
  auto unfoldedConfig = jsonConfig;
  JsonUtils::unfoldConfig(unfoldedConfig);
  std::stringstream ss;
  ss << unfoldedConfig << std::endl;
  TNamed unfoldedConfigString("unfoldedConfig", ss.str().c_str());
  GenericToolbox::writeInTFile(GenericToolbox::mkdirTFile(out, "gundamFitter"), &unfoldedConfigString);


  LogInfo << "FitterEngine setup..." << std::endl;

  // Fitter
  FitterEngine fitter;
  fitter.setConfig(JsonUtils::fetchSubEntry(jsonConfig, {"fitterEngineConfig"}));
  fitter.setSaveDir(GenericToolbox::mkdirTFile(out, "FitterEngine"));

  fitter.getPropagator().setLoadAsimovData( clParser.isOptionTriggered("asimov") );

  fitter.initialize();

  fitter.updateChi2Cache();
  LogInfo << "Initial χ² = " << fitter.getChi2Buffer() << std::endl;
  LogInfo << "Initial χ²(stat) = " << fitter.getChi2StatBuffer() << std::endl;

















  // Get best fit parameter values and postfit covariance matrix
  auto fFitterOutput = clParser.getOptionVal<std::string>("inputFile");
  TFile* postfit_file = TFile::Open(fFitterOutput.c_str(), "READ");
  TH1D* postfit_params = (TH1D*)postfit_file->Get("FitterEngine/postFit/Migrad/errors/Template Parameters/values/postFitErrors_TH1D");
  TMatrixTSym<double>* postfit_cov = (TMatrixTSym<double>*)postfit_file->Get("FitterEngine/postFit/Migrad/errors/Template Parameters/matrices/Covariance_TMatrixD");



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

