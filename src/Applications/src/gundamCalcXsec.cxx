
#include "GlobalVariables.h"
#include "VersionConfig.h"
#include "JsonUtils.h"
#include "GundamGreetings.h"
#include "Propagator.h"

#ifdef GUNDAM_USING_CACHE_MANAGER
#include "CacheManager.h"
#endif

#include "Logger.h"
#include "CmdLineParser.h"
#include "GenericToolbox.h"
#include "GenericToolbox.Root.h"

#include <TFile.h>
#include "TDirectory.h"
#include "TH1D.h"
#include "TH2D.h"

#include <string>
#include <vector>


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
  GenericToolbox::writeInTFile(GenericToolbox::mkdirTFile(out, "gundamCalcXsec"), &commandLineString);

  // Fit file
  LogThrowIf(not GenericToolbox::doesTFileIsValid(clParser.getOptionVal<std::string>("fitterOutputFile")),
             "Can't open input fitter file: " << clParser.getOptionVal<std::string>("fitterOutputFile"));
  auto* fitFile = TFile::Open(clParser.getOptionVal<std::string>("fitterOutputFile").c_str());
  std::string configStr{fitFile->Get<TNamed>("gundamFitter/unfoldedConfig_TNamed")->GetTitle()};

  // Get config from the fit
  auto configFit = JsonUtils::readConfigJsonStr(configStr); // works with yaml
  auto configPropagator = JsonUtils::fetchValue<nlohmann::json>(JsonUtils::fetchValue<nlohmann::json>(configFit, "fitterEngineConfig"), "propagatorConfig");

  // Create a propagator object
  Propagator p;

  p.setConfig(configPropagator);
  p.readConfig();

  // modify according to the Xsec needs
  // binning?

  // Get best fit parameter values and postfit covariance matrix
  LogInfo << "Injecting postfit values of fitted parameters..." << std::endl;
  auto* postFitDir = fitFile->Get<TDirectory>("FitterEngine/postFit/Hesse/errors");
  for( auto& parSet : p.getParameterSetsList() ){
    auto* postFitParHist = postFitDir->Get<TH1D>(Form("%s/values/postFitErrors_TH1D", parSet.getName().c_str()));
    LogThrowIf(parSet.getNbParameters() != postFitParHist->GetNbinsX(),
               "Expecting " << parSet.getNbParameters() << " postfit parameter values, but got: " << postFitParHist->GetNbinsX());
    for( auto& par : parSet.getParameterList() ){
      par.setPriorValue(postFitParHist->GetBinContent( 1+par.getParameterIndex() ));
    }
    parSet.moveFitParametersToPrior();
  }

  LogInfo << "Using postfit covariance matrix as the global covariance matrix for the Propagator..." << std::endl;
  auto* postFitCovMat = fitFile->Get<TH2D>("FitterEngine/postFit/Hesse/hessian/postfitCovarianceOriginal_TH2D");
  for( int iBin = 0 ; iBin < postFitCovMat->GetNbinsX() ; iBin++ ){
    for( int jBin = 0 ; jBin < postFitCovMat->GetNbinsX() ; jBin++ ){
      (*p.getGlobalCovarianceMatrix())[iBin][jBin] = postFitCovMat->GetBinContent(1+iBin, 1+jBin);
    }
  }

  // load the data
  p.initialize();




  // Set best fit parameters and reweight events
  std::vector<FitParameterSet> _parameterSetsList_ =  p.getParameterSetsList();
  int i = 0;
  for( auto& parSet : _parameterSetsList_ ){
    for( auto& par : parSet.getParameterList() ){
//      par.setParameterValue( postfit_params->GetBinContent(i+1) );
      ++i;
    }
  }

  p.propagateParametersOnSamples();




  // TODO: Get number of selected true signal events in each truth bin (after best fit reweight)
  // TODO: Get number of   all    true signal events in each truth bin (after best fit reweight)




  // Cholesky decompose postfit covariance matrix
//  TMatrixT<double>* choleskyMatrix = GenericToolbox::getCholeskyMatrix(postfit_cov);



  // Create toys which throw the fit parameters according to chol.-decomp. postfit cov matrix
  // get number of toys from cmlParser
//  for (int iToy = 0; iToy < 1000; iToy++) {
//    std::vector<double> throws = GenericToolbox::throwCorrelatedParameters(choleskyMatrix);
//    for (int j = 0; j < throws.size(); j++){
//      throws.at(j) += postfit_params->GetBinContent(j+1);
//      if ( throws.at(j) < 0) {
//        throws.at(j) = 0.01;
//      }
//    }

    // Reweight events based on toy params
//    int i = 0;
//    for( auto& parSet : _parameterSetsList_ ){
//      for( auto& par : parSet.getParameterList() ){
//        par.setParameterValue( throws.at(i) );
//        ++i;
//      }
//    }
//
//    // TODO: Get number of selected true signal events in each truth bin (after each toy reweight)
//    // TODO: Get number of   all    true signal events in each truth bin (after each toy reweight)
//
//
//
//    // TODO: Apply efficiency correction
//    // TODO: Apply normalizations for number of targets, integrated flux and bin widths
//
//  }
//

  // TODO: calculated covariance matrix between computed cross section values for the different toys



//  postfit_file->Close();















  LogWarning << "Closing output file \"" << out->GetName() << "\"..." << std::endl;
  out->Close();
  LogInfo << "Closed." << std::endl;

  // --------------------------
  // Goodbye:
  // --------------------------
  g.goodbye();

  GlobalVariables::getParallelWorker().reset();
}

