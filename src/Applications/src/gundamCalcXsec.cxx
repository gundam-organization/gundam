
#include "GlobalVariables.h"
#include "GundamGreetings.h"
#include "GundamUtils.h"
#include "Propagator.h"
#include "ConfigUtils.h"

#ifdef GUNDAM_USING_CACHE_MANAGER
#include "CacheManager.h"
#endif

#include "Logger.h"
#include "CmdLineParser.h"
#include "GenericToolbox.h"
#include "GenericToolbox.Json.h"
#include "GenericToolbox.Root.h"

#include <TFile.h>
#include "TDirectory.h"
#include "TH1D.h"
#include "TH2D.h"

#include <string>
#include <vector>


LoggerInit([]{
  Logger::getUserHeader() << "[" << FILENAME << "]";
});


int main(int argc, char** argv){

  GundamGreetings g;
  g.setAppName("cross-section calculator tool");
  g.hello();


  // --------------------------
  // Read Command Line Args:
  // --------------------------
  CmdLineParser clParser;

  clParser.addDummyOption("Main options:");
  clParser.addOption("configFile", {"-c", "--config-file"}, "Specify path to the fitter config file");
  clParser.addOption("fitterOutputFile", {"-f"}, "Specify the fitter output file");
  clParser.addOption("outputFile", {"-o", "--out-file"}, "Specify the CalcXsec output file");
  clParser.addOption("nbThreads", {"-t", "--nb-threads"}, "Specify nb of parallel threads");
  clParser.addOption("nToys", {"-n"}, "Specify number of toys");
  clParser.addOption("randomSeed", {"-s", "--seed"}, "Set random seed");

  clParser.addDummyOption("Trigger options:");
  clParser.addTriggerOption("dryRun", {"-d", "--dry-run"}, "Only overrides fitter config and print it.");

  LogInfo << "Usage: " << std::endl;
  LogInfo << clParser.getConfigSummary() << std::endl << std::endl;

  clParser.parseCmdLine(argc, argv);

  LogThrowIf(clParser.isNoOptionTriggered(), "No option was provided.");

  LogInfo << "Provided arguments: " << std::endl;
  LogInfo << clParser.getValueSummary() << std::endl << std::endl;


  // Sanity checks
  LogThrowIf(not clParser.isOptionTriggered("configFile"), "Xsec calculator config file not provided.");
  LogThrowIf(not clParser.isOptionTriggered("fitterOutputFile"), "Did not provide the output fitter file.");
  LogThrowIf(not clParser.isOptionTriggered("nToys"), "Did not provide number of toys.");


  // Global parameters
  if( clParser.isOptionTriggered("randomSeed") ){
    LogAlert << "Using user-specified random seed: " << clParser.getOptionVal<ULong_t>("randomSeed") << std::endl;
    gRandom->SetSeed(clParser.getOptionVal<ULong_t>("randomSeed"));
  }
  else{
    ULong_t seed = time(nullptr);
    LogInfo << "Using \"time(nullptr)\" random seed: " << seed << std::endl;
    gRandom->SetSeed(seed);
  }
  GlobalVariables::setNbThreads(clParser.getOptionVal("nbThreads", 1));
  LogInfo << "Running the fitter with " << GlobalVariables::getNbThreads() << " parallel threads." << std::endl;


  // Reading fitter file
  LogInfo << "Opening fitter output file: " << clParser.getOptionVal<std::string>("fitterOutputFile") << std::endl;
  auto fitterFile = std::unique_ptr<TFile>( TFile::Open( clParser.getOptionVal<std::string>("fitterOutputFile").c_str() ) );
  LogThrowIf( fitterFile == nullptr, "Could not open fitter output file." );

  using namespace GundamUtils;
  ObjectReader::throwIfNotFound = true;

  nlohmann::json fitterConfig;
  ObjectReader::readObject<TNamed>(fitterFile.get(), "gundamFitter/unfoldedConfig_TNamed", [&](TNamed* config_){
    fitterConfig = GenericToolbox::Json::readConfigJsonStr( config_->GetTitle() );
  });
  ConfigUtils::ConfigHandler cHandler{ fitterConfig };

  // Disabling defined samples:
  LogInfo << "Removing defined samples..." << std::endl;
  ConfigUtils::applyOverrides(
      cHandler.getConfig(),
      GenericToolbox::Json::readConfigJsonStr(R"({"fitterEngineConfig":{"propagatorConfig":{"fitSampleSetConfig":{"fitSampleList":[]}}}})")
  );

  // Disabling defined plots:
  LogInfo << "Removing defined plots..." << std::endl;
  ConfigUtils::applyOverrides(
      cHandler.getConfig(),
      GenericToolbox::Json::readConfigJsonStr(R"({"fitterEngineConfig":{"propagatorConfig":{"plotGeneratorConfig":{}}}})")
  );

  // Defining signal samples
  cHandler.override( clParser.getOptionVal<std::string>("configFile") );

  if( clParser.isOptionTriggered("dryRun") ){
    std::cout << cHandler.toString() << std::endl;

    LogAlert << "Exiting as dry-run is set." << std::endl;
    return EXIT_SUCCESS;
  }


  auto configPropagator = GenericToolbox::Json::fetchValuePath<nlohmann::json>( cHandler.getConfig(), "fitterEngineConfig/propagatorConfig" );

  // Create a propagator object
  Propagator propagator;

  // Read the whole fitter config with the override parameters
  propagator.readConfig( configPropagator );

  // We are only interested in our MC. Data has already been used to get the post-fit error/values
  propagator.setLoadAsimovData( true );

  // Disabling eigen decomposed parameters
  propagator.setEnableEigenToOrigInPropagate( false );

  // Load post-fit parameters as "prior" so we can reset the weight to this point when throwing toys
  ObjectReader::readObject<TNamed>( fitterFile.get(), "FitterEngine/postFit/parState_TNamed", [&](TNamed* parState_){
    propagator.injectParameterValues( parState_->GetTitle() );
    for( auto& parSet : propagator.getParameterSetsList() ){
      if( not parSet.isEnabled() ){ continue; }
      for( auto& par : parSet.getParameterList() ){
        if( not par.isEnabled() ){ continue; }
        par.setPriorValue( par.getParameterValue() );
      }
    }
  });

  // Load the post-fit covariance matrix
  ObjectReader::readObject<TH2D>(
      fitterFile.get(), "FitterEngine/postFit/Hesse/hessian/postfitCovarianceOriginal_TH2D",
      [&](TH2D* hCovPostFit_){
    propagator.setGlobalCovarianceMatrix(std::make_shared<TMatrixD>(hCovPostFit_->GetNbinsX(), hCovPostFit_->GetNbinsX()));
    for( int iBin = 0 ; iBin < hCovPostFit_->GetNbinsX() ; iBin++ ){
      for( int jBin = 0 ; jBin < hCovPostFit_->GetNbinsX() ; jBin++ ){
        (*propagator.getGlobalCovarianceMatrix())[iBin][jBin] = hCovPostFit_->GetBinContent(1 + iBin, 1 + jBin);
      }
    }
  });

  // Sample binning using parameterSetName
  for( auto& sample : propagator.getFitSampleSet().getFitSampleList() ){
    auto associatedParSet = GenericToolbox::Json::fetchValue<std::string>(sample.getConfig(), "parameterSetName");

    // Looking for parSet
    auto foundDialCollection = std::find_if(
        propagator.getDialCollections().begin(), propagator.getDialCollections().end(),
        [&](const DialCollection& dialCollection_){
          auto* parSetPtr{dialCollection_.getSupervisedParameterSet()};
          if( parSetPtr == nullptr ){ return false; }
          return ( parSetPtr->getName() == associatedParSet );
    });
    LogThrowIf(
        foundDialCollection == propagator.getDialCollections().end(),
        "Could not find " << associatedParSet << " among fit dial collections: "
        << GenericToolbox::iterableToString(propagator.getDialCollections(), [](const DialCollection& dialCollection_){
          return dialCollection_.getTitle();
        }
    ));

    LogThrowIf(foundDialCollection->getDialBinSet().isEmpty(), "Could not find binning");
    sample.setBinningFilePath( foundDialCollection->getDialBinSet().getFilePath() );
  }

  // Load everything
  propagator.initialize();


  // Creating output file
  std::string outFilePath{};
  if( clParser.isOptionTriggered("outputFile") ){ outFilePath = clParser.getOptionVal<std::string>("outputFile"); }
  else{
    // appendixDict["optionName"] = "Appendix"
    // this list insure all appendices will appear in the same order
    std::vector<std::pair<std::string, std::string>> appendixDict{
        {"configFile", "%s"},
        {"fitterOutputFile", "Fit_%s"},
        {"nToys", "nToys_%s"},
        {"randomSeed", "Seed_%s"},
    };

    outFilePath = "xsecCalc_" + GundamUtils::generateFileName(clParser, appendixDict) + ".root";
  }
  auto outFile = std::unique_ptr<TFile>( TFile::Open( outFilePath.c_str(), "RECREATE" ) );

  GenericToolbox::writeInTFile(
      GenericToolbox::mkdirTFile(outFile.get(), "gundamCalcXsec"),
      TNamed("gundamVersion", GundamUtils::getVersionFullStr().c_str())
  );
  GenericToolbox::writeInTFile(
      GenericToolbox::mkdirTFile(outFile.get(), "gundamCalcXsec"),
      TNamed("commandLine", clParser.getCommandLineString().c_str())
  );

  LogInfo << "Generating loaded sample plots..." << std::endl;
  propagator.getPlotGenerator().generateSamplePlots(
    GenericToolbox::mkdirTFile(outFile.get(), "XsecExtractor/postFit/samples")
  );

  LogInfo << "Creating throws tree" << std::endl;
  auto* signalThrowTree = new TTree("signalThrowTree", "signalThrowTree");
  std::vector<GenericToolbox::RawDataArray> signalThrowData{propagator.getFitSampleSet().getFitSampleList().size()};
  std::vector<std::vector<double>> bufferList{propagator.getFitSampleSet().getFitSampleList().size()};
  for( size_t iSignal = 0 ; iSignal < propagator.getFitSampleSet().getFitSampleList().size() ; iSignal++ ){

    int nBins = propagator.getFitSampleSet().getFitSampleList()[iSignal].getMcContainer().histogram->GetNbinsX();
    bufferList[iSignal].resize(nBins, 0);

    signalThrowData[iSignal].resetCurrentByteOffset();
    std::vector<std::string> leafNameList{};
    leafNameList.reserve(nBins);
    for( int iBin = 0 ; iBin < nBins ; iBin++ ){
      leafNameList.emplace_back(Form("bin_%i/D", iBin));
      signalThrowData[iSignal].writeRawData( double(0) );
    }

    signalThrowData[iSignal].lockArraySize();
    signalThrowTree->Branch(
        GenericToolbox::generateCleanBranchName(propagator.getFitSampleSet().getFitSampleList()[iSignal].getName()).c_str(),
        &signalThrowData[iSignal].getRawDataArray()[0],
        GenericToolbox::joinVectorString(leafNameList, ":").c_str()
    );
  }

  std::vector<double> numberOfTargets(signalThrowData.size());
  std::vector<double> integratedFlux(signalThrowData.size());
  for( size_t iSignal = 0 ; iSignal < signalThrowData.size() ; iSignal++ ){
    numberOfTargets[iSignal] = GenericToolbox::Json::fetchValue<double>(propagator.getFitSampleSet().getFitSampleList()[iSignal].getConfig(), "numberOfTargets", 1);
    integratedFlux[iSignal] = GenericToolbox::Json::fetchValue<double>(propagator.getFitSampleSet().getFitSampleList()[iSignal].getConfig(), "integratedFlux", 1);
  }

  int nToys{100};
  if(clParser.isOptionTriggered("nToys")) nToys = clParser.getOptionVal<int>("nToys");

  bool enableEventMcThrow{true};
  bool enableStatThrowInToys{true};
  auto xsecCalcConfig = GenericToolbox::Json::fetchValue( cHandler.getConfig(), "xsecCalcConfig", nlohmann::json() );
  enableStatThrowInToys = GenericToolbox::Json::fetchValue( xsecCalcConfig, "enableStatThrowInToys", enableStatThrowInToys);
  enableEventMcThrow = GenericToolbox::Json::fetchValue( xsecCalcConfig, "enableEventMcThrow", enableEventMcThrow);

  std::stringstream ss; ss << LogWarning.getPrefixString() << "Generating " << nToys << " toys...";
  for( int iToy = 0 ; iToy < nToys ; iToy++ ){
    GenericToolbox::displayProgressBar(iToy, nToys, ss.str());

    // Do the throwing:
    propagator.throwParametersFromGlobalCovariance();
    propagator.propagateParametersOnSamples();
    if( enableStatThrowInToys ){
      for( auto& sample : propagator.getFitSampleSet().getFitSampleList() ){
        if( enableEventMcThrow ){
          // Take into account the finite amount of event in MC
          sample.getMcContainer().throwEventMcError();
        }
        // Asimov bin content -> toy data
        sample.getMcContainer().throwStatError();
      }
    }

    // Compute N_truth_i(flux, xsec) and N_selected_truth_i(flux, xsec, det) -> efficiency
    // mask detector parameters?


    // Then compute N_selected_truth_i(flux, xsec, det, c_i) -> N_i


    // Write N_i / efficiency / T / phi
    for( size_t iSignal = 0 ; iSignal < propagator.getFitSampleSet().getFitSampleList().size() ; iSignal++ ){
      signalThrowData[iSignal].resetCurrentByteOffset();
      for( int iBin = 0 ; iBin < propagator.getFitSampleSet().getFitSampleList()[iSignal].getMcContainer().histogram->GetNbinsX() ; iBin++ ){
        signalThrowData[iSignal].writeRawData(
            propagator.getFitSampleSet().getFitSampleList()[iSignal].getMcContainer().histogram->GetBinContent(1+iBin)
            / numberOfTargets[iSignal] / integratedFlux[iSignal]
        );
      }
    }

    // Write the branches
    signalThrowTree->Fill();
  }

  GenericToolbox::writeInTFile(GenericToolbox::mkdirTFile(outFile.get(), "XsecExtractor/throws"), signalThrowTree, "signalThrow");
  auto* meanValuesVector = GenericToolbox::generateMeanVectorOfTree(signalThrowTree);
  auto* globalCovMatrix = GenericToolbox::generateCovarianceMatrixOfTree(signalThrowTree);

  auto* globalCovMatrixHist = GenericToolbox::convertTMatrixDtoTH2D(globalCovMatrix);
  auto* globalCorMatrixHist = GenericToolbox::convertTMatrixDtoTH2D(GenericToolbox::convertToCorrelationMatrix(globalCovMatrix));

  std::vector<TH1D> binValues{};
  binValues.reserve( propagator.getFitSampleSet().getFitSampleList().size() );
  int iGlobal{-1};
  for( size_t iSignal = 0 ; iSignal < propagator.getFitSampleSet().getFitSampleList().size() ; iSignal++ ){
    binValues.emplace_back(
        propagator.getFitSampleSet().getFitSampleList()[iSignal].getName().c_str(),
        propagator.getFitSampleSet().getFitSampleList()[iSignal].getName().c_str(),
        propagator.getFitSampleSet().getFitSampleList()[iSignal].getMcContainer().histogram->GetNbinsX(),
      0,
        propagator.getFitSampleSet().getFitSampleList()[iSignal].getMcContainer().histogram->GetNbinsX()
    );

    std::string sampleTitle{ propagator.getFitSampleSet().getFitSampleList()[iSignal].getName() };

    for( int iBin = 0 ; iBin < propagator.getFitSampleSet().getFitSampleList()[iSignal].getMcContainer().histogram->GetNbinsX() ; iBin++ ){
      iGlobal++;

      std::string binTitle = propagator.getFitSampleSet().getFitSampleList()[iSignal].getBinning().getBinsList()[iBin].getSummary();
      double binVolume = propagator.getFitSampleSet().getFitSampleList()[iSignal].getBinning().getBinsList()[iBin].getVolume();

      binValues[iSignal].SetBinContent(1+iBin, (*meanValuesVector)[iGlobal] / binVolume  );
      binValues[iSignal].SetBinError(1+iBin, TMath::Sqrt( (*globalCovMatrix)[iGlobal][iGlobal] ) / binVolume );
      binValues[iSignal].GetXaxis()->SetBinLabel( 1+iBin, binTitle.c_str() );

      globalCovMatrixHist->GetXaxis()->SetBinLabel(1+iGlobal, GenericToolbox::joinPath(sampleTitle, binTitle).c_str());
      globalCorMatrixHist->GetXaxis()->SetBinLabel(1+iGlobal, GenericToolbox::joinPath(sampleTitle, binTitle).c_str());
      globalCovMatrixHist->GetYaxis()->SetBinLabel(1+iGlobal, GenericToolbox::joinPath(sampleTitle, binTitle).c_str());
      globalCorMatrixHist->GetYaxis()->SetBinLabel(1+iGlobal, GenericToolbox::joinPath(sampleTitle, binTitle).c_str());
    }

    binValues[iSignal].SetMarkerStyle(kFullDotLarge);
    binValues[iSignal].SetMarkerColor(kGreen-3);
    binValues[iSignal].SetMarkerSize(0.5);
    binValues[iSignal].SetLineWidth(2);
    binValues[iSignal].SetLineColor(kGreen-3);
    binValues[iSignal].SetDrawOption("E1");
    binValues[iSignal].GetXaxis()->LabelsOption("v");
    binValues[iSignal].GetXaxis()->SetLabelSize(0.02);
    binValues[iSignal].GetYaxis()->SetTitle("#delta#sigma");

    GenericToolbox::writeInTFile(
        GenericToolbox::mkdirTFile(outFile.get(), "XsecExtractor/histograms"),
        &binValues[iSignal],
        GenericToolbox::generateCleanBranchName(propagator.getFitSampleSet().getFitSampleList()[iSignal].getName())
    );
  }


  globalCovMatrixHist->GetXaxis()->SetLabelSize(0.02);
  globalCovMatrixHist->GetYaxis()->SetLabelSize(0.02);
  GenericToolbox::writeInTFile(GenericToolbox::mkdirTFile(outFile.get(), "XsecExtractor/matrices"), globalCovMatrixHist, "covarianceMatrix");

  globalCorMatrixHist->GetXaxis()->SetLabelSize(0.02);
  globalCorMatrixHist->GetYaxis()->SetLabelSize(0.02);
  globalCorMatrixHist->GetZaxis()->SetRangeUser(-1, 1);
  GenericToolbox::writeInTFile(GenericToolbox::mkdirTFile(outFile.get(), "XsecExtractor/matrices"), globalCorMatrixHist, "correlationMatrix");

  LogWarning << "Closing output file \"" << outFile->GetName() << "\"..." << std::endl;
  outFile->Close();
  LogInfo << "Closed." << std::endl;

  // --------------------------
  // Goodbye:
  // --------------------------
  g.goodbye();

  GlobalVariables::getParallelWorker().reset();
}

