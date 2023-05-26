
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

  std::string configStr{};
  ObjectReader::readObject<TNamed>(fitterFile.get(), "gundamFitter/unfoldedConfig_TNamed", [&](TNamed* config_){
    configStr = config_->GetTitle();
  });
  ConfigUtils::ConfigHandler cHandler{ configStr };


  // Editing fitter config to add truth samples
  ConfigUtils::ConfigHandler cXsecHandler{ clParser.getOptionVal<std::string>("configFile") };
  cHandler.override( cXsecHandler.toString() );
  auto configPropagator = GenericToolbox::Json::fetchValuePath<nlohmann::json>( cHandler.getConfig(), "fitterEngineConfig/propagatorConfig" );

  // Create a propagator object
  Propagator propagator;

  // Read the whole fitter config with the overrided parameters
  propagator.readConfig( configPropagator );

  // We are only interested in our MC. Data has already been used to get the post-fit error/values
  propagator.setLoadAsimovData( true );

  // Load post-fit parameters as "prior" so we can reset the weight to this point when throwing toys
  ObjectReader::readObject<TNamed>( fitterFile.get(), "FitterEngine/postFit/parState_TNamed", [&](TNamed* parState_){
    propagator.injectParameterValues( parState_->GetTitle() );
    for( auto& parSet : propagator.getParameterSetsList() ){
      if( not parSet.isEnabled() ){ continue; }

      if( parSet.isUseEigenDecompInFit() ){
        // TODO: SHOULD NOT USE EIGEN DECOMP
      }

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











  bool enableEventMcThrow{true};
  bool enableStatThrowInToys{true};

  enableStatThrowInToys = GenericToolbox::Json::fetchValue(configXsecExtractor, "enableStatThrowInToys", enableStatThrowInToys);
  enableEventMcThrow = GenericToolbox::Json::fetchValue(configXsecExtractor, "enableEventMcThrow", enableEventMcThrow);

  // Need this number later -> STILL NEEDED?
  size_t nFitSample{propagator.getFitSampleSet().getFitSampleList().size() };

  // Get best fit parameter values and postfit covariance matrix
  LogInfo << "Injecting post-fit values of fitted parameters..." << std::endl;
  auto* postFitDir = fitFile->Get<TDirectory>("FitterEngine/postFit/Hesse/errors");
  LogThrowIf(postFitDir == nullptr, "Could not find FitterEngine/postFit/Hesse/errors");
  for( auto& parSet : propagator.getParameterSetsList() ){
    if( not parSet.isEnabled() ) continue;
    auto* postFitParHist = postFitDir->Get<TH1D>(Form("%s/values/postFitErrors_TH1D", parSet.getName().c_str()));
    LogThrowIf( postFitParHist == nullptr, " Could not find " << Form("%s/values/postFitErrors_TH1D", parSet.getName().c_str()));
    LogThrowIf(parSet.getNbParameters() != postFitParHist->GetNbinsX(),
               "Expecting " << parSet.getNbParameters() << " postfit parameter values, but got: " << postFitParHist->GetNbinsX());
    for( auto& par : parSet.getParameterList() ){
      par.setPriorValue( postFitParHist->GetBinContent( 1+par.getParameterIndex() ) );
    }
  }

  LogInfo << "Using post-fit covariance matrix as the global covariance matrix for the Propagator..." << std::endl;
  auto* postFitCovMat = fitFile->Get<TH2D>("FitterEngine/postFit/Hesse/hessian/postfitCovarianceOriginal_TH2D");
  LogThrowIf(postFitCovMat == nullptr, "Could not find postfit cov matrix");
  propagator.setGlobalCovarianceMatrix(std::make_shared<TMatrixD>(postFitCovMat->GetNbinsX(), postFitCovMat->GetNbinsX()));
  for( int iBin = 0 ; iBin < postFitCovMat->GetNbinsX() ; iBin++ ){
    for( int jBin = 0 ; jBin < postFitCovMat->GetNbinsX() ; jBin++ ){
      (*propagator.getGlobalCovarianceMatrix())[iBin][jBin] = postFitCovMat->GetBinContent(1 + iBin, 1 + jBin);
    }
  }

  // load the data
  propagator.initialize();

  LogWarning << std::endl << GenericToolbox::addUpDownBars("Propagator initialized. Now filling selected events...") << std::endl;

  LogInfo << "Filling selected true signal events..." << std::endl;

  for( auto& signalSamplePair : signalSampleList ){
    // reserving events for templateSampleDetected by the maximum size it could be
    signalSamplePair.second->getMcContainer().reserveEventMemory(
        0,
        signalSamplePair.first->getMcContainer().eventList.size(),
        signalSamplePair.first->getMcContainer().eventList[0]
    );
    int iTemplateEvt{0};


    auto isInTemplateBin = [&](const PhysicsEvent& ev_, const DataBin& b_){
      for( size_t iVar = 0 ; iVar < b_.getVariableNameList().size() ; iVar++ ){
        if( not b_.isBetweenEdges(iVar, ev_.getVarAsDouble(b_.getVariableNameList()[iVar])) ){
          return false;
        }
      } // Var
      return true;
    };
    for( int iFitSample = 0 ; iFitSample < nFitSample ; iFitSample++ ){
      for( auto& event : propagator.getFitSampleSet().getFitSampleList()[iFitSample].getMcContainer().eventList ){
        for( size_t iBin = 0 ; iBin < signalSamplePair.first->getBinning().getBinsList().size() ; iBin++ ){
          if( isInTemplateBin(event, signalSamplePair.first->getBinning().getBinsList()[iBin]) ){
            // copy event in template bin
            signalSamplePair.second->getMcContainer().eventList[iTemplateEvt++] = event;
            signalSamplePair.second->getMcContainer().eventList[iTemplateEvt-1].setSampleBinIndex(int(iBin));
            break;
          }
        } // template bins
      } // sample event
    } // sample loop
    signalSamplePair.second->getMcContainer().shrinkEventList(iTemplateEvt);


    // copying to data container..
    signalSamplePair.second->getDataContainer().eventList.insert(
        std::end(signalSamplePair.second->getDataContainer().eventList),
        std::begin(signalSamplePair.second->getMcContainer().eventList),
        std::end(signalSamplePair.second->getMcContainer().eventList)
    );
  }

  LogInfo << "Updating sample bin cache for signal sample (reconstructed)..." << std::endl;
  propagator.getFitSampleSet().updateSampleBinEventList();
  propagator.getFitSampleSet().updateSampleHistograms();

  for( auto& signalSamplePair : signalSampleList ) {
    signalSamplePair.second->getDataContainer().isLocked = true;
  }

  propagator.propagateParametersOnSamples();

  // redefine histograms for the plot generator
  propagator.getPlotGenerator().defineHistogramHolders();

  for(size_t iSample = nFitSample ; iSample < propagator.getFitSampleSet().getFitSampleList().size() ; iSample++ ){
    auto* sample = &propagator.getFitSampleSet().getFitSampleList()[iSample];
    propagator.getTreeWriter().writeEvents(GenericToolbox::mkdirTFile(out, "XsecExtractor/postFit/events/" + sample->getName()), "MC", sample->getMcContainer().eventList);
  }

  LogInfo << "Generating loaded sample plots..." << std::endl;
  propagator.getPlotGenerator().generateSamplePlots(GenericToolbox::mkdirTFile(out, "XsecExtractor/postFit/samples"));

  LogInfo << "Creating throws tree" << std::endl;
  auto* signalThrowTree = new TTree("signalThrowTree", "signalThrowTree");
  std::vector<GenericToolbox::RawDataArray> signalThrowData{signalSampleList.size()};
  std::vector<std::vector<double>> bufferList{signalSampleList.size()};
  for( size_t iSignal = 0 ; iSignal < signalSampleList.size() ; iSignal++ ){

    int nBins = signalSampleList[iSignal].first->getMcContainer().histogram->GetNbinsX();
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
        GenericToolbox::generateCleanBranchName(signalSampleList[iSignal].first->getName()).c_str(),
        &signalThrowData[iSignal].getRawDataArray()[0],
        GenericToolbox::joinVectorString(leafNameList, ":").c_str()
    );
  }

  std::vector<double> numberOfTargets(signalSampleList.size());
  std::vector<double> integratedFlux(signalSampleList.size());
  for( size_t iSignal = 0 ; iSignal < signalSampleList.size() ; iSignal++ ){
    numberOfTargets[iSignal] = GenericToolbox::Json::fetchValue<double>(signalDefinitions[iSignal], "numberOfTargets", 1);
    integratedFlux[iSignal] = GenericToolbox::Json::fetchValue<double>(signalDefinitions[iSignal], "integratedFlux", 1);
  }

  int nToys{100};
  if(clParser.isOptionTriggered("nToys")) nToys = clParser.getOptionVal<int>("nToys");

  std::stringstream ss; ss << LogWarning.getPrefixString() << "Generating " << nToys << " toys...";
  for( int iToy = 0 ; iToy < nToys ; iToy++ ){
    GenericToolbox::displayProgressBar(iToy, nToys, ss.str());

    // Do the throwing:
    propagator.throwParametersFromGlobalCovariance();
    propagator.propagateParametersOnSamples();
//    if( enableStatThrowInToys ){
//      for( auto& sample : propagator.getFitSampleSet().getFitSampleList() ){
//        if( enableEventMcThrow ){
//          // Take into account the finite amount of event in MC
//          sample.getMcContainer().throwEventMcError();
//        }
//        // Asimov bin content -> toy data
//        sample.getMcContainer().throwStatError();
//      }
//    }

    // Compute N_truth_i(flux, xsec) and N_selected_truth_i(flux, xsec, det) -> efficiency
    // mask detector parameters?


    // Then compute N_selected_truth_i(flux, xsec, det, c_i) -> N_i


    // Write N_i / efficiency / T / phi
    for( size_t iSignal = 0 ; iSignal < signalSampleList.size() ; iSignal++ ){
      signalThrowData[iSignal].resetCurrentByteOffset();
      for( int iBin = 0 ; iBin < signalSampleList[iSignal].first->getMcContainer().histogram->GetNbinsX() ; iBin++ ){
        LogDebug(iBin == 0) << std::endl << iBin << " -> " << signalSampleList[iSignal].first->getMcContainer().histogram->GetBinContent(1+iBin)
//                    / numberOfTargets[iSignal] / integratedFlux[iSignal]
                    << std::endl;
        signalThrowData[iSignal].writeRawData(
            signalSampleList[iSignal].first->getMcContainer().histogram->GetBinContent(1+iBin)
            / numberOfTargets[iSignal] / integratedFlux[iSignal]
        );
      }
    }

    // Write the branches
    signalThrowTree->Fill();
  }

  GenericToolbox::writeInTFile(GenericToolbox::mkdirTFile(out, "XsecExtractor/throws"), signalThrowTree, "signalThrow");
  auto* meanValuesVector = GenericToolbox::generateMeanVectorOfTree(signalThrowTree);
  auto* globalCovMatrix = GenericToolbox::generateCovarianceMatrixOfTree(signalThrowTree);

  auto* globalCovMatrixHist = GenericToolbox::convertTMatrixDtoTH2D(globalCovMatrix);
  auto* globalCorMatrixHist = GenericToolbox::convertTMatrixDtoTH2D(GenericToolbox::convertToCorrelationMatrix(globalCovMatrix));

  std::vector<TH1D> binValues{};
  binValues.reserve( signalSampleList.size() );
  int iGlobal{-1};
  for( size_t iSignal = 0 ; iSignal < signalSampleList.size() ; iSignal++ ){
    binValues.emplace_back(
      signalSampleList[iSignal].first->getName().c_str(),
      signalSampleList[iSignal].first->getName().c_str(),
      signalSampleList[iSignal].first->getMcContainer().histogram->GetNbinsX(),
      0,
      signalSampleList[iSignal].first->getMcContainer().histogram->GetNbinsX()
    );

    std::string sampleTitle{ signalSampleList[iSignal].first->getName() };

    for( int iBin = 0 ; iBin < signalSampleList[iSignal].first->getMcContainer().histogram->GetNbinsX() ; iBin++ ){
      iGlobal++;

      std::string binTitle = signalSampleList[iSignal].first->getBinning().getBinsList()[iBin].getSummary();
      double binVolume = signalSampleList[iSignal].first->getBinning().getBinsList()[iBin].getVolume();

      binValues[iSignal].SetBinContent(1+iBin, (*meanValuesVector)[iGlobal] / binVolume  );
      binValues[iSignal].SetBinError(1+iBin, TMath::Sqrt( (*globalCovMatrix)[iGlobal][iGlobal] ) / binVolume );
      binValues[iSignal].GetXaxis()->SetBinLabel( 1+iBin, binTitle.c_str() );

      globalCovMatrixHist->GetXaxis()->SetBinLabel(1+iGlobal, (sampleTitle + "/" + binTitle).c_str());
      globalCorMatrixHist->GetXaxis()->SetBinLabel(1+iGlobal, (sampleTitle + "/" + binTitle).c_str());
      globalCovMatrixHist->GetYaxis()->SetBinLabel(1+iGlobal, (sampleTitle + "/" + binTitle).c_str());
      globalCorMatrixHist->GetYaxis()->SetBinLabel(1+iGlobal, (sampleTitle + "/" + binTitle).c_str());
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
        GenericToolbox::mkdirTFile(out, "XsecExtractor/histograms"),
        &binValues[iSignal],
        GenericToolbox::generateCleanBranchName(signalSampleList[iSignal].first->getName())
    );
  }


  globalCovMatrixHist->GetXaxis()->SetLabelSize(0.02);
  globalCovMatrixHist->GetYaxis()->SetLabelSize(0.02);
  GenericToolbox::writeInTFile(GenericToolbox::mkdirTFile(out, "XsecExtractor/matrices"), globalCovMatrixHist, "covarianceMatrix");

  globalCorMatrixHist->GetXaxis()->SetLabelSize(0.02);
  globalCorMatrixHist->GetYaxis()->SetLabelSize(0.02);
  globalCorMatrixHist->GetZaxis()->SetRangeUser(-1, 1);
  GenericToolbox::writeInTFile(GenericToolbox::mkdirTFile(out, "XsecExtractor/matrices"), globalCorMatrixHist, "correlationMatrix");

  LogWarning << "Closing output file \"" << out->GetName() << "\"..." << std::endl;
  out->Close();
  LogInfo << "Closed." << std::endl;

  // --------------------------
  // Goodbye:
  // --------------------------
  g.goodbye();

  GlobalVariables::getParallelWorker().reset();
}

