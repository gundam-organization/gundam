
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
  auto configXsecExtractor = JsonUtils::readConfigFile(configFilePath); // works with yaml

  if( JsonUtils::doKeyExist(configXsecExtractor, "minGundamVersion") ){
    LogThrowIf(
      not g.isNewerOrEqualVersion(JsonUtils::fetchValue<std::string>(configXsecExtractor, "minGundamVersion")),
      "Version check FAILED: " << GundamVersionConfig::getVersionStr() << " < " << JsonUtils::fetchValue<std::string>(configXsecExtractor, "minGundamVersion")
    );
    LogInfo << "Version check passed: " << GundamVersionConfig::getVersionStr() << " >= " << JsonUtils::fetchValue<std::string>(configXsecExtractor, "minGundamVersion") << std::endl;
  }



  // Open output of the fitter
  // Get configuration of fitter (TNamed) -> parse to json config file



  std::string outFileName = configFilePath;

  outFileName = clParser.getOptionVal("outputFile", outFileName + ".root");
  if( JsonUtils::doKeyExist(configXsecExtractor, "outputFolder") ){
    GenericToolbox::mkdirPath(JsonUtils::fetchValue<std::string>(configXsecExtractor, "outputFolder"));
    outFileName.insert(0, JsonUtils::fetchValue<std::string>(configXsecExtractor, "outputFolder") + "/");
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
  auto configPropagator = JsonUtils::fetchValuePath<nlohmann::json>( configFit, "fitterEngineConfig/propagatorConfig" );

  bool enableEventMcThrow{true};
  bool enableStatThrowInToys{true};

  enableStatThrowInToys = JsonUtils::fetchValue(configXsecExtractor, "enableStatThrowInToys", enableStatThrowInToys);
  enableEventMcThrow = JsonUtils::fetchValue(configXsecExtractor, "enableEventMcThrow", enableEventMcThrow);

  // Create a propagator object
  Propagator p;

  // Read the whole fitter config
  p.readConfig(configPropagator);

  // We are only interested in our MC. Data has already been used to get the post-fit error/values
  p.setLoadAsimovData( true );

  // Need this number later -> STILL NEEDED?
  size_t nFitSample{ p.getFitSampleSet().getFitSampleList().size() };


  for( auto& dataset : p.getDataSetList() ){ LogDebug << dataset.getDataDispenserDict().at("Asimov").getTitle() << std::endl; }

  // As the signal sample might use external data-set, we need to make sure the fit sample will be filled up with
  // the original ones:
  LogInfo << "Defining explicit dataset names from fit samples..." << std::endl;
  for( auto& sample : p.getFitSampleSet().getFitSampleList() ){
    std::vector<std::string> explicitDatasetNameList;
    for( auto& dataset : p.getDataSetList() ){
      if( sample.isDatasetValid( dataset.getName() ) ){
        explicitDatasetNameList.emplace_back( dataset.getName() );
      }
    }
    LogInfo << "Sample \"" << sample.getName() << "\" will be loaded from datasets: "
    << GenericToolbox::parseVectorAsString(explicitDatasetNameList) << std::endl;
    sample.setEnabledDatasetList( explicitDatasetNameList );
  }

  if( JsonUtils::doKeyExist(configXsecExtractor, "signalDatasets") ){
    LogWarning << "Defining additional datasets for signals..." << std::endl;
    auto signalDatasetList = JsonUtils::fetchValue<std::vector<nlohmann::json>>(configXsecExtractor, "signalDatasets");

    // WARNING THIS CHANGES THE SIZE OF THE VECTOR:
    p.getDataSetList().reserve( p.getDataSetList().size() + signalDatasetList.size() );

    // So DatasetLoaders get moved in memory so the _owner_ reference within the DataDispenser are pointing to garbage in memory
    // We need to update this reference.
    for( auto& dataset : p.getDataSetList() ){ dataset.updateDispenserOwnership(); }

    for( auto& signalDatasetConfig : signalDatasetList ){
      p.getDataSetList().emplace_back(signalDatasetConfig, p.getDataSetList().size());
    }
  }

  // Add template sample
  auto signalDefinitions = JsonUtils::fetchValue<std::vector<nlohmann::json>>(configXsecExtractor, "signalDefinitions");
  LogInfo << "Adding 2 x " << signalDefinitions.size() << " signal samples (true + reconstructed) to the propagator..." << std::endl;
  std::vector<std::pair<FitSample*, FitSample*>> signalSampleList;
  p.getFitSampleSet().getFitSampleList().reserve(
      p.getFitSampleSet().getFitSampleList().size() + signalDefinitions.size()*2
  );

  LogInfo << "Defining sample for signal definitions..." << std::endl;
  for( auto& signalDef : signalDefinitions ){

    auto parameterSetName = JsonUtils::fetchValue<std::string>(signalDef, "parameterSetName");
    LogInfo << "Adding signal samples: \"" << parameterSetName << "\"" << std::endl;
    signalSampleList.emplace_back();
    p.getFitSampleSet().getFitSampleList().emplace_back();
    signalSampleList.back().first = &p.getFitSampleSet().getFitSampleList().back();
    signalSampleList.back().first->readConfig(); // read empty config
    signalSampleList.back().first->setName( parameterSetName );

    auto templateBinningFilePath = JsonUtils::fetchValue<std::string>(
        p.getFitParameterSetPtr( parameterSetName )->getDialSetDefinitions()[0], "parametersBinningPath"
    );
    auto applyOnCondition = JsonUtils::fetchValue<std::string>(
        p.getFitParameterSetPtr( parameterSetName )->getDialSetDefinitions()[0], "applyCondition"
    );

    signalSampleList.back().first->setBinningFilePath( templateBinningFilePath );
    signalSampleList.back().first->setVarSelectionFormulaStr( applyOnCondition );

    if( JsonUtils::doKeyExist(signalDef, "datasetSources") ){
      auto datasetSrcList = JsonUtils::fetchValue<std::vector<std::string>>(signalDef, "datasetSources");
      LogInfo << "Signal sample \"" << parameterSetName << "\" (truth) will load data from datasets: "
      << GenericToolbox::parseVectorAsString( datasetSrcList ) << std::endl;
      signalSampleList.back().first->setEnabledDatasetList( datasetSrcList );
    }


    // Add template sample which will take care of the detection efficiency
    // Same as templateSample but will need to make sure events belong to any of the fit sample
    // Don't fill the events yet -> add a dummy cut
    p.getFitSampleSet().getFitSampleList().emplace_back( *signalSampleList.back().first );
    signalSampleList.back().second = &p.getFitSampleSet().getFitSampleList().back();
    signalSampleList.back().second->setName( parameterSetName + " (Reconstructed)" );
    signalSampleList.back().second->setSelectionCutStr("0"); // dummy cut to select no event during the data loading


    // Events in templateSampleDetected will be loaded using the original fit samples
    // To identify the right template bin for each detected event, one need to make sure the variables are kept in memory
    DataBinSet templateBinning;
    templateBinning.readBinningDefinition( templateBinningFilePath );
    for( auto& varStorageList : templateBinning.getBinVariables() ){
      if( not GenericToolbox::doesElementIsInVector(varStorageList, p.getFitSampleSet().getAdditionalVariablesForStorage()) ){
        p.getFitSampleSet().getAdditionalVariablesForStorage().emplace_back(varStorageList);
      }
    }
  }

  // Get best fit parameter values and postfit covariance matrix
  LogInfo << "Injecting post-fit values of fitted parameters..." << std::endl;
  auto* postFitDir = fitFile->Get<TDirectory>("FitterEngine/postFit/Hesse/errors");
  LogThrowIf(postFitDir == nullptr, "Could not find FitterEngine/postFit/Hesse/errors");
  for( auto& parSet : p.getParameterSetsList() ){
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
  p.setGlobalCovarianceMatrix(std::make_shared<TMatrixD>(postFitCovMat->GetNbinsX(), postFitCovMat->GetNbinsX()));
  for( int iBin = 0 ; iBin < postFitCovMat->GetNbinsX() ; iBin++ ){
    for( int jBin = 0 ; jBin < postFitCovMat->GetNbinsX() ; jBin++ ){
      (*p.getGlobalCovarianceMatrix())[iBin][jBin] = postFitCovMat->GetBinContent(1+iBin, 1+jBin);
    }
  }

  // load the data
  p.initialize();

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
      for( auto& event : p.getFitSampleSet().getFitSampleList()[iFitSample].getMcContainer().eventList ){
        signalSamplePair.first->getBinning().getBinsList();
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
  p.getFitSampleSet().updateSampleBinEventList();
  p.getFitSampleSet().updateSampleHistograms();

  for( auto& signalSamplePair : signalSampleList ) {
    signalSamplePair.second->getDataContainer().isLocked = true;
  }

  p.propagateParametersOnSamples();

  // redefine histograms for the plot generator
  p.getPlotGenerator().defineHistogramHolders();

  for( size_t iSample = nFitSample ; iSample < p.getFitSampleSet().getFitSampleList().size() ; iSample++ ){
    auto* sample = &p.getFitSampleSet().getFitSampleList()[iSample];
    p.getTreeWriter().writeEvents(GenericToolbox::mkdirTFile(out, "XsecExtractor/postFit/events/" + sample->getName()), "MC", sample->getMcContainer().eventList);
  }

  LogInfo << "Generating loaded sample plots..." << std::endl;
  p.getPlotGenerator().generateSamplePlots(GenericToolbox::mkdirTFile(out, "XsecExtractor/postFit/samples"));

  LogInfo << "Creating throws tree" << std::endl;
  auto* signalThrowTree = new TTree("signalThrowTree", "signalThrowTree");
  std::vector<GenericToolbox::RawDataArray> signalThrowData{signalSampleList.size()};
  std::vector<std::vector<double>> bufferList{signalSampleList.size()};
  for( size_t iSignal = 0 ; iSignal < signalSampleList.size() ; iSignal++ ){

    int nBins = signalSampleList[iSignal].first->getMcContainer().histogram->GetNbinsX();
    bufferList[iSignal].resize(nBins, 0);

    signalThrowData[iSignal].resetCurrentByteOffset();
    std::vector<std::string> leafNameList(nBins);
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
    numberOfTargets[iSignal] = JsonUtils::fetchValue<double>(signalDefinitions[iSignal], "numberOfTargets", 1);
    integratedFlux[iSignal] = JsonUtils::fetchValue<double>(signalDefinitions[iSignal], "integratedFlux", 1);
  }

  int nToys{100};
  if(clParser.isOptionTriggered("nToys")) nToys = clParser.getOptionVal<int>("nToys");

  std::stringstream ss; ss << LogWarning.getPrefixString() << "Generating " << nToys << " toys...";
  for( int iToy = 0 ; iToy < nToys ; iToy++ ){
    GenericToolbox::displayProgressBar(iToy, nToys, ss.str());

    // Do the throwing:
    p.throwParametersFromGlobalCovariance();
    p.propagateParametersOnSamples();
    if( enableStatThrowInToys ){
      for( auto& sample : p.getFitSampleSet().getFitSampleList() ){
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
    for( size_t iSignal = 0 ; iSignal < signalSampleList.size() ; iSignal++ ){
      signalThrowData[iSignal].resetCurrentByteOffset();
      for( int iBin = 0 ; iBin < signalSampleList[iSignal].first->getMcContainer().histogram->GetNbinsX() ; iBin++ ){
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
      TH1D(
        signalSampleList[iSignal].first->getName().c_str(),
        signalSampleList[iSignal].first->getName().c_str(),
        signalSampleList[iSignal].first->getMcContainer().histogram->GetNbinsX(),
        0,
        signalSampleList[iSignal].first->getMcContainer().histogram->GetNbinsX()
      )
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

