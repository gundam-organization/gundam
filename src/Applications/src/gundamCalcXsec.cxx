
#include "GundamGlobals.h"
#include "GundamApp.h"
#include "GundamUtils.h"
#include "RootUtils.h"
#include "FitterEngine.h"
#include "ConfigUtils.h"

#include "Logger.h"
#include "CmdLineParser.h"
#include "GenericToolbox.Root.h"
#include "GenericToolbox.Utils.h"

#include <TFile.h>
#include "TH1D.h"
#include "TH2D.h"

#include <string>
#include <vector>


int main(int argc, char** argv){

  GundamApp app{"cross-section calculator tool"};

  // --------------------------
  // Read Command Line Args:
  // --------------------------
  CmdLineParser clParser;

  clParser.addDummyOption("Main options:");
  clParser.addOption("configFile", {"-c", "--config-file"}, "Specify path to the fitter config file");
  clParser.addOption("fitterFile", {"-f"}, "Specify the fitter output file");
  clParser.addOption("outputFile", {"-o", "--out-file"}, "Specify the CalcXsec output file");
  clParser.addOption("nbThreads", {"-t", "--nb-threads"}, "Specify nb of parallel threads");
  clParser.addOption("nToys", {"-n"}, "Specify number of toys");
  clParser.addOption("randomSeed", {"-s", "--seed"}, "Set random seed");

  clParser.addDummyOption("Trigger options:");
  clParser.addTriggerOption("dryRun", {"-d", "--dry-run"}, "Only overrides fitter config and print it.");
  clParser.addTriggerOption("useBfAsXsec", {"--use-bf-as-xsec"}, "Use best-fit as x-sec value instead of mean of toys.");
  clParser.addTriggerOption("usePreFit", {"--use-prefit"}, "Use prefit covariance matrices for the toy throws.");
  clParser.addTriggerOption("debugVerbose", {"--debug"}, "Add debug verbose.");

  LogInfo << "Usage: " << std::endl;
  LogInfo << clParser.getConfigSummary() << std::endl << std::endl;

  clParser.parseCmdLine(argc, argv);

  LogThrowIf(clParser.isNoOptionTriggered(), "No option was provided.");

  LogInfo << "Provided arguments: " << std::endl;
  LogInfo << clParser.getValueSummary() << std::endl << std::endl;


  GundamGlobals::setIsDebug(clParser.isOptionTriggered("debugVerbose"));

  // Sanity checks
  LogThrowIf(not clParser.isOptionTriggered("configFile"), "Xsec calculator config file not provided.");
  LogThrowIf(not clParser.isOptionTriggered("fitterFile"), "Did not provide the output fitter file.");
  LogThrowIf(not clParser.isOptionTriggered("nToys"), "Did not provide number of toys.");


  // Global parameters
  gRandom = new TRandom3(0);     // Initialize with a UUID
  if( clParser.isOptionTriggered("randomSeed") ){
    LogAlert << "Using user-specified random seed: " << clParser.getOptionVal<ULong_t>("randomSeed") << std::endl;
    gRandom->SetSeed(clParser.getOptionVal<ULong_t>("randomSeed"));
  }
  else{
    ULong_t seed = time(nullptr);
    LogInfo << "Using \"time(nullptr)\" random seed: " << seed << std::endl;
    gRandom->SetSeed(seed);
  }

  GundamGlobals::setNumberOfThreads( clParser.getOptionVal("nbThreads", 1) );
  LogInfo << "Running the fitter with " << GundamGlobals::getNbCpuThreads() << " parallel threads." << std::endl;

  // Reading fitter file
  std::string fitterFile{clParser.getOptionVal<std::string>("fitterFile")};
  std::unique_ptr<TFile> fitterRootFile{nullptr};
  JsonType fitterConfig; // will be used to load the propagator

  if( GenericToolbox::hasExtension(fitterFile, "root") ){
    LogWarning << "Opening fitter output file: " << fitterFile << std::endl;
    fitterRootFile = std::unique_ptr<TFile>( TFile::Open( fitterFile.c_str() ) );
    LogThrowIf( fitterRootFile == nullptr, "Could not open fitter output file." );

    RootUtils::ObjectReader::throwIfNotFound = true;

    RootUtils::ObjectReader::readObject<TNamed>(
        fitterRootFile.get(),
        {{"gundam/config/unfoldedJson_TNamed"},
         {"gundam/config_TNamed"},
         {"gundamFitter/unfoldedConfig_TNamed"}},
        [&](TNamed* config_){
      fitterConfig = GenericToolbox::Json::readConfigJsonStr( config_->GetTitle() );
    });
  }
  else{
    LogWarning << "Reading fitter config file: " << fitterFile << std::endl;
    fitterConfig = GenericToolbox::Json::readConfigFile( fitterFile );

    clParser.getOptionPtr("usePreFit")->setIsTriggered( true );
  }

  LogAlertIf(clParser.isOptionTriggered("usePreFit")) << "Pre-fit mode enabled: will throw toys according to the prior covariance matrices..." << std::endl;

  ConfigUtils::ConfigReader xsecConfig(ConfigUtils::readConfigFile( clParser.getOptionVal<std::string>("configFile") ));

  ConfigUtils::ConfigReader engineConfig;
  {
    ConfigUtils::ConfigBuilder cHandler{ fitterConfig };

    // Disabling defined fit samples:
    LogInfo << "Removing defined samples..." << std::endl;
    GenericToolbox::Json::clearEntry( cHandler.getConfig(), "fitterEngineConfig/likelihoodInterfaceConfig/propagatorConfig/sampleSetConfig/sampleList" );
    GenericToolbox::Json::clearEntry( cHandler.getConfig(), "fitterEngineConfig/likelihoodInterfaceConfig/dataSetManagerConfig/propagatorConfig/sampleSetConfig/sampleList" );
    GenericToolbox::Json::clearEntry( cHandler.getConfig(), "fitterEngineConfig/likelihoodInterfaceConfig/dataSetManagerConfig/propagatorConfig/fitSampleSetConfig/fitSampleList" );
    GenericToolbox::Json::clearEntry( cHandler.getConfig(), "fitterEngineConfig/propagatorConfig/fitSampleSetConfig/fitSampleList" );

    // Disabling defined plots:
    LogInfo << "Removing defined plots..." << std::endl;
    GenericToolbox::Json::clearEntry( cHandler.getConfig(), "fitterEngineConfig/likelihoodInterfaceConfig/propagatorConfig/plotGeneratorConfig" );
    GenericToolbox::Json::clearEntry( cHandler.getConfig(), "fitterEngineConfig/likelihoodInterfaceConfig/dataSetManagerConfig/propagatorConfig/plotGeneratorConfig" );
    GenericToolbox::Json::clearEntry( cHandler.getConfig(), "fitterEngineConfig/propagatorConfig/plotGeneratorConfig" );

    // Defining signal samples
    cHandler.override( xsecConfig.getConfig() );

    engineConfig.setConfig(cHandler.getConfig());
  }



  LogInfo << "Override done." << std::endl;


  LogInfo << "Fetching propagator config into fitter config..." << std::endl;

  // it will handle all the deprecated config options and names properly
  FitterEngine fitter{nullptr};
  fitter.configure( engineConfig.fetchValue<ConfigUtils::ConfigReader>( "fitterEngineConfig" ) );

  // We are only interested in our MC. Data has already been used to get the post-fit error/values
  fitter.getLikelihoodInterface().setForceAsimovData( true );

  // Disabling eigen decomposed parameters
  fitter.getLikelihoodInterface().getModelPropagator().setEnableEigenToOrigInPropagate( false );

  // Sample binning using parameterSetName
  for( auto& sample : fitter.getLikelihoodInterface().getModelPropagator().getSampleSet().getSampleList() ){

    if( clParser.isOptionTriggered("usePreFit") ){
      sample.setName( sample.getName() + " (pre-fit)" );
    }

    // binning already set?
    if( not sample.getBinningFilePath().empty() ){ continue; }

    LogScopeIndent;
    LogInfo << sample.getName() << ": binning not set, looking for parSetBinning..." << std::endl;
    auto associatedParSet = sample.getConfig().fetchValue(
      {{"parSetBinning"}, {"parameterSetName"}}, std::string()
    );

    LogThrowIf(associatedParSet.empty(), "Could not find parSetBinning.");

    // Looking for parSet
    auto foundDialCollection = std::find_if(
        fitter.getLikelihoodInterface().getModelPropagator().getDialCollectionList().begin(),
        fitter.getLikelihoodInterface().getModelPropagator().getDialCollectionList().end(),
        [&](const DialCollection& dialCollection_){
          auto* parSetPtr{dialCollection_.getSupervisedParameterSet()};
          if( parSetPtr == nullptr ){ return false; }
          return ( parSetPtr->getName() == associatedParSet );
        });
    LogThrowIf(
        foundDialCollection == fitter.getLikelihoodInterface().getModelPropagator().getDialCollectionList().end(),
        "Could not find " << associatedParSet << " among fit dial collections: "
                          << GenericToolbox::toString(fitter.getLikelihoodInterface().getModelPropagator().getDialCollectionList(),
                                                      [](const DialCollection& dialCollection_){
                                                        return dialCollection_.getTitle();
                                                      }
                          ));

    LogThrowIf(foundDialCollection->getDialBinSet().getBinList().empty(), "Could not find binning");
    sample.setBinningFilePath( foundDialCollection->getDialBinSet().getFilePath() );

  }

  // Load everything
  fitter.getLikelihoodInterface().initialize();

  Propagator& propagator{fitter.getLikelihoodInterface().getModelPropagator()};


  if( clParser.isOptionTriggered("dryRun") ){
    std::cout << engineConfig.toString() << std::endl;

    LogAlert << "Exiting as dry-run is set." << std::endl;
    return EXIT_SUCCESS;
  }


  if( not clParser.isOptionTriggered("usePreFit") and fitterRootFile != nullptr ){

    // Load post-fit parameters as "prior" so we can reset the weight to this point when throwing toys
    LogWarning << std::endl << GenericToolbox::addUpDownBars("Injecting post-fit parameters...") << std::endl;
    RootUtils::ObjectReader::readObject<TNamed>( fitterRootFile.get(), "FitterEngine/postFit/parState_TNamed", [&](TNamed* parState_){
      propagator.getParametersManager().injectParameterValues( GenericToolbox::Json::readConfigJsonStr( parState_->GetTitle() ) );
      for( auto& parSet : propagator.getParametersManager().getParameterSetsList() ){
        if( not parSet.isEnabled() ){ continue; }
        for( auto& par : parSet.getParameterList() ){
          if( not par.isEnabled() ){ continue; }
          par.setPriorValue( par.getParameterValue() );
        }
      }
    });

    // Load the post-fit covariance matrix
    LogWarning << std::endl << GenericToolbox::addUpDownBars("Injecting post-fit covariance matrix...") << std::endl;
    RootUtils::ObjectReader::readObject<TH2D>(
        fitterRootFile.get(), "FitterEngine/postFit/Hesse/hessian/postfitCovarianceOriginal_TH2D",
        [&](TH2D* hCovPostFit_){
          propagator.getParametersManager().setGlobalCovarianceMatrix(std::make_shared<TMatrixD>(hCovPostFit_->GetNbinsX(), hCovPostFit_->GetNbinsX()));
          for( int iBin = 0 ; iBin < hCovPostFit_->GetNbinsX() ; iBin++ ){
            for( int jBin = 0 ; jBin < hCovPostFit_->GetNbinsX() ; jBin++ ){
              (*propagator.getParametersManager().getGlobalCovarianceMatrix())[iBin][jBin] = hCovPostFit_->GetBinContent(1 + iBin, 1 + jBin);
            }
          }
        }
    );
  }



  // Creating output file
  std::string outFilePath{};
  if( clParser.isOptionTriggered("outputFile") ){ outFilePath = clParser.getOptionVal<std::string>("outputFile"); }
  else{
    // appendixDict["optionName"] = "Appendix"
    // this list insure all appendices will appear in the same order
    std::vector<GundamUtils::AppendixEntry> appendixDict{
        {"configFile", ""},
        {"fitterFile", "Fit"},
        {"nToys", "nToys"},
        {"randomSeed", "Seed"},
        {"usePreFit", "PreFit"},
    };

    outFilePath = "gundamCalcXsec_" + GundamUtils::generateFileName(clParser, appendixDict) + ".root";

    auto outFolder(xsecConfig.fetchValue<std::string>("outputFolder", "./"));
    outFilePath = GenericToolbox::joinPath(outFolder, outFilePath);
  }

  app.setCmdLinePtr( &clParser );
  app.setConfigString( xsecConfig.toString() );
  app.openOutputFile( outFilePath );
  app.writeAppInfo();

  auto* calcXsecDir{ GenericToolbox::mkdirTFile(app.getOutfilePtr(), "calcXsec") };
  bool useBestFitAsCentralValue{
    clParser.isOptionTriggered("useBfAsXsec")
    or xsecConfig.fetchValue<bool>("useBestFitAsCentralValue", false)
  };

  LogInfo << "Creating throws tree" << std::endl;
  auto* xsecThrowTree = new TTree("xsecThrow", "xsecThrow");
  xsecThrowTree->SetDirectory( GenericToolbox::mkdirTFile(calcXsecDir, "throws") ); // temp saves will be done here

  auto* xsecAtBestFitTree = new TTree("xsecAtBestFitTree", "xsecAtBestFitTree");
  xsecAtBestFitTree->SetDirectory( GenericToolbox::mkdirTFile(calcXsecDir, "throws") ); // temp saves will be done here

  LogInfo << "Creating normalizer objects..." << std::endl;
  // flux renorm with toys
  struct ParSetNormaliser{
    void configure(const ConfigUtils::ConfigReader& config_){
      LogScopeIndent;

      name = config_.fetchValue<std::string>("name");
      LogInfo << "ParSetNormaliser config \"" << name << "\": " << std::endl;

      // mandatory
      filePath = config_.fetchValue<std::string>("filePath");
      histogramPath = config_.fetchValue<std::string>("histogramPath");
      axisVariable = config_.fetchValue<std::string>("axisVariable");

      // optionals
      for( auto& parSelConfig : config_.loop("parSelections") ){
        parSelectionList.emplace_back();
        parSelConfig.fillValue(parSelectionList.back().name, "name");
        parSelConfig.fillValue(parSelectionList.back().value, "value");
      }

      // init
      LogScopeIndent;
      LogInfo << GET_VAR_NAME_VALUE(filePath) << std::endl;
      LogInfo << GET_VAR_NAME_VALUE(histogramPath) << std::endl;
      LogInfo << GET_VAR_NAME_VALUE(axisVariable) << std::endl;

      if( not parSelectionList.empty() ){
        LogInfo << "parSelections:" << std::endl;
        for( auto& parSelection : parSelectionList ){
          LogScopeIndent;
          LogInfo << parSelection.name << " -> " << parSelection.value << std::endl;
        }
      }

    }
    void initialize(){
      LogThrowIf(dialCollectionPtr == nullptr, "Associated dial collection not provided.");
      LogThrowIf(dialCollectionPtr->isEventByEvent(), "Dial collection is event by event.");
      LogThrowIf(dialCollectionPtr->getSupervisedParameter() != nullptr, "Need a dial collection that handle a whole parSet.");

      file = std::make_shared<TFile>( filePath.c_str() );
      LogThrowIf(file == nullptr, "Could not open file");

      histogram = file->Get<TH1D>( histogramPath.c_str() );
      LogThrowIf(histogram == nullptr, "Could not find histogram.");
    }
    [[nodiscard]] double getNormFactor() const {
      double out{0};

      for( int iBin = 0 ; iBin < histogram->GetNbinsX() ; iBin++ ){
        double binValue{histogram->GetBinContent(1+iBin)};


        // do we skip this bin? if not, apply coefficient
        bool skipBin{true};
        for( size_t iParBin = 0 ; iParBin < dialCollectionPtr->getDialBinSet().getBinList().size() ; iParBin++ ){
          const Bin& parBin = dialCollectionPtr->getDialBinSet().getBinList()[iParBin];

          bool isParBinValid{true};

          // first check the conditions
          for( auto& selection : parSelectionList ){
            if( parBin.isVariableSet(selection.name) and not parBin.isBetweenEdges(selection.name, selection.value) ){
              isParBinValid = false;
              break;
            }
          }

          // checking if the hist bin correspond to this
          if( parBin.isVariableSet(axisVariable) and not parBin.isBetweenEdges(axisVariable, histogram->GetBinCenter(1+iBin)) ){
            isParBinValid = false;
          }

          if( isParBinValid ){
            // ok, then apply the weight
            binValue *= dialCollectionPtr->getSupervisedParameterSet()->getParameterList()[iParBin].getParameterValue();

            skipBin = false;
            break;
          }
        }
        if( skipBin ){ continue; }

        // ok, add the fluctuated value
        out += binValue;
      }

      return out;
    }

    // config
    std::string name{};
    std::string filePath{};
    std::string histogramPath{};
    std::string axisVariable{};

    struct ParSelection{
      std::string name{};
      double value{};
    };
    std::vector<ParSelection> parSelectionList{};

    // internals
    std::shared_ptr<TFile> file{nullptr};
    TH1D* histogram{nullptr};
    const DialCollection* dialCollectionPtr{nullptr}; // where the binning is defined
  };
  std::vector<ParSetNormaliser> parSetNormList;
  for( auto& parSet : propagator.getParametersManager().getParameterSetsList() ){
    for( auto& parSetNormConfig : parSet.getConfig().loop("normalisations") ){
      parSetNormList.emplace_back();
      parSetNormList.back().configure( parSetNormConfig );

      for( auto& dialCollection : propagator.getDialCollectionList() ){
        if( dialCollection.getSupervisedParameterSet() == &parSet ){
          parSetNormList.back().dialCollectionPtr = &dialCollection;
          break;
        }
      }

      parSetNormList.back().initialize();
    }
  }



  // to be filled up
  struct BinNormaliser{
    void configure(const ConfigUtils::ConfigReader& config_){
      LogScopeIndent;

      name = config_.fetchValue<std::string>("name");

      if( not config_.fetchValue("isEnabled", bool(true)) ){
        LogWarning << "Skipping disabled re-normalization config \"" << name << "\"" << std::endl;
        return;
      }

      LogInfo << "Re-normalization config \"" << name << "\": ";

      if     ( config_.hasKey( "meanValue" ) ){
        normParameter.min  = config_.fetchValue<double>("meanValue");
        normParameter.max = config_.fetchValue("stdDev", double(0.));
        LogInfo << "mean ± sigma = " << normParameter.min << " ± " << normParameter.max;
      }
      else if( config_.hasKey("disabledBinDim" ) ){
        disabledBinDim = config_.fetchValue<std::string>("disabledBinDim");
        LogInfo << "disabledBinDim = " << disabledBinDim;
      }
      else if( config_.hasKey("parSetNormName" ) ){
        parSetNormaliserName = config_.fetchValue<std::string>("parSetNormName");
        LogInfo << "parSetNormName = " << parSetNormaliserName;
      }
      else{
        LogInfo << std::endl;
        LogThrow("Unrecognized config.");
      }

      LogInfo << std::endl;
    }

    std::string name{};
    GenericToolbox::Range normParameter{};
    std::string disabledBinDim{};
    std::string parSetNormaliserName{};

  };

  struct CrossSectionData{
    Sample* samplePtr{nullptr};
    ConfigUtils::ConfigReader config{};
    GenericToolbox::RawDataArray branchBinsData{};

    TH1D histogram{};
    std::vector<BinNormaliser> normList{};
  };
  std::vector<CrossSectionData> crossSectionDataList{};

  LogInfo << "Initializing xsec samples..." << std::endl;
  crossSectionDataList.reserve(propagator.getSampleSet().getSampleList().size() );
  for( auto& sample : propagator.getSampleSet().getSampleList() ){
    crossSectionDataList.emplace_back();
    auto& xsecEntry = crossSectionDataList.back();

    LogScopeIndent;
    LogInfo << "Defining xsec entry: " << sample.getName() << std::endl;
    xsecEntry.samplePtr = &sample;
    xsecEntry.config = sample.getConfig();
    xsecEntry.branchBinsData.resetCursor();
    std::vector<std::string> leafNameList{};
    leafNameList.reserve( sample.getHistogram().getNbBins() );
    for( int iBin = 0 ; iBin < sample.getHistogram().getNbBins(); iBin++ ){
      leafNameList.emplace_back(Form("bin_%i/D", iBin));
      xsecEntry.branchBinsData.writeRawData( double(0) );
    }
    xsecEntry.branchBinsData.lock();

    xsecThrowTree->Branch(
        GenericToolbox::generateCleanBranchName( sample.getName() ).c_str(),
        xsecEntry.branchBinsData.getRawDataArray().data(),
        GenericToolbox::joinVectorString(leafNameList, ":").c_str()
    );
    xsecAtBestFitTree->Branch(
        GenericToolbox::generateCleanBranchName( sample.getName() ).c_str(),
        xsecEntry.branchBinsData.getRawDataArray().data(),
        GenericToolbox::joinVectorString(leafNameList, ":").c_str()
    );

    auto normConfigList = xsecEntry.config.loop("normaliseParameterList");
    xsecEntry.normList.reserve( normConfigList.size() );
    for( auto& normConfig : normConfigList ){
      xsecEntry.normList.emplace_back();
      xsecEntry.normList.back().configure( normConfig );
    }

    xsecEntry.histogram = TH1D(
        sample.getName().c_str(),
        sample.getName().c_str(),
        sample.getHistogram().getNbBins(),
        0,
        sample.getHistogram().getNbBins()
    );
  }

  int nToys{ clParser.getOptionVal<int>("nToys") };

  // no bin volume of events -> use the current weight container
  for( auto& xsec : crossSectionDataList ){
    {
      auto& mcEvList{xsec.samplePtr->getEventList()};
      std::for_each(mcEvList.begin(), mcEvList.end(), []( Event& ev_){ ev_.getWeights().current = 0; });
    }
//    {
//      auto& dataEvList{xsec.samplePtr->getDataContainer().getEventList()};
//      std::for_each(dataEvList.begin(), dataEvList.end(), []( Event& ev_){ ev_.getWeights().current = 0; });
//    }
  }

  bool enableEventMcThrow{true};
  bool enableStatThrowInToys{true};
  auto xsecCalcConfig   = xsecConfig.fetchValue( "xsecCalcConfig", ConfigReader() );
  enableStatThrowInToys = xsecCalcConfig.fetchValue("enableStatThrowInToys", enableStatThrowInToys);
  enableEventMcThrow    = xsecCalcConfig.fetchValue("enableEventMcThrow", enableEventMcThrow);

  auto writeBinDataFct = std::function<void()>([&]{
    for( auto& xsec : crossSectionDataList ){

      xsec.branchBinsData.resetCursor();
      for( int iBin = 0 ; iBin < xsec.samplePtr->getHistogram().getNbBins() ; iBin++ ){
        double binData{ xsec.samplePtr->getHistogram().getBinContentList()[iBin].sumWeights };

        // special re-norm
        for( auto& normData : xsec.normList ){
          if( not std::isnan( normData.normParameter.min ) ){
            double norm{normData.normParameter.min};
            if( normData.normParameter.max != 0 ){ norm += normData.normParameter.max * gRandom->Gaus(); }
            binData /= norm;
          }
          else if( not normData.parSetNormaliserName.empty() ){
            ParSetNormaliser* parSetNormPtr{nullptr};
            for( auto& parSetNorm : parSetNormList ){
              if( parSetNorm.name == normData.parSetNormaliserName ){
                parSetNormPtr = &parSetNorm;
                break;
              }
            }
            LogThrowIf(parSetNormPtr == nullptr, "Could not find parSetNorm obj with name: " << normData.parSetNormaliserName);

            binData /= parSetNormPtr->getNormFactor();
          }
        }

        // no bin volume of events
        {
          auto& mcEvList{xsec.samplePtr->getEventList()};
          std::for_each(mcEvList.begin(), mcEvList.end(), [&]( Event& ev_){
            if( iBin != ev_.getIndices().bin ){ return; }
            ev_.getWeights().current += binData;
          });
        }

        // set event weight
//        {
//          auto& dataEvList{xsec.samplePtr->getDataContainer().getEventList()};
//          std::for_each(dataEvList.begin(), dataEvList.end(), [&]( Event& ev_){
//            if( iBin != ev_.getIndices().bin ){ return; }
//            ev_.getWeights().current = binData;
//          });
//        }

        // bin volume
        auto& bin = xsec.samplePtr->getHistogram().getBinContextList()[iBin].bin;
        double binVolume{1};

        for( auto& edges : bin.getEdgesList() ){
          if( edges.isConditionVar ){ continue; } // no volume, just a condition variable

          // is this bin excluded from the normalisation ?
          if( GenericToolbox::doesElementIsInVector(edges.varName, xsec.normList, [](const BinNormaliser& n){ return n.disabledBinDim; }) ){
            continue;
          }

          binVolume *= (edges.max - edges.min);
        }

        binData /= binVolume;
        xsec.branchBinsData.writeRawData( binData );
      }
    }
  });

  {
    LogWarning << "Calculating weight at best-fit" << std::endl;
    for( auto& parSet : propagator.getParametersManager().getParameterSetsList() ){ parSet.moveParametersToPrior(); }
    propagator.propagateParameters();
    writeBinDataFct();
    xsecAtBestFitTree->Fill();
    GenericToolbox::writeInTFileWithObjTypeExt( GenericToolbox::mkdirTFile(calcXsecDir, "throws"), xsecAtBestFitTree );
  }


  //////////////////////////////////////
  // THROWS LOOP
  /////////////////////////////////////
  LogWarning << std::endl << GenericToolbox::addUpDownBars( "Generating toys..." ) << std::endl;
  propagator.getParametersManager().initializeStrippedGlobalCov();

  // stats printing
  GenericToolbox::Time::AveragedTimer<1> totalTimer{};
  GenericToolbox::Time::AveragedTimer<1> throwTimer{};
  GenericToolbox::Time::AveragedTimer<1> propagateTimer{};
  GenericToolbox::Time::AveragedTimer<1> otherTimer{};
  GenericToolbox::Time::AveragedTimer<1> writeTimer{};
  GenericToolbox::TablePrinter t{};
  std::stringstream progressSs;
  std::stringstream ss; ss << LogWarning.getPrefixString() << "Generating " << nToys << " toys...";
  for( int iToy = 0 ; iToy < nToys ; iToy++ ){

    t.reset();
    t << "Total time" << GenericToolbox::TablePrinter::NextColumn;
    t << "Throw toys" << GenericToolbox::TablePrinter::NextColumn;
    t << "Propagate pars" << GenericToolbox::TablePrinter::NextColumn;
    t << "Re-normalize" << GenericToolbox::TablePrinter::NextColumn;
    t << "Write throws" << GenericToolbox::TablePrinter::NextLine;

    t << totalTimer << GenericToolbox::TablePrinter::NextColumn;
    t << throwTimer << GenericToolbox::TablePrinter::NextColumn;
    t << propagateTimer << GenericToolbox::TablePrinter::NextColumn;
    t << otherTimer << GenericToolbox::TablePrinter::NextColumn;
    t << writeTimer << GenericToolbox::TablePrinter::NextLine;

    totalTimer.stop();
    totalTimer.start();

    // loading...
    progressSs.str("");
    progressSs << t.generateTableString() << std::endl;
    progressSs << ss.str();
    GenericToolbox::displayProgressBar( iToy+1, nToys, progressSs.str() );

    // Do the throwing:
    throwTimer.start();
    propagator.getParametersManager().throwParametersFromGlobalCovariance( not GundamGlobals::isDebug() );
    throwTimer.stop();

    propagateTimer.start();
    propagator.propagateParameters();

    if( enableStatThrowInToys ){
      for( auto& xsec : crossSectionDataList ){
        if( enableEventMcThrow and not xsec.samplePtr->isEventMcThrowDisabled() ){
          // Take into account the finite amount of event in MC
          xsec.samplePtr->getHistogram().throwEventMcError();
        }
        // Asimov bin content -> toy data
        xsec.samplePtr->getHistogram().throwStatError();
      }
    }
    propagateTimer.stop();

    otherTimer.start();
    // TODO: parallelize this
    writeBinDataFct();
    otherTimer.stop();

    // Write the branches
    writeTimer.start();
    xsecThrowTree->Fill();
    writeTimer.stop();
  }


  LogInfo << "Writing throws..." << std::endl;
  GenericToolbox::writeInTFileWithObjTypeExt( GenericToolbox::mkdirTFile(calcXsecDir, "throws"), xsecThrowTree );

  LogInfo << "Calculating mean & covariance matrix..." << std::endl;
  auto* meanValuesVector = GenericToolbox::generateMeanVectorOfTree(
      useBestFitAsCentralValue ? xsecAtBestFitTree : xsecThrowTree
  );
  auto* globalCovMatrix = GenericToolbox::generateCovarianceMatrixOfTree( xsecThrowTree );

  auto* globalCovMatrixHist = GenericToolbox::convertTMatrixDtoTH2D(globalCovMatrix);
  auto* globalCorMatrixHist = GenericToolbox::convertTMatrixDtoTH2D(GenericToolbox::convertToCorrelationMatrix(globalCovMatrix));

  std::vector<TH1D> binValues{};
  binValues.reserve(propagator.getSampleSet().getSampleList().size() );
  int iBinGlobal{-1};

  for( auto& xsec : crossSectionDataList ){

    for( int iBin = 0 ; iBin < xsec.samplePtr->getHistogram().getNbBins() ; iBin++ ){
      iBinGlobal++;

      std::string binTitle = xsec.samplePtr->getHistogram().getBinContextList()[iBin].bin.getSummary();
      double binVolume = xsec.samplePtr->getHistogram().getBinContextList()[iBin].bin.getVolume();

      xsec.histogram.SetBinContent( 1+iBin, (*meanValuesVector)[iBinGlobal] );
      xsec.histogram.SetBinError( 1+iBin, std::sqrt( (*globalCovMatrix)[iBinGlobal][iBinGlobal] ) );
      xsec.histogram.GetXaxis()->SetBinLabel( 1+iBin, binTitle.c_str() );

      globalCovMatrixHist->GetXaxis()->SetBinLabel(1+iBinGlobal, GenericToolbox::joinPath(xsec.samplePtr->getName(), binTitle).c_str());
      globalCorMatrixHist->GetXaxis()->SetBinLabel(1+iBinGlobal, GenericToolbox::joinPath(xsec.samplePtr->getName(), binTitle).c_str());
      globalCovMatrixHist->GetYaxis()->SetBinLabel(1+iBinGlobal, GenericToolbox::joinPath(xsec.samplePtr->getName(), binTitle).c_str());
      globalCorMatrixHist->GetYaxis()->SetBinLabel(1+iBinGlobal, GenericToolbox::joinPath(xsec.samplePtr->getName(), binTitle).c_str());
    }

    xsec.histogram.SetMarkerStyle(kFullDotLarge);
    xsec.histogram.SetMarkerColor(kGreen-3);
    xsec.histogram.SetMarkerSize(0.5);
    xsec.histogram.SetLineWidth(2);
    xsec.histogram.SetLineColor(kGreen-3);
    xsec.histogram.SetDrawOption("E1");
    xsec.histogram.GetXaxis()->LabelsOption("v");
    xsec.histogram.GetXaxis()->SetLabelSize(0.02);
    xsec.histogram.GetYaxis()->SetTitle( xsec.samplePtr->getConfig().fetchValue("yAxis", std::string("#delta#sigma")).c_str() );

    GenericToolbox::writeInTFileWithObjTypeExt(
        GenericToolbox::mkdirTFile(calcXsecDir, "histograms"),
        &xsec.histogram, GenericToolbox::generateCleanBranchName( xsec.samplePtr->getName() )
    );

  }

  globalCovMatrixHist->GetXaxis()->SetLabelSize(0.02);
  globalCovMatrixHist->GetYaxis()->SetLabelSize(0.02);
  GenericToolbox::writeInTFileWithObjTypeExt(GenericToolbox::mkdirTFile(calcXsecDir, "matrices"), globalCovMatrixHist, "covarianceMatrix");

  globalCorMatrixHist->GetXaxis()->SetLabelSize(0.02);
  globalCorMatrixHist->GetYaxis()->SetLabelSize(0.02);
  globalCorMatrixHist->GetZaxis()->SetRangeUser(-1, 1);
  GenericToolbox::writeInTFileWithObjTypeExt(GenericToolbox::mkdirTFile(calcXsecDir, "matrices"), globalCorMatrixHist, "correlationMatrix");

  // now propagate to the engine for the plot generator
  LogInfo << "Re-normalizing the samples for the plot generator..." << std::endl;
  for( auto& xsec : crossSectionDataList ){
    // this gives the average as the event weights were summed together
    {
      auto &mcEvList{xsec.samplePtr->getEventList()};
      std::vector<size_t> nEventInBin(xsec.histogram.GetNbinsX(), 0);
      for( size_t iBin = 0 ; iBin < nEventInBin.size() ; iBin++ ){
        nEventInBin[iBin] = std::count_if(mcEvList.begin(), mcEvList.end(), [iBin]( Event &ev_) {
          return ev_.getIndices().bin == iBin;
        });
      }

      std::for_each(mcEvList.begin(), mcEvList.end(), [&]( Event &ev_) {
        ev_.getWeights().current /= nToys;
        ev_.getWeights().current /= double(nEventInBin[ev_.getIndices().bin]);
      });
    }
//    {
//      auto &dataEvList{xsec.samplePtr->getDataContainer().getEventList()};
//      std::vector<size_t> nEventInBin(xsec.histogram.GetNbinsX(), 0);
//      for( size_t iBin = 0 ; iBin < nEventInBin.size() ; iBin++ ){
//        nEventInBin[iBin] = std::count_if(dataEvList.begin(), dataEvList.end(), [iBin]( Event &ev_) {
//          return ev_.getIndices().bin== iBin;
//        });
//      }
//
//      std::for_each(dataEvList.begin(), dataEvList.end(), [&]( Event &ev_) {
//        ev_.getWeights().current /= nToys;
//        ev_.getWeights().current /= double(nEventInBin[ev_.getIndices().bin]);
//      });
//    }
  }

  LogInfo << "Generating xsec sample plots..." << std::endl;
  // manual trigger to tweak the error bars
  fitter.getLikelihoodInterface().getPlotGenerator().generateSampleHistograms( GenericToolbox::mkdirTFile(calcXsecDir, "plots/histograms") );

  for( auto& histHolder : fitter.getLikelihoodInterface().getPlotGenerator().getHistHolderList(0) ){
    if( not histHolder.isData ){ continue; } // only data will print errors

    const CrossSectionData* xsecDataPtr{nullptr};
    for( auto& xsecData : crossSectionDataList ){
      /*
      if( xsecData.samplePtr  == histHolder.samplePtr){
        xsecDataPtr = &xsecData;
        break;
      }
      */
      if( (xsecData.samplePtr)->getName()  == (histHolder.samplePtr)->getName()){
        xsecDataPtr = &xsecData;
        break;
      }      
    }
    LogThrowIf(xsecDataPtr==nullptr, "corresponding data not found");

    // alright, now rescale error bars
    for( int iBin = 0 ; iBin < histHolder.histPtr->GetNbinsX() ; iBin++ ){
      // relative error should be set
      histHolder.histPtr->SetBinError(
          1+iBin,
          histHolder.histPtr->GetBinContent(1+iBin)
          * xsecDataPtr->histogram.GetBinError(1+iBin)
          / xsecDataPtr->histogram.GetBinContent(1+iBin)
      );
    }
  }

  fitter.getLikelihoodInterface().getPlotGenerator().generateCanvas(
      fitter.getLikelihoodInterface().getPlotGenerator().getHistHolderList(0),
      GenericToolbox::mkdirTFile(calcXsecDir, "plots/canvas")
  );


  LogInfo << "Writing event samples in TTrees..." << std::endl;
  fitter.getLikelihoodInterface().writeEvents({calcXsecDir, "events"});

}
