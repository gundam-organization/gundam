
#include "GundamGlobals.h"
#include "GundamApp.h"
#include "GundamUtils.h"
#include "FitterEngine.h"
#include "ConfigUtils.h"

#include "Logger.h"
#include "CmdLineParser.h"
#include "GenericToolbox.Json.h"
#include "GenericToolbox.Root.h"
#include "GenericToolbox.Utils.h"

#include <TFile.h>
#include "TH1D.h"
#include "TH2D.h"

#include <string>
#include <vector>


LoggerInit([]{
  Logger::getUserHeader() << "[" << FILENAME << "]";
});


int main(int argc, char** argv){

  using namespace GundamUtils;

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

  LogInfo << "Usage: " << std::endl;
  LogInfo << clParser.getConfigSummary() << std::endl << std::endl;

  clParser.parseCmdLine(argc, argv);

  LogThrowIf(clParser.isNoOptionTriggered(), "No option was provided.");

  LogInfo << "Provided arguments: " << std::endl;
  LogInfo << clParser.getValueSummary() << std::endl << std::endl;


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

  GundamGlobals::getParallelWorker().setNThreads( clParser.getOptionVal("nbThreads", 1) );
  LogInfo << "Running the fitter with " << GundamGlobals::getParallelWorker().getNbThreads() << " parallel threads." << std::endl;

  // Reading fitter file
  std::string fitterFile{clParser.getOptionVal<std::string>("fitterFile")};
  std::unique_ptr<TFile> fitterRootFile{nullptr};
  JsonType fitterConfig; // will be used to load the propagator

  if( GenericToolbox::hasExtension(fitterFile, "root") ){
    LogWarning << "Opening fitter output file: " << fitterFile << std::endl;
    fitterRootFile = std::unique_ptr<TFile>( TFile::Open( fitterFile.c_str() ) );
    LogThrowIf( fitterRootFile == nullptr, "Could not open fitter output file." );

    ObjectReader::throwIfNotFound = true;

    ObjectReader::readObject<TNamed>(fitterRootFile.get(), {{"gundam/config_TNamed"}, {"gundamFitter/unfoldedConfig_TNamed"}}, [&](TNamed* config_){
      fitterConfig = GenericToolbox::Json::readConfigJsonStr( config_->GetTitle() );
    });
  }
  else{
    LogWarning << "Reading fitter config file: " << fitterFile << std::endl;
    fitterConfig = GenericToolbox::Json::readConfigFile( fitterFile );

    clParser.getOptionPtr("usePreFit")->setIsTriggered( true );
  }

  LogAlertIf(clParser.isOptionTriggered("usePreFit")) << "Pre-fit mode enabled: will throw toys according to the prior covariance matrices..." << std::endl;

  ConfigUtils::ConfigHandler cHandler{ fitterConfig };

  // Disabling defined fit samples:
  LogInfo << "Removing defined samples..." << std::endl;
  ConfigUtils::clearEntry( cHandler.getConfig(), "fitterEngineConfig/likelihoodInterfaceConfig/dataSetManagerConfig/propagatorConfig/sampleSetConfig/sampleList" );
  ConfigUtils::clearEntry( cHandler.getConfig(), "fitterEngineConfig/likelihoodInterfaceConfig/dataSetManagerConfig/propagatorConfig/fitSampleSetConfig/fitSampleList" );
  ConfigUtils::clearEntry( cHandler.getConfig(), "fitterEngineConfig/propagatorConfig/fitSampleSetConfig/fitSampleList" );

  // Disabling defined plots:
  LogInfo << "Removing defined plots..." << std::endl;
  ConfigUtils::clearEntry( cHandler.getConfig(), "fitterEngineConfig/likelihoodInterfaceConfig/dataSetManagerConfig/propagatorConfig/plotGeneratorConfig" );
  ConfigUtils::clearEntry( cHandler.getConfig(), "fitterEngineConfig/propagatorConfig/plotGeneratorConfig" );

  // Defining signal samples
  JsonType xsecConfig{ ConfigUtils::readConfigFile( clParser.getOptionVal<std::string>("configFile") ) };
  cHandler.override( xsecConfig );
  LogInfo << "Override done." << std::endl;


  LogInfo << "Fetching propagator config into fitter config..." << std::endl;

  // it will handle all the deprecated config options and names properly
  FitterEngine fitter{nullptr};
  fitter.readConfig( GenericToolbox::Json::fetchValuePath<JsonType>( cHandler.getConfig(), "fitterEngineConfig" ) );

  DataSetManager& dataSetManager{fitter.getLikelihoodInterface().getDataSetManager()};

  // We are only interested in our MC. Data has already been used to get the post-fit error/values
  dataSetManager.getPropagator().setLoadAsimovData( true );

  // Disabling eigen decomposed parameters
  dataSetManager.getPropagator().setEnableEigenToOrigInPropagate( false );

  // Sample binning using parameterSetName
  for( auto& sample : dataSetManager.getPropagator().getSampleSet().getSampleList() ){

    if( clParser.isOptionTriggered("usePreFit") ){
      sample.setName( sample.getName() + " (pre-fit)" );
    }

    // binning already set?
    if( not sample.getBinningFilePath().empty() ){ continue; }

    LogScopeIndent;
    LogInfo << sample.getName() << ": binning not set, looking for parSetBinning..." << std::endl;
    auto associatedParSet = GenericToolbox::Json::fetchValue(
        sample.getConfig(),
        {{"parSetBinning"}, {"parameterSetName"}},
        std::string()
    );

    LogThrowIf(associatedParSet.empty(), "Could not find parSetBinning.");

    // Looking for parSet
    auto foundDialCollection = std::find_if(
        dataSetManager.getPropagator().getDialCollectionList().begin(),
        dataSetManager.getPropagator().getDialCollectionList().end(),
        [&](const DialCollection& dialCollection_){
          auto* parSetPtr{dialCollection_.getSupervisedParameterSet()};
          if( parSetPtr == nullptr ){ return false; }
          return ( parSetPtr->getName() == associatedParSet );
        });
    LogThrowIf(
        foundDialCollection == dataSetManager.getPropagator().getDialCollectionList().end(),
        "Could not find " << associatedParSet << " among fit dial collections: "
                          << GenericToolbox::toString(dataSetManager.getPropagator().getDialCollectionList(),
                                                      [](const DialCollection& dialCollection_){
                                                        return dialCollection_.getTitle();
                                                      }
                          ));

    LogThrowIf(foundDialCollection->getDialBinSet().getBinList().empty(), "Could not find binning");
    sample.setBinningFilePath( foundDialCollection->getDialBinSet().getFilePath() );

  }

  // Load everything
  dataSetManager.initialize();

  Propagator& propagator{dataSetManager.getPropagator()};


  if( clParser.isOptionTriggered("dryRun") ){
    std::cout << cHandler.toString() << std::endl;

    LogAlert << "Exiting as dry-run is set." << std::endl;
    return EXIT_SUCCESS;
  }


  if( not clParser.isOptionTriggered("usePreFit") and fitterRootFile != nullptr ){

    // Load post-fit parameters as "prior" so we can reset the weight to this point when throwing toys
    LogWarning << std::endl << GenericToolbox::addUpDownBars("Injecting post-fit parameters...") << std::endl;
    ObjectReader::readObject<TNamed>( fitterRootFile.get(), "FitterEngine/postFit/parState_TNamed", [&](TNamed* parState_){
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
    ObjectReader::readObject<TH2D>(
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
    std::vector<std::pair<std::string, std::string>> appendixDict{
        {"configFile", ""},
        {"fitterFile", "Fit"},
        {"nToys", "nToys"},
        {"randomSeed", "Seed"},
        {"usePreFit", "PreFit"},
    };

    outFilePath = "gundamCalcXsec_" + GundamUtils::generateFileName(clParser, appendixDict) + ".root";

    std::string outFolder(GenericToolbox::Json::fetchValue<std::string>(xsecConfig, "outputFolder", "./"));
    outFilePath = GenericToolbox::joinPath(outFolder, outFilePath);
  }

  app.setCmdLinePtr( &clParser );
  app.setConfigString( ConfigUtils::ConfigHandler{xsecConfig}.toString() );
  app.openOutputFile( outFilePath );
  app.writeAppInfo();

  auto* calcXsecDir{ GenericToolbox::mkdirTFile(app.getOutfilePtr(), "calcXsec") };
  bool useBestFitAsCentralValue{
    clParser.isOptionTriggered("useBfAsXsec")
    or GenericToolbox::Json::fetchValue<bool>(xsecConfig, "useBestFitAsCentralValue", false)
  };

  LogInfo << "Creating throws tree" << std::endl;
  auto* xsecThrowTree = new TTree("xsecThrow", "xsecThrow");
  xsecThrowTree->SetDirectory( GenericToolbox::mkdirTFile(calcXsecDir, "throws") ); // temp saves will be done here

  auto* xsecAtBestFitTree = new TTree("xsecAtBestFitTree", "xsecAtBestFitTree");
  xsecAtBestFitTree->SetDirectory( GenericToolbox::mkdirTFile(calcXsecDir, "throws") ); // temp saves will be done here

  LogInfo << "Creating normalizer objects..." << std::endl;
  // flux renorm with toys
  struct ParSetNormaliser{
    void readConfig(const JsonType& config_){
      LogScopeIndent;

      name = GenericToolbox::Json::fetchValue<std::string>(config_, "name");
      LogInfo << "ParSetNormaliser config \"" << name << "\": " << std::endl;

      // mandatory
      filePath = GenericToolbox::Json::fetchValue<std::string>(config_, "filePath");
      histogramPath = GenericToolbox::Json::fetchValue<std::string>(config_, "histogramPath");
      axisVariable = GenericToolbox::Json::fetchValue<std::string>(config_, "axisVariable");

      // optionals
      for( auto& parSelConfig : GenericToolbox::Json::fetchValue<JsonType>(config_, "parSelections") ){
        parSelections.emplace_back();
        parSelections.back().first = GenericToolbox::Json::fetchValue<std::string>(parSelConfig, "name");
        parSelections.back().second = GenericToolbox::Json::fetchValue<double>(parSelConfig, "value");
      }
      parSelections = GenericToolbox::Json::fetchValue(config_, "parSelections", parSelections);

      // init
      LogScopeIndent;
      LogInfo << GET_VAR_NAME_VALUE(filePath) << std::endl;
      LogInfo << GET_VAR_NAME_VALUE(histogramPath) << std::endl;
      LogInfo << GET_VAR_NAME_VALUE(axisVariable) << std::endl;

      if( not parSelections.empty() ){
        LogInfo << "parSelections:" << std::endl;
        for( auto& parSelection : parSelections ){
          LogScopeIndent;
          LogInfo << parSelection.first << " -> " << parSelection.second << std::endl;
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
          const DataBin& parBin = dialCollectionPtr->getDialBinSet().getBinList()[iParBin];

          bool isParBinValid{true};

          // first check the conditions
          for( auto& selection : parSelections ){
            if( parBin.isVariableSet(selection.first) and not parBin.isBetweenEdges(selection.first, selection.second) ){
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
    std::vector<std::pair<std::string, double>> parSelections{};

    // internals
    std::shared_ptr<TFile> file{nullptr};
    TH1D* histogram{nullptr};
    const DialCollection* dialCollectionPtr{nullptr}; // where the binning is defined
  };
  std::vector<ParSetNormaliser> parSetNormList;
  for( auto& parSet : propagator.getParametersManager().getParameterSetsList() ){
    if( GenericToolbox::Json::doKeyExist(parSet.getConfig(), "normalisations") ){
      for( auto& parSetNormConfig : GenericToolbox::Json::fetchValue<JsonType>(parSet.getConfig(), "normalisations") ){
        parSetNormList.emplace_back();
        parSetNormList.back().readConfig( parSetNormConfig );

        for( auto& dialCollection : propagator.getDialCollectionList() ){
          if( dialCollection.getSupervisedParameterSet() == &parSet ){
            parSetNormList.back().dialCollectionPtr = &dialCollection;
            break;
          }
        }

        parSetNormList.back().initialize();
      }
    }
  }



  // to be filled up
  struct BinNormaliser{
    void readConfig(const JsonType& config_){
      LogScopeIndent;

      name = GenericToolbox::Json::fetchValue<std::string>(config_, "name");

      if( not GenericToolbox::Json::fetchValue(config_, "isEnabled", bool(true)) ){
        LogWarning << "Skipping disabled re-normalization config \"" << name << "\"" << std::endl;
        return;
      }

      LogInfo << "Re-normalization config \"" << name << "\": ";

      if     ( GenericToolbox::Json::doKeyExist( config_, "meanValue" ) ){
        normParameter.first  = GenericToolbox::Json::fetchValue<double>(config_, "meanValue");
        normParameter.second = GenericToolbox::Json::fetchValue(config_, "stdDev", double(0.));
        LogInfo << "mean ± sigma = " << normParameter.first << " ± " << normParameter.second;
      }
      else if( GenericToolbox::Json::doKeyExist( config_, "disabledBinDim" ) ){
        disabledBinDim = GenericToolbox::Json::fetchValue<std::string>(config_, "disabledBinDim");
        LogInfo << "disabledBinDim = " << disabledBinDim;
      }
      else if( GenericToolbox::Json::doKeyExist( config_, "parSetNormName" ) ){
        parSetNormaliserName = GenericToolbox::Json::fetchValue<std::string>(config_, "parSetNormName");
        LogInfo << "parSetNormName = " << parSetNormaliserName;
      }
      else{
        LogInfo << std::endl;
        LogThrow("Unrecognized config.");
      }

      LogInfo << std::endl;
    }

    std::string name{};
    std::pair<double, double> normParameter{std::nan("mean unset"), std::nan("stddev unset")};
    std::string disabledBinDim{};
    std::string parSetNormaliserName{};

  };

  struct CrossSectionData{
    Sample* samplePtr{nullptr};
    JsonType config{};
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
    xsecEntry.branchBinsData.resetCurrentByteOffset();
    std::vector<std::string> leafNameList{};
    leafNameList.reserve( sample.getMcContainer().getHistogram().nBins );
    for( int iBin = 0 ; iBin < sample.getMcContainer().getHistogram().nBins; iBin++ ){
      leafNameList.emplace_back(Form("bin_%i/D", iBin));
      xsecEntry.branchBinsData.writeRawData( double(0) );
    }
    xsecEntry.branchBinsData.lockArraySize();

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

    auto normConfigList = GenericToolbox::Json::fetchValue( xsecEntry.config, "normaliseParameterList", JsonType() );
    xsecEntry.normList.reserve( normConfigList.size() );
    for( auto& normConfig : normConfigList ){
      xsecEntry.normList.emplace_back();
      xsecEntry.normList.back().readConfig( normConfig );
    }

    xsecEntry.histogram = TH1D(
        sample.getName().c_str(),
        sample.getName().c_str(),
        sample.getMcContainer().getHistogram().nBins,
        0,
        sample.getMcContainer().getHistogram().nBins
    );
  }

  int nToys{ clParser.getOptionVal<int>("nToys") };

  // no bin volume of events -> use the current weight container
  for( auto& xsec : crossSectionDataList ){
    {
      auto& mcEvList{xsec.samplePtr->getMcContainer().getEventList()};
      std::for_each(mcEvList.begin(), mcEvList.end(), []( Event& ev_){ ev_.getWeights().current = 0; });
    }
    {
      auto& dataEvList{xsec.samplePtr->getDataContainer().getEventList()};
      std::for_each(dataEvList.begin(), dataEvList.end(), []( Event& ev_){ ev_.getWeights().current = 0; });
    }
  }

  bool enableEventMcThrow{true};
  bool enableStatThrowInToys{true};
  auto xsecCalcConfig   = GenericToolbox::Json::fetchValue( cHandler.getConfig(), "xsecCalcConfig", JsonType() );
  enableStatThrowInToys = GenericToolbox::Json::fetchValue( xsecCalcConfig, "enableStatThrowInToys", enableStatThrowInToys);
  enableEventMcThrow    = GenericToolbox::Json::fetchValue( xsecCalcConfig, "enableEventMcThrow", enableEventMcThrow);

  auto writeBinDataFct = std::function<void()>([&]{
    for( auto& xsec : crossSectionDataList ){

      xsec.branchBinsData.resetCurrentByteOffset();
      for( int iBin = 0 ; iBin < xsec.samplePtr->getMcContainer().getHistogram().nBins ; iBin++ ){
        double binData{ xsec.samplePtr->getMcContainer().getHistogram().binList[iBin].content };

        // special re-norm
        for( auto& normData : xsec.normList ){
          if( not std::isnan( normData.normParameter.first ) ){
            double norm{normData.normParameter.first};
            if( normData.normParameter.second != 0 ){ norm += normData.normParameter.second * gRandom->Gaus(); }
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
          auto& mcEvList{xsec.samplePtr->getMcContainer().getEventList()};
          std::for_each(mcEvList.begin(), mcEvList.end(), [&]( Event& ev_){
            if( iBin != ev_.getIndices().bin ){ return; }
            ev_.getWeights().current += binData;
          });
        }

        // set event weight
        {
          auto& dataEvList{xsec.samplePtr->getDataContainer().getEventList()};
          std::for_each(dataEvList.begin(), dataEvList.end(), [&]( Event& ev_){
            if( iBin != ev_.getIndices().bin ){ return; }
            ev_.getWeights().current = binData;
          });
        }

        // bin volume
        auto& bin = xsec.samplePtr->getBinning().getBinList()[iBin];
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
    GenericToolbox::writeInTFile( GenericToolbox::mkdirTFile(calcXsecDir, "throws"), xsecAtBestFitTree );
  }


  //////////////////////////////////////
  // THROWS LOOP
  /////////////////////////////////////
  LogWarning << std::endl << GenericToolbox::addUpDownBars( "Generating toys..." ) << std::endl;

  std::stringstream ss; ss << LogWarning.getPrefixString() << "Generating " << nToys << " toys...";
  for( int iToy = 0 ; iToy < nToys ; iToy++ ){

    // loading...
    GenericToolbox::displayProgressBar( iToy+1, nToys, ss.str() );

    // Do the throwing:
    propagator.getParametersManager().throwParametersFromGlobalCovariance();
    propagator.propagateParameters();

    if( enableStatThrowInToys ){
      for( auto& xsec : crossSectionDataList ){
        if( enableEventMcThrow ){
          // Take into account the finite amount of event in MC
          xsec.samplePtr->getMcContainer().throwEventMcError();
        }
        // Asimov bin content -> toy data
        xsec.samplePtr->getMcContainer().throwStatError();
      }
    }

    writeBinDataFct();

    // Write the branches
    xsecThrowTree->Fill();
  }


  LogInfo << "Writing throws..." << std::endl;
  GenericToolbox::writeInTFile( GenericToolbox::mkdirTFile(calcXsecDir, "throws"), xsecThrowTree );

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

    for( int iBin = 0 ; iBin < xsec.samplePtr->getMcContainer().getHistogram().nBins ; iBin++ ){
      iBinGlobal++;

      std::string binTitle = xsec.samplePtr->getBinning().getBinList()[iBin].getSummary();
      double binVolume = xsec.samplePtr->getBinning().getBinList()[iBin].getVolume();

      xsec.histogram.SetBinContent( 1+iBin, (*meanValuesVector)[iBinGlobal] );
      xsec.histogram.SetBinError( 1+iBin, TMath::Sqrt( (*globalCovMatrix)[iBinGlobal][iBinGlobal] ) );
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
    xsec.histogram.GetYaxis()->SetTitle( GenericToolbox::Json::fetchValue(xsec.samplePtr->getConfig(), "yAxis", "#delta#sigma").c_str() );

    GenericToolbox::writeInTFile(
        GenericToolbox::mkdirTFile(calcXsecDir, "histograms"),
        &xsec.histogram, GenericToolbox::generateCleanBranchName( xsec.samplePtr->getName() )
    );

  }

  globalCovMatrixHist->GetXaxis()->SetLabelSize(0.02);
  globalCovMatrixHist->GetYaxis()->SetLabelSize(0.02);
  GenericToolbox::writeInTFile(GenericToolbox::mkdirTFile(calcXsecDir, "matrices"), globalCovMatrixHist, "covarianceMatrix");

  globalCorMatrixHist->GetXaxis()->SetLabelSize(0.02);
  globalCorMatrixHist->GetYaxis()->SetLabelSize(0.02);
  globalCorMatrixHist->GetZaxis()->SetRangeUser(-1, 1);
  GenericToolbox::writeInTFile(GenericToolbox::mkdirTFile(calcXsecDir, "matrices"), globalCorMatrixHist, "correlationMatrix");

  // now propagate to the engine for the plot generator
  LogInfo << "Re-normalizing the samples for the plot generator..." << std::endl;
  for( auto& xsec : crossSectionDataList ){
    // this gives the average as the event weights were summed together
    {
      auto &mcEvList{xsec.samplePtr->getMcContainer().getEventList()};
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
    {
      auto &dataEvList{xsec.samplePtr->getDataContainer().getEventList()};
      std::vector<size_t> nEventInBin(xsec.histogram.GetNbinsX(), 0);
      for( size_t iBin = 0 ; iBin < nEventInBin.size() ; iBin++ ){
        nEventInBin[iBin] = std::count_if(dataEvList.begin(), dataEvList.end(), [iBin]( Event &ev_) {
          return ev_.getIndices().bin== iBin;
        });
      }

      std::for_each(dataEvList.begin(), dataEvList.end(), [&]( Event &ev_) {
        ev_.getWeights().current /= nToys;
        ev_.getWeights().current /= double(nEventInBin[ev_.getIndices().bin]);
      });
    }
  }

  LogInfo << "Generating xsec sample plots..." << std::endl;
  // manual trigger to tweak the error bars
  propagator.getPlotGenerator().generateSampleHistograms( GenericToolbox::mkdirTFile(calcXsecDir, "plots/histograms") );

  for( auto& histHolder : propagator.getPlotGenerator().getHistHolderList(0) ){
    if( not histHolder.isData ){ continue; } // only data will print errors

    const CrossSectionData* xsecDataPtr{nullptr};
    for( auto& xsecData : crossSectionDataList ){
      if( xsecData.samplePtr  == histHolder.samplePtr){
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

  propagator.getPlotGenerator().generateCanvas(
      propagator.getPlotGenerator().getHistHolderList(0),
      GenericToolbox::mkdirTFile(calcXsecDir, "plots/canvas")
  );


  LogInfo << "Writing event samples in TTrees..." << std::endl;
  dataSetManager.getTreeWriter().writeSamples(
      GenericToolbox::mkdirTFile(calcXsecDir, "events"),
      dataSetManager.getPropagator()
  );

}
