
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
#include "TLegend.h"
#include "TGraphErrors.h"

#include <string>
#include <vector>


LoggerInit([]{
  Logger::getUserHeader() << "[" << FILENAME << "]";
});

void readBinningFromFile(const char* filename, std::vector<Double_t>& binEdges) ;


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

//    if( clParser.isOptionTriggered("usePreFit") ){
//      sample.setName( sample.getName() + " (pre-fit)" );
//    }

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

  std::vector<double> listOfParametersAtBestFit{}; // this will contain prior values in the pre-fit case!

  if( not clParser.isOptionTriggered("usePreFit") and fitterRootFile != nullptr ){

    // Load post-fit parameters as "prior" so we can reset the weight to this point when throwing toys
    LogWarning << std::endl << GenericToolbox::addUpDownBars("Injecting post-fit parameters...") << std::endl;
    ObjectReader::readObject<TNamed>( fitterRootFile.get(), "FitterEngine/postFit/parState_TNamed", [&](TNamed* parState_){
      propagator.getParametersManager().injectParameterValues( GenericToolbox::Json::readConfigJsonStr( parState_->GetTitle() ) );
      for( auto& parSet : propagator.getParametersManager().getParameterSetsList() ){
        if( not parSet.isEnabled() ){ continue; }
        for( auto& par : parSet.getParameterList() ){
          if( not par.isEnabled() ){ continue; }
          LogInfo << " moving par prior: " << par.getFullTitle() << " from " << par.getPriorValue() << " to " << par.getParameterValue() << std::endl;
          par.setPriorValue( par.getParameterValue() );
          listOfParametersAtBestFit.emplace_back( par.getParameterValue() );
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
  }else{
    for( auto& parSet : propagator.getParametersManager().getParameterSetsList() ){
      if( not parSet.isEnabled() ){ continue; }
      for( auto& par : parSet.getParameterList() ){
        if( not par.isEnabled() ){ continue; }
        listOfParametersAtBestFit.emplace_back( par.getPriorValue() );
      }
    }
  }


  // Creating output file
  std::string outFilePath{};
  if( clParser.isOptionTriggered("outputFile") ){ outFilePath = clParser.getOptionVal<std::string>("outputFile"); }
  else{
    // appendixDict["optionName"] = "Appendix"
    // this list insure all appendices will appear in the same order
    std::vector<std::pair<std::string, std::string>> appendixDict{
        {"configFile", "%s"},
        {"fitterFile", "Fit_%s"},
        {"nToys", "nToys_%s"},
        {"randomSeed", "Seed_%s"},
        {"usePreFit", "PreFit"},
    };

    outFilePath = "xsecCalc_" + GundamUtils::generateFileName(clParser, appendixDict) + ".root";

    std::string outFolder{GenericToolbox::Json::fetchValue<std::string>(xsecConfig, "outputFolder", "./")};
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
      LogThrowIf(not dialCollectionPtr->isBinned(), "Dial collection is not binned.");
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
      leafNameList.emplace_back(Form("bin_%i/D", iBin ));
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
//      LogInfo << xsec.samplePtr->getName() << std::endl;
      xsec.branchBinsData.resetCurrentByteOffset();
//      xsec.samplePtr->getMcContainer().throwStatError();
//      xsec.samplePtr->getMcContainer().throwEventMcError();
      for( int iBin = 0 ; iBin < xsec.samplePtr->getMcContainer().getHistogram().nBins ; iBin++ ){
        double binData{ xsec.samplePtr->getMcContainer().getHistogram().binList[iBin].content };

        LogInfo << iBin <<" Before norm: "<< binData << std::endl;

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
            // get rid of events out of range by setting the weight to 0
              double Pmu = ev_.getVariables().fetchVariable("Pmu").getVarAsDouble();
              double CosThetamu = ev_.getVariables().fetchVariable("CosThetamu").getVarAsDouble();
              if(Pmu <= 0 || Pmu >= 30000 || CosThetamu <= -1 || CosThetamu >= 1){
                ev_.getWeights().current = 0.;
              }
              //if( iBin != ev_.getIndices().bin ){ return; }
              //ev_.getWeights().current = ;
          });
        }

        // set event weight
        {
          auto& dataEvList{xsec.samplePtr->getDataContainer().getEventList()};
          std::for_each(dataEvList.begin(), dataEvList.end(), [&]( Event& ev_){
            if( iBin != ev_.getIndices().bin ){ return; }
            //ev_.getWeights().current = binData;
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
        LogInfo <<  iBin << " After bin volume norm: "<< binData << " bin volume: "<< binVolume << std::endl;
      }
    }
  });



  {
    LogWarning << "Calculating weight at best-fit" << std::endl;
    // the following lines has to be avoided otherwise the parameter values are set back to the prior for the flux paramters.
    // I don't know why this happens since the parameter priors should be moved to the best fit values already....
    //for( auto& parSet : propagator.getParametersManager().getParameterSetsList() ){ parSet.moveParametersToPrior(); }
    //  Print parameters at best-fit
    int iPar = 0;
    for( auto& parSet : propagator.getParametersManager().getParameterSetsList() ){
      if (not parSet.isEnabled()) continue;
      LogInfo << "Best-fit parameters for " << parSet.getName() << ": " << std::endl;
      for( auto& par : parSet.getParameterList() ){
        if (not par.isEnabled()) continue;
        par.setParameterValue(listOfParametersAtBestFit[iPar]);
        LogInfo << par.getTitle() << " = " << par.getParameterValue() << std::endl;
        iPar++;
      }
    }
    propagator.propagateParameters();
    // Set weight to 0 for events out of range
    for( auto& xsec : crossSectionDataList ) {
      LogInfo << "Events at best fit point for sample " << xsec.samplePtr->getName() << " = " <<
              xsec.samplePtr->getMcContainer().getEventList().size()
              << " binned events: " << xsec.samplePtr->getMcContainer().getNbBinnedEvents()
              << std::endl;
      int maxEventPrint = 60;
      int iEvent = 0;
      int inRange = 0;
      int outOfRange = 0;
      for (auto &ev: xsec.samplePtr->getMcContainer().getEventList()) {
        if(iEvent < maxEventPrint) {
          LogInfo << "Event: " << iEvent << " going into bin " << ev.getIndices().bin
                  << " | EventWeight: " << ev.getEventWeight()
                  << " | Pmu: " << ev.getVariables().fetchVariable("Pmu").getVarAsDouble()
                  << " | CosThetamu: " << ev.getVariables().fetchVariable("CosThetamu").getVarAsDouble()
                  << " | summary: " << ev.getSummary()
                  << std::endl;
        }
        double Pmu = ev.getVariables().fetchVariable("Pmu").getVarAsDouble();
        double CosThetamu = ev.getVariables().fetchVariable("CosThetamu").getVarAsDouble();
        if(Pmu <= 0 || Pmu >= 30000 || CosThetamu <= -1 || CosThetamu >= 1){
//          LogInfo << "Pmu or CosThetamu out of range: Pmu=" << Pmu << " CosThetamu=" << CosThetamu << std::endl;
// need to remove these events!!!!
          ev.getWeights().current = 0.;
          //xsec.samplePtr->getMcContainer().getEventList().erase(xsec.samplePtr->getMcContainer().getEventList().begin() + iEvent);
          outOfRange++;
        }else{
          inRange++;
        }
        iEvent++;
      }
      LogInfo << "Events in range: " << inRange << " out of range: " << outOfRange << std::endl;
      auto eventListAtBestFit = xsec.samplePtr->getMcContainer().getEventList();
      xsec.samplePtr->getMcContainer().refillHistogram();
    }


    writeBinDataFct();
    xsecAtBestFitTree->Fill();
    GenericToolbox::writeInTFile( GenericToolbox::mkdirTFile(calcXsecDir, "throws"), xsecAtBestFitTree );
  }

  std::vector<double> vectorOfParametersThrow_Mean((*propagator.getParametersManager().getGlobalCovarianceMatrix()).GetNcols(), 0);
  std::vector<double> vectorOfParametersThrow_RMS((*propagator.getParametersManager().getGlobalCovarianceMatrix()).GetNcols(), 0);

  int i_aux = 0;
  for( auto& parSet : propagator.getParametersManager().getParameterSetsList() ){
    if( not parSet.isEnabled() ) continue;
    for( auto& par : parSet.getParameterList() ){
      if( not par.isEnabled() ) continue;
      if(i_aux>706 || i_aux<10){
        LogInfo << "i = "<<i_aux<<" par = "<<par.getParameterValue()<<std::endl;
      }
      LogInfo << "Parameter "<<i_aux<<" mean = "<<par.getPriorValue()<<" RMS = "<<TMath::Sqrt((*propagator.getParametersManager().getGlobalCovarianceMatrix())[i_aux][i_aux])<<std::endl;
      i_aux++;
    }
  }
  if (i_aux!=(*propagator.getParametersManager().getGlobalCovarianceMatrix()).GetNcols()){
    LogInfo << "ERROR: i = "<<i_aux<<" != "<<(*propagator.getParametersManager().getGlobalCovarianceMatrix()).GetNcols()<<std::endl;
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
    int  i = 0;
    for( auto& parSet : propagator.getParametersManager().getParameterSetsList() ){
      if( not parSet.isEnabled() ) continue;
      for( auto& par : parSet.getParameterList() ){
        if( not par.isEnabled() ) continue;
        vectorOfParametersThrow_Mean[i] += par.getParameterValue();
        vectorOfParametersThrow_RMS[i] += par.getParameterValue()*par.getParameterValue();
        i++;
//        if(i>706){
//          LogInfo << "i = "<<i<< "par = "<<par.getParameterValue()<<std::endl;
//        }
      }
    }
//    LogInfo<<"\n";

    // print info about the throw
    if( iToy == 0 ){
      LogInfo << "First throw: " << std::endl;
    }

    // disable stats throw
//    if( enableStatThrowInToys ){
//      for( auto& xsec : crossSectionDataList ){
//        if( enableEventMcThrow ){
//          // Take into account the finite amount of event in MC
//          xsec.samplePtr->getMcContainer().throwEventMcError();
//        }
//        // Asimov bin content -> toy data
//        xsec.samplePtr->getMcContainer().throwStatError();
//      }
//    }

    writeBinDataFct();




    // Write the branches
    xsecThrowTree->Fill();
  }

  // compute mean and rms of parameter throws
  for(int i=0;i<vectorOfParametersThrow_Mean.size();i++){
    vectorOfParametersThrow_Mean[i] /= nToys;
    vectorOfParametersThrow_RMS[i] = TMath::Sqrt(vectorOfParametersThrow_RMS[i]/nToys - vectorOfParametersThrow_Mean[i]*vectorOfParametersThrow_Mean[i]);
    LogInfo << "Parameter "<<i<<" mean = "<<vectorOfParametersThrow_Mean[i]<<" RMS = "<<vectorOfParametersThrow_RMS[i]<<std::endl;
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
    LogInfo << "Sample: " << xsec.samplePtr->getName() << std::endl;
    for( int iBin = 0 ; iBin < xsec.samplePtr->getMcContainer().getHistogram().nBins ; iBin++ ){
      iBinGlobal++;

      std::string binTitle = xsec.samplePtr->getBinning().getBinList()[iBin].getSummary();
      double binVolume = xsec.samplePtr->getBinning().getBinList()[iBin].getVolume();

      xsec.histogram.SetBinContent( 1+iBin, (*meanValuesVector)[iBinGlobal] );
      xsec.histogram.SetBinError( 1+iBin, TMath::Sqrt( (*globalCovMatrix)[iBinGlobal][iBinGlobal] ) );
      LogInfo<<"Bin "<<iBin<<" mean = "<<(*meanValuesVector)[iBinGlobal]<<" RMS = "<<TMath::Sqrt( (*globalCovMatrix)[iBinGlobal][iBinGlobal] )<<std::endl;
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

//  // now propagate to the engine for the plot generator
//  LogInfo << "Re-normalizing the samples for the plot generator..." << std::endl;
//  for( auto& xsec : crossSectionDataList ){
//    // this gives the average as the event weights were summed together
//    {
//      auto &mcEvList{xsec.samplePtr->getMcContainer().getEventList()};
//      std::vector<size_t> nEventInBin(xsec.histogram.GetNbinsX(), 0);
//      for( size_t iBin = 0 ; iBin < nEventInBin.size() ; iBin++ ){
//        nEventInBin[iBin] = std::count_if(mcEvList.begin(), mcEvList.end(), [iBin]( Event &ev_) {
//          return ev_.getIndices().bin == iBin;
//        });
//      }
//
//      std::for_each(mcEvList.begin(), mcEvList.end(), [&]( Event &ev_) {
//        //ev_.getWeights().current /= nToys;
//        ev_.getWeights().current /= double(nEventInBin[ev_.getIndices().bin]);
//      });
//    }
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
//        //ev_.getWeights().current /= nToys;
//        ev_.getWeights().current /= double(nEventInBin[ev_.getIndices().bin]);
//      });
//    }
//  }

//  LogInfo << "Generating xsec sample plots..." << std::endl;
//  // manual trigger to tweak the error bars
//  propagator.getPlotGenerator().generateSampleHistograms( GenericToolbox::mkdirTFile(calcXsecDir, "plots/histograms") );
//
//  for( auto& histHolder : propagator.getPlotGenerator().getHistHolderList(0) ){
//    if( not histHolder.isData ){ continue; } // only data will print errors
//
//    const CrossSectionData* xsecDataPtr{nullptr};
//    for( auto& xsecData : crossSectionDataList ){
//      if( xsecData.samplePtr  == histHolder.samplePtr){
//        xsecDataPtr = &xsecData;
//        break;
//      }
//    }
//    LogThrowIf(xsecDataPtr==nullptr, "corresponding data not found");
//
//    // alright, now rescale error bars
//    for( int iBin = 0 ; iBin < histHolder.histPtr->GetNbinsX() ; iBin++ ){
//      // relative error should be set
//      histHolder.histPtr->SetBinError(
//          1+iBin,
//          histHolder.histPtr->GetBinContent(1+iBin)
//          * xsecDataPtr->histogram.GetBinError(1+iBin)
//          / xsecDataPtr->histogram.GetBinContent(1+iBin)
//      );
//    }
//  }

  //// SECTION TO MAKE THE HISTOGRAMS FROM THE DATA TTREE FOR THE CLOSURE TEST

  // here, fetch data from fitter file, to draw on top of MC predictions for each specific sample
  LogInfo << "Fetching data from fitter file..." << std::endl;
  int i{0};
  // Access the fitter output file
  LogInfo<<"Opening fitter output file: "<<fitterFile<<std::endl;
  std::unique_ptr<TFile> fitterFilePtr{ TFile::Open(fitterFile.c_str()) };

  // Define a closureVariable struct to store the data needed to generate the histograms
  struct closureVariable {
      Sample* samplePtr{nullptr};
      std::string varToPlot{};
      std::string variableFormula{};
      std::string binningFile{};
      std::vector<double> binErrors{}; // fetch this from the covariance matrix
      TH1D* histogram{}; // histogram generated from the fitted toy data
      TH1D* mcHistogram{}; // histogram generated from the MC at the best fit point
      TH1D* meanValueHistogram{}; // histogram generated from the mean value of the systematic throws
      TH1D* bestFitHistogram{}; // histogram generated from the best fit point
      std::vector<TH1D*> mcHistogramReactionCodes{};
      bool rescaleAsBinWidth{false};
      TCanvas* canvas{};
  };

  /// PSEUDO CODE:
  //
  // for(sample : samples){
  //    for(varToPlot : varToPlotVector){
  //    closureVariable closureVar;
  //    closureVar.samplePtr = sample;
  //    closureVar: generate the histogram using the binning file defined in the plotGenerator
  //    closureVar.varToPlot = varToPlot;
  //    closureVar: connect varToPlot to a leafvar
  //    use the leafvar definition to fill the histogram properly
  //    }
  // }



  if( propagator.getPlotGenerator().getConfig().at("histogramsDefinition").is_array() ){
    std::cout << "\"Closure\" variable to plot: " << propagator.getPlotGenerator().getConfig().at("histogramsDefinition").size() <<std::endl;
  }


  std::string prePostFit ="";
  if( clParser.isOptionTriggered("usePreFit") ){
    prePostFit = "preFit";
  } else {
    prePostFit = "postFit";
  }
  // reset aprams to best fit (or prior in the prefit case)
  int iParam=0;
  for( auto& parSet : propagator.getParametersManager().getParameterSetsList() ){
    if( not parSet.isEnabled() ) continue;
    for( auto& par : parSet.getParameterList() ){
      if( not par.isEnabled() ) continue;
      par.setParameterValue(listOfParametersAtBestFit[iParam]);
      iParam++;
    }
  }
  propagator.propagateParameters();
  writeBinDataFct();


  std::vector<closureVariable> closureVarList;
  std::vector<std::string> varToPlotVector = propagator.getPlotGenerator().fetchListOfVarToPlot();
  size_t nHist = propagator.getPlotGenerator().getConfig().at("histogramsDefinition").size();
  iBinGlobal = -1;
  for( auto& sample : propagator.getSampleSet().getSampleList() ) {
    // roll back the sample to the best fit point


    LogInfo << "Fetching data histogram for sample: " << sample.getName() << std::endl;
    TTree *dataTree = (TTree *) fitterFilePtr->Get(
            ("FitterEngine/" + prePostFit + "/events/" + sample.getName() + "/Data_TTree").std::string::c_str());
    TTree *mcTree = (TTree *) fitterFilePtr->Get(
            ("FitterEngine/" + prePostFit + "/events/" + sample.getName() + "/MC_TTree").std::string::c_str());
    LogErrorIf(dataTree == nullptr) << "Could not find data tree for sample: " << sample.getName() << std::endl;
//     Save data tree in the output file (instead of the data tree that is just a copy of the mc tree)
//    GenericToolbox::writeInTFile(
//            GenericToolbox::mkdirTFile(calcXsecDir, "events"),
//            *dataTree,
//            GenericToolbox::generateCleanBranchName( (sample.getName()+"/Data_TTree").c_str() )
//    );
    LogErrorIf(mcTree == nullptr) << "Could not find mc tree for sample: " << sample.getName() << std::endl;



    nHist = 1; // otherwise the code does not work :)
    for (size_t iHist = 0; iHist < nHist; iHist++) { // this loop is over the variables to plot, as defined in the config file
      closureVariable closureVar;
      closureVar.samplePtr = &sample;
      closureVar.varToPlot = GenericToolbox::Json::fetchValue<std::string>(propagator.getPlotGenerator().getConfig().at("histogramsDefinition")[iHist], "varToPlot");
      bool useSampleBinning = GenericToolbox::Json::fetchValue<bool>(propagator.getPlotGenerator().getConfig().at("histogramsDefinition")[iHist], "useSampleBinning");
      if (useSampleBinning) {
        closureVar.binningFile = sample.getBinningFilePath();
      }else {
        closureVar.binningFile = GenericToolbox::Json::fetchValue<std::string>(propagator.getPlotGenerator().getConfig().at("histogramsDefinition")[iHist], "binningFile");
      }
      // debug: print out summary of first 20 events in the mcTree
      int maxLogEvents = 100;
      int eventLogCounter = 0;
      int iEventXsec = 0;
      int iEventFiter = 0;
      for(int iEvent=0; iEvent<mcTree->GetEntries(); iEvent++){
        mcTree->GetEntry(iEvent + iEventFiter);
        if(iEvent<60) {
          LogInfo << "Event: " << iEvent
                  << " | nominal weight: " << mcTree->GetLeaf("Event.nominalWeight")->GetValue()
                  << " | EventWeight: " << mcTree->GetLeaf("Event.eventWeight")->GetValue()
                  << " | Pmu: " << mcTree->GetLeaf("Leaves.Pmu")->GetValue()
                  << " | CosThetamu: " << mcTree->GetLeaf("Leaves.CosThetamu")->GetValue()
                  << std::endl;
          LogInfo << " From xsec event list: " << sample.getMcContainer().getEventList().at(iEvent).getSummary()
                  << std::endl;
        }
        if (iEvent + iEventXsec >= sample.getMcContainer().getEventList().size()) {
          break;
        }
          double weightFromXsec = sample.getMcContainer().getEventList().at(iEvent + iEventXsec).getEventWeight();
          double weightFromFitter = mcTree->GetLeaf("Event.eventWeight")->GetValue();
          double Pmu_xsec = sample.getMcContainer().getEventList().at(iEvent + iEventXsec).getVariables().fetchVariable("Pmu").getVarAsDouble();
          double CosThetamu_xsec = sample.getMcContainer().getEventList().at(iEvent + iEventXsec).getVariables().fetchVariable("CosThetamu").getVarAsDouble();
          // so that if the weight is 0, the event is simply skipped
          while ( Pmu_xsec <= 0 || Pmu_xsec >= 30000 || CosThetamu_xsec <= -1 || CosThetamu_xsec >= 1  || weightFromXsec == 0. ) {
            iEventXsec++;
            if (iEvent + iEventXsec >= sample.getMcContainer().getEventList().size()) {
              break;
            }
            Pmu_xsec = sample.getMcContainer().getEventList().at(iEvent + iEventXsec).getVariables().fetchVariable("Pmu").getVarAsDouble();
            CosThetamu_xsec = sample.getMcContainer().getEventList().at(iEvent + iEventXsec).getVariables().fetchVariable("CosThetamu").getVarAsDouble();
            weightFromXsec = sample.getMcContainer().getEventList().at(iEvent + iEventXsec).getEventWeight();
          }
          double Pmu_fitter = mcTree->GetLeaf("Leaves.Pmu")->GetValue();
          double CosThetamu_fitter = mcTree->GetLeaf("Leaves.CosThetamu")->GetValue();
          while ( Pmu_fitter <= 0 || Pmu_fitter >= 30000 || CosThetamu_fitter <= -1 || CosThetamu_fitter >= 1  ||  weightFromFitter == 0. ) {
            iEventFiter++;
            if (iEvent + iEventFiter >= mcTree->GetEntries()) {
              break;
            }
            mcTree->GetEntry(iEvent + iEventFiter);
            Pmu_fitter = mcTree->GetLeaf("Leaves.Pmu")->GetValue();
            CosThetamu_fitter = mcTree->GetLeaf("Leaves.CosThetamu")->GetValue();
            weightFromFitter = mcTree->GetLeaf("Event.eventWeight")->GetValue();
          }

          if ( abs(weightFromXsec - weightFromFitter) > 1e-4 && eventLogCounter < maxLogEvents  ) {
            if ( abs(Pmu_fitter-Pmu_xsec)<1.e-6 ) {
              std::cout << "Event " << iEvent << " has different weights: " << weightFromXsec << " vs "
                        << weightFromFitter << std::endl;
            }else{
             // std::cout << "    event shift " <<iEvent<< std::endl;
            }
          eventLogCounter++;
          }
      }
      closureVar.rescaleAsBinWidth = GenericToolbox::Json::fetchValue<bool>(propagator.getPlotGenerator().getConfig().at("histogramsDefinition")[iHist], "rescaleAsBinWidth");
      // Generate the histogram (using the binning file defined in the plotGenerator)
      std::vector<double> binEdges;
      readBinningFromFile(closureVar.binningFile.c_str(), binEdges);
      closureVar.histogram = new TH1D(
              (closureVar.samplePtr->getName() + "_" + closureVar.varToPlot).c_str(),
              (closureVar.samplePtr->getName() + " data ;" + closureVar.varToPlot).c_str(),
              binEdges.size() - 1,
              &binEdges[0]
      );
      closureVar.mcHistogram = new TH1D(
              (closureVar.samplePtr->getName() + "_mc_" + closureVar.varToPlot).c_str(),
              (closureVar.samplePtr->getName() + " mc ;" + closureVar.varToPlot).c_str(),
              binEdges.size() - 1,
              &binEdges[0]
      );
      closureVar.meanValueHistogram = new TH1D(
              (closureVar.samplePtr->getName() + "_mean_" + closureVar.varToPlot).c_str(),
              (closureVar.samplePtr->getName() + " mean of throws ;" + closureVar.varToPlot).c_str(),
              binEdges.size() - 1,
              &binEdges[0]
      );
      closureVar.bestFitHistogram = new TH1D(
              (closureVar.samplePtr->getName() + "_bestFit_" + closureVar.varToPlot).c_str(),
              (closureVar.samplePtr->getName() + " best fit ;" + closureVar.varToPlot).c_str(),
              binEdges.size() - 1,
              &binEdges[0]
      );
      // load the formula
      for (int i = 0; i < propagator.getConfig()["dataSetList"][0]["mc"].at("overrideLeafDict").size(); i++) {
        std::string eventVar = propagator.getConfig()["dataSetList"][0]["mc"].at("overrideLeafDict")[i].at("eventVar");
        if (eventVar == closureVar.varToPlot) {
          closureVar.variableFormula = propagator.getConfig()["dataSetList"][0]["mc"].at("overrideLeafDict")[i].at(
                  "leafVar");
          break;
        }
      }
      // Fill the mean value histogram
      std::vector<double> meanValues(closureVar.meanValueHistogram->GetNbinsX(),0);
      LogInfo << "Entries of throws tree: " << xsecThrowTree->GetEntries() << std::endl;
      for (int iToy = 0; iToy < xsecThrowTree->GetEntries(); iToy++) {
        xsecThrowTree->GetEntry(iToy);
        // check that the number of leaves in the ttree is the same as the number of bins in the histogram
        std::string branchName = GenericToolbox::generateCleanBranchName(sample.getName()).c_str();
        TBranch *branch = xsecThrowTree->GetBranch(branchName.c_str());
        int leavesInBranch = 0;
        if (branch == nullptr) {
          LogError << "Could not find branch for sample: "
                   << GenericToolbox::generateCleanBranchName(sample.getName()).c_str() << std::endl;
          return 1;
        } else {
          leavesInBranch = xsecThrowTree->GetBranch(
                  GenericToolbox::generateCleanBranchName(sample.getName()).c_str())->GetListOfLeaves()->GetEntries();
//          LogInfo << "The branch " << GenericToolbox::generateCleanBranchName(sample.getName()).c_str() << " has "
//                  << leavesInBranch << " leaves." << std::endl;
//          // print list of leaves:
//          branch->GetListOfLeaves()->Print();
        }
        int binsInHistogram = closureVar.meanValueHistogram->GetNbinsX();
        if (leavesInBranch != binsInHistogram) {
          LogError << "Number of leaves in the tree (" << leavesInBranch
                   << ") is different from the number of bins in the histogram (" << binsInHistogram << ")."
                   << std::endl;
          return 1;
        } else {
          for (int iBin = 0; iBin < binsInHistogram; iBin++) {
            std::string leafName = Form("bin_%i", iBin);
            TLeaf* leaf = (TLeaf*)branch->GetLeaf(leafName.c_str());
            if (leaf == nullptr) {
              LogError << "Could not find leaf named " << leafName << " in branch " << branchName << std::endl;
              return 1;
            } else {
              double binContent = leaf->GetValue();
              meanValues.at(iBin) += binContent;
            }
          }
        }
      }
      for (int iBin = 0; iBin < closureVar.meanValueHistogram->GetNbinsX(); iBin++) {
        meanValues.at(iBin) /= xsecThrowTree->GetEntries();
      }
      for(int iBin=0;iBin<closureVar.meanValueHistogram->GetNbinsX();iBin++){
        closureVar.meanValueHistogram->SetBinContent(iBin+1, meanValues[iBin]);
      }
      // fill the best fit histogram
      xsecAtBestFitTree->GetEntry(0);
      // check that the number of leaves in the ttree is the same as the number of bins in the histogram
      std::string branchName = GenericToolbox::generateCleanBranchName(sample.getName()).c_str();
      TBranch *branch = xsecAtBestFitTree->GetBranch(branchName.c_str());
      int leavesInBranch = 0;
      if (branch == nullptr) {
        LogError << "Could not find branch for sample: "
                 << GenericToolbox::generateCleanBranchName(sample.getName()).c_str() << std::endl;
        return 1;
      } else {
        leavesInBranch = xsecAtBestFitTree->GetBranch(
                GenericToolbox::generateCleanBranchName(sample.getName()).c_str())->GetListOfLeaves()->GetEntries();
//          LogInfo << "The branch " << GenericToolbox::generateCleanBranchName(sample.getName()).c_str() << " has "
//                  << leavesInBranch << " leaves." << std::endl;
//          // print list of leaves:
//          branch->GetListOfLeaves()->Print();
      }
      int binsInHistogram = closureVar.meanValueHistogram->GetNbinsX();
      if (leavesInBranch != binsInHistogram) {
        LogError << "Number of leaves in the tree (" << leavesInBranch
                 << ") is different from the number of bins in the histogram (" << binsInHistogram << ")."
                 << std::endl;
        return 1;
      } else {
        for (int iBin = 0; iBin < binsInHistogram; iBin++) {
          std::string leafName = Form("bin_%i", iBin);
          TLeaf* leaf = (TLeaf*)branch->GetLeaf(leafName.c_str());
          if (leaf == nullptr) {
            LogError << "Could not find leaf named " << leafName << " in branch " << branchName << std::endl;
            return 1;
          } else {
            double binContent = leaf->GetValue();
            closureVar.bestFitHistogram->SetBinContent(iBin+1, binContent);
          }
        }
      }
      // LogInfo<< "    Variable formula: " << closureVar.variableFormula << std::endl; // a bit verbose, do not print this
      LogErrorIf(closureVar.variableFormula.empty()) << "Could not find leafVar for varToPlot: " << closureVar.varToPlot
                                                     << std::endl;
      // Fill the histogram
      closureVar.canvas = new TCanvas(("canvas_" + closureVar.varToPlot + "_" +
                                       GenericToolbox::generateCleanBranchName(sample.getName())).c_str(),
                                      ("canvas_" + closureVar.varToPlot + "_" + sample.getName()).c_str(), 800, 1600);
      closureVar.canvas->Divide(1,2);
      closureVar.canvas->cd(1);
      dataTree->Draw(("(" + closureVar.variableFormula + ")>>" + closureVar.histogram->GetName()).c_str(),
                     "Event.eventWeight*(Leaves.Pmu > 0 && Leaves.Pmu < 30000 && Leaves.CosThetamu > -1 && Leaves.CosThetamu < 1)", "goff");
      mcTree->Draw(("(" + closureVar.variableFormula + ")>>" + closureVar.mcHistogram->GetName()).c_str(),
                   "Event.eventWeight*(Leaves.Pmu > 0 && Leaves.Pmu < 30000 && Leaves.CosThetamu > -1 && Leaves.CosThetamu < 1)", "goff");
      // manually set the bin errors from the covariance matrix
      LogInfo << "Variable: " << closureVar.varToPlot << " | Sample: " << closureVar.samplePtr->getName()
              << " | Binning file: " << closureVar.binningFile << std::endl;

      // the bin errors are the square root of the diagonal of the covariance matrix,
      // but the covariance matrix stores all samples in sequence, so we need to match the correct sample
      // to the correct covariance matrix row
      for(int iBinLocal = 0; iBinLocal < closureVar.histogram->GetNbinsX(); iBinLocal++){
        closureVar.mcHistogram->SetBinError(iBinLocal+1, TMath::Sqrt( (*globalCovMatrix)[iBinGlobal+1][iBinGlobal+1])  );
        closureVar.binErrors.push_back( TMath::Sqrt( (*globalCovMatrix)[iBinGlobal+1][iBinGlobal+1]) );
        iBinGlobal++;
        LogInfo << iBinLocal << iBinGlobal << " | Bin error: " << closureVar.mcHistogram->GetBinError(iBinLocal+1) << std::endl;
      }

        LogInfo << "Before rescaling" << std::endl;
        for (int iBin = 0; iBin < closureVar.histogram->GetNbinsX(); iBin++) {
          // print the bin content and error of histogram and mcHistogram (before rescaling)
          LogInfo << "Bin " << iBin << " | Data: " << closureVar.histogram->GetBinContent(iBin + 1) << " +/- "
                  << closureVar.histogram->GetBinError(iBin + 1) << " | MC: " << closureVar.mcHistogram->GetBinContent(iBin + 1) << " +/- "
                  << closureVar.mcHistogram->GetBinError(iBin + 1) << std::endl;
          // rescale to bin width
          if (closureVar.rescaleAsBinWidth) {
            double binWidth = closureVar.mcHistogram->GetBinWidth(iBin + 1);
            if (binWidth == 0) {
              LogError << "Bin " << iBin << " has a width of 0. Skipping rescaling." << std::endl;
            } else {
              closureVar.histogram->SetBinContent(iBin + 1,
                                                  closureVar.histogram->GetBinContent(iBin + 1) / binWidth);
              closureVar.mcHistogram->SetBinContent(iBin + 1,
                                                    closureVar.mcHistogram->GetBinContent(iBin + 1) / binWidth);
              // No need to normalize to bin width the meanValue and bestFit histograms, that is already done
              // The same arguments is valid for the errors on mcHistogram, that are taken from the
              // width-normalized throws-generated histograms. This is why the following lines are commented.
              // DO NOT UNCOMMENT
              //              closureVar.mcHistogram->SetBinError(iBin + 1,
              //                                    closureVar.mcHistogram->GetBinError(iBin + 1) / binWidth );
            }
          }
        }
        // another loop just to print the bin contents and errors after rescaling
        LogInfo << "After rescaling: bin contents" << std::endl;
        for (int iBin = 0; iBin < closureVar.histogram->GetNbinsX(); iBin++) {
          LogInfo << "Bin " << iBin << " | Data: " << closureVar.histogram->GetBinContent(iBin + 1)
                  << " | MC: " << closureVar.mcHistogram->GetBinContent(iBin + 1)
                  << " | Mean: " << closureVar.meanValueHistogram->GetBinContent(iBin + 1)
                  << " | Best fit: " << closureVar.bestFitHistogram->GetBinContent(iBin + 1)
                  << " | BF/MC: " << closureVar.bestFitHistogram->GetBinContent(iBin + 1) / closureVar.mcHistogram->GetBinContent(iBin + 1)
                  << " | bin width mc: " << closureVar.mcHistogram->GetBinWidth(iBin + 1)
                  << " | bin width bf: " << closureVar.bestFitHistogram->GetBinWidth(iBin + 1) << std::endl;

        }
        LogInfo << "After rescaling: bin errors" << std::endl;
        for (int iBin = 0; iBin < closureVar.histogram->GetNbinsX(); iBin++) {
          LogInfo << "Bin " << iBin << " | Data: " << closureVar.histogram->GetBinError(iBin + 1)
                                    << " | MC: " << closureVar.mcHistogram->GetBinError(iBin + 1)
                                    << " | Mean: " << closureVar.meanValueHistogram->GetBinError(iBin + 1)
                                    << " | Best fit: " << closureVar.bestFitHistogram->GetBinError(iBin + 1) << std::endl;
        }

        // make histo with the relative difference between data and MC, overlaid
        // with the relative uncertainty on MC
        std::string sampleNameDiff = GenericToolbox::generateCleanBranchName( "hRelativeDiff" + closureVar.varToPlot + "_" + sample.getName()  );
        TH1F* hRelativeDiff = (TH1F*)closureVar.histogram->Clone( sampleNameDiff.c_str() );
        TH1F* hMcErrors = (TH1F*)closureVar.histogram->Clone( (sampleNameDiff+"_err").c_str() );
        for (int iBin = 0; iBin < closureVar.histogram->GetNbinsX(); iBin++) {
          double data = closureVar.histogram->GetBinContent(iBin + 1);
          double mc = closureVar.mcHistogram->GetBinContent(iBin + 1);
          double mcError = closureVar.mcHistogram->GetBinError(iBin + 1);
          double relativeDiff;
          double relativeError;
          if(mc!=0){
            relativeDiff = (data - mc) / mc;
            relativeError = mcError / mc;
          }else if(data!=0){
            relativeDiff = (data - mc) / data; // -> this will always be 1.
            relativeError = 0;
          }else{
            relativeDiff = 0;
            relativeError = 0;
          }
          hRelativeDiff->SetBinContent(iBin + 1, relativeDiff);
          hRelativeDiff->SetBinError(iBin + 1, 0);
          hMcErrors->SetBinContent(iBin + 1, 0);
          hMcErrors->SetBinError(iBin + 1, relativeError);
          std::cout << iBin << "  " << hRelativeDiff->GetBinCenter(iBin+1) << " " << hRelativeDiff->GetBinContent(iBin + 1) << "  " << hMcErrors->GetBinContent(iBin + 1) << "  " << hMcErrors->GetBinError(iBin + 1)
          << std::endl;
        }
      closureVar.canvas->cd(2);
      hRelativeDiff->Draw("hist");
      hMcErrors->Draw("e2 same");
      hRelativeDiff->SetMarkerStyle(kFullDotLarge);
      hRelativeDiff->SetMarkerColor(kRed);
      hRelativeDiff->SetMarkerSize(1);
      hRelativeDiff->SetLineWidth(2);
      hRelativeDiff->SetLineColor(kRed);
      hMcErrors->SetFillColor(kBlue);
      hMcErrors->SetFillStyle(3001);
      hMcErrors->GetYaxis()->SetRangeUser(-1,1);
      hRelativeDiff->GetYaxis()->SetRangeUser(-1,1);
      hRelativeDiff->GetYaxis()->SetTitle("(data - mc) / mc");
      hRelativeDiff->GetXaxis()->SetTitle(closureVar.varToPlot.c_str());
      hRelativeDiff->SetTitle("Relative difference between data and MC");

      // Only look at the shape
//      closureVar.histogram->Scale(1.0/closureVar.histogram->Integral());
//      closureVar.mcHistogram->Scale(1.0/closureVar.mcHistogram->Integral());
//      closureVar.meanValueHistogram->Scale(1.0/closureVar.meanValueHistogram->Integral());
//      closureVar.bestFitHistogram->Scale(1.0/closureVar.bestFitHistogram->Integral());


      // cosmetics + draw
      closureVar.canvas->cd(1);
      closureVar.histogram->SetMarkerStyle(kFullDotLarge);
      closureVar.histogram->SetMarkerColor(kRed);
      closureVar.histogram->SetMarkerSize(1);
      closureVar.histogram->SetLineWidth(2);
      closureVar.histogram->SetLineColor(kRed);
      closureVar.histogram->SetDrawOption("hist");
      closureVar.histogram->SetTitle( (closureVar.varToPlot+" for "+closureVar.samplePtr->getName()  ).c_str() );
      closureVar.histogram->GetXaxis()->SetTitle(closureVar.varToPlot.c_str());
      closureVar.mcHistogram->DrawCopy("hist");
      closureVar.mcHistogram->SetFillColor(kBlue);
      closureVar.mcHistogram->SetFillStyle(3001);
      closureVar.mcHistogram->Draw("e2 same");
      closureVar.histogram->Draw("hist same");
      closureVar.meanValueHistogram->SetMarkerStyle(kFullDotLarge);
      closureVar.meanValueHistogram->SetMarkerColor(kBlack);
      closureVar.meanValueHistogram->SetLineColor(kBlack);
      closureVar.meanValueHistogram->Draw("hist same");
      closureVar.bestFitHistogram->SetMarkerStyle(kFullDotLarge);
      closureVar.bestFitHistogram->SetMarkerColor(kMagenta);
      closureVar.bestFitHistogram->SetLineColor(kMagenta);
      closureVar.bestFitHistogram->Draw("hist same");
      // legend
      TLegend *legend = new TLegend(0.7, 0.7, 0.9, 0.9);
      legend->AddEntry(closureVar.histogram, "Data", "l");
      legend->AddEntry(closureVar.mcHistogram, "MC", "l");
      legend->AddEntry(closureVar.meanValueHistogram, "Mean of throws", "l");
      legend->AddEntry(closureVar.bestFitHistogram, "Best fit", "l");
      legend->Draw();




      // save
      GenericToolbox::writeInTFile(
              GenericToolbox::mkdirTFile(calcXsecDir, "plots/closure_histograms"),
              closureVar.histogram,
              GenericToolbox::generateCleanBranchName( "data_" + closureVar.varToPlot + "_" + sample.getName()  )
      );
      GenericToolbox::writeInTFile(
              GenericToolbox::mkdirTFile(calcXsecDir, "plots/closure_histograms"),
              closureVar.mcHistogram,
              GenericToolbox::generateCleanBranchName( "mc_" + closureVar.varToPlot + "_" + sample.getName()  )
      );
      GenericToolbox::writeInTFile(
              GenericToolbox::mkdirTFile(calcXsecDir, "plots/closure_histograms"),
              closureVar.canvas,
              GenericToolbox::generateCleanBranchName( "overlay_" + closureVar.varToPlot + "_" + sample.getName()  )
      );


      // Now generate, for each sample, a canvas with all the histograms: stack of all the MC with different reaction
      // codes generated before, and the data histogram just generated from the fitter output file

      closureVarList.push_back(closureVar);
    }

  } // end of loop over samples



//  // print info about the histos in histHolder
//  for(auto histHolder : propagator.getPlotGenerator().getHistHolderList()){
//    LogInfo << "HistHolder: " << histHolder.samplePtr->getName() << " | " << histHolder.histPtr->GetName() << std::endl;
//    LogInfo << "  isData? " << histHolder.isData << std::endl;
//  }
//  // Generate a temporary histHolderList that includes also the closure variable histograms
//  std::vector<HistHolder> myHistHolderList = propagator.getPlotGenerator().getHistHolderList();
//  for(auto closureVar : closureVarList){
//    HistHolder histHolder;
//    // copy a random histHolder from the plot generator
//    histHolder = myHistHolderList.at(0);
//    std::shared_ptr<TH1D> histPtr(closureVar.histogram);
//    histHolder.histPtr = histPtr;
//    histHolder.isData = true;
//    myHistHolderList.push_back(histHolder);
//  }
//  // print info about the histos in histHolder
// LogInfo << "Printing histHolderList" << std::endl;
//  for(auto histHolder : myHistHolderList){
//    LogInfo << "HistHolder: " << histHolder.samplePtr->getName() << " | " << histHolder.histPtr->GetName() << std::endl;
//    LogInfo << "  isData? " << histHolder.isData << std::endl;
//  }

//  LogInfo << "Generating canvases " << std::endl;
//  propagator.getPlotGenerator().generateCanvas(
//          propagator.getPlotGenerator().getHistHolderList(),
//          GenericToolbox::mkdirTFile(calcXsecDir, "plots/canvas")
//  );

//
//  // overlay the data histograms on the MC histograms
//  for(auto closureVar : closureVarList) {
//    std::string cleanSampleName = GenericToolbox::generateCleanBranchName(closureVar.samplePtr->getName());
//    TCanvas * c_MC = (TCanvas*)(app.getOutfilePtr()->Get( ("calcXsec/plots/canvas/"+closureVar.varToPlot+"/ReactionCode/sample_"+cleanSampleName+"_TCanvas").c_str() ) );
//    if(!c_MC){
//      LogError << "Could not find canvas for variable: " << closureVar.varToPlot << " and sample: " << cleanSampleName << std::endl;
//      continue;
//    }else{
//      // change canvas name
//       c_MC->SetName( ("Closure_"+closureVar.varToPlot+"_"+cleanSampleName+"_TCanvas").c_str() );
//      c_MC->SetTitle( ("Closure_"+closureVar.varToPlot+"_"+cleanSampleName+"_TCanvas").c_str() );
//      LogInfo<<"Creating canvas: "<<c_MC->GetName()<<std::endl;
//      int nPads = c_MC->GetListOfPrimitives()->GetSize();
////      c_MC->Draw("goff");
//      c_MC->cd(0);
//      closureVar.histogram->Draw("hist same");
//      //c_MC->SaveAs(  "temp_"+TString(c_MC->GetTitle())+".png" );
//
//      GenericToolbox::writeInTFile(
//              GenericToolbox::mkdirTFile(calcXsecDir, "plots/canvas"),
//              c_MC,
//              GenericToolbox::generateCleanBranchName( c_MC->GetTitle() )
//      );
//    }
//  }


  LogInfo << "Writing event samples in TTrees..." << std::endl;
  dataSetManager.getTreeWriter().writeSamples(
      GenericToolbox::mkdirTFile(calcXsecDir, "events"),
      dataSetManager.getPropagator()
  );



} // end of main





void readBinningFromFile(const char* filename, std::vector<Double_t>& binEdges) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Error opening file: " << filename << std::endl;
    return;
  }

  std::string line;
  // Skip the first line
  std::getline(file, line);
  double lastRightEdge;

  while (std::getline(file, line)) {
    Double_t lower, upper;
    std::istringstream iss(line);
    // skip empty lines
    if( line.empty() ){ continue; }
    // skip commented lines
    if( line[0] == '#' ){ continue; }
    if (!(iss >> lower >> upper)) {
      LogError << "Error reading bin edges from file: " << filename << std::endl;
      break;
    }
    binEdges.push_back(lower);
    lastRightEdge = upper;
  }

  // Add the last upper edge
  binEdges.push_back(lastRightEdge);

  file.close();
}