
#include "GlobalVariables.h"
#include "GundamApp.h"
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

  GundamApp app{"cross-section calculator tool"};

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
  ObjectReader::readObject<TNamed>(fitterFile.get(), {{"gundam/config_TNamed"}, {"gundamFitter/unfoldedConfig_TNamed"}}, [&](TNamed* config_){
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
  nlohmann::json xsecConfig{ConfigUtils::readConfigFile( clParser.getOptionVal<std::string>("configFile") )};
  cHandler.override( xsecConfig );

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

    std::string outFolder{GenericToolbox::Json::fetchValue<std::string>(xsecConfig, "outputFolder", "./")};
    outFilePath = GenericToolbox::joinPath(outFolder, outFilePath);
  }

  app.setCmdLinePtr( &clParser );
  app.setConfigString( ConfigUtils::ConfigHandler{xsecConfig}.toString() );
  app.openOutputFile( outFilePath );
  app.writeAppInfo();

  auto* calcXsecDir{ GenericToolbox::mkdirTFile(app.getOutfilePtr(), "calcXsec") };

  LogInfo << "Generating loaded sample plots..." << std::endl;
  propagator.getPlotGenerator().generateSamplePlots(
    GenericToolbox::mkdirTFile(calcXsecDir, "postFit/samples")
  );

  LogInfo << "Creating throws tree" << std::endl;
  auto* xsecThrowTree = new TTree("xsecThrow", "xsecThrow");


  LogInfo << "Creating normalizer objects..." << std::endl;
  // flux renorm with toys
  struct ParSetNormaliser{

    void readConfig(const nlohmann::json& config_){
      LogScopeIndent;

      name = GenericToolbox::Json::fetchValue<std::string>(config_, "name");
      LogInfo << "ParSetNormaliser config \"" << name << "\": " << std::endl;

      // mandatory
      filePath = GenericToolbox::Json::fetchValue<std::string>(config_, "filePath");
      histogramPath = GenericToolbox::Json::fetchValue<std::string>(config_, "histogramPath");
      axisVariable = GenericToolbox::Json::fetchValue<std::string>(config_, "axisVariable");

      // optionals
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
      LogThrowIf(dialCollectionPtr->getSupervisedParameter() == nullptr, "Need a dial collection that handle a whole parSet.");

      file = std::make_shared<TFile>( filePath.c_str() );
      LogThrowIf(file == nullptr, "Could not open file");

      histogram = file->Get<TH1D>( histogramPath.c_str() );
      LogThrowIf(histogram == nullptr, "Could not find histogram.");
    }


    [[nodiscard]] double getNormFactor() const {
      double out{0};

      LogDebug << __METHOD_NAME__ << std::endl;

      for( int iBin = 0 ; iBin < histogram->GetNbinsX() ; iBin++ ){
        LogDebug << "GETBINCONTENT" << std::endl;
        double binValue{histogram->GetBinContent(1+iBin)};


        // do we skip this bin? if not, apply coefficient
        bool skipBin{true};
        for( size_t iParBin = 0 ; iParBin < dialCollectionPtr->getDialBinSet().getBinsList().size() ; iParBin++ ){
          const DataBin& parBin = dialCollectionPtr->getDialBinSet().getBinsList()[iParBin];

          LogDebug << __LINE__ << std::endl;
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
  for( auto& parSet : propagator.getParameterSetsList() ){
    if( GenericToolbox::Json::doKeyExist(parSet.getConfig(), "normalisations") ){
      for( auto& parSetNormConfig : GenericToolbox::Json::fetchValue<nlohmann::json>(parSet.getConfig(), "normalisations") ){
        parSetNormList.emplace_back();
        parSetNormList.back().readConfig( parSetNormConfig );

        for( auto& dialCollection : propagator.getDialCollections() ){
          if( dialCollection.getSupervisedParameterSet() == &parSet ){
            parSetNormList.back().dialCollectionPtr = &dialCollection;
          }
        }

        parSetNormList.back().initialize();
      }
    }
  }



  // to be filled up
  struct BinNormaliser{
    void readConfig(const nlohmann::json& config_){
      LogScopeIndent;
      
      name = GenericToolbox::Json::fetchValue<std::string>(config_, "name");

      if( not GenericToolbox::Json::fetchValue(config_, "isEnabled", true) ){
        LogWarning << "Skipping disabled re-normalization config \"" << name << "\"" << std::endl;
        return;
      }

      LogInfo << "Re-normalization config \"" << name << "\": ";

      if     ( GenericToolbox::Json::doKeyExist( config_, "meanValue" ) ){
        normParameter.first  = GenericToolbox::Json::fetchValue<double>(config_, "meanValue");
        normParameter.second = GenericToolbox::Json::fetchValue(config_, "stdDev", 0);
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
    FitSample* samplePtr{nullptr};
    nlohmann::json config{};
    GenericToolbox::RawDataArray branchBinsData{};

    TH1D histogram{};
    std::vector<BinNormaliser> normList{};
  };
  std::vector<CrossSectionData> crossSectionDataList{};

  LogInfo << "Initializing xsec samples..." << std::endl;
  crossSectionDataList.reserve( propagator.getFitSampleSet().getFitSampleList().size() );
  for( auto& sample : propagator.getFitSampleSet().getFitSampleList() ){
    crossSectionDataList.emplace_back();
    auto& xsecEntry = crossSectionDataList.back();

    LogScopeIndent;
    LogInfo << "Defining xsec entry: " << sample.getName() << std::endl;
    xsecEntry.samplePtr = &sample;
    xsecEntry.config = sample.getConfig();
    xsecEntry.branchBinsData.resetCurrentByteOffset();
    std::vector<std::string> leafNameList{};
    leafNameList.reserve( sample.getMcContainer().histogram->GetNbinsX() );
    for( int iBin = 0 ; iBin < sample.getMcContainer().histogram->GetNbinsX() ; iBin++ ){
      leafNameList.emplace_back(Form("bin_%i/D", iBin));
      xsecEntry.branchBinsData.writeRawData( double(0) );
    }
    xsecEntry.branchBinsData.lockArraySize();

    xsecThrowTree->Branch(
        GenericToolbox::generateCleanBranchName( sample.getName() ).c_str(),
        xsecEntry.branchBinsData.getRawDataArray().data(),
        GenericToolbox::joinVectorString(leafNameList, ":").c_str()
    );

    auto normConfigList = GenericToolbox::Json::fetchValue( xsecEntry.config, "normaliseParameterList", nlohmann::json() );
    xsecEntry.normList.reserve( normConfigList.size() );
    for( auto& normConfig : normConfigList ){
      xsecEntry.normList.emplace_back();
      xsecEntry.normList.back().readConfig( normConfig );
    }

    xsecEntry.histogram = TH1D(
        sample.getName().c_str(),
        sample.getName().c_str(),
        sample.getMcContainer().histogram->GetNbinsX(),
        0,
        sample.getMcContainer().histogram->GetNbinsX()
    );
  }

  int nToys{ clParser.getOptionVal<int>("nToys") };

  bool enableEventMcThrow{true};
  bool enableStatThrowInToys{true};
  auto xsecCalcConfig   = GenericToolbox::Json::fetchValue( cHandler.getConfig(), "xsecCalcConfig", nlohmann::json() );
  enableStatThrowInToys = GenericToolbox::Json::fetchValue( xsecCalcConfig, "enableStatThrowInToys", enableStatThrowInToys);
  enableEventMcThrow    = GenericToolbox::Json::fetchValue( xsecCalcConfig, "enableEventMcThrow", enableEventMcThrow);

  std::stringstream ss; ss << LogWarning.getPrefixString() << "Generating " << nToys << " toys...";
  for( int iToy = 0 ; iToy < nToys ; iToy++ ){

    // loading...
    GenericToolbox::displayProgressBar( iToy+1, nToys, ss.str() );

    // Do the throwing:
    propagator.throwParametersFromGlobalCovariance();
    propagator.propagateParametersOnSamples();

    for( auto& xsec : crossSectionDataList ){

      if( enableStatThrowInToys ){
        if( enableEventMcThrow ){
          // Take into account the finite amount of event in MC
          xsec.samplePtr->getMcContainer().throwEventMcError();
        }
        // Asimov bin content -> toy data
        xsec.samplePtr->getMcContainer().throwStatError();
      }

      xsec.branchBinsData.resetCurrentByteOffset();
      for( int iBin = 0 ; iBin < xsec.samplePtr->getMcContainer().histogram->GetNbinsX() ; iBin++ ){
        double binData{ xsec.samplePtr->getMcContainer().histogram->GetBinContent(1+iBin) };

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

        // bin volume
        auto& bin = xsec.samplePtr->getMcContainer().binning.getBinsList()[iBin];
        double binVolume{1};

        for( size_t iDim = 0 ; iDim < bin.getEdgesList().size() ; iDim++ ){
          auto& edges = bin.getEdgesList()[iDim];
          if( edges.first == edges.second ) continue; // no volume, just a condition variable

          // is this bin excluded from norm?
          if( not bin.getVariableNameList()[iDim].empty() and std::any_of(
                xsec.normList.begin(), xsec.normList.end(), [&](const BinNormaliser& normData_){
                  return (
                      not normData_.disabledBinDim.empty()
                      and normData_.disabledBinDim == bin.getVariableNameList()[iDim] );
                }
              )
          ){ continue; }
          binVolume *= std::max( edges.first, edges.second ) - std::min(edges.first, edges.second);
        }

        binData /= binVolume;

        xsec.branchBinsData.writeRawData( binData );
      }
    }

    // Write the branches
    xsecThrowTree->Fill();
  }

  LogInfo << "Writing throws..." << std::endl;
  GenericToolbox::writeInTFile( GenericToolbox::mkdirTFile(calcXsecDir, "throws"), xsecThrowTree );

  LogInfo << "Calculating mean & covariance matrix..." << std::endl;
  auto* meanValuesVector = GenericToolbox::generateMeanVectorOfTree( xsecThrowTree );
  auto* globalCovMatrix = GenericToolbox::generateCovarianceMatrixOfTree( xsecThrowTree );

  auto* globalCovMatrixHist = GenericToolbox::convertTMatrixDtoTH2D(globalCovMatrix);
  auto* globalCorMatrixHist = GenericToolbox::convertTMatrixDtoTH2D(GenericToolbox::convertToCorrelationMatrix(globalCovMatrix));

  std::vector<TH1D> binValues{};
  binValues.reserve( propagator.getFitSampleSet().getFitSampleList().size() );
  int iBinGlobal{-1};

  for( auto& xsec : crossSectionDataList ){

    for( int iBin = 0 ; iBin < xsec.samplePtr->getMcContainer().histogram->GetNbinsX() ; iBin++ ){
      iBinGlobal++;

      std::string binTitle = xsec.samplePtr->getBinning().getBinsList()[iBin].getSummary();
      double binVolume = xsec.samplePtr->getBinning().getBinsList()[iBin].getVolume();

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

  GlobalVariables::getParallelWorker().reset();
}

