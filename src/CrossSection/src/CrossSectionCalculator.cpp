//
// Created by Nadrino on 25/06/2025.
//

#include "CrossSectionCalculator.h"

#include "FitterEngine.h"
#include "RootUtils.h"

#include "GenericToolbox.Root.h"


void CrossSectionCalculator::configureImpl(){

  _config_.clearFields();
  _config_.defineFields({
    {"outputFolder"},
    {"useBestFitAsCentralValue"},
    {"enableStatThrowInToys", {"xsecCalcConfig/enableStatThrowInToys"}},
    {"enableEventMcThrow", {"xsecCalcConfig/enableEventMcThrow"}},
  });
  _config_.checkConfiguration();

  _config_.fillValue(_outputFolder_, "outputFolder");
  _config_.fillValue(_useBestFitAsCentralValue_, "useBestFitAsCentralValue");
  _config_.fillValue(_enableStatThrowInToys_, "enableStatThrowInToys");
  _config_.fillValue(_enableEventMcThrow_, "enableEventMcThrow");

  if( GenericToolbox::hasExtension(_fitterFilePath_, "root") ){
    LogWarning << "Opening fitter output file: " << _fitterFilePath_ << std::endl;
    std::unique_ptr<TFile> fitterRootFile{nullptr};
    fitterRootFile = std::unique_ptr<TFile>( TFile::Open( _fitterFilePath_.c_str() ) );
    LogThrowIf( fitterRootFile == nullptr, "Could not open fitter output file." );

    RootUtils::ObjectReader::throwIfNotFound = true;

    RootUtils::ObjectReader::readObject<TNamed>(
        fitterRootFile.get(),
        {{"gundam/config/unfoldedJson_TNamed"},
         {"gundam/config_TNamed"},
         {"gundamFitter/unfoldedConfig_TNamed"}},
        [&](TNamed* config_){
      _fitterEngineConfig_ = ConfigReader(GenericToolbox::Json::readConfigJsonStr( config_->GetTitle() )).getConfig();
    });

    if( not _usePrefit_ and fitterRootFile != nullptr ){

      // Load post-fit parameters as "prior" so we can reset the weight to this point when throwing toys
      RootUtils::ObjectReader::readObject<TNamed>( fitterRootFile.get(), "FitterEngine/postFit/parState_TNamed", [&](TNamed* parState_){
        _postFitParState_ = GenericToolbox::Json::readConfigJsonStr( parState_->GetTitle() );
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
  }
  else{
    LogWarning << "Reading fitter config file (no-post-fit will be available): " << _fitterFilePath_ << std::endl;
    _fitterEngineConfig_ = ConfigReader(GenericToolbox::Json::readConfigFile( _fitterFilePath_ )).getConfig();
    _usePrefit_ = true;
  }

  // overrides
  LogInfo << "Overriding fitter engine config..." << std::endl;
  ConfigReader engineConfig;
  LogExitIf(_fitterEngineConfig_.empty(), "Fitter engine config is empty.");
  {
    ConfigUtils::ConfigBuilder cHandler{ _fitterEngineConfig_ };

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
    cHandler.override( _config_.getConfig() );

    engineConfig.setConfig(cHandler.getConfig());
    LogInfo << "Override done." << std::endl;
  }

  // now forward to the fitter engine
  engineConfig.defineFields({
    {"fitterEngineConfig"},
  });
  _fitterEngine_.configure( engineConfig.fetchValue<ConfigReader>( "fitterEngineConfig" ) );

  // We are only interested in our MC. Data has already been used to get the post-fit error/values
  _fitterEngine_.getLikelihoodInterface().setForceAsimovData( true );

  // Disabling eigen decomposed parameters
  _fitterEngine_.getLikelihoodInterface().getModelPropagator().setEnableEigenToOrigInPropagate( false );

  // LEGACY: If we want to define the sample binning using template parameters
  for( auto& sample : _fitterEngine_.getLikelihoodInterface().getModelPropagator().getSampleSet().getSampleList() ){

    // binning already set in the xsec config?
    if( not sample.getBinningFilePath().empty() ){ continue; }

    LogExitIf( not sample.getConfig().hasField("parSetBinning"), "No binning defined for " << sample.getName() );

    LogScopeIndent;
    LogAlert << sample.getName() << ": USING parSetBinning to define the binning. This is a deprecated. You need to directly provide the binning." << std::endl;

    auto associatedParSet = sample.getConfig().fetchValue<std::string>("parSetBinning");

    // Looking for parSet
    auto foundDialCollection = std::find_if(
        _fitterEngine_.getLikelihoodInterface().getModelPropagator().getDialCollectionList().begin(),
        _fitterEngine_.getLikelihoodInterface().getModelPropagator().getDialCollectionList().end(),
        [&](const DialCollection& dialCollection_){
          auto* parSetPtr{dialCollection_.getSupervisedParameterSet()};
          if( parSetPtr == nullptr ){ return false; }
          return ( parSetPtr->getName() == associatedParSet );
        });
    LogThrowIf(
        foundDialCollection == _fitterEngine_.getLikelihoodInterface().getModelPropagator().getDialCollectionList().end(),
        "Could not find " << associatedParSet << " among fit dial collections: "
                          << GenericToolbox::toString(_fitterEngine_.getLikelihoodInterface().getModelPropagator().getDialCollectionList(),
                                                      [](const DialCollection& dialCollection_){
                                                        return dialCollection_.getTitle();
                                                      }
                          ));

    LogThrowIf(foundDialCollection->getDialBinSet().getBinList().empty(), "Could not find binning");
    JsonType json(foundDialCollection->getDialBinSet().getFilePath());
    sample.setBinningFilePath( ConfigReader(json) );

  }

}
void CrossSectionCalculator::initializeImpl(){

  // Load everything
  _fitterEngine_.getLikelihoodInterface().initialize();

  auto& propagator{_fitterEngine_.getLikelihoodInterface().getModelPropagator()};

  if( not _usePrefit_ ){
    LogWarning << std::endl << GenericToolbox::addUpDownBars("Injecting post-fit parameters...") << std::endl;
    propagator.getParametersManager().injectParameterValues( _postFitParState_ );

    LogInfo << "Anchoring parameter prior with the current value..." << std::endl;
    propagator.getParametersManager().setParametersPriorWithCurrentValue();
  }

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

}

void CrossSectionCalculator::throwToys(int nToys_){

  auto& propagator{_fitterEngine_.getLikelihoodInterface().getModelPropagator()};

  LogWarning << std::endl << GenericToolbox::addUpDownBars( "Generating toys..." ) << std::endl;
  propagator.getParametersManager().initializeStrippedGlobalCov();

  LogInfo << "Creating throws tree" << std::endl;
  auto* xsecThrowTree = new TTree("xsecThrow", "xsecThrow");
  xsecThrowTree->SetDirectory( _savePath_.getSubDir("throws").getDir() ); // temp saves will be done here

  for( auto& xsecEntry : crossSectionDataList ) {
    xsecThrowTree->Branch(
        GenericToolbox::generateCleanBranchName( xsecEntry.samplePtr->getName() ).c_str(),
        xsecEntry.branchBinsData.getRawDataArray().data(),
        GenericToolbox::joinVectorString(leafNameList, ":").c_str()
    );
  }

  // stats printing
  GenericToolbox::Time::AveragedTimer<1> totalTimer{};
  GenericToolbox::Time::AveragedTimer<1> throwTimer{};
  GenericToolbox::Time::AveragedTimer<1> propagateTimer{};
  GenericToolbox::Time::AveragedTimer<1> otherTimer{};
  GenericToolbox::Time::AveragedTimer<1> writeTimer{};
  GenericToolbox::TablePrinter t{};
  std::stringstream progressSs;
  std::stringstream ss; ss << LogWarning.getPrefixString() << "Generating " << nToys_ << " toys...";
  for( int iToy = 0 ; iToy < nToys_ ; iToy++ ){

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
    GenericToolbox::displayProgressBar( iToy+1, nToys_, progressSs.str() );

    // Do the throwing:
    throwTimer.start();
    propagator.getParametersManager().throwParametersFromGlobalCovariance( not GundamGlobals::isDebug() );
    throwTimer.stop();

    propagateTimer.start();
    propagator.propagateParameters();

    if( _enableStatThrowInToys_ ){
      for( auto& xsec : crossSectionDataList ){
        if( _enableEventMcThrow_ and not xsec.samplePtr->isEventMcThrowDisabled() ){
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
  GenericToolbox::writeInTFileWithObjTypeExt( _savePath_.getSubDir("throws").getDir(), xsecThrowTree );


}