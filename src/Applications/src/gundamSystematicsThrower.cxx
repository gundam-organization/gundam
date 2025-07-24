//
// Created by Lorenzo Giannessi on 05.09.23.
//

#include "GundamGlobals.h"
#include "GundamApp.h"
#include "GundamUtils.h"
#include "RootUtils.h"
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
#include "../../Fitter/Engine/include/FitterEngine.h"
#include "../../StatisticalInference/Likelihood/include/LikelihoodInterface.h"

#include <string>
#include <vector>
#include <TObjString.h>


LoggerInit([]{
    Logger::getUserHeader() << "[" << FILENAME << "]";
});

double getParameterValueFromTextFile(std::string fileName, std::string parameterName);

int main(int argc, char** argv){


    GundamApp app{"Systematics thrower/marginaliser"};

    // --------------------------
    // Read Command Line Args:
    // --------------------------
    CmdLineParser clParser;

    clParser.addDummyOption("Main options:");

    // I need a config file where the list of parameters to marginalise over are specified (look at the XSec config file to get inspired)
    clParser.addOption("configFile", {"-c"}, "Specify the parameters to marginalise over");

    clParser.addOption("fitterOutputFile", {"-f"}, "Specify the fitter output file");

    clParser.addOption("outputFile", {"-o", "--out-file"}, "Specify the Marginaliser output file");

    clParser.addOption("nbThreads", {"-t", "--nb-threads"}, "Specify nb of parallel threads");
    clParser.addOption("nToys", {"-n"}, "Specify number of toys");
    clParser.addOption("randomSeed", {"-s", "--seed"}, "Set random seed");
    clParser.addTriggerOption("usingGpu", {"--gpu"}, "Use GPU parallelization");
    clParser.addOption("parInject", {"--parameters-inject"}, "Input txt file for injecting params");
    clParser.addOption("pedestal", {"--pedestal"}, "Add pedestal to the sampling distribution (percents)");
    clParser.addOption("weightCap", {"--weight-cap"}, "Cap the weight of the throws (default: 1.e8)", 1);
    clParser.addOption("tStudent", {"--t-student"}, "ndf for tStudent throw (default: gaussian throws)", 1);

    clParser.addDummyOption("Trigger options:");
    clParser.addTriggerOption("dryRun", {"-d", "--dry-run"}, "Only overrides fitter config and print it.");
    clParser.addTriggerOption("throwStats", {"--throw-stats"}, "Throw statistical errors in the toys.");

    // I probably don't need this
    //clParser.addTriggerOption("useBfAsXsec", {"--use-bf-as-xsec"}, "Use best-fit as x-sec value instead of mean of toys.");

    LogInfo << "Usage: " << std::endl;
    LogInfo << clParser.getConfigSummary() << std::endl << std::endl;

    clParser.parseCmdLine(argc, argv);

    LogThrowIf(clParser.isNoOptionTriggered(), "No option was provided.");

    LogInfo << "Provided arguments: " << std::endl;
    LogInfo << clParser.getValueSummary() << std::endl << std::endl;


    // Sanity checks
    LogThrowIf(not clParser.isOptionTriggered("configFile"), "Marginaliser config file not provided.");
    LogThrowIf(not clParser.isOptionTriggered("fitterOutputFile"), "Did not provide the output fitter file.");
    // Do I need to throw toys actually??? YES: throwing toys in your case is throwing according to the post-fit covariance matrix.
    // You do it to compute the weights that eventually go in the computation of the marginalised likelihood.
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

    GundamGlobals::setNumberOfThreads(clParser.getOptionVal("nbThreads", 1));
    LogInfo << "Running the fitter with " << GundamGlobals::getNbCpuThreads()<< " parallel threads." << std::endl;

    // Reading fitter file
    LogInfo << "Opening fitter output file: " << clParser.getOptionVal<std::string>("fitterOutputFile") << std::endl;
    std::string fitterFile{clParser.getOptionVal<std::string>("fitterOutputFile")};
    std::unique_ptr<TFile> fitterRootFile{nullptr};
    ConfigReader fitterConfig; // will be used to load the propagator

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
                  fitterConfig = ConfigReader(GenericToolbox::Json::readConfigJsonStr( config_->GetTitle() ));
              });
    }else{
      LogError << "Fitter output file is not a ROOT file: " << fitterFile << std::endl;
      LogExit("Fitter output file must be a ROOT file.");
    }

    // Print out the config
    LogInfo << "Fitter config loaded: " << std::endl;
    fitterConfig.defineFields({
                                    {"fitterEngineConfig"},
                            });
    LogInfo << fitterConfig.toString() << std::endl;


    ConfigUtils::ConfigBuilder cHandler{ fitterConfig.getConfig() };
    LogInfo << "Fitter config loaded." << std::endl;
    // Reading marginaliser config file
    ConfigReader margConfig{ ConfigUtils::readConfigFile( clParser.getOptionVal<std::string>("configFile") ) };
  // Check if the config is an array (baobab compatibility)
//    if(margConfig.size()==1) {
//        // broken json library puts everything in the same level
//        // normal behavior
//        margConfig = margConfig[0];
//    }
    cHandler.override( margConfig.getConfig() );
    LogInfo << "Override config done." << std::endl;



    if( clParser.isOptionTriggered("dryRun") ){
        std::cout << cHandler.toString() << std::endl;

        LogAlert << "Exiting as dry-run is set." << std::endl;
        return EXIT_SUCCESS;
    }
    bool injectParamsManually = false;
    std::string parInjectFile;
    if( clParser.isOptionTriggered("parInject") ){
        parInjectFile = clParser.getOptionVal<std::string>("parInject");
        injectParamsManually = true;
        LogInfo << "Injecting parameters from file: " << parInjectFile << std::endl;
    }else{
        LogInfo << "Injecting best fit parameters as prior" << std::endl;
    }
    double pedestalEntity = 0;
    bool usePedestal = false;
    double pedestalLeftEdge = -3, pedestalRightEdge = -pedestalLeftEdge;
    if(clParser.isOptionTriggered("pedestal")){
        usePedestal = true;
        pedestalEntity = clParser.getOptionVal<double>("pedestal")/100.;
        LogInfo << "Using Gauss+pedestal sampling: "<< pedestalEntity*100 << "% of the throws will be drawn from a uniform distribution."<<std::endl;
        LogInfo <<"Using default pedestal interval: ["<<pedestalLeftEdge<<"sigma, "<<pedestalRightEdge<<"sigma]";
        LogInfo <<". User defined interval to be implemented"<<std::endl;
    }
    double weightCap = 1.e8;
    int countBigThrows = 0;
    if(clParser.isOptionTriggered("weightCap")){
        weightCap = clParser.getOptionVal<double>("weightCap");
        LogInfo << "Using weight cap: "<< weightCap << std::endl;
    }else{
        LogInfo << "Using default weight cap: "<< weightCap << std::endl;
    }
    bool tStudent = false;
    double tStudentNu = -1;
    if(clParser.isOptionTriggered("tStudent")){
        tStudentNu = clParser.getOptionVal<double>("tStudent");
        LogInfo << "Throwing according to a multivariate t-student with ndf: "<< tStudentNu << std::endl;
    }
    if(tStudentNu==-1){
        tStudent = false;
    }else if(tStudentNu<=2){
        LogError<< "Not possible to use ndf for t-student smaller than 2!"<<std::endl;
        tStudent = false;
    }else{
        tStudent = true;
    }
//    bool enableStatThrowInToys = false;
//    bool enableEventMcThrow = false;
//    if(clParser.isOptionTriggered("throwStats")) {
//      enableStatThrowInToys = true;
//      LogInfo << "Throwing statistical errors in the toys." << std::endl;
//      enableEventMcThrow = true; // default
//      if(GenericToolbox::Json::fetchValue<bool>(margConfig, "disableEventMcThrow", false)){
//          enableEventMcThrow = false;
//          LogInfo << "Disabling event MC throw." << std::endl;
//      }
//    }else{
//      LogInfo << "Statistical errors will not be thrown in the toys." << std::endl;
//      enableStatThrowInToys = false;
//      enableEventMcThrow = false; // default
//    }

    auto configPropagator = GenericToolbox::Json::fetchValue<nlohmann::json>( cHandler.getConfig(), "fitterEngineConfig/propagatorConfig" );

    // Initialize the fitterEngine
    LogInfo << "FitterEngine setup..." << std::endl;
    FitterEngine fitter{nullptr};
    fitter.configure( fitterConfig.fetchValue<ConfigReader>( "fitterEngineConfig" ) );

    // We load the Asimov (prior) histograms. Then the histograms will be filled manually with the histos in the fitter output file
    fitter.getLikelihoodInterface().setForceAsimovData( true );

    LogInfo<< "setForceAsimovData - done." << std::endl;

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
      auto associatedParSet = sample.getConfig().fetchValue("parSetBinning", std::string());

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
      JsonType json(foundDialCollection->getDialBinSet().getFilePath());
      sample.setBinningFilePath( ConfigReader(json) );

    }

    LogInfo<< "Initializing the fitter ..." << std::endl;
    // Load everything
    fitter.getLikelihoodInterface().initialize();

    // At this point the "data" histograms are just built with Asimov priors.
    // We have to replace the bin contents based on the actual data from teh fitter output file.
    // The easiest way is to grab the histograms from the fitter output file for each sample, and
    // overwrite the histograms in the propagator.
    LogInfo << "Loading DATA histograms from fitter output file..." << std::endl;
    // Loop through the samples
    for( auto& sample : fitter.getLikelihoodInterface().getModelPropagator().getSampleSet().getSampleList() ){

        std::string samplePath = "FitterEngine/preFit/data/" + sample.getName();
        LogInfo << "Loading data histogram for sample: " << sample.getName() << std::endl;
        Histogram hist = sample.getHistogram();
        // Load the histogram from the fitter output file
        RootUtils::ObjectReader::readObject<TH1D>(
                fitterRootFile.get(), samplePath,
            [&](TH1D* hist_) {
                LogInfo << "Loaded histogram: " << hist_->GetName() << std::endl;
                // Copy the histogram content to the sample histogram
                // check number of bins
                LogThrowIf(hist_->GetNbinsX() != hist.getBinContentList().size(),
                           "Histogram " + std::string(hist_->GetName()) +
                           " has different number of bins than the sample histogram.");
                LogIndent;
                for (int iBin = 0; iBin < hist.getBinContentList().size(); iBin++) {
                  LogInfo << "Bin " << iBin << " from " <<hist.getBinContentList().at(iBin).sumWeights << " to " << hist_->GetBinContent(1 + iBin) << std::endl;
                  hist.getBinContentList().at(iBin).sumWeights = hist_->GetBinContent(1 + iBin);
                  // bin error = ?
                }
                LogUnIndent;
            }
        );
    }

    Propagator& propagator = fitter.getLikelihoodInterface().getModelPropagator();

    // you need this to use the custom systematic throwers
    propagator.getParametersManager().setThrowerAsCustom();

    propagator.getParametersManager().getParameterSetsList();

    std::vector<std::string> parameterNames;
    std::vector<double> bestFitValues;
    std::vector<double> priorValues;
    std::vector<double> priorSigmas;
    int debug_enabled_params{0};
    int debug_cov_rows{0};

    fitter.getLikelihoodInterface().getModelPropagator().propagateParameters();
    std::future<bool> propagated = fitter.getLikelihoodInterface().getModelPropagator().applyParameters();
    fitter.getLikelihoodInterface().evalLikelihood(propagated);
    double priorLLH = fitter.getLikelihoodInterface().getBuffer().totalLikelihood;
    double prior_statLH = fitter.getLikelihoodInterface().evalStatLikelihood(propagated);
    double prior_systLH = fitter.getLikelihoodInterface().evalPenaltyLikelihood();;
    LogInfo<<" LH at prior:\nLH_stat="<<prior_statLH
           <<"\nLH_syst="<<prior_systLH
           <<"\nTotal LH="<<priorLLH<<std::endl;

    // Load post-fit parameters as "prior" so we can reset the weight to this point when throwing toys
    // also save the values in a vector so we can use them to compute the LLH at the best fit point
    int NParametersFromFitterFile = 0;
    RootUtils::ObjectReader::readObject<TNamed>( fitterRootFile.get(), "FitterEngine/postFit/parState_TNamed", [&](TNamed* parState_){
        propagator.getParametersManager().injectParameterValues( GenericToolbox::Json::readConfigJsonStr( parState_->GetTitle() ) );
        for( auto& parSet : propagator.getParametersManager().getParameterSetsList() ){
            if( not parSet.isEnabled() ){ continue; }
//            LogInfo<< parSet.getName()<<std::endl;
            for( auto& par : parSet.getParameterList() ){
                if( not par.isEnabled() ){ continue; }
//                LogInfo<<debug_enabled_params<<" "<<par.getFullTitle()<<std::endl;
                debug_enabled_params++;
                parameterNames.emplace_back(par.getFullTitle());
                bestFitValues.emplace_back(par.getParameterValue());
                priorValues.emplace_back(par.getPriorValue());
                priorSigmas.emplace_back(par.getStdDevValue());
                LogInfo <<" DEBUG: Parameter: " << par.getFullTitle()
                        << " | Best fit value: " << par.getParameterValue()
                        << " | Prior value: " << par.getPriorValue() << std::endl;
                // set prior to best fit
                par.setPriorValue(par.getParameterValue());
                NParametersFromFitterFile++;
            }
        }
    });
    LogInfo << "DEBUG: Number of parameters read from the fitter file: " << NParametersFromFitterFile << std::endl;


    // Creating output file
    std::string outFilePath{};
    if( clParser.isOptionTriggered("outputFile") ){
      outFilePath = clParser.getOptionVal<std::string>("outputFile");
    }
    else {
        // appendixDict["optionName"] = "Appendix"
        // this list insure all appendices will appear in the same order
        std::vector<GundamUtils::AppendixEntry> appendixDict{
                {"configFile",       ""},
                {"fitterOutputFile", ""},
                {"nToys",            "nToys"},
                {"randomSeed",       "Seed"},
                {"parInject",        "parInject"},
                {"pedestal",         "pedestal"},
                {"tStudent",         "tStudentNu"},
                {"throwStats",       "stats"}
        };
        outFilePath = GundamUtils::generateFileName(clParser, appendixDict) + ".root";

        auto outFolder(margConfig.fetchValue<std::string>("outputFolder", "./"));
        outFilePath = GenericToolbox::joinPath(outFolder, outFilePath);
    }




    // Load the post-fit covariance matrix
    RootUtils::ObjectReader::readObject<TH2D>(
          fitterRootFile.get(), "FitterEngine/postFit/Hesse/hessian/postfitCovarianceOriginal_TH2D",
          [&](TH2D* hCovPostFit_){
              propagator.getParametersManager().setGlobalCovarianceMatrix(std::make_shared<TMatrixD>(hCovPostFit_->GetNbinsX(), hCovPostFit_->GetNbinsX()));
              for( int iBin = 0 ; iBin < hCovPostFit_->GetNbinsX() ; iBin++ ){
                  for( int jBin = 0 ; jBin < hCovPostFit_->GetNbinsX() ; jBin++ ){
                      (*propagator.getParametersManager().getGlobalCovarianceMatrix())[iBin][jBin] = hCovPostFit_->GetBinContent(1 + iBin, 1 + jBin);
                  }
//                  LogInfo<<iBin<<" "<<hCovPostFit_->GetXaxis()->GetBinLabel(1 + iBin)<<std::endl;
              }
              debug_cov_rows = hCovPostFit_->GetNbinsX();
          });

    LogInfo << "DEBUG: Number of enabled parameters: " << debug_enabled_params << "\nNumber of rows in the covariance matrix: " << debug_cov_rows << std::endl;



    app.setCmdLinePtr( &clParser );
    app.setConfigString( ConfigUtils::ConfigBuilder{margConfig.getConfig()}.toString()  );
    app.openOutputFile( outFilePath );
    app.writeAppInfo();


    // also save the value of the LLH at the best fit point:
    for( auto& parSet : propagator.getParametersManager().getParameterSetsList() ){
      if( not parSet.isEnabled() ){ continue; }
      for( auto& par : parSet.getParameterList() ){
        if( not par.isEnabled() ){ continue; }
        debug_enabled_params++;
        LogInfo <<" DEBUG: Parameter: " << par.getFullTitle()
                << " | current value: " << par.getParameterValue()
                << " | best fit value: " << par.getPriorValue() << std::endl;
      }
    }
    fitter.getLikelihoodInterface().getModelPropagator().propagateParameters();
    propagated = fitter.getLikelihoodInterface().getModelPropagator().applyParameters();
    fitter.getLikelihoodInterface().evalLikelihood(propagated);
    double bestFitLLH = fitter.getLikelihoodInterface().getBuffer().totalLikelihood;
    double bestFit_statLH = fitter.getLikelihoodInterface().evalStatLikelihood(propagated);
    double bestFit_systLH = fitter.getLikelihoodInterface().evalPenaltyLikelihood();;
    LogInfo<<" Best fit Likelihood:\nLH_stat="<<bestFit_statLH
           <<"\nLH_syst="<<bestFit_systLH
           <<"\nTotal LH="<<bestFitLLH<<std::endl;
    // write the post fit covariance matrix in the output file
    TDirectory* postFitInfo = app.getOutfilePtr()->mkdir("postFitInfo");
    postFitInfo->cd();
    RootUtils::ObjectReader::readObject<TH2D>(
            fitterRootFile.get(), "FitterEngine/postFit/Hesse/hessian/postfitCovarianceOriginal_TH2D",
            [&](TH2D* hCovPostFit_) {
                hCovPostFit_->SetName("postFitCovarianceOriginal_TH2D");
                hCovPostFit_->Write();// save the TH2D cov. matrix also in the output file
            });
    // write the best fit parameters vector to the output file
    RootUtils::ObjectReader::readObject<TNamed>( fitterRootFile.get(), "FitterEngine/postFit/parState_TNamed", [&](TNamed* parState_){
        parState_->SetName("postFitParameters_TNamed");
        parState_->Write();
    });

    unsigned int nParameters = parameterNames.size();
    TVectorD bestFitParameters_TVectorD(nParameters,bestFitValues.data());
    TVectorD priorParameters_TVectorD(nParameters,priorValues.data());
    TVectorD priorSigmas_TVectorD(nParameters,priorSigmas.data());
    app.getOutfilePtr()->WriteObject(&parameterNames, "parameterFullTitles");
    app.getOutfilePtr()->WriteObject(&bestFitParameters_TVectorD, "bestFitParameters_TVectorD");
    app.getOutfilePtr()->WriteObject(&priorParameters_TVectorD, "priorParameters_TVectorD");
    app.getOutfilePtr()->WriteObject(&priorSigmas_TVectorD, "priorSigmas_TVectorD");
    app.getOutfilePtr()->cd();

    auto* marginalisationDir{ GenericToolbox::mkdirTFile(app.getOutfilePtr(), "marginalisation") };
    LogInfo << "Creating throws tree" << std::endl;
    auto* margThrowTree = new TTree("margThrow", "margThrow");
    margThrowTree->SetDirectory( GenericToolbox::mkdirTFile(marginalisationDir, "throws") ); // temp saves will be done here
    auto* ThrowsPThetaFormat = new TTree("PThetaThrows", "PThetaThrows");



    // make a TTree with the following branches, to be filled with the throws
    // 1. marginalised parameters drew: vector<double>
    // 2. non-marginalised parameters drew: vector<double>
    // 3. LLH value: double
    // 4. "g" value: the chi square value as extracted from the covariance matrix: double
    // 5. value of the priors for the marginalised parameters: vector<double>

    // branches for margThrowTree
    std::vector<double> *parameters = nullptr;
    std::vector<bool> *margThis = nullptr;
    std::vector<double> *prior = nullptr;
    std::vector<double> *weightsChiSquare = nullptr;
//    weightsChiSquare.reserve(1000);
    double LLH, gLLH, priorSum, LLHwrtBestFit;
    double LH_stat, LH_syst, LH_systWrtBestFit, LH_statWrtBestFit;
    double totalWeight; // use in the pedestal case, just as a placeholder
    margThrowTree->Branch("Parameters", &parameters);
    margThrowTree->Branch("Marginalise", &margThis);
    margThrowTree->Branch("prior", &prior);
    margThrowTree->Branch("LLH", &LLH);
    margThrowTree->Branch("LH_stat", &LH_stat);
    margThrowTree->Branch("LH_syst", &LH_syst);
    margThrowTree->Branch("LLHwrtBestFit", &LLHwrtBestFit);
    margThrowTree->Branch("LH_statWrtBestFit", &LH_statWrtBestFit);
    margThrowTree->Branch("LH_systWrtBestFit", &LH_systWrtBestFit);
    margThrowTree->Branch("gLLH", &gLLH);
    margThrowTree->Branch("priorSum", &priorSum);
    margThrowTree->Branch("weightsChiSquare", &weightsChiSquare);

    // branches for ThrowsPThetaFormat
    std::vector<double> survivingParameterValues;
    double LhOverGauss;
    ThrowsPThetaFormat->Branch("ParValues", &survivingParameterValues);
    ThrowsPThetaFormat->Branch("weight", &LhOverGauss);


    int nToys{ clParser.getOptionVal<int>("nToys") };

    // Get parameters to be marginalised
    std::vector<std::string> marginalisedParameters;
    std::vector<std::string> marginalisedParameterSets;
    marginalisedParameters = margConfig.fetchValue<std::vector<std::string>>("parameterList");
    marginalisedParameterSets = margConfig.fetchValue<std::vector<std::string>>("parameterSetList");

    LogInfo<<"----------------------- INFO ABOUT PTheta marginalised TTree -----------------------"<<std::endl;
    LogInfo<<"Marginalised parameters: "<<GenericToolbox::parseVectorAsString(marginalisedParameters,true,true)<<std::endl;
    LogInfo<<"Marginalised parameter Sets: "<<GenericToolbox::parseVectorAsString(marginalisedParameterSets,true,true)<<std::endl;
    // object array with the names of the parameters that "survive" the marginalisation
    TObjArray *marg_param_list;
    marg_param_list = new TObjArray();
    // loop over parameters and compare their titles with the ones in the marginalisedParameters string
    // if they match, set the corresponding margThis to true
    // Also display the prior information
    LogInfo<<"Prior information: "<<std::endl;
    for( auto& parSet : propagator.getParametersManager().getParameterSetsList() ) {
        bool setMatches = false;
        if (not parSet.isEnabled()) {
            LogInfo << "Parameter set " << parSet.getName() << " is disabled" << std::endl;
            continue;
        } else {
            LogInfo << "Set: " << parSet.getName().c_str();
            for (int i = 0; i < marginalisedParameterSets.size(); i++) {
                if (0 == std::strcmp(parSet.getName().c_str(), marginalisedParameterSets[i].c_str())) {
                    setMatches = (true);
                    break;
                } else {
                    setMatches = (false);
                }
            }
            if (setMatches) {
                LogInfo << " will be marginalized out.   \n";
                // loop over the parameters in the set and set the corresponding margThis to true
                for (auto &par: parSet.getParameterList()) {
                    par.setMarginalised(true);
                }
            } else {
                LogInfo << " will not be marginalized out.   \n";
            }
        }
        // Do the same for single parameters
        for (auto &par: parSet.getParameterList()) {
            if (not par.isEnabled()) {
                LogInfo << "Parameter " << par.getName() << " is disabled" << std::endl;
                continue;
            }
            bool matches = false;
            for (int i = 0; i < marginalisedParameters.size(); i++) {
                if (0 == std::strcmp(par.getFullTitle().c_str(), marginalisedParameters[i].c_str())) {
                    matches = (true);
                    break;
                } else {
                    matches = (false);
                }
            }
            par.setMarginalised(matches or setMatches);
            if (!par.isMarginalised()){
                marg_param_list->Add(new TObjString(par.getFullTitle().c_str()));
            }
            if(par.isMarginalised()){
                LogInfo << "Parameter " << par.getFullTitle()
                << " -> type: " << par.getPriorType() << " mu=" << par.getPriorValue()
                << " sigma= " << par.getStdDevValue() << " limits: "<< par.getPhysicalLimits().min << " - "
                                                                    << par.getPhysicalLimits().max
                <<" -> will be marg. out\n";
            }else{
                LogInfo << "Parameter " << par.getFullTitle()
                        << " -> type: " << par.getPriorType() << " mu=" << par.getPriorValue()
                        << " sigma= " << par.getStdDevValue() << " limits: " << par.getPhysicalLimits().min << " - "
                                                                             << par.getPhysicalLimits().max
                <<" -> will NOT be marg. out\n";
            }
        }
    }
    LogInfo<<"----------------------- END of INFO ABOUT PTheta marginalised TTree -----------------------"<<std::endl;


    // write the list of parameters that will not be marginalised to the output file
    app.getOutfilePtr()->WriteObject(marg_param_list, "marg_param_list");

    // print marg_param_list for debug
//    LogInfo<<" DEBUG: Number of parameter in marg_param_list: "<<marg_param_list->GetEntries()<<std::endl;

    // initializing variables for "parametersInject" mode
    double LLH_sum{0};// needed when injecting parameters manually
    double injectedLLH{-1};// needed when injecting parameters manually
    double epsilonNormAverage{0};// needed when injecting parameters manually
    if (injectParamsManually){
        LogInfo<< "Injecting parameters from file: " << parInjectFile << std::endl;
    }


    double weightSum = 0, weightSquareSum = 0, ESS = 0;
    int weightSumE50 = 0, weightSquareSumE50 = 0;
    std::stringstream ss; ss << LogWarning.getPrefixString() << "Generating " << nToys << " toys...";
    //////////////////////////////////////
    // THROWS LOOP
    /////////////////////////////////////
    for( int iToy = 0 ; iToy < nToys ; iToy++ ){

        // loading...
        GenericToolbox::displayProgressBar( iToy+1, nToys, ss.str() );

        // reset weights vector
        weightsChiSquare->clear();
        // Do the throwing
        if (usePedestal){
            // if the pedestal option is enabled, a uniform distribution is added to the gaussian sampling distribution
            propagator.getParametersManager().throwParametersFromGlobalCovariance(*weightsChiSquare, pedestalEntity, pedestalLeftEdge, pedestalRightEdge);
        }else{
            if(!tStudent) {
                // standard case: throw according to the covariance matrix
                propagator.getParametersManager().throwParametersFromGlobalCovariance(*weightsChiSquare);
            }else{
                propagator.getParametersManager().throwParametersFromTStudent(*weightsChiSquare,tStudentNu);
            }
        }
        // Sanity check on the length of the weights vector. It should be as long as the number of parameters
        if(weightsChiSquare->size() != nParameters){
          for(int i=0;i<weightsChiSquare->size();i++){
            LogInfo<<"weightsChiSquare["<<i<<"]: "<<weightsChiSquare->at(i)<<std::endl;
          }
          std::cout<<"("<<nParameters<<" parameters)"<<std::endl;
        }
        LogThrowIf(weightsChiSquare->size() != nParameters, "ERROR: The weights vector has a different size than the number of parameters.");

//        // Debug: print out parameters
//        for (auto &parSet: propagator.getParametersManager().getParameterSetsList()) {
//            if (not parSet.isEnabled()) { continue; }
//            for (auto &par: parSet.getParameterList()) {
//                if (not par.isEnabled()) continue;
//                LogInfo <<"{gundamMarginalise} " << par.getTitle() << " -> " << par.getParameterValue() << std::endl;
//            }
//        }

        if(injectParamsManually) {
            // count the number of parameters to be injected
            int nStripped{0};
            for( auto& parSet : propagator.getParametersManager().getParameterSetsList() ){
                if( not parSet.isEnabled() ) continue;
                for( auto& par : parSet.getParameterList() ){
                    if( not par.isEnabled() ) continue;
                    nStripped++;
                }
            }
            // allocate a vector of parameter pointers
            std::vector<Parameter*> strippedParameterList;
            strippedParameterList.reserve( nStripped );
            for( auto& parSet : propagator.getParametersManager().getParameterSetsList() ){
                if( not parSet.isEnabled() ) continue;
                for( auto& par : parSet.getParameterList() ){
                    if( not par.isEnabled() ) continue;
                    strippedParameterList.emplace_back(&par);
                }
            }
            // change the parameter values
            double epsilonNorm=0;
            for( int iPar = 0 ; iPar < nStripped ; iPar++ ) {
                double sigma = strippedParameterList[iPar]->getStdDevValue();
                double epsilon = gRandom->Gaus(0, sigma/(nStripped));
                epsilonNorm += epsilon*epsilon;
                if (iToy==0) epsilon = 0;
                //LogInfo<<strippedParameterList[iPar]->getFullTitle()<<" e: "<<epsilon<<std::endl;
                strippedParameterList[iPar]->setParameterValue(
                        epsilon + getParameterValueFromTextFile(parInjectFile, strippedParameterList[iPar]->getFullTitle())
                );
            }
            epsilonNorm = sqrt(epsilonNorm);
            LogInfo<<"epsilonNorm: "<<epsilonNorm;
            epsilonNormAverage += epsilonNorm;

            // print out the parameter values
            // If is in eigen space, propagateOriginalToEigen
            for (auto &parSet: propagator.getParametersManager().getParameterSetsList()) {
                if (not parSet.isEnabled()) { continue; }
                if (parSet.isEnableEigenDecomp()){
                    parSet.propagateOriginalToEigen();
                }
            }

        }// end if(injectParamsManually)

        // Propagate the parameters and compute the LH
        fitter.getLikelihoodInterface().getModelPropagator().propagateParameters();
        propagated = fitter.getLikelihoodInterface().getModelPropagator().applyParameters();
        fitter.getLikelihoodInterface().evalLikelihood(propagated);
        LLH = fitter.getLikelihoodInterface().getBuffer().totalLikelihood;
        LH_stat = fitter.getLikelihoodInterface().evalStatLikelihood(propagated);
        LH_syst = fitter.getLikelihoodInterface().evalPenaltyLikelihood();
        LLHwrtBestFit = LLH - bestFitLLH;
        LH_statWrtBestFit = LH_stat - bestFit_statLH;
        LH_systWrtBestFit = LH_syst - bestFit_systLH;
        // LogInfo<<"Done.  ";
        LogInfo << iToy << "\t: LH_stat:  " << LH_stat     << "  LH_syst:  " << LH_syst           << "  LH tot:  " << LLH << std::endl;
        LogInfo << "wrtBF\t: LH_stat: (" << LH_statWrtBestFit << ") LH_syst: (" << LH_systWrtBestFit << ") LH tot: (" << LLHwrtBestFit << ")" << std::endl;

        // LogInfo<<"LLH: "<<LLH<<std::endl;
        // make the LH a probability distribution (but still work with the log)
        // This is an approximation, it works only in case of gaussian LH
        // LLH /= -2.0;
        // LLH += nParameters/2.*log(1.0/(2.0*TMath::Pi()));
        // LLH = -LLH;


        LLH_sum += LLH;
        if(iToy==0 and injectParamsManually){
            injectedLLH = LLH;
        }
        if(injectParamsManually)
            LogInfo<<" dLLH: "<<LLH-injectedLLH<<std::endl;

        //LogInfo<<"LLH: "<<LLH;
        gLLH = 0;
        priorSum = 0;

        parameters->clear();
        margThis->clear();
        prior->clear();
        survivingParameterValues.clear();
        int iPar=0;
        for( auto& parSet : propagator.getParametersManager().getParameterSetsList() ) {
            if (not parSet.isEnabled()) { continue; }
//            LogInfo<< parSet.getName()<<std::endl;
            for (auto &par: parSet.getParameterList()) {
                if (not par.isEnabled()) continue;
//                LogInfo<<"  "<<par.getTitle()<<" -> "<<par.getParameterValue()<<std::endl;
                parameters->push_back(par.getParameterValue());
                margThis->push_back(par.isMarginalised());
                prior->push_back(par.getDistanceFromNominal() * par.getDistanceFromNominal());
                priorSum += prior->back();
                gLLH += weightsChiSquare->at(iPar);
                if(not par.isMarginalised())
                    survivingParameterValues.push_back(par.getParameterValue());
                iPar++;
            }
        }
        LhOverGauss = exp(LLH-gLLH);
        if ( LLH-gLLH > log(weightCap)) {
//            LogInfo << "Throw " << iToy << " over weight cap: LLH-gLLH = " << LLH - gLLH << std::endl;
            countBigThrows++;
        }else{
            weightSum += LhOverGauss;
            weightSquareSum += LhOverGauss*LhOverGauss;
        }
        //debug
//        LogInfo<<"LogLH: "<<LLH<<" sampl: "<<gLLH<<" weight: "<<LhOverGauss <<std::endl    ;

        while(weightSum>1.e50){
            weightSum /= 1.e50;
            weightSumE50++;
        }
        while(weightSquareSum>1.e50){
            weightSquareSum /= 1.e50;
            weightSquareSumE50++;
        }
        // Fill the TTrees
        margThrowTree->Fill();
        ThrowsPThetaFormat->Fill();

        // reset the parameters to the best fit values
        for( auto& parSet : propagator.getParametersManager().getParameterSetsList() ){
            if( not parSet.isEnabled() ) continue;
            for( auto& par : parSet.getParameterList() ){
                if( not par.isEnabled() ) continue;
                par.setParameterValue(par.getPriorValue());
            }
        }

    }// end of main throws loop


    LogInfo<<"weight cap: "<<weightCap<<std::endl;
    LogInfo<<"Number of throws with overweight: LLH-gLLH > log(weightCap): "<<countBigThrows<<" - "<<(double)countBigThrows/nToys*100<<" % of total"<<std::endl;
    LogInfo<<"Weight sum: "<<weightSum<<" x (10^50)^"<<weightSumE50<<"  |  weight^2 sum: "<< weightSquareSum<<" x (10^50)^"<<weightSquareSumE50<<std::endl;
    ESS = (weightSumE50*2-weightSquareSumE50) * 1.e50 * weightSum*weightSum/weightSquareSum;
    LogInfo<<"Nb. of throws: "<<nToys<<"\nESS: "<<weightSum*weightSum/weightSquareSum<<" x (10^50)^"<<(weightSumE50*2-weightSquareSumE50)<<std::endl;

    // write ESS in output file
    TVectorD ESS_writable(1); ESS_writable[0] = ESS;
    TVectorD weightSum_writable(1); weightSum_writable[0] = weightSum;
    TVectorD weightSquareSum_writable(1); weightSquareSum_writable[0] = weightSquareSum;
    // I think this is stupid, I cannot write a double???
    app.getOutfilePtr()->WriteObject(&ESS_writable,"ESS");
    app.getOutfilePtr()->WriteObject(&weightSum_writable,"weight_sum");
    app.getOutfilePtr()->WriteObject(&weightSquareSum_writable,"weight2_sum");


    double averageLLH = LLH_sum/nToys;
    epsilonNormAverage /= nToys;
    if(injectParamsManually){
        LogInfo<<"Injected LLH: "<<injectedLLH<<std::endl;
        LogInfo<<"Average  LLH: "<<averageLLH<<std::endl;
        LogInfo<<"Average  epsilonNorm: "<<epsilonNormAverage<<std::endl;
    }

    margThrowTree->Write();
    ThrowsPThetaFormat->Write();

    // Compute the determinant of the covariance matrix
//    double det = 1.0;
//    TMatrixD eigenVectors = (*propagator.getParametersManager().getGlobalCovarianceMatrix());
//    TVectorD eigenValues(parameters.size());
//    eigenVectors.EigenVectors(eigenValues);
//    //LogInfo<<"Eigenvalues: "<<std::endl;
//    for(int i=0;i<eigenValues.GetNrows();i++){
//        det *= pow(eigenValues[i],1./2);
//        //LogInfo<<eigenValues[i]<<" "<<det<<std::endl;
//    }
//    LogInfo<<"SQUARE ROOT OF the determinant of the covariance matrix: "<<det<<std::endl;

    //GundamGlobals::getParallelWorker().reset();
}



double getParameterValueFromTextFile(std::string fileName="LargeWeight_parVector.txt", std::string parameterName="Parameter Flux Systematics/#98") {
    std::ifstream file(fileName);
    std::string line;
    std::string parName;
    double parValue;
    // title and parvalue are separateed by a colon
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string title, value;

        // Splitting the line by ':'
        std::getline(iss, title, ':');
        std::getline(iss, value);


        parValue = std::stod(value);
        if (title == parameterName) {
            return parValue;
        }
    }
    LogInfo << "Parameter \"" << parameterName << "\" not found in file " << fileName << std::endl;
    return -999;
}

