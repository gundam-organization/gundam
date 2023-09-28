//
// Created by Lorenzo Giannessi on 05.09.23.
//

#include "GundamGlobals.h"
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

    GundamApp app{"contours marginalisation tool"};

    // --------------------------
    // Read Command Line Args:
    // --------------------------
    CmdLineParser clParser;

    clParser.addDummyOption("Main options:");

    // I need a config file where the list of parameters to marginalise over are specified (look at the XSec config file to get inspired)
    // TODO: understand the format/syntax of this file. This should be a required file.
    clParser.addOption("configFile", {"-c"}, "Specify the parameters to marginalise over");

    // (I think) I need the output file from a fitter to use as input here
    clParser.addOption("fitterOutputFile", {"-f"}, "Specify the fitter output file");

    // Think carefully what do you need to put in the output file
    // 1. Marginalised covariance matrix
    // what else?
    clParser.addOption("outputFile", {"-o", "--out-file"}, "Specify the Marginaliser output file");

    clParser.addOption("nbThreads", {"-t", "--nb-threads"}, "Specify nb of parallel threads");
    clParser.addOption("nToys", {"-n"}, "Specify number of toys");
    clParser.addOption("randomSeed", {"-s", "--seed"}, "Set random seed");

    clParser.addDummyOption("Trigger options:");
    clParser.addTriggerOption("dryRun", {"-d", "--dry-run"}, "Only overrides fitter config and print it.");

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

    GundamGlobals::setNbThreads(clParser.getOptionVal("nbThreads", 1));
    LogInfo << "Running the fitter with " << GundamGlobals::getNbThreads() << " parallel threads." << std::endl;

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
    nlohmann::json margConfig{ ConfigUtils::readConfigFile( clParser.getOptionVal<std::string>("configFile") ) };
    cHandler.override( margConfig );
    LogInfo << "Override done." << std::endl;

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
        propagator.injectParameterValues( GenericToolbox::Json::readConfigJsonStr( parState_->GetTitle() ) );
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
    else {
        // appendixDict["optionName"] = "Appendix"
        // this list insure all appendices will appear in the same order
        std::vector<std::pair<std::string, std::string>> appendixDict{
                {"configFile",       "%s"},
                {"fitterOutputFile", "Fit_%s"},
                {"nToys",            "nToys_%s"},
                {"randomSeed",       "Seed_%s"},
        };
        outFilePath = "marg_" + GundamUtils::generateFileName(clParser, appendixDict) + ".root";

        std::string outFolder{GenericToolbox::Json::fetchValue<std::string>(margConfig, "outputFolder", "./")};
        outFilePath = GenericToolbox::joinPath(outFolder, outFilePath);
    }

    auto* marginalisationDir{ GenericToolbox::mkdirTFile(app.getOutfilePtr(), "marginalisation") };
    LogInfo << "Creating throws tree" << std::endl;
    auto* margThrowTree = new TTree("margThrow", "margThrow");
    margThrowTree->SetDirectory( GenericToolbox::mkdirTFile(marginalisationDir, "throws") ); // temp saves will be done here


    int nToys{ clParser.getOptionVal<int>("nToys") };


    //////////////////////////////////////
    // THROWS LOOP
    /////////////////////////////////////
    std::stringstream ss; ss << LogWarning.getPrefixString() << "Generating " << nToys << " toys...";
    for( int iToy = 0 ; iToy < nToys ; iToy++ ){

        // loading...
        GenericToolbox::displayProgressBar( iToy+1, nToys, ss.str() );

        // Do the throwing:
        // TODO: ask Adrien: when I do the throwing at this stage, the parameters to "be thrown" are already set? should I already set before the parameters to marginalise over?
        propagator.throwParametersFromGlobalCovariance();
        //propagator.propagateParametersOnSamples();
        propagator.updateLlhCache();
        double LLH_value = propagator.getLlhBuffer();
        // get the prior covariance matrix for a subset of parameters (to marginalise over them)


        // for now set it to false. Still need to understand/implement this
        bool enableStatThrowInToys{false};
        if( enableStatThrowInToys ){
//            for( auto& xsec : crossSectionDataList ){
//                if( enableEventMcThrow ){
//                    // Take into account the finite amount of event in MC
//                    xsec.samplePtr->getMcContainer().throwEventMcError();
//                }
//                // Asimov bin content -> toy data
//                xsec.samplePtr->getMcContainer().throwStatError();
//            }
        }

        // i don't know what is this for, so I comment it for now
//        writeBinDataFct();

        // Write the branches
        margThrowTree->Fill();
    }


    GundamGlobals::getParallelWorker().reset();
}
