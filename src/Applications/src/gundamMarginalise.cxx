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

double getParameterValueFromTextFile(std::string fileName, std::string parameterName);


int main(int argc, char** argv){

    std::cout<<"Hello world 11.12.23"<<std::endl;

    GundamApp app{"contours marginalisation tool"};

    // --------------------------
    // Read Command Line Args:
    // --------------------------
    CmdLineParser clParser;

    clParser.addDummyOption("Main options:");

    // I need a config file where the list of parameters to marginalise over are specified (look at the XSec config file to get inspired)
    clParser.addOption("configFile", {"-c"}, "Specify the parameters to marginalise over");

    // (I think) I need the output file from a fitter to use as input here
    clParser.addOption("fitterOutputFile", {"-f"}, "Specify the fitter output file");

    // Think carefully what do you need to put in the output file
    // 1. Marginalised covariance matrix: gotten from teh outoutFitter file
    // what else?
    clParser.addOption("outputFile", {"-o", "--out-file"}, "Specify the Marginaliser output file");

    clParser.addOption("nbThreads", {"-t", "--nb-threads"}, "Specify nb of parallel threads");
    clParser.addOption("nToys", {"-n"}, "Specify number of toys");
    clParser.addOption("randomSeed", {"-s", "--seed"}, "Set random seed");
    clParser.addTriggerOption("usingGpu", {"--gpu"}, "Use GPU parallelization");
    clParser.addOption("parInject", {"--parameters-inject"}, "Input txt file for injecting params");

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

    GundamGlobals::getParallelWorker().setNThreads(clParser.getOptionVal("nbThreads", 1));
    LogInfo << "Running the fitter with " << GundamGlobals::getParallelWorker().getNbThreads() << " parallel threads." << std::endl;

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
    // Check if the config is an array (baobab compatibility)

    ConfigUtils::ConfigHandler cHandler{ fitterConfig };

//    // Disabling defined samples:
//    LogInfo << "Removing defined samples..." << std::endl;
//    ConfigUtils::applyOverrides(
//            cHandler.getConfig(),
//            GenericToolbox::Json::readConfigJsonStr(R"({"fitterEngineConfig":{"propagatorConfig":{"fitSampleSetConfig":{"fitSampleList":[]}}}})")
//    );

//    // Disabling defined plots:
//    LogInfo << "Removing defined plots..." << std::endl;
//    ConfigUtils::applyOverrides(
//            cHandler.getConfig(),
//            GenericToolbox::Json::readConfigJsonStr(R"({"fitterEngineConfig":{"propagatorConfig":{"plotGeneratorConfig":{}}}})")
//    );



    // Reading marginaliser config file
    nlohmann::json margConfig{ ConfigUtils::readConfigFile( clParser.getOptionVal<std::string>("configFile") ) };
    if(margConfig.size()==1) {
        // broken json library puts everything in the same level
        // normal behavior
        margConfig = margConfig[0];
    }
    cHandler.override( margConfig );
    LogInfo << "Override done." << std::endl;


        // read the parameters to include in the TTree

    // read the parameters to marginalise over


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

    auto configPropagator = GenericToolbox::Json::fetchValuePath<nlohmann::json>( cHandler.getConfig(), "fitterEngineConfig/propagatorConfig" );

    // Create a propagator object
    Propagator propagator;

//    // Read the whole fitter config with the override parameters
    propagator.readConfig( configPropagator );

    // We are only interested in our MC. Data has already been used to get the post-fit error/values
    propagator.setLoadAsimovData( true );

    // Disabling eigen decomposed parameters
    propagator.setEnableEigenToOrigInPropagate( false );

    // Load everything
    propagator.initialize();

    propagator.getParametersManager().getParameterSetsList();
    for( auto& parSet : propagator.getParametersManager().getParameterSetsList() ){
        if( not parSet.isEnabled() ){ continue; }
        LogInfo <<parSet.getName()<<std::endl;
        for( auto& par : parSet.getParameterList() ){
            if( not par.isEnabled() ){ continue; }
            LogInfo<<par.getTitle()<<" -> "<<par.getParameterValue()<<std::endl;
            par.setPriorValue( par.getParameterValue() );
        }
    }


    std::vector<double> parametersBestFit;
    std::vector<std::string> parameterFullTitles;
    // Load post-fit parameters as "prior" so we can reset the weight to this point when throwing toys
    // also save the values in a vector so we can use them to compute the LLH at the best fit point
    ObjectReader::readObject<TNamed>( fitterFile.get(), "FitterEngine/postFit/parState_TNamed", [&](TNamed* parState_){
        propagator.getParametersManager().injectParameterValues( GenericToolbox::Json::readConfigJsonStr( parState_->GetTitle() ) );
        for( auto& parSet : propagator.getParametersManager().getParameterSetsList() ){
            if( not parSet.isEnabled() ){ continue; }
//            LogInfo<< parSet.getName()<<std::endl;
            for( auto& par : parSet.getParameterList() ){
                if( not par.isEnabled() ){ continue; }
//                LogInfo<<"  "<<par.getTitle()<<" -> "<<par.getParameterValue()<<std::endl;
                parametersBestFit.push_back( par.getParameterValue() );
                parameterFullTitles.push_back( par.getFullTitle() );
                par.setPriorValue(par.getParameterValue());
            }
        }
    });
    // also save the value of the LLH at the best fit point:
    propagator.propagateParametersOnSamples();
    propagator.updateLlhCache();
    double bestFitLLH = propagator.getLlhBuffer();
    LogInfo<<"LLH (computed on injected parameters): "<<bestFitLLH<<std::endl;

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
        outFilePath = GundamUtils::generateFileName(clParser, appendixDict) + ".root";

        std::string outFolder{GenericToolbox::Json::fetchValue<std::string>(margConfig, "outputFolder", "./")};
        outFilePath = GenericToolbox::joinPath(outFolder, outFilePath);
    }

    // Load the post-fit covariance matrix
    ObjectReader::readObject<TH2D>(
            fitterFile.get(), "FitterEngine/postFit/Hesse/hessian/postfitCovarianceOriginal_TH2D",
            [&](TH2D* hCovPostFit_){
                propagator.getParametersManager().setGlobalCovarianceMatrix(std::make_shared<TMatrixD>(hCovPostFit_->GetNbinsX(), hCovPostFit_->GetNbinsX()));
                for( int iBin = 0 ; iBin < hCovPostFit_->GetNbinsX() ; iBin++ ){
                    for( int jBin = 0 ; jBin < hCovPostFit_->GetNbinsX() ; jBin++ ){
                        (*propagator.getParametersManager().getGlobalCovarianceMatrix())[iBin][jBin] = hCovPostFit_->GetBinContent(1 + iBin, 1 + jBin);
                    }
                }
            });


//    // Sample binning using parameterSetName
//    for( auto& sample : propagator.getFitSampleSet().getFitSampleList() ){
//        auto associatedParSet = GenericToolbox::Json::fetchValue<std::string>(sample.getConfig(), "parameterSetName");
//
//        // Looking for parSet
//        auto foundDialCollection = std::find_if(
//                propagator.getDialCollections().begin(), propagator.getDialCollections().end(),
//                [&](const DialCollection& dialCollection_){
//                    auto* parSetPtr{dialCollection_.getSupervisedParameterSet()};
//                    if( parSetPtr == nullptr ){ return false; }
//                    return ( parSetPtr->getName() == associatedParSet );
//                });
//        LogThrowIf(
//                foundDialCollection == propagator.getDialCollections().end(),
//                "Could not find " << associatedParSet << " among fit dial collections: "
//                                  << GenericToolbox::iterableToString(propagator.getDialCollections(), [](const DialCollection& dialCollection_){
//                                                                          return dialCollection_.getTitle();
//                                                                      }
//                                  ));
//
//        LogThrowIf(foundDialCollection->getDialBinSet().isEmpty(), "Could not find binning");
//        sample.setBinningFilePath( foundDialCollection->getDialBinSet().getFilePath() );
//    }





    app.setCmdLinePtr( &clParser );
    app.setConfigString( ConfigUtils::ConfigHandler{margConfig}.toString() );
    app.openOutputFile( outFilePath );
    app.writeAppInfo();

    // write the post fit covariance matrix in the output file
    TDirectory* postFitInfo = app.getOutfilePtr()->mkdir("postFitInfo");
    postFitInfo->cd();
    ObjectReader::readObject<TH2D>(
            fitterFile.get(), "FitterEngine/postFit/Hesse/hessian/postfitCovarianceOriginal_TH2D",
            [&](TH2D* hCovPostFit_) {
                hCovPostFit_->SetName("postFitCovarianceOriginal_TH2D");
                hCovPostFit_->Write();// save the TH2D cov. matrix also in the output file
            });
    // write the best fit parameters vector to the output file
    // TODO: implement using TH1D: (FitterEngine/postFit/Migrad/errors/<parListTitle>/values/postFitErrors_TH1D)
    ObjectReader::readObject<TNamed>( fitterFile.get(), "FitterEngine/postFit/parState_TNamed", [&](TNamed* parState_){
        parState_->SetName("postFitParameters_TNamed");
        parState_->Write();
    });
    TVectorD bestFitParameters (parametersBestFit.size(), parametersBestFit.data());
    bestFitParameters.Write("bestFitParameters_TVectorD");
    app.getOutfilePtr()->WriteObject(&parameterFullTitles, "parameterFullTitles");
    app.getOutfilePtr()->cd();


    auto* marginalisationDir{ GenericToolbox::mkdirTFile(app.getOutfilePtr(), "marginalisation") };
    LogInfo << "Creating throws tree" << std::endl;
    auto* margThrowTree = new TTree("margThrow", "margThrow");
    margThrowTree->SetDirectory( GenericToolbox::mkdirTFile(marginalisationDir, "throws") ); // temp saves will be done here
    // make a TTree with the following branches, to be filled with the throws
    // 1. marginalised parameters drew: vector<double>
    // 2. non-marginalised parameters drew: vector<double>
    // 3. LLH value: double
    // 4. "g" value: the chi square value as extracted from the covariance matrix: double
    // 5. value of the priors for the marginalised parameters: vector<double>

    std::vector<double> parameters;
    std::vector<bool> margThis;
    std::vector<double> prior;
    std::vector<double> weightsChiSquare;
    std::vector<double> dialResponse;
//    weightsChiSquare.reserve(1000);
    double LLH, gLLH, priorSum, LLHwrtBestFit;
    margThrowTree->Branch("Parameters", &parameters);
    margThrowTree->Branch("Marginalise", &margThis);
    margThrowTree->Branch("prior", &prior);
    margThrowTree->Branch("LLH", &LLH);
    margThrowTree->Branch("LLHwrtBestFit", &LLHwrtBestFit);
    margThrowTree->Branch("gLLH", &gLLH);
    margThrowTree->Branch("priorSum", &priorSum);
    margThrowTree->Branch("weightsChiSquare", &weightsChiSquare);
    margThrowTree->Branch("dialResponse", &dialResponse);


    int nToys{ clParser.getOptionVal<int>("nToys") };

    // Get parameters to be marginalised
    std::vector<std::string> marginalisedParameters;
    std::vector<std::string> marginalisedParameterSets;
        marginalisedParameters = GenericToolbox::Json::fetchValue<std::vector<std::string>>(margConfig, "parameterList");
        marginalisedParameterSets = GenericToolbox::Json::fetchValue<std::vector<std::string>>(margConfig, "parameterSetList");

    LogInfo<<"Marginalised parameters: "<<GenericToolbox::parseVectorAsString(marginalisedParameters,true,true)<<std::endl;
    LogInfo<<"Marginalised parameter Sets: "<<GenericToolbox::parseVectorAsString(marginalisedParameterSets,true,true)<<std::endl;
    // loop over parameters and compare their titles with the ones in the marginalisedParameters string
    // if they match, set the corresponding margThis to true
    // Also display the prior information
    LogInfo<<"Prior information: "<<std::endl;
    for( auto& parSet : propagator.getParametersManager().getParameterSetsList() ) {
        if (not parSet.isEnabled()) {
            LogInfo << "Parameter set " << parSet.getName() << " is disabled" << std::endl;
            continue;
        } else {
            bool setMatches = false;
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
                continue; // skip the single params thing and go to the next ParameterSet
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
            par.setMarginalised(matches);
            if(par.isMarginalised()){
                LogInfo << "Parameter " << par.getFullTitle() << " will be marginalized out.   \n";
            }else{
                LogInfo << "Parameter " << par.getFullTitle() << " will not be marginalized out.   \n";
            }
        }
    }

//    LogInfo << par.getFullTitle() << " -> type: " << par.getPriorType() << " mu=" << par.getPriorValue()
//            << " sigma= " << par.getStdDevValue() << " limits: " << par.getMinValue() << " - "
//            << par.getMaxValue() << " limits (phys): " << par.getMinPhysical() << " - "
//            << par.getMaxPhysical() << " limits (mirr): " << par.getMinMirror() << " - "
//            << par.getMaxMirror() <<" --- marg? "<<par.isMarginalised() << std::endl;


    //////////////////////////////////////
    // THROWS LOOP
    /////////////////////////////////////
    std::stringstream ss; ss << LogWarning.getPrefixString() << "Generating " << nToys << " toys...";




    for( int iToy = 0 ; iToy < nToys ; iToy++ ){

        // loading...
        GenericToolbox::displayProgressBar( iToy+1, nToys, ss.str() );

        // reset weights vector
        weightsChiSquare.clear();
        // Do the throwing:
        //propagator.getParametersManager().throwParametersFromGlobalCovariance(weightsChiSquare);
        //propagator.propagateParametersOnSamples(); // Not necessary (included in updateLlhCache())

        if(injectParamsManually) {
            LogInfo<< "Injecting parameters from file: " << parInjectFile << std::endl;

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
            std::vector<Parameter*> strippedParameterList.reserve( nStripped );
            for( auto& parSet : propagator.getParametersManager().getParameterSetsList() ){
                if( not parSet.isEnabled() ) continue;
                for( auto& par : parSet.getParameterList() ){
                    if( not par.isEnabled() ) continue;
                    strippedParameterList.emplace_back(&par);
                }
            }
            // change the parameter values
            for( int iPar = 0 ; iPar < nStripped ; iPar++ ) {
                strippedParameterList[iPar]->setParameterValue(
                        getParameterValueFromTextFile(parInjectFile, strippedParameterList[iPar]->getFullTitle())
                );
            }
            // If is in eigen space, propagateOriginalToEigen
            for (auto &parSet: propagator.getParametersManager().getParameterSetsList()) {
                if (not parSet.isEnabled()) { continue; }
                if (parSet.isUseEigenDecompInFit()){
                    parSet.propagateEigenToOriginal();
                }
            }

        }// end if(injectParamsManually)

        LogInfo<<"Computing LH..."<<std::endl;
        propagator.updateLlhCache();
        LLH = propagator.getLlhBuffer();
        LogInfo<<"LLH: "<<LLH<<std::endl;
        LLHwrtBestFit = LLH - bestFitLLH;
        gLLH = 0;
        priorSum = 0;

        parameters.clear();
        margThis.clear();
        prior.clear();
        int iPar=0;
        for( auto& parSet : propagator.getParametersManager().getParameterSetsList() ) {
            if (not parSet.isEnabled()) { continue; }
//            LogInfo<< parSet.getName()<<std::endl;
            for (auto &par: parSet.getParameterList()) {
                if (not par.isEnabled()) { continue; }
//                LogInfo<<"  "<<par.getTitle()<<" -> "<<par.getParameterValue()<<std::endl;
                parameters.push_back(par.getParameterValue());
                margThis.push_back(par.isMarginalised());
                prior.push_back(par.getDistanceFromNominal() * par.getDistanceFromNominal());
                priorSum += prior.back();
                gLLH += weightsChiSquare[iPar];
                //LogInfo<<iPar<<": "<<weightsChiSquare[iPar]<<std::endl;
                iPar++;
            }
        }

//        LogInfo<<"LLH: "<<LLH/2.<<"  gLLH: "<<gLLH/2.<<"\tdifference: "<<(-LLH/2.+gLLH/2.)  <<std::endl;
//        if((-LLH+gLLH)/2. > 10){
//            LogInfo<<"WARNING: BIG THROW!!"<<std::endl;
//        }

        //LogInfo<<"->   gLLH: "<<gLLH<<std::endl;
        //LogDebugIf(gLLH<50)<<gLLH<<std::endl;
        // print the parameters

//        for(int iPar=0;iPar<propagator.getParameterSetPtr().size();iPar++){
//            propagator.getFitParameterSetPtr("all")->getParameterList().at(0).getParameterValue();
//
//        }


        // for now set it to false. Still need to understand/implement this
        bool enableStatThrowInToys{false};
//        if( enableStatThrowInToys ){
//            for( auto& xsec : crossSectionDataList ){
//                if( enableEventMcThrow ){
//                    // Take into account the finite amount of event in MC
//                    xsec.samplePtr->getMcContainer().throwEventMcError();
//                }
//                // Asimov bin content -> toy data
//                xsec.samplePtr->getMcContainer().throwStatError();
//            }
//        }

        // i don't know what is this for, so I comment it for now
//        writeBinDataFct();

        // Write the branches
        margThrowTree->Fill();

        //debug
//        if(iToy==0 || iToy==1){
//            LogInfo<<"DEBUG: Toy "<<iToy<<std::endl;
//            for(int i=0;i<parameters.size();i++){
//                LogInfo<<i<<": "<<parameters[i]<<" "<<margThis[i]<<std::endl;
//            }
//        }
    }

    margThrowTree->Write();

    double det = 1.0;
    TMatrixD eigenVectors = (*propagator.getParametersManager().getGlobalCovarianceMatrix());
    TVectorD eigenValues(parameters.size());
    eigenVectors.EigenVectors(eigenValues);
    //LogInfo<<"Eigenvalues: "<<std::endl;
    for(int i=0;i<eigenValues.GetNrows();i++){
        det *= pow(eigenValues[i],1./2);
        //LogInfo<<eigenValues[i]<<" "<<det<<std::endl;
    }
    LogInfo<<"SQUARE ROOT OF the determinant of the covariance matrix: "<<det<<std::endl;


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