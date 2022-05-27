#include <string>
#include <iostream>
#include <sstream>

#include <TFile.h>
#include <TTree.h>

#include <Logger.h>

#include "AnaFitParameters.h"
#include "AnaSample.hh"
#include "AnaTreeMC.hh"

#include "DetParameters.h"
#include "FluxParameters.h"
#include "OptParser.hh"
#include "ND280Fitter.h"
#include "XsecParameters.h"

#include "GlobalVariables.h"

#include <GenericToolbox.h>
#include <GenericToolbox.Root.h>


//! Global Variables
bool __isDryRun__           = false;
bool __skipOneSigmaChecks__ = false;

int __PRNG_seed__ = -1;
std::string __jsonConfigPath__;

std::string __commandLine__;
std::string __outFilePath__;
int __argc__;
char **__argv__;

// RAM monitoring
size_t __ramBaseline__ = 0;
size_t __ramStep__ = 0;
std::vector<std::string> __memoryMonitorList__;



//! Local Functions
std::string remindUsage();
void resetParameters();
void getUserParameters();

void monitorRAMPoint(std::string pointTitle_);

LoggerInit([]{
  Logger::setUserHeaderStr("[xsllhND280UpFit.cxx]");
} );

int main(int argc, char** argv)
{
    // RAM Monitoring
    __ramBaseline__ = GenericToolbox::getProcessMemoryUsage();
    __memoryMonitorList__.emplace_back("RAM baseline: "+GenericToolbox::parseSizeUnits(__ramBaseline__));

    /////////////////////////////
    // Init
    ////////////////////////////

    __argc__ = argc;
    __argv__ = argv;

    resetParameters();
    getUserParameters();
    remindUsage(); // display used parameters

    OptParser options_parser;
    LogInfo << "Reading json parameter files..." << std::endl;
    if(!options_parser.ParseJSON(__jsonConfigPath__))
    {
        LogError << "JSON parsing failed. Exiting." << std::endl;
        exit(EXIT_FAILURE);
    }
    options_parser.PrintOptions();

    // Setup data trees
    std::string data_tfile_path = options_parser.fname_data;
    TTree* data_ttree = nullptr;
    if(not data_tfile_path.empty()){
        LogInfo << "Opening data file: " << data_tfile_path << std::endl;
        if(not GenericToolbox::doesTFileIsValid(data_tfile_path)){
            LogError << data_tfile_path << " can't be opened." << std::endl;
            exit(EXIT_FAILURE);
        }
        TFile* data_tfile = TFile::Open(data_tfile_path.c_str(), "READ");
        data_ttree  = (TTree*)(data_tfile->Get("selectedEvents"));
    }


    // Setup output tfile
    if(__outFilePath__.empty()){
        __outFilePath__ = __jsonConfigPath__ + ".root";
    }

    LogInfo << "Opening output file: " << __outFilePath__ << std::endl;
    TFile* output_tfile  = TFile::Open(__outFilePath__.c_str(), "RECREATE");

    monitorRAMPoint("RAM taken by initialization: ");


    /////////////////////////////
    // SAMPLES DEFINITION
    ////////////////////////////

    // Add analysis samples:
    std::vector<AnaSample*> analysisSampleList;
    LogInfo << "Add analysis samples..." << std::endl;
    for(const auto& sample : options_parser.samples) {
        if(sample.use_sample and sample.cut_branch >= 0)
        {
            LogWarning << "Adding new sample to fit." << std::endl
                       << "Name: " << sample.name << std::endl
                       << "Detector: " << sample.detector << std::endl
                       << "Phase Space: " << GenericToolbox::parseVectorAsString(sample.fit_phase_space) << std::endl
                       << "Binning file: " << sample.binning << std::endl
                       << "CutB: " << sample.cut_branch << std::endl
                       << "Use Sample: " << std::boolalpha << sample.use_sample << std::endl;

            auto* analysisSample = new AnaSample(sample, data_ttree);

            analysisSample->SetLLHFunction(options_parser.min_settings.likelihood);
            analysisSampleList.emplace_back(analysisSample);
        }
    }

    //read MC events
    std::string mc_file_path = options_parser.fname_mc;
    AnaTreeMC selected_events_AnaTreeMC(mc_file_path, "selectedEvents");
    LogInfo << "Reading and collecting events..." << std::endl;
    selected_events_AnaTreeMC.GetEvents(analysisSampleList, options_parser.signal_definition, false);

//    LogInfo << "Getting sample breakdown by topology..." << std::endl;
//    std::vector<std::string> sampleTopologyList = options_parser.sample_topology;
//    std::vector<int> topology_HL_codes = options_parser.topology_HL_code;
//    // Mapping the Highland topology codes to consecutive integers and then getting the topology breakdown for each sample:
//    for(auto& analysis_sample : analysisSampleList) {
//        analysis_sample->SetTopologyHLCode(topology_HL_codes);
//        analysis_sample->GetSampleBreakdown(output_tfile, "nominal", sampleTopologyList, false);
//    }

    monitorRAMPoint("RAM taken by samples definition");


    /////////////////////////////
    // SYSTEMATICS DEFINITION
    ////////////////////////////

    //define fit param classes
    std::vector<AnaFitParameters*> fitSystematicList;

    //////////////////////
    // Flux parameters:
    LogInfo << "Reading flux parameters..." << std::endl;
    FluxParameters flux_parameters("Flux Systematics");
    if( options_parser.flux_cov.do_fit ) {
        LogWarning << "Setup Flux Covariance." << std::endl
                   << "Opening " << options_parser.flux_cov.fname << " for flux covariance."
                  << std::endl;

        if(not GenericToolbox::doesTFileIsValid(options_parser.flux_cov.fname)){
            LogError << options_parser.flux_cov.fname << " can't be opened." << std::endl;
            exit(EXIT_FAILURE);
        }
        TFile* file_flux_cov = TFile::Open(options_parser.flux_cov.fname.c_str(), "READ");

        auto* cov_flux = (TMatrixDSym*) file_flux_cov -> Get(options_parser.flux_cov.matrix.c_str());
        if(cov_flux == nullptr){
            LogError << options_parser.flux_cov.fname << ": " << options_parser.flux_cov.matrix << " can't be opened." << std::endl;
            exit(EXIT_FAILURE);
        }
        file_flux_cov -> Close();

//        if(options_parser.flux_cov.rng_start)
//            flux_parameters.SetRNGstart();

        flux_parameters.SetCovarianceMatrix(*cov_flux, options_parser.flux_cov.decompose);
        flux_parameters.SetThrow(options_parser.flux_cov.do_throw);
        flux_parameters.SetInfoFrac(options_parser.flux_cov.info_frac);
        for(const auto& detector : options_parser.detectors)
        {
            if(detector.use_detector)
                flux_parameters.AddDetector(detector.name, options_parser.flux_cov.binning);
        }
        flux_parameters.InitEventMap(analysisSampleList, 0);
        fitSystematicList.emplace_back(&flux_parameters);
    }

    monitorRAMPoint("RAM taken by flux systematics");


    //////////////////////
    // Cross-section parameters:
    LogInfo << "Reading Cross-section parameters..." << std::endl;
    XsecParameters xsec_parameters("Cross-Section Systematics");
    if(options_parser.xsec_cov.do_fit)
    {
        LogWarning << "Setup Xsec Covariance." << std::endl
                   << "Opening " << options_parser.xsec_cov.fname << " for xsec covariance."
                  << std::endl;

        if(not GenericToolbox::doesTFileIsValid(options_parser.xsec_cov.fname)){
            LogError << options_parser.xsec_cov.fname << " can't be opened." << std::endl;
            exit(EXIT_FAILURE);
        }


        if(options_parser.xsec_cov.rng_start)
            xsec_parameters.SetRNGstart();
        xsec_parameters.SetThrow(options_parser.xsec_cov.do_throw);

        TFile* file_xsec_cov = TFile::Open(options_parser.xsec_cov.fname.c_str(), "READ");
        auto* cov_xsec = (TMatrixDSym*) file_xsec_cov -> Get(options_parser.xsec_cov.matrix.c_str());
        auto* xsec_param_names = dynamic_cast<TObjArray *>(file_xsec_cov->Get("xsec_param_names"));
        if(cov_xsec == nullptr){
            LogError << options_parser.xsec_cov.fname << ": " << options_parser.xsec_cov.matrix << " can't be opened." << std::endl;
            exit(EXIT_FAILURE);
        }
        file_xsec_cov -> Close();

        std::vector<int> keptCovIndices;
        for(const auto& detector : options_parser.detectors)
        {
            if(detector.use_detector){
                xsec_parameters.AddDetector(detector.name, detector.xsec);
                auto dialsList = xsec_parameters.GetDetectorDials(detector.name);
                for(const auto& dial : dialsList){
                    for(int iSpline = 0 ; iSpline < xsec_param_names->GetEntries() ; iSpline++ ){
                        std::string xsecSplineName = xsec_param_names->At(iSpline)->GetName();
                        if(xsecSplineName == dial.GetName()){
                            keptCovIndices.emplace_back(iSpline);
                        }
                    }
                }
                break; // 1 detector only for the moment
            }

        }
        auto* fitCovMatrix = new TMatrixD(keptCovIndices.size(), keptCovIndices.size());
        for( int iLine = 0 ; iLine < int(keptCovIndices.size()) ; iLine++ ){
            for( int iCol = 0 ; iCol < int(keptCovIndices.size()) ; iCol++ ){
                (*fitCovMatrix)[iLine][iCol] = (*cov_xsec)[keptCovIndices[iLine]][keptCovIndices[iCol]];
            }
        }
        xsec_parameters.SetCovarianceMatrix(*GenericToolbox::convertToSymmetricMatrix(fitCovMatrix), options_parser.xsec_cov.decompose);
        monitorRAMPoint("RAM taken by Cross-section systematics (init)");

        xsec_parameters.InitEventMap(analysisSampleList, 0);
        monitorRAMPoint("RAM taken by Cross-section systematics (event map)");

        fitSystematicList.emplace_back(&xsec_parameters);
    }

    monitorRAMPoint("RAM taken by Cross-section systematics (rest)");


    // Detector parameters:
    DetParameters detpara("Detector Systematics");
    if(options_parser.det_cov.do_fit)
    {
        LogWarning << "Setup Detector Covariance." << std::endl
                   << "Opening " << options_parser.det_cov.fname << " for detector covariance."
                  << std::endl;

        TFile* file_det_cov = TFile::Open(options_parser.det_cov.fname.c_str(), "READ");
        if(file_det_cov == nullptr)
        {
            LogError << "Could not open file! Exiting." << std::endl;
            return 1;
        }
        auto* cov_det = (TMatrixDSym*)file_det_cov -> Get(options_parser.det_cov.matrix.c_str());
        file_det_cov -> Close();

        if(options_parser.det_cov.rng_start)
            detpara.SetRNGstart();

        detpara.SetCovarianceMatrix(*cov_det, options_parser.det_cov.decompose);
        detpara.SetThrow(options_parser.det_cov.do_throw);
        detpara.SetInfoFrac(options_parser.det_cov.info_frac);
        for(const auto& detector : options_parser.detectors)
        {
            if(detector.use_detector)
                detpara.AddDetector(detector.name, analysisSampleList, true);
        }
        detpara.InitEventMap(analysisSampleList, 0);
        fitSystematicList.emplace_back(&detpara);
    }

    monitorRAMPoint("RAM taken by Detector systematics");

    if(__PRNG_seed__ == -1)
        __PRNG_seed__ = options_parser.rng_seed;

    GenericToolbox::mkdirTFile(output_tfile, "fitter");

    //Instantiate fitter obj
    ND280Fitter fitter;

    fitter.SetOutputTDirectory(output_tfile->GetDirectory("fitter"));
    fitter.SetPrngSeed(__PRNG_seed__);

    fitter.SetAnaFitParametersList(fitSystematicList);
    fitter.SetAnaSamplesList(analysisSampleList);
    fitter.SetSelectedDataType(options_parser.fit_type);
    fitter.SetApplyStatisticalFluctuationsOnSamples(options_parser.stat_fluc);

    fitter.SetMinimizationSettings(options_parser.min_settings);
    fitter.SetDisableSystFit(options_parser.zero_syst);

    xsec_parameters.SetEnableZeroWeightFenceGate(true);
    fitter.initialize();
    fitter.WritePrefitData();
    if(not __skipOneSigmaChecks__) fitter.MakeOneSigmaChecks();
    xsec_parameters.SetEnableZeroWeightFenceGate(false);

    monitorRAMPoint("RAM taken by the fitter");

    bool did_converge = false;
    if(not __isDryRun__) {

        // Do the Fit
        did_converge = fitter.Fit();

        if(not did_converge)
            LogError    << "Fit did not coverge." << std::endl;
        else
            LogWarning  << "Fit has converged." << std::endl;

        fitter.WritePostFitData();

    }
    else{
        LogWarning << "Dry run is enabled. The fit is ignored." << std::endl;
    }
    output_tfile -> Close();

    for(const auto& monitorLine : __memoryMonitorList__){
        LogDebug << monitorLine << std::endl;
    }

    LogInfo << "Output file: " << __outFilePath__ << std::endl;

    // Print Arigatou Gozaimashita with Rainbowtext :)
    LogInfo << color::RainbowText("\u3042\u308a\u304c\u3068\u3046\u3054\u3056\u3044\u307e\u3057\u305f\uff01")
              << std::endl;

    return did_converge ? EXIT_SUCCESS : 121;
}


std::string remindUsage()
{

    std::stringstream remind_usage_ss;
    remind_usage_ss << "*********************************************************" << std::endl;
    remind_usage_ss << " > Command Line Arguments" << std::endl;
    remind_usage_ss << "  -j : JSON input (Current : " << __jsonConfigPath__ << ")" << std::endl;
    remind_usage_ss << "  -o : Override output file path (Current : " << __outFilePath__ << ")" << std::endl;
    remind_usage_ss << "  -t : Override number of threads (Current : " << GlobalVariables::getNbThreads() << ")" << std::endl;
    remind_usage_ss << "  -d : Enable dry run (Current : " << __isDryRun__ << ")" << std::endl;
    remind_usage_ss << "  --skip-one-sigma-checks : Skip one sigma checks (Current : " << __skipOneSigmaChecks__ << ")" << std::endl;
    remind_usage_ss << "*********************************************************" << std::endl;

    LogInfo << remind_usage_ss.str();
    return remind_usage_ss.str();

}
void resetParameters()
{
    __jsonConfigPath__   = "";
    __outFilePath__      = "";
}
void getUserParameters(){

    if(__argc__ == 1){
        remindUsage();
        exit(EXIT_FAILURE);
    }

    LogWarning << "Sanity check" << std::endl;

    const std::string XSLLHFITTER = std::getenv("XSLLHFITTER")? std::getenv("XSLLHFITTER"): "";;
    if(XSLLHFITTER.empty())
    {
        LogError << "Environment variable \"XSLLHFITTER\" not set." << std::endl
                  << "Cannot determine source tree location." << std::endl;
        remindUsage();
        exit(EXIT_FAILURE);
    }

    LogWarning << "Reading user parameters" << std::endl;

    for(int i_arg = 0; i_arg < __argc__; i_arg++){
        __commandLine__ += __argv__[i_arg];
        __commandLine__ += " ";
    }

    for(int i_arg = 0; i_arg < __argc__; i_arg++){

        if     (std::string(__argv__[i_arg]) == "-j"){
            if (i_arg < __argc__ - 1) {
                int j_arg = i_arg + 1;
                __jsonConfigPath__ = std::string(__argv__[j_arg]);
            }
            else {
                LogError << "Give an argument after " << __argv__[i_arg] << std::endl;
                throw std::logic_error(std::string(__argv__[i_arg]) + " : no argument found");
            }
        }
        else if(std::string(__argv__[i_arg]) == "-o"){
            if (i_arg < __argc__ - 1) {
                int j_arg = i_arg + 1;
                __outFilePath__ = std::string(__argv__[j_arg]);
            }
            else {
                LogError << "Give an argument after " << __argv__[i_arg] << std::endl;
                throw std::logic_error(std::string(__argv__[i_arg]) + " : no argument found");
            }
        }
        else if(std::string(__argv__[i_arg]) == "-t"){
            if (i_arg < __argc__ - 1) {
                int j_arg = i_arg + 1;
                GlobalVariables::setNbThreads(std::stoi(__argv__[j_arg]));
            }
            else {
                LogError << "Give an argument after " << __argv__[i_arg] << std::endl;
                throw std::logic_error(std::string(__argv__[i_arg]) + " : no argument found");
            }
        }
        else if(std::string(__argv__[i_arg]) == "-d"){
            __isDryRun__ = true;
        }
        else if(std::string(__argv__[i_arg]) == "--skip-one-sigma-checks"){
            __skipOneSigmaChecks__ = true;
        }

    }

}

void monitorRAMPoint(std::string pointTitle_){
    if(__ramStep__ == 0) __ramStep__ = __ramBaseline__;
    size_t ramMonitorPoint = GenericToolbox::getProcessMemoryUsage();
    __memoryMonitorList__.emplace_back(
        pointTitle_ + ": "
        + GenericToolbox::parseSizeUnits(ramMonitorPoint - __ramStep__)
        + " - Total: " + GenericToolbox::parseSizeUnits(ramMonitorPoint)
        );
    LogDebug << __memoryMonitorList__.back() << std::endl;
    __ramStep__ = ramMonitorPoint;
}