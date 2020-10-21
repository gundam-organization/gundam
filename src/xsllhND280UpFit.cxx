#include <string>
#include <iostream>
#include <sstream>

#include <TFile.h>
#include <TTree.h>

#include <Logger.h>

#include <AnaFitParameters.hh>
#include <AnaSample.hh>
#include <AnaTreeMC.hh>
#include <DetParameters.hh>
#include <FitParameters.hh>
#include <FluxParameters.hh>
#include <LocalGenericToolbox.h>
#include <OptParser.hh>
#include <ND280Fitter.hh>
#include <XsecParameters.hh>


//! Global Variables
bool __is_dry_run__ = false;
int __nb_threads__ = -1;
int __PRNG_seed__ = -1;
std::string __json_config_path__;

std::string __commandLine__;
std::string __output_file_path__;
int __argc__;
char **__argv__;




//! Local Functions
std::string remindUsage();
void resetParameters();
void getUserParameters();

int main(int argc, char** argv)
{
    Logger::setUserHeaderStr("[xsllhND280UpFit.cxx]");
    size_t ramBaseline = GenericToolbox::getProcessMemoryUsage();
    size_t ramStep = ramBaseline;
    size_t ramTemp = ramBaseline;
    LogDebug << "RAM baseline: " << GenericToolbox::parseSizeUnits(ramBaseline) << std::endl;

    __argc__ = argc;
    __argv__ = argv;

    resetParameters();
    getUserParameters();
    remindUsage(); // display used parameters


    OptParser options_parser;
    LogInfo << "Reading json parameter files..." << std::endl;
    if(!options_parser.ParseJSON(__json_config_path__))
    {
        LogError << "JSON parsing failed. Exiting." << std::endl;
        exit(EXIT_FAILURE);
    }
    options_parser.PrintOptions();

    // Setup data trees
    std::string data_tfile_path = options_parser.fname_data;
    LogInfo << "Opening data file: " << data_tfile_path << std::endl;
    if(not LocalGenericToolbox::do_tfile_is_valid(data_tfile_path)){
        LogError << data_tfile_path << " can't be opened." << std::endl;
        exit(EXIT_FAILURE);
    }
    TFile* data_tfile = TFile::Open(data_tfile_path.c_str(), "READ");
    auto* data_ttree  = (TTree*)(data_tfile->Get("selectedEvents"));

    // Setup output tfile
    if(__output_file_path__.empty()) __output_file_path__ = options_parser.fname_output;
    LogInfo << "Opening output file: " << __output_file_path__ << std::endl;
    TFile* output_tfile          = TFile::Open(__output_file_path__.c_str(), "RECREATE");

    ramTemp = GenericToolbox::getProcessMemoryUsage();
    LogDebug << "RAM taken by initialization: " << GenericToolbox::parseSizeUnits(ramTemp - ramStep) << std::endl;
    ramStep = ramTemp;

    // Add analysis samples:
    const double data_POT  = options_parser.data_POT;
    const double mc_POT = options_parser.mc_POT;
    std::vector<AnaSample*> analysis_sample_list;
    LogInfo << "Add analysis samples..." << std::endl;
    for(const auto& sample : options_parser.samples) {
        if(sample.use_sample and sample.cut_branch >= 0)
        {
            LogWarning << "Adding new sample to fit." << std::endl
                       << "Name: " << sample.name << std::endl
                       << "CutB: " << sample.cut_branch << std::endl
                       << "Detector: " << sample.detector << std::endl
                       << "Use Sample: " << std::boolalpha << sample.use_sample << std::endl;

            auto analysis_sample = new AnaSample(sample, data_ttree);

            analysis_sample->SetLLHFunction(options_parser.min_settings.likelihood);
//            analysis_sample-> SetNorm(data_POT/mc_POT); // internally done
            analysis_sample_list.emplace_back(analysis_sample);
        }
    }

    //read MC events
    std::string mc_file_path = options_parser.fname_mc;
    AnaTreeMC selected_events_AnaTreeMC(mc_file_path, "selectedEvents");
    LogInfo << "Reading and collecting events..." << std::endl;
    selected_events_AnaTreeMC.GetEvents(analysis_sample_list, options_parser.signal_definition, false);

    LogInfo << "Getting sample breakdown by topology..." << std::endl;
    std::vector<std::string> sample_topology_list = options_parser.sample_topology;
    std::vector<int> topology_HL_codes = options_parser.topology_HL_code;
    // Mapping the Highland topology codes to consecutive integers and then getting the topology breakdown for each sample:
    for(auto& analysis_sample : analysis_sample_list) {
        analysis_sample->SetTopologyHLCode(topology_HL_codes);
        analysis_sample->GetSampleBreakdown(output_tfile, "nominal", sample_topology_list, false);
    }

    ramTemp = GenericToolbox::getProcessMemoryUsage();
    LogDebug << "RAM taken by samples initialization: " << GenericToolbox::parseSizeUnits(ramTemp - ramStep) << std::endl;
    ramStep = ramTemp;



    //*************** FITTER SETTINGS **************************
    //In the bit below we choose which params are used in the fit
    //For stats only just use fit params
    //**********************************************************

    //define fit param classes
    std::vector<AnaFitParameters*> fit_parameters_list;



    // Fit parameters (template parameters):
//    LogInfo << "Reading template parameters..." << std::endl;
//    FitParameters sigfitpara("par_fit");
//    if(options_parser.rng_template)
//        sigfitpara.SetRNGstart();
//    if(options_parser.regularise)
//        sigfitpara.SetRegularisation(options_parser.reg_strength, options_parser.reg_method);
//    for(const auto& detector : options_parser.detectors)
//    {
//        if(detector.use_detector){
//            sigfitpara.AddDetector(detector.name, options_parser.signal_definition);
//        }
//    }
//    sigfitpara.InitEventMap(analysis_sample_list, 0);
//    fit_parameters_list.emplace_back(&sigfitpara);



    // Flux parameters:
    LogInfo << "Reading flux parameters..." << std::endl;
    FluxParameters flux_parameters("Flux Systematics");
    if( options_parser.flux_cov.do_fit ) {
        LogWarning << "Setup Flux Covariance." << std::endl
                   << "Opening " << options_parser.flux_cov.fname << " for flux covariance."
                  << std::endl;

        if(not LocalGenericToolbox::do_tfile_is_valid(options_parser.flux_cov.fname)){
            LogError << options_parser.flux_cov.fname << " can't be opened." << std::endl;
            exit(EXIT_FAILURE);
        }
        TFile* file_flux_cov = TFile::Open(options_parser.flux_cov.fname.c_str(), "READ");
        //TH1D* nd_numu_bins_hist = (TH1D*)file_flux_cov->Get(parser.flux_cov.binning.c_str());
        //TAxis* nd_numu_bins = nd_numu_bins_hist->GetXaxis();

        //std::vector<double> enubins;
        //enubins.push_back(nd_numu_bins -> GetBinLowEdge(1));
        //for(int i = 0; i < nd_numu_bins -> GetNbins(); ++i)
        //    enubins.push_back(nd_numu_bins -> GetBinUpEdge(i+1));

        auto* cov_flux = (TMatrixDSym*)file_flux_cov -> Get(options_parser.flux_cov.matrix.c_str());
        if(cov_flux == nullptr){
            LogError << options_parser.flux_cov.fname << ": " << options_parser.flux_cov.matrix << " can't be opened." << std::endl;
            exit(EXIT_FAILURE);
        }
        file_flux_cov -> Close();

        if(options_parser.flux_cov.rng_start)
            flux_parameters.SetRNGstart();

        flux_parameters.SetCovarianceMatrix(*cov_flux, options_parser.flux_cov.decompose);
        flux_parameters.SetThrow(options_parser.flux_cov.do_throw);
        flux_parameters.SetInfoFrac(options_parser.flux_cov.info_frac);
        for(const auto& detector : options_parser.detectors)
        {
            if(detector.use_detector)
                flux_parameters.AddDetector(detector.name, options_parser.flux_cov.binning);
        }
        flux_parameters.InitEventMap(analysis_sample_list, 0);
        fit_parameters_list.emplace_back(&flux_parameters);
    }

    ramTemp = GenericToolbox::getProcessMemoryUsage();
    LogDebug << "RAM taken by flux param: " << GenericToolbox::parseSizeUnits(ramTemp - ramStep) << std::endl;
    ramStep = ramTemp;


    // Cross-section parameters:
    LogInfo << "Reading Cross-section parameters..." << std::endl;
    XsecParameters xsec_parameters("Cross-Section Systematics");
    if(options_parser.xsec_cov.do_fit)
    {
        LogWarning << "Setup Xsec Covariance." << std::endl
                   << "Opening " << options_parser.xsec_cov.fname << " for xsec covariance."
                  << std::endl;

        if(not LocalGenericToolbox::do_tfile_is_valid(options_parser.xsec_cov.fname)){
            LogError << options_parser.xsec_cov.fname << " can't be opened." << std::endl;
            exit(EXIT_FAILURE);
        }
        TFile* file_xsec_cov = TFile::Open(options_parser.xsec_cov.fname.c_str(), "READ");
        auto* cov_xsec = (TMatrixDSym*)file_xsec_cov -> Get(options_parser.xsec_cov.matrix.c_str());
        if(cov_xsec == nullptr){
            LogError << options_parser.xsec_cov.fname << ": " << options_parser.xsec_cov.matrix << " can't be opened." << std::endl;
            exit(EXIT_FAILURE);
        }
        file_xsec_cov -> Close();

        ramTemp = GenericToolbox::getProcessMemoryUsage();
        LogDebug << "RAM taken by Xsec covMatrix: " << GenericToolbox::parseSizeUnits(ramTemp - ramStep) << std::endl;
        ramStep = ramTemp;

        if(options_parser.xsec_cov.rng_start)
            xsec_parameters.SetRNGstart();
        xsec_parameters.SetCovarianceMatrix(*cov_xsec, options_parser.xsec_cov.decompose);
        xsec_parameters.SetThrow(options_parser.xsec_cov.do_throw);
        xsec_parameters.SetNbThreads(__nb_threads__);

        for(const auto& detector : options_parser.detectors)
        {
            if(detector.use_detector)
                xsec_parameters.AddDetector(detector.name, detector.xsec);
        }

        ramTemp = GenericToolbox::getProcessMemoryUsage();
        LogDebug << "RAM taken by Xsec init: " << GenericToolbox::parseSizeUnits(ramTemp - ramStep) << std::endl;
        ramStep = ramTemp;

        xsec_parameters.InitEventMap(analysis_sample_list, 0);

        ramTemp = GenericToolbox::getProcessMemoryUsage();
        LogDebug << "RAM taken by Xsec init event map: " << GenericToolbox::parseSizeUnits(ramTemp - ramStep) << std::endl;
        ramStep = ramTemp;

        fit_parameters_list.emplace_back(&xsec_parameters);
    }

    ramTemp = GenericToolbox::getProcessMemoryUsage();
    LogDebug << "RAM taken by Xsec param: " << GenericToolbox::parseSizeUnits(ramTemp - ramStep) << std::endl;
    ramStep = ramTemp;


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
                detpara.AddDetector(detector.name, analysis_sample_list, true);
        }
        detpara.InitEventMap(analysis_sample_list, 0);
        fit_parameters_list.emplace_back(&detpara);
    }

    ramTemp = GenericToolbox::getProcessMemoryUsage();
    LogDebug << "RAM taken by detector param: " << GenericToolbox::parseSizeUnits(ramTemp - ramStep) << std::endl;
    ramStep = ramTemp;



    if(__PRNG_seed__ == -1)
        __PRNG_seed__ = options_parser.rng_seed;
    if(__nb_threads__ == -1)
        __nb_threads__ = options_parser.num_threads;



    output_tfile->mkdir("fitter");

    //Instantiate fitter obj
    ND280Fitter fitter;

    fitter.SetOutputTDirectory(output_tfile->GetDirectory("fitter"));
    fitter.SetPrngSeed(__PRNG_seed__);
    fitter.SetNbThreads(__nb_threads__);

//    fitter.SetMcNormalizationFactor(data_POT/mc_POT);
    fitter.SetAnaFitParametersList(fit_parameters_list);
    fitter.SetAnaSamplesList(analysis_sample_list);
    fitter.SetSelectedDataType(options_parser.fit_type);
    fitter.SetApplyStatisticalFluctuationsOnSamples(options_parser.stat_fluc);

    fitter.SetMinimizationSettings(options_parser.min_settings);
    fitter.SetDisableSystFit(options_parser.zero_syst);
    fitter.SetSaveEventTree(options_parser.save_events);

    fitter.Initialize();
    fitter.WritePrefitData();

    ramTemp = GenericToolbox::getProcessMemoryUsage();
    LogDebug << "RAM taken by the fitter: " << GenericToolbox::parseSizeUnits(ramTemp - ramStep) << std::endl;
    ramStep = ramTemp;
    LogDebug << "Total RAM before fitting: " << GenericToolbox::parseSizeUnits(ramStep) << std::endl;
    LogDebug << "Total RAM before fitting (removing program baseline) : " << GenericToolbox::parseSizeUnits(ramStep - ramBaseline) << std::endl;

    bool did_converge = false;
    if(not __is_dry_run__) {

        // Do the Fit
        did_converge = fitter.Fit();

        if(not did_converge)
            LogError    << "Fit did not coverge." << std::endl;
        else
            LogWarning  << "Fit has converged." << std::endl;

        fitter.WritePostFitData();

        std::vector<int> par_scans = options_parser.par_scan_list;
        if(!par_scans.empty())
            fitter.ParameterScans(par_scans, options_parser.par_scan_steps);
    }
    else{
        LogWarning << "Dry run is enabled. The fit is ignored." << std::endl;
    }
    output_tfile -> Close();

    LogInfo << "Output file: " << __output_file_path__ << std::endl;

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
    remind_usage_ss << "  -j : JSON input (Current : " << __json_config_path__ << ")" << std::endl;
    remind_usage_ss << "  -o : Override output file path (Current : " << __output_file_path__ << ")" << std::endl;
    remind_usage_ss << "  -t : Override number of threads (Current : " << __nb_threads__ << ")" << std::endl;
    remind_usage_ss << "  -d : Enable dry run (Current : " << __is_dry_run__ << ")" << std::endl;
    remind_usage_ss << "*********************************************************" << std::endl;

    LogInfo << remind_usage_ss.str();
    return remind_usage_ss.str();

}
void resetParameters()
{
    __json_config_path__ = "";
    __output_file_path__ = "";
}
void getUserParameters(){

    if(__argc__ == 1){
        remindUsage();
        exit(EXIT_FAILURE);
    }

    LogWarning << "Sanity check" << std::endl;

    const std::string XSLLHFITTER = std::getenv("XSLLHFITTER");
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
                __json_config_path__ = std::string(__argv__[j_arg]);
            }
            else {
                LogError << "Give an argument after " << __argv__[i_arg] << std::endl;
                throw std::logic_error(std::string(__argv__[i_arg]) + " : no argument found");
            }
        }
        else if(std::string(__argv__[i_arg]) == "-o"){
            if (i_arg < __argc__ - 1) {
                int j_arg = i_arg + 1;
                __output_file_path__ = std::string(__argv__[j_arg]);
            }
            else {
                LogError << "Give an argument after " << __argv__[i_arg] << std::endl;
                throw std::logic_error(std::string(__argv__[i_arg]) + " : no argument found");
            }
        }
        else if(std::string(__argv__[i_arg]) == "-t"){
            if (i_arg < __argc__ - 1) {
                int j_arg = i_arg + 1;
                __nb_threads__ = std::stoi(__argv__[j_arg]);
            }
            else {
                LogError << "Give an argument after " << __argv__[i_arg] << std::endl;
                throw std::logic_error(std::string(__argv__[i_arg]) + " : no argument found");
            }
        }
        else if(std::string(__argv__[i_arg]) == "-d"){
            __is_dry_run__ = true;
        }

    }

}