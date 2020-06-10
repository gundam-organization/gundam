#include <string>
#include <iostream>
#include <sstream>

#include <TFile.h>
#include <TTree.h>

#include <OptParser.hh>
#include <AnaSample.hh>
#include <AnaTreeMC.hh>
#include <AnaFitParameters.hh>
#include <FitParameters.hh>
#include <FluxParameters.hh>
#include <XsecParameters.hh>
#include <DetParameters.hh>
#include <XsecFitter.hh>
#include <GenericToolbox.h>

const std::string ERROR    = "\033[1;31m[xsllhND280UpFit.cxx] \033[00m";
const std::string INFO  = "\033[1;32m[xsllhND280UpFit.cxx] \033[00m";
const std::string WARNING   = "\033[1;33m[xsllhND280UpFit.cxx] \033[00m";
const std::string ALERT = "\033[1;35m[xsllhND280UpFit.cxx] \033[00m";


//! Global Variables
bool __is_dry_run__ = false;
int __nb_threads__ = -1;
int __PRNG_seed__ = -1;
std::string __json_config_path__;

std::string __command_line__;
int __argc__;
char **__argv__;




//! Local Functions
std::string remind_usage();
void reset_parameters();
void get_user_parameters();

int main(int argc, char** argv)
{
    __argc__ = argc;
    __argv__ = argv;

    reset_parameters();
    get_user_parameters();
    remind_usage(); // display used parameters


    OptParser options_parser;
    std::cout << INFO << "Reading json parameter files..." << std::endl;
    if(!options_parser.ParseJSON(__json_config_path__))
    {
        std::cerr << ERROR << "JSON parsing failed. Exiting." << std::endl;
        exit(EXIT_FAILURE);
    }
    options_parser.PrintOptions();

    // Setup data trees
    std::string data_tfile_path = options_parser.fname_data;
    std::cout << INFO << "Opening data file: " << data_tfile_path << std::endl;
    if(not GenericToolbox::do_tfile_is_valid(data_tfile_path)){
        std::cerr << ERROR << data_tfile_path << " can't be opened." << std::endl;
        exit(EXIT_FAILURE);
    }
    TFile* data_tfile = TFile::Open(data_tfile_path.c_str(), "READ");
    auto* data_ttree  = (TTree*)(data_tfile->Get("selectedEvents"));

    // Setup output tfile
    std::string output_file_path = options_parser.fname_output;
    std::cout << INFO << "Opening output file: " << output_file_path << std::endl;
    TFile* output_tfile          = TFile::Open(output_file_path.c_str(), "RECREATE");

    // Add analysis samples:
    const double data_POT  = options_parser.data_POT;
    const double mc_POT = options_parser.mc_POT;
    std::vector<AnaSample*> analysis_sample_list;
    std::cout << INFO << "Add analysis samples..." << std::endl;
    for(const auto& sample : options_parser.samples) {
        if(sample.use_sample and sample.cut_branch >= 0)
        {
            std::cout << WARNING << "Adding new sample to fit." << std::endl
                      << WARNING << "Name: " << sample.name << std::endl
                      << WARNING << "CutB: " << sample.cut_branch << std::endl
                      << WARNING << "Detector: " << sample.detector << std::endl
                      << WARNING << "Use Sample: " << std::boolalpha << sample.use_sample << std::endl;

            auto analysis_sample = new AnaSample(
                sample.cut_branch,
                sample.name,
                sample.detector,
                sample.binning,
                data_ttree
                );

            analysis_sample-> SetLLHFunction(options_parser.min_settings.likelihood);
            analysis_sample-> SetNorm(data_POT/mc_POT);
            analysis_sample_list.emplace_back(analysis_sample);
        }
    }

    //read MC events
    std::string mc_file_path = options_parser.fname_mc;
    AnaTreeMC selected_events_AnaTreeMC(mc_file_path, "selectedEvents");
    std::cout << INFO << "Reading and collecting events..." << std::endl;
    selected_events_AnaTreeMC.GetEvents(analysis_sample_list, options_parser.signal_definition, false);


    std::cout << INFO << "Getting sample breakdown by topology..." << std::endl;
    std::vector<std::string> sample_topology_list = options_parser.sample_topology;
    std::vector<int> topology_HL_codes = options_parser.topology_HL_code;
    // Mapping the Highland topology codes to consecutive integers and then getting the topology breakdown for each sample:
    for(auto& analysis_sample : analysis_sample_list) {
        analysis_sample->SetTopologyHLCode(topology_HL_codes);
        analysis_sample->GetSampleBreakdown(output_tfile, "nominal", sample_topology_list, false);
    }


    //*************** FITTER SETTINGS **************************
    //In the bit below we choose which params are used in the fit
    //For stats only just use fit params
    //**********************************************************

    //define fit param classes
    std::vector<AnaFitParameters*> fit_parameters_list;
    // Fit parameters (template parameters):
    std::cout << INFO << "Reading template parameters..." << std::endl;
    FitParameters sigfitpara("par_fit");
    if(options_parser.rng_template)
        sigfitpara.SetRNGstart();
    if(options_parser.regularise)
        sigfitpara.SetRegularisation(options_parser.reg_strength, options_parser.reg_method);
    for(const auto& detector : options_parser.detectors)
    {
        if(detector.use_detector){
            sigfitpara.AddDetector(detector.name, options_parser.signal_definition);
        }
    }
    sigfitpara.InitEventMap(analysis_sample_list, 0);
//    fit_parameters_list.emplace_back(&sigfitpara);



    // Flux parameters:
    std::cout << INFO << "Reading flux parameters..." << std::endl;
    FluxParameters flux_parameters("par_flux");
    if(options_parser.flux_cov.do_fit)
    {
        std::cout << WARNING << "Setup Flux Covariance." << std::endl
                  << WARNING << "Opening " << options_parser.flux_cov.fname << " for flux covariance."
                  << std::endl;

        if(not GenericToolbox::do_tfile_is_valid(options_parser.flux_cov.fname)){
            std::cerr << ERROR << options_parser.flux_cov.fname << " can't be opened." << std::endl;
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
            std::cerr << ERROR << options_parser.flux_cov.fname << ": " << options_parser.flux_cov.matrix << " can't be opened." << std::endl;
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


    // Cross-section parameters:
    std::cout << INFO << "Reading Cross-section parameters..." << std::endl;
    XsecParameters xsec_parameters("par_xsec");
    if(options_parser.xsec_cov.do_fit)
    {
        std::cout << WARNING << "Setup Xsec Covariance." << std::endl
                  << WARNING << "Opening " << options_parser.xsec_cov.fname << " for xsec covariance."
                  << std::endl;

        if(not GenericToolbox::do_tfile_is_valid(options_parser.xsec_cov.fname)){
            std::cerr << ERROR << options_parser.xsec_cov.fname << " can't be opened." << std::endl;
            exit(EXIT_FAILURE);
        }
        TFile* file_xsec_cov = TFile::Open(options_parser.xsec_cov.fname.c_str(), "READ");
        auto* cov_xsec = (TMatrixDSym*)file_xsec_cov -> Get(options_parser.xsec_cov.matrix.c_str());
        if(cov_xsec == nullptr){
            std::cerr << ERROR << options_parser.xsec_cov.fname << ": " << options_parser.xsec_cov.matrix << " can't be opened." << std::endl;
            exit(EXIT_FAILURE);
        }
        file_xsec_cov -> Close();

        if(options_parser.xsec_cov.rng_start)
            xsec_parameters.SetRNGstart();

        xsec_parameters.SetCovarianceMatrix(*cov_xsec, options_parser.xsec_cov.decompose);
        xsec_parameters.SetThrow(options_parser.xsec_cov.do_throw);
        for(const auto& detector : options_parser.detectors)
        {
            if(detector.use_detector)
                xsec_parameters.AddDetector(detector.name, detector.xsec);
        }
        xsec_parameters.InitEventMap(analysis_sample_list, 0);
        fit_parameters_list.emplace_back(&xsec_parameters);
    }


    // Detector parameters:
    DetParameters detpara("par_det");
    if(options_parser.det_cov.do_fit)
    {
        std::cout << WARNING << "Setup Detector Covariance." << std::endl
                  << WARNING << "Opening " << options_parser.det_cov.fname << " for detector covariance."
                  << std::endl;

        TFile* file_det_cov = TFile::Open(options_parser.det_cov.fname.c_str(), "READ");
        if(file_det_cov == nullptr)
        {
            std::cout << ERROR << "Could not open file! Exiting." << std::endl;
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



    if(__PRNG_seed__ == -1)
        __PRNG_seed__ = options_parser.rng_seed;
    if(__nb_threads__ == -1)
        __nb_threads__ = options_parser.num_threads;



    //Instantiate fitter obj
    XsecFitter xsecfit(output_tfile, __PRNG_seed__, __nb_threads__);
    //xsecfit.SetSaveFreq(10000);
    xsecfit.SetMinSettings(options_parser.min_settings);
    xsecfit.SetPOTRatio(data_POT/mc_POT);
    xsecfit.SetTopology(sample_topology_list);
    xsecfit.SetZeroSyst(options_parser.zero_syst);
    xsecfit.SetSaveEvents(options_parser.save_events);

    // Initialize fitter with fitpara vector (vector of AnaFitParameters objects):
    xsecfit.InitFitter(fit_parameters_list);
    std::cout << INFO << "Fitter initialised." << std::endl;


    bool did_converge = false;
    if(!__is_dry_run__)
    {
        // Run the fitter with the given samples, fit type and statistical fluctuations as specified in the .json config file:
        did_converge = xsecfit.Fit(analysis_sample_list, options_parser.fit_type, options_parser.stat_fluc);

        if(!did_converge)
            std::cout << WARNING << "Fit did not coverge." << std::endl;
        else
            std::cout << WARNING << "Fit has converged." << std::endl;

        xsecfit.WriteCovarianceMatrices();

        std::vector<int> par_scans = options_parser.par_scan_list;
        if(!par_scans.empty())
            xsecfit.ParameterScans(par_scans, options_parser.par_scan_steps);
    }
    output_tfile -> Close();

    // Print Arigatou Gozaimashita with Rainbowtext :)
    std::cout << WARNING << color::RainbowText("\u3042\u308a\u304c\u3068\u3046\u3054\u3056\u3044\u307e\u3057\u305f\uff01")
              << std::endl;

    return did_converge ? EXIT_SUCCESS : 121;
}


std::string remind_usage(){

    std::stringstream remind_usage_ss;
    remind_usage_ss << "*********************************************************" << std::endl;
    remind_usage_ss << " > Command Line Arguments" << std::endl;
    remind_usage_ss << "  -j : JSON input (Current : " << __json_config_path__ << ")" << std::endl;
    remind_usage_ss << "*********************************************************" << std::endl;

    std::cerr << remind_usage_ss.str();
    return remind_usage_ss.str();

}
void reset_parameters(){
    __json_config_path__ = "";
}
void get_user_parameters(){

    if(__argc__ == 1){
        remind_usage();
        exit(EXIT_FAILURE);
    }

    std::cout << WARNING << "Sanity check" << std::endl;

    const std::string XSLLHFITTER = std::getenv("XSLLHFITTER");
    if(XSLLHFITTER.empty())
    {
        std::cerr << ERROR << "Environment variable \"XSLLHFITTER\" not set." << std::endl
                  << ERROR << "Cannot determine source tree location." << std::endl;
        remind_usage();
        exit(EXIT_FAILURE);
    }

    std::cout << WARNING << "Reading user parameters" << std::endl;

    for(int i_arg = 0; i_arg < __argc__; i_arg++){
        __command_line__ += __argv__[i_arg];
        __command_line__ += " ";
    }

    for(int i_arg = 0; i_arg < __argc__; i_arg++){

        if(std::string(__argv__[i_arg]) == "-j"){
            if (i_arg < __argc__ - 1) {
                int j_arg = i_arg + 1;
                __json_config_path__ = std::string(__argv__[j_arg]);
            } else {
                std::cout << ERROR << "Give an argument after " << __argv__[i_arg] << std::endl;
                throw std::logic_error(std::string(__argv__[i_arg]) + " : no argument found");
            }
        }

    }

}