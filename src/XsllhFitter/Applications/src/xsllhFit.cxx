#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>

#include "AnaSample.hh"
#include "AnaTreeMC.hh"
#include "ColorOutput.hh"
#include "DetParameters.h"
#include "FitParameters.h"
#include "FluxParameters.h"
#include "OptParser.hh"
#include "XsecFitter.hh"
#include "XsecParameters.h"

int main(int argc, char** argv)
{
    const std::string TAG = color::CYAN_STR + "[xsFit]: " + color::RESET_STR;
    const std::string ERR = color::RED_STR + color::BOLD_STR
                            + "[ERROR]: " + color::RESET_STR;

    std::cout << "------------------------------------------------\n"
              << TAG << color::RainbowText("Welcome to the Super-xsLLhFitter.\n")
              << TAG << color::RainbowText("Initializing the fit machinery...") << std::endl;

    const std::string xslf_env = std::getenv("XSLLHFITTER");
    if(xslf_env.empty())
    {
        std::cerr << ERR << "Environment variable \"XSLLHFITTER\" not set." << std::endl
                  << ERR << "Cannot determine source tree location." << std::endl;
        return 1;
    }

    std::string json_file;
    std::string fname_output;
    bool dry_run = false;
    int seed = -1;
    int threads = -1;

    char option;
    while((option = getopt(argc, argv, "j:o:s:t:nh")) != -1)
    {
        switch(option)
        {
            case 'j':
                json_file = optarg;
                break;
            case 'o':
                fname_output = optarg;
                break;
            case 's':
                seed = std::stoi(optarg);
                break;
            case 't':
                threads = std::stoi(optarg);
                break;
            case 'n':
                dry_run = true;
                break;
            case 'h':
                std::cout << "USAGE: "
                          << argv[0] << "\nOPTIONS:\n"
                          << "-j : JSON input\n"
                          << "-o : Output file (overrides JSON config)\n"
                          << "-s : RNG seed (overrides JSON config)\n"
                          << "-t : Num. __nb_threads__ (overrides JSON config)\n"
                          << "-n : Dry run - Set up but do not run fit.\n";
            default:
                return 0;
        }
    }

    OptParser parser;
    if(!parser.ParseJSON(json_file))
    {
        std::cerr << ERR << "JSON parsing failed. Exiting.\n";
        return 1;
    }
    parser.PrintOptions();

    // Initializing some variables from the .json config file:
    std::string input_dir = parser.input_dir;
    std::string fname_data = parser.fname_data;
    std::string fname_mc   = parser.fname_mc;
    std::vector<std::string> topology = parser.sample_topology;
    std::vector<int> topology_HL_codes = parser.topology_HL_code;

    const double potD  = parser.data_POT;
    const double potMC = parser.mc_POT;

    if(fname_output.empty())
        fname_output = parser.fname_output;
    else
        std::cout << TAG << "Output file set by CLI to: " << fname_output << std::endl;

    if(seed < 0)
        seed = parser.rng_seed;
    else
        std::cout << TAG << "RNG seed set by CLI to: " << seed << std::endl;

    if(threads < 0)
        threads = parser.num_threads;
    else
        std::cout << TAG << "Threads set by CLI to: " << seed << std::endl;

    //Setup data trees
    TFile* fdata = TFile::Open(fname_data.c_str(), "READ");
    TTree* tdata = (TTree*)(fdata->Get("selectedEvents"));

    std::cout << TAG << "Configure file parsing finished." << std::endl;
    std::cout << TAG << "Opening " << fname_data << " for data selection.\n"
              << TAG << "Opening " << fname_mc << " for MC selection." << std::endl;

    TFile* fout = TFile::Open(fname_output.c_str(), "RECREATE");
    std::cout << TAG << "Open output file: " << fname_output << std::endl;

    // Add analysis samples:
    std::vector<AnaSample*> samples;

    for(const auto& opt : parser.samples)
    {
        if(opt.use_sample == true && opt.cut_branch >= 0)
        {
            std::cout << TAG << "Adding new sample to fit.\n"
                      << TAG << "Name: " << opt.name << std::endl
                      << TAG << "CutB: " << opt.cut_branch << std::endl
                      << TAG << "Detector: " << opt.detector << std::endl
                      << TAG << "Use Sample: " << std::boolalpha << opt.use_sample << std::endl;

            auto s = new AnaSample(opt.cut_branch, opt.name, opt.detector, opt.binning, tdata);
            s -> SetLLHFunction(parser.min_settings.likelihood);
            s -> SetNorm(potD/potMC);
            samples.push_back(s);
        }
    }

    //read MC events
    AnaTreeMC selTree(fname_mc.c_str(), "selectedEvents");
    std::cout << TAG << "Reading and collecting events." << std::endl;
    selTree.GetEvents(samples, parser.signal_definition, false);

    std::cout << TAG << "Getting sample breakdown by topology." << std::endl;

    // Mapping the Highland topology codes to consecutive integers and then getting the topology breakdown for each sample:
    for(auto& sample : samples)
    {
        sample -> SetTopologyHLCode(topology_HL_codes);
        sample -> GetSampleBreakdown(fout, "nominal", topology, false);
    }


    //*************** FITTER SETTINGS **************************
    //In the bit below we choose which params are used in the fit
    //For stats only just use fit params
    //**********************************************************

    //define fit param classes
    std::vector<AnaFitParameters*> fitpara;

    // Fit parameters (template parameters):
    FitParameters sigfitpara("par_fit");
    if(parser.rng_template)
        sigfitpara.SetRNGstart();
    if(parser.regularise)
        sigfitpara.SetRegularisation(parser.reg_strength, parser.reg_method);
    for(const auto& opt : parser.detectors)
    {
        if(opt.use_detector)
            sigfitpara.AddDetector(opt.name, parser.signal_definition);
        //sigfitpara.AddDetector(opt.name, opt.binning);
    }
    sigfitpara.InitEventMap(samples, 0);
//    fitpara.push_back(&sigfitpara);

    // Flux parameters:
    FluxParameters fluxpara("par_flux");
    if(parser.flux_cov.do_fit)
    {
        std::cout << TAG << "Setup Flux Covariance." << std::endl
                  << TAG << "Opening " << parser.flux_cov.fname << " for flux covariance."
                  << std::endl;

        TFile* file_flux_cov = TFile::Open(parser.flux_cov.fname.c_str(), "READ");
        if(file_flux_cov == nullptr)
        {
            std::cout << ERR << "Could not open file! Exiting." << std::endl;
            return 1;
        }
        //TH1D* nd_numu_bins_hist = (TH1D*)file_flux_cov->Get(parser.flux_cov.binning.c_str());
        //TAxis* nd_numu_bins = nd_numu_bins_hist->GetXaxis();

        //std::vector<double> enubins;
        //enubins.push_back(nd_numu_bins -> GetBinLowEdge(1));
        //for(int i = 0; i < nd_numu_bins -> GetNbins(); ++i)
        //    enubins.push_back(nd_numu_bins -> GetBinUpEdge(i+1));

        TMatrixDSym* cov_flux = (TMatrixDSym*)file_flux_cov -> Get(parser.flux_cov.matrix.c_str());
        file_flux_cov -> Close();

        if(parser.flux_cov.rng_start)
            fluxpara.SetRNGstart();

        fluxpara.SetCovarianceMatrix(*cov_flux, parser.flux_cov.decompose);
        fluxpara.SetThrow(parser.flux_cov.do_throw);
        fluxpara.SetInfoFrac(parser.flux_cov.info_frac);
        for(const auto& opt : parser.detectors)
        {
            if(opt.use_detector)
                fluxpara.AddDetector(opt.name, parser.flux_cov.binning);
        }
        fluxpara.InitEventMap(samples, 0);
        fitpara.push_back(&fluxpara);
    }


    XsecParameters xsecpara("par_xsec");
    if(parser.xsec_cov.do_fit)
    {
        std::cout << TAG << "Setup Xsec Covariance." << std::endl
                  << TAG << "Opening " << parser.xsec_cov.fname << " for xsec covariance."
                  << std::endl;

        TFile* file_xsec_cov = TFile::Open(parser.xsec_cov.fname.c_str(), "READ");
        if(file_xsec_cov == nullptr)
        {
            std::cout << ERR << "Could not open file! Exiting." << std::endl;
            return 1;
        }
        TMatrixDSym* cov_xsec = (TMatrixDSym*)file_xsec_cov -> Get(parser.xsec_cov.matrix.c_str());
        file_xsec_cov -> Close();

        if(parser.xsec_cov.rng_start)
            xsecpara.SetRNGstart();

        xsecpara.SetCovarianceMatrix(*cov_xsec, parser.xsec_cov.decompose);
        xsecpara.SetThrow(parser.xsec_cov.do_throw);
        for(const auto& opt : parser.detectors)
        {
            if(opt.use_detector)
                xsecpara.AddDetector(opt.name, opt.xsec);
        }
        xsecpara.InitEventMap(samples, 0);
        fitpara.push_back(&xsecpara);
    }

    DetParameters detpara("par_det");
    if(parser.det_cov.do_fit)
    {
        std::cout << TAG << "Setup Detector Covariance." << std::endl
                  << TAG << "Opening " << parser.det_cov.fname << " for detector covariance."
                  << std::endl;

        TFile* file_det_cov = TFile::Open(parser.det_cov.fname.c_str(), "READ");
        if(file_det_cov == nullptr)
        {
            std::cout << ERR << "Could not open file! Exiting." << std::endl;
            return 1;
        }
        TMatrixDSym* cov_det = (TMatrixDSym*)file_det_cov -> Get(parser.det_cov.matrix.c_str());
        file_det_cov -> Close();

        if(parser.det_cov.rng_start)
            detpara.SetRNGstart();

        detpara.SetCovarianceMatrix(*cov_det, parser.det_cov.decompose);
        detpara.SetThrow(parser.det_cov.do_throw);
        detpara.SetInfoFrac(parser.det_cov.info_frac);
        for(const auto& opt : parser.detectors)
        {
            if(opt.use_detector)
                detpara.AddDetector(opt.name, samples, true);
        }
        detpara.InitEventMap(samples, 0);
        fitpara.push_back(&detpara);
    }

    //Instantiate fitter obj
    XsecFitter xsecfit(fout, seed, threads);
    //xsecfit.SetSaveFreq(10000);
    xsecfit.SetMinSettings(parser.min_settings);
    xsecfit.SetPOTRatio(potD/potMC);
    xsecfit.SetTopology(topology);
    xsecfit.SetZeroSyst(parser.zero_syst);
    xsecfit.SetSaveEvents(parser.save_events);

    // generateFormula fitter with fitpara vector (vector of AnaFitParameters objects):
    xsecfit.InitFitter(fitpara);
    std::cout << TAG << "Fitter initialised." << std::endl;

    bool did_converge = false;
    if(!dry_run)
    {
        // Run the fitter with the given samples, fit type and statistical fluctuations as specified in the .json config file:
        did_converge = xsecfit.Fit(samples, parser.fit_type, parser.stat_fluc);

        if(!did_converge)
            std::cout << TAG << "Fit did not coverge." << std::endl;
        else
            std::cout << TAG << "Fit has converged." << std::endl;

        xsecfit.WriteCovarianceMatrices();

        std::vector<int> par_scans = parser.par_scan_list;
        if(!par_scans.empty())
            xsecfit.ParameterScans(par_scans, parser.par_scan_steps);
    }
    fout -> Close();

    // Print Arigatou Gozaimashita with Rainbowtext :)
    std::cout << TAG << color::RainbowText("\u3042\u308a\u304c\u3068\u3046\u3054\u3056\u3044\u307e\u3057\u305f\uff01")
              << std::endl;

    return did_converge ? 0 : 121;
}
