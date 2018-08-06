#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>

#include "AnaSample.hh"
#include "AnyTreeMC.hh"
#include "ColorOutput.hh"
#include "FitParameters.hh"
#include "FluxParameters.hh"
#include "OptParser.hh"
#include "XsecFitter.hh"
#include "XsecParameters.hh"

const std::string TAG = color::CYAN_STR + "[IngridFit]: " + color::RESET_STR;
const std::string ERR = color::RED_STR + color::BOLD_STR
                        + "[ERROR]: " + color::RESET_STR;

int main(int argc, char** argv)
{
    //std::cout << std::fixed << std::setprecision(3);
    std::cout << "------------------------------------------------\n"
              << TAG << "Welcome to the Super-xsLLhFitter.\n"
              << TAG << "Initializing the fit machinery..." << std::endl;

    const std::string xslf_env = std::getenv("XSLLHFITTER");
    if(xslf_env.empty())
    {
        std::cerr << ERR << "Environment variable \"XSLLHFITTER\" not set." << std::endl
                  << ERR << "Cannot determine source tree location." << std::endl;
        return 1;
    }

    std::string json_file;
    char option;
    while((option = getopt(argc, argv, "j:h")) != -1)
    {
        switch(option)
        {
            case 'j':
                json_file = optarg;
                break;
            case 'h':
                std::cout << "USAGE: "
                          << argv[0] << "\nOPTIONS:\n"
                          << "-j : JSON input\n";
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

    std::string input_dir = parser.input_dir;
    std::string fname_data = parser.fname_data;
    std::string fname_mc   = parser.fname_mc;
    std::string fname_output = parser.fname_output;
    std::string paramVectorFname = "fitresults.root";

    std::vector<int> signal_topology = parser.sample_signal;
    std::vector<std::string> topology = parser.sample_topology;

    const double potD  = parser.data_POT;
    const double potMC = parser.mc_POT;
    int seed = parser.rng_seed;
    int threads = parser.num_threads;
    bool stat_fluc = false;

    int isBuffer = false; // Is the final bin just for including events that go beyond xsec binning
    // e.g. events with over 5GeV pmu if binning in pmu

    //Setup data trees
    TFile* fdata = TFile::Open(fname_data.c_str(), "READ");
    TTree* tdata = (TTree*)(fdata->Get("selectedEvents"));

    std::cout << TAG << "Opening " << fname_data << " for data selection.\n"
              << TAG << "Opening " << fname_mc << " for MC selection." << std::endl;

    /*************************************** FLUX *****************************************/
    std::cout << TAG << "Setup Flux " << std::endl;

    //input File
    TFile *finfluxcov = TFile::Open(parser.flux_cov.fname.c_str(), "READ"); //contains flux systematics info
    std::cout << TAG << "Opening " << parser.flux_cov.fname << " for flux covariance." << std::endl;
    //setup enu bins and covm for flux
    TH1D *nd_numu_bins_hist = (TH1D*)finfluxcov->Get(parser.flux_cov.binning.c_str());
    TAxis *nd_numu_bins = nd_numu_bins_hist->GetXaxis();

    std::vector<double> enubins;
    enubins.push_back(nd_numu_bins -> GetBinLowEdge(1));
    for(int i = 0; i < nd_numu_bins -> GetNbins(); ++i)
        enubins.push_back(nd_numu_bins -> GetBinUpEdge(i+1));

    //Cov mat stuff:
    TMatrixDSym* cov_flux_in = (TMatrixDSym*)finfluxcov -> Get(parser.flux_cov.matrix.c_str());
    TMatrixDSym cov_flux = *cov_flux_in;
    finfluxcov -> Close();

    /*************************************** FLUX END *************************************/

    TFile *fout = TFile::Open(fname_output.c_str(), "RECREATE");
    std::cout << TAG << "Open output file: " << fname_output << std::endl;

    // Add analysis samples:

    std::vector<AnaSample*> samples;

    for(const auto& opt : parser.samples)
    {
        std::cout << TAG << "Adding new sample to fit.\n"
                  << TAG << "Name: " << opt.name << std::endl
                  << TAG << "CutB: " << opt.cut_branch << std::endl
                  << TAG << "Detector: " << opt.detector << std::endl
                  << TAG << "Use Sample: " << opt.use_sample << std::endl;

        auto s = new AnaSample(opt.cut_branch, opt.name, opt.detector, opt.binning, tdata);
        s -> SetNorm(potD/potMC);
        if(opt.cut_branch >= 0 && opt.use_sample == true)
            samples.push_back(s);
    }

    //read MC events
    AnyTreeMC selTree(fname_mc.c_str(), "selectedEvents");
    std::cout << TAG << "Reading and collecting events." << std::endl;
    selTree.GetEvents(samples, signal_topology, false);

    std::cout << TAG << "Getting sample breakdown by reaction." << std::endl;
    for(auto& sample : samples)
        sample -> GetSampleBreakdown(fout, "nominal", topology, false);


    //*************** FITTER SETTINGS **************************
    //In the bit below we choose which params are used in the fit
    //For stats only just use fit params
    //**********************************************************

    //define fit param classes
    std::vector<AnaFitParameters*> fitpara;

    // When filling the fitparas note that there are some assumptions later in the code
    // that the fit parameters are stored at index 0. For this reason always fill the
    // fit parameters first.

    //Fit parameters
    FitParameters sigfitpara("par_fit");
    for(const auto& opt : parser.samples)
    {
        if(opt.cut_branch >= 0 && opt.use_sample == true)
            sigfitpara.AddDetector(opt.detector, opt.binning, opt.det_offset);
    }
    sigfitpara.InitEventMap(samples, 0);
    fitpara.push_back(&sigfitpara);

    //Flux parameters
    FluxParameters fluxpara("par_flux");
    fluxpara.SetCovarianceMatrix(cov_flux);
    for(const auto& opt : parser.samples)
    {
        if(opt.cut_branch >= 0 && opt.use_sample == true)
            fluxpara.AddDetector(opt.detector, enubins, opt.flux_offset);
    }
    fluxpara.InitEventMap(samples, 0);
    //fitpara.push_back(&fluxpara);

    XsecParameters xsecpara("par_xsec");
    for(const auto& opt : parser.samples)
    {
        if(opt.cut_branch >= 0 && opt.use_sample == true)
            xsecpara.AddDetector(opt.detector, opt.xsec_config);
    }
    xsecpara.InitParameters();
    fitpara.push_back(&xsecpara);

    //Instantiate fitter obj
    XsecFitter xsecfit(seed, threads);
    xsecfit.SetPOTRatio(potD/potMC);

    //init w/ para vector
    xsecfit.InitFitter(fitpara, 0, paramVectorFname);
    std::cout << TAG << "Fitter initialised." << std::endl;

    /*
    for(int i = 0; i < sigfitpara.GetNpar(); ++i)
    {
        xsecfit.FixParameter("par_fit_ND280_" + std::to_string(i), 1.0);
        xsecfit.FixParameter("par_fit_INGRID_" + std::to_string(i), 1.0);
    }
    */
    //set frequency to save output
    xsecfit.SetSaveMode(fout, 1);

    //fitmode: 1 = generate toy dataset from nuisances (WITH stat fluct)
    //         2 = fake data from MC or real data
    //         3 = no nuisance sampling only stat fluctuation
    //         4 = fake data from MC or real data with statistical fluctuations applied to that data
    //         5 = generate toy dataset from nuisances and regularised c_i (WITH stat fluct)
    //         6 = generate toy dataset from nuisances and random c_i (WITH stat fluct)
    //         7 = generate toy dataset from nuisances and regularised c_i (WITH stat fluct) but fit without reg
    //         8 = Asimov (Make fake data that == MC)


    //fitmethod: 1 = MIGRAD only
    //           2 = MIGRAD + HESSE
    //           3 = MINOS

    //statFluct (only relevent if fitmode is set to gen fake data with nuisances):
    //           0 = Do not apply Stat Fluct to fake data
    //           1 = Apply Stat Fluct to fake data

    int fit_mode = 2;
    if(stat_fluc == true)
        fit_mode = 3;

    //xsecfit.Fit(samples, topology, fit_mode, 2, 0);
    fout -> Close();

    return 0;
}
