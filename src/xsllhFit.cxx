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
#include "DetParameters.hh"
#include "FitParameters.hh"
#include "FluxParameters.hh"
#include "OptParser.hh"
#include "XsecFitter.hh"
#include "XsecParameters.hh"

int main(int argc, char** argv)
{
    const std::string TAG = color::CYAN_STR + "[xsFit]: " + color::RESET_STR;
    const std::string ERR = color::RED_STR + color::BOLD_STR
                            + "[ERROR]: " + color::RESET_STR;

    //std::cout << std::fixed << std::setprecision(3);
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
    bool dry_run = false;
    char option;
    while((option = getopt(argc, argv, "j:nh")) != -1)
    {
        switch(option)
        {
            case 'j':
                json_file = optarg;
                break;
            case 'n':
                dry_run = true;
                break;
            case 'h':
                std::cout << "USAGE: "
                          << argv[0] << "\nOPTIONS:\n"
                          << "-j : JSON input\n"
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
    std::cout << TAG << "Setup Flux" << std::endl;

    //input File
    TFile* finfluxcov = TFile::Open(parser.flux_cov.fname.c_str(), "READ"); //contains flux systematics info
    std::cout << TAG << "Opening " << parser.flux_cov.fname << " for flux covariance." << std::endl;
    //setup enu bins and covm for flux
    TH1D* nd_numu_bins_hist = (TH1D*)finfluxcov->Get(parser.flux_cov.binning.c_str());
    TAxis* nd_numu_bins = nd_numu_bins_hist->GetXaxis();

    std::vector<double> enubins;
    enubins.push_back(nd_numu_bins -> GetBinLowEdge(1));
    for(int i = 0; i < nd_numu_bins -> GetNbins(); ++i)
        enubins.push_back(nd_numu_bins -> GetBinUpEdge(i+1));

    //Cov mat stuff:
    TMatrixDSym* cov_flux_in = (TMatrixDSym*)finfluxcov -> Get(parser.flux_cov.matrix.c_str());
    TMatrixDSym cov_flux = *cov_flux_in;
    finfluxcov -> Close();

    /*************************************** FLUX END *************************************/
    std::cout << TAG << "Setup Xsec Covariance" << std::endl;
    std::ifstream fin(parser.xsec_cov.fname, std::ios::in);

    TMatrixDSym cov_xsec;
    if(!fin.is_open())
    {
        std::cerr << ERR << "Failed to open " << parser.xsec_cov.fname << std::endl;
        return 1;
    }
    else
    {
        unsigned int dim = 0;
        std::string line;
        if(std::getline(fin, line))
        {
            std::stringstream ss(line);
            ss >> dim;
        }

        cov_xsec.ResizeTo(dim, dim);
        for(unsigned int i = 0; i < dim; ++i)
        {
            std::getline(fin, line);
            std::stringstream ss(line);
            double val = 0;

            for(unsigned int j = 0; j < dim; ++j)
            {
                ss >> val;
                cov_xsec(i,j) = val;
            }
        }
    }

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
                  << TAG << "Use Sample: " << std::boolalpha << opt.use_sample << std::endl;

        auto s = new AnaSample(opt.cut_branch, opt.name, opt.detector, opt.binning, tdata);
        s -> SetNorm(potD/potMC);
        if(opt.cut_branch >= 0 && opt.use_sample == true)
            samples.push_back(s);
    }

    //read MC events
    AnaTreeMC selTree(fname_mc.c_str(), "selectedEvents");
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
    FitParameters sigfitpara("par_fit", false);
    for(const auto& opt : parser.detectors)
    {
        if(opt.use_detector)
            sigfitpara.AddDetector(opt.name, opt.binning);
    }
    sigfitpara.InitEventMap(samples, 0);
    fitpara.push_back(&sigfitpara);

    //Flux parameters
    FluxParameters fluxpara("par_flux");
    fluxpara.SetCovarianceMatrix(cov_flux, parser.flux_cov.decompose);
    fluxpara.SetInfoFrac(parser.flux_cov.info_frac);
    for(const auto& opt : parser.detectors)
    {
        if(opt.use_detector)
            fluxpara.AddDetector(opt.name, enubins);
    }
    fluxpara.InitEventMap(samples, 0);
    fitpara.push_back(&fluxpara);

    XsecParameters xsecpara("par_xsec");
    xsecpara.SetCovarianceMatrix(cov_xsec, parser.xsec_cov.decompose);
    for(const auto& opt : parser.detectors)
    {
        if(opt.use_detector)
            xsecpara.AddDetector(opt.name, opt.xsec);
    }
    xsecpara.InitEventMap(samples, 0);
    fitpara.push_back(&xsecpara);

    /*
    TMatrixDSym cov_det(348);
    cov_det.Zero();
    for(int i = 0; i < 348; ++i)
        cov_det(i,i) = 0.01;
    */

    std::cout << TAG << "Setup Detector Covariance" << std::endl;
    TFile* file_detcov = TFile::Open(parser.det_cov.fname.c_str(), "READ");
    TMatrixDSym* cov_det_in = (TMatrixDSym*)file_detcov -> Get(parser.det_cov.matrix.c_str());
    TMatrixDSym cov_det = *cov_det_in;
    file_detcov -> Close();

    DetParameters detpara("par_det");
    detpara.SetCovarianceMatrix(cov_det, parser.det_cov.decompose);
    detpara.SetInfoFrac(parser.det_cov.info_frac);
    for(const auto& opt : parser.detectors)
    {
        if(opt.use_detector)
            detpara.AddDetector(opt.name, samples, true);
    }
    detpara.InitEventMap(samples, 0);
    fitpara.push_back(&detpara);

    //Instantiate fitter obj
    XsecFitter xsecfit(fout, seed, threads);
    //xsecfit.SetSaveFreq(10000);
    xsecfit.SetPOTRatio(potD/potMC);

    //init w/ para vector
    xsecfit.InitFitter(fitpara, paramVectorFname);
    std::cout << TAG << "Fitter initialised." << std::endl;

    /*
    for(int i = 0; i < sigfitpara.GetNpar(); ++i)
    {
        xsecfit.FixParameter("par_fit_ND280_" + std::to_string(i), 1.0);
        xsecfit.FixParameter("par_fit_INGRID_" + std::to_string(i), 1.0);
    }
    */

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

    if(!dry_run)
        xsecfit.Fit(samples, topology, fit_mode, 2, 0);
    fout -> Close();

    return 0;
}
