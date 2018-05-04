/******************************************************

Bare bones of what needs to be set up to fit some set of
parameters defined in /fitparams/src/FitParameters

******************************************************/

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>

#include <TCanvas.h>
#include <TH1F.h>

#include "FitParameters.hh"
#include "XsecFitter.hh"
#include "AnyTreeMC.hh"
#include "AnySample.hh"
#include "FluxParameters.hh"

int main(int argc, char *argv[])
{
    std::cout << std::fixed << std::setprecision(3);

    const char* xslf_env = std::getenv("XSLLHFITTER");
    if(!xslf_env)
    {
        std::cerr << "[ERROR]: Environment variable \"XSLLHFITTER\" not set." << std::endl
                  << "[ERROR]: Cannot determine source tree location." << std::endl;
        return 1;
    }

    std::string input_dir = std::string(xslf_env) + "/inputs/ingridFit/";
    std::string fname_data = input_dir + "fixed_nominal.root"; //"/NeutAir5_2DV2.root";
    std::string fname_mc   = input_dir + "fixed_nominal.root"; //"/NeutAir5_2DV2.root";
    std::string fname_binning = input_dir + "ingrid_binning.txt"; //"/dptbinning2DPS_shortest_inclusive.txt";
    std::string fname_fluxcov = input_dir + "flux_numu_cov.root";
    std::string paramVectorFname = "fitresults.root";
    std::string fname_output = "ingridFit.root";

    const double potD  = 331.6; //in units 10^19 Neut Air
    const double potMC = 331.6; //in units 10^19 Neut Air
    int seed = 1019;
    bool fit_nd280 = false;
    bool fit_ingrid = false;
    bool stat_fluc = false;

    int nprimbins = 8;
    int isBuffer = false; // Is the final bin just for including events that go beyond xsec binning
    // e.g. events with over 5GeV pmu if binning in pmu

    //Argument parser for now. Possibly try cxxopts library.
    char option;
    while((option = getopt(argc, argv, "d:m:o:b:f:NIsS:h")) != -1)
    {
        switch(option)
        {
            case 'd':
                fname_data = input_dir + optarg;
                break;
            case 'm':
                fname_mc = input_dir + optarg;
                break;
            case 'o':
                fname_output = optarg;
                break;
            case 'b':
                fname_binning = input_dir + optarg;
                break;
            case 'f':
                fname_fluxcov = input_dir + optarg;
                break;
            case 'N':
                fit_nd280 = true;
                break;
            case 'I':
                fit_ingrid = true;
                break;
            case 's':
                stat_fluc = true;
                break;
            case 'S':
                seed = std::stoi(optarg);
                break;
            case 'h':
                std::cout << "USAGE: "
                          << argv[0] << "\nOPTIONS:\n"
                          << "-d : set data file\n"
                          << "-m : set mc file\n"
                          << "-o : set output file\n"
                          << "-b : set binning file\n"
                          << "-f : set flux covariance\n"
                          << "-N : fit nd280 sample\n"
                          << "-I : fit ingrid sample\n"
                          << "-s : set statistical fluctuations\n"
                          << "-S : set random seed\n";
            default:
                return 0;
        }
    }

    if(fit_nd280 == false && fit_ingrid == false)
        fit_nd280 = true;

    // Setup data trees
    TFile* fdata = TFile::Open(fname_data.c_str(), "READ");
    TTree* tdata = (TTree*)(fdata->Get("selectedEvents"));

    //Set up bin edges
    std::vector< std::pair<double, double> > v_D1edges;
    std::vector< std::pair<double, double> > v_D2edges;

    std::cout << "------------------------------------------------\n"
              << "[IngridFit]: Welcome to the Super-xsLLhFitter.\n"
              << "[IngridFit]: Initializing the fit machinery..." << std::endl;
    std::cout << "[IngridFit]: Opening " << fname_data << " for data selection.\n"
              << "[IngridFit]: Opening " << fname_mc << " for MC selection." << std::endl;
    std::cout << "[IngridFit]: Opening " << fname_binning << " for analysis binning." << std::endl;
    std::ifstream fin(fname_binning, std::ios::in);
    if(!fin.is_open())
    {
        std::cerr << "[ERROR]: Failed to open binning file: " << fname_binning
                  << "[ERROR]: Terminating execution." << std::endl;
        return 1;
    }

    else
    {
        std::string line;
        while(getline(fin, line))
        {
            std::stringstream ss(line);
            double D1_1, D1_2, D2_1, D2_2;
            if(!(ss>>D2_1>>D2_2>>D1_1>>D1_2))
            {
                std::cerr << "[IngridFit]: Bad line format: " << line << std::endl;
                continue;
            }
            v_D1edges.emplace_back(std::make_pair(D1_1,D1_2));
            v_D2edges.emplace_back(std::make_pair(D2_1,D2_2));
        }
        fin.close();
    }

    const int ing_fit_offset = v_D1edges.size();
    //Set up systematics:

    /*************************************** FLUX *****************************************/
    std::cout << "[IngridFit]: Setup Flux " << std::endl;

    //input File
    TFile *finfluxcov = TFile::Open(fname_fluxcov.c_str(), "READ"); //contains flux systematics info
    std::cout << "[IngridFit]: Opening " << fname_fluxcov << " for flux covariance." << std::endl;
    //setup enu bins and covm for flux
    TH1D *nd_numu_bins_hist = (TH1D*)finfluxcov->Get("flux_binning");
    TAxis *nd_numu_bins = nd_numu_bins_hist->GetXaxis();

    std::vector<double> enubins;
    enubins.push_back(nd_numu_bins -> GetBinLowEdge(1));
    for(int i = 0; i < nd_numu_bins -> GetNbins(); ++i)
        enubins.push_back(nd_numu_bins -> GetBinUpEdge(i+1));

    //Cov mat stuff:
    TMatrixDSym* cov_flux_in = (TMatrixDSym*)finfluxcov -> Get("flux_numu_cov");
    TMatrixDSym cov_flux = *cov_flux_in;
    finfluxcov -> Close();

    /*************************************** FLUX END *************************************/

    TFile *fout = TFile::Open(fname_output.c_str(), "RECREATE");
    std::cout << "[IngridFit]: Open output file: " << fname_output << std::endl;

    // Add analysis samples:

    std::vector<AnaSample*> samples;

    // The sample ID (first arg) should match the cutBranch corresponding to it

    AnySample sam2(1, "MuTPCpTPC", "ND280", v_D1edges, v_D2edges, tdata, isBuffer, false);
    sam2.SetNorm(potD/potMC);
    if(fit_nd280 == true)
        samples.push_back(&sam2);

    //AnySample sam6(5, "CC1pi", v_D1edges, v_D2edges, tdata, isBuffer, true);
    //sam6.SetNorm(potD/potMC);
    //samples.push_back(&sam6);

    //AnySample sam7(6, "DIS", v_D1edges, v_D2edges, tdata, isBuffer, true);
    //sam7.SetNorm(potD/potMC);
    //samples.push_back(&sam7);

    AnySample sam8(10, "Ingrid", "INGRID", v_D1edges, v_D2edges, tdata, isBuffer, false);
    sam8.SetNorm(potD/potMC);
    if(fit_ingrid == true)
        samples.push_back(&sam8);

    //read MC events
    AnyTreeMC selTree(fname_mc.c_str());
    std::cout << "[IngridFit]: Reading and collecting events." << std::endl;
    std::vector<int> signal_topology = {1, 2};
    selTree.GetEvents(samples, signal_topology);

    std::cout << "[IngridFit]: Getting sample breakdown by reaction." << std::endl;
    std::vector<std::string> topology = {"cc0pi0p", "cc0pi1p", "cc0pinp", "cc1pi+", "ccother", "backg", "Null", "OOFV"};

    for(auto& sample : samples)
        sample -> GetSampleBreakdown(fout, "nominal", topology, true);


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
    sigfitpara.AddDetector("ND280", fname_binning, 0);
    sigfitpara.AddDetector("INGRID", fname_binning, ing_fit_offset);
    sigfitpara.InitEventMap(samples, 0);
    fitpara.push_back(&sigfitpara);

    //Flux parameters
    FluxParameters fluxpara("par_flux");
    fluxpara.SetCovarianceMatrix(cov_flux);
    fluxpara.AddDetector("ND280", enubins, 0);
    fluxpara.AddDetector("INGRID", enubins, 20);
    fluxpara.InitEventMap(samples, 0);
    fitpara.push_back(&fluxpara);

    //Instantiate fitter obj
    XsecFitter xsecfit(seed);
    xsecfit.SetPOTRatio(potD/potMC);

    //init w/ para vector
    xsecfit.InitFitter(fitpara, 0, 0, nprimbins, paramVectorFname);
    std::cout << "[IngridFit]: Fitter initialised." << std::endl;

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

    xsecfit.Fit(samples, topology, fit_mode, 2, 0);
    fout -> Close();

    return 0;
}
