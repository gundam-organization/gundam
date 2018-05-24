#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>

#include "AnySample.hh"
#include "AnyTreeMC.hh"
#include "FitParameters.hh"
#include "FluxParameters.hh"
#include "OptParser.hh"
#include "ToyThrower.hh"
#include "XsecFitter.hh"

int main(int argc, char** argv)
{
    //std::cout << std::fixed << std::setprecision(3);
    std::cout << "------------------------------------------------\n"
              << "[CalcXsec]: Welcome to the Super-xsLLhFitter.\n"
              << "[CalcXsec]: Initializing the fit machinery..." << std::endl;

    const std::string xslf_env = std::getenv("XSLLHFITTER");
    if(xslf_env.empty())
    {
        std::cerr << "[ERROR]: Environment variable \"XSLLHFITTER\" not set." << std::endl
                  << "[ERROR]: Cannot determine source tree location." << std::endl;
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
        std::cerr << "[ERROR] JSON parsing failed. Exiting.\n";
        return 1;
    }

    std::string input_dir  = parser.input_dir;
    std::string fname_data = parser.fname_data;
    std::string fname_mc   = parser.fname_mc;
    std::string fname_postfit = parser.fname_output;
    std::string fname_xsec = parser.fname_xsec;
    std::string paramVectorFname = "fitresults.root";

    std::vector<int> signal_topology = parser.sample_signal;
    std::vector<std::string> topology = parser.sample_topology;

    const double potD  = parser.data_POT;
    const double potMC = parser.mc_POT;
    int seed = parser.rng_seed;
    int threads = parser.num_threads;

    TFile* fdata = TFile::Open(fname_data.c_str(), "READ");
    TTree* tdata = (TTree*)(fdata->Get("selectedEvents"));

    TFile* fpostfit = TFile::Open(fname_postfit.c_str(), "READ");
    TFile* foutput  = TFile::Open(fname_xsec.c_str(), "RECREATE");

    std::cout << "[CalcXsec]: Opening " << fname_data << " for data selection.\n"
              << "[CalcXsec]: Opening " << fname_mc << " for MC selection.\n"
              << "[CalcXsec]: Opening " << fname_postfit << " for post-fit results.\n"
              << "[CalcXsec]: Opening " << fname_xsec << " to store xsec results." << std::endl;

    std::cout << "[CalcXsec]: Setup Flux " << std::endl;

    TFile *finfluxcov = TFile::Open(parser.flux_cov.fname.c_str(), "READ");
    std::cout << "[CalcXsec]: Opening " << parser.flux_cov.fname << " for flux covariance." << std::endl;
    TH1D *nd_numu_bins_hist = (TH1D*)finfluxcov->Get(parser.flux_cov.binning.c_str());
    TAxis *nd_numu_bins = nd_numu_bins_hist->GetXaxis();

    std::vector<double> enubins;
    enubins.push_back(nd_numu_bins -> GetBinLowEdge(1));
    for(int i = 0; i < nd_numu_bins -> GetNbins(); ++i)
        enubins.push_back(nd_numu_bins -> GetBinUpEdge(i+1));

    TMatrixDSym* cov_flux_in = (TMatrixDSym*)finfluxcov -> Get(parser.flux_cov.matrix.c_str());
    TMatrixDSym cov_flux = *cov_flux_in;
    finfluxcov -> Close();

    std::vector<AnaSample*> samples;
    for(const auto& opt : parser.samples)
    {
        std::cout << "[CalcXsec]: Adding new sample to fit.\n"
                  << "[CalcXsec]: Name: " << opt.name << std::endl
                  << "[CalcXsec]: CutB: " << opt.cut_branch << std::endl
                  << "[CalcXsec]: Detector: " << opt.detector << std::endl
                  << "[CalcXsec]: Use Sample: " << opt.use_sample << std::endl;
        std::cout << "[CalcXsec]: Opening " << opt.binning << " for analysis binning." << std::endl;

        std::vector< std::pair<double, double> > D1_edges;
        std::vector< std::pair<double, double> > D2_edges;
        std::ifstream fin(opt.binning, std::ios::in);
        if(!fin.is_open())
        {
            std::cerr << "[ERROR]: Failed to open binning file: " << opt.binning
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
                    std::cerr << "[CalcXsec]: Bad line format: " << line << std::endl;
                    continue;
                }
                D1_edges.emplace_back(std::make_pair(D1_1,D1_2));
                D2_edges.emplace_back(std::make_pair(D2_1,D2_2));
            }
            fin.close();
        }

        auto s = new AnySample(opt.cut_branch, opt.name, opt.detector, D1_edges, D2_edges, tdata, false, opt.use_sample);
        s -> SetNorm(potD/potMC);
        samples.push_back(s);
    }

    AnyTreeMC selTree(fname_mc.c_str());
    std::cout << "[CalcXsec]: Reading and collecting events." << std::endl;
    selTree.GetEvents(samples, signal_topology);

    std::cout << "[CalcXsec]: Getting sample breakdown by reaction." << std::endl;
    for(auto& sample : samples)
        sample -> GetSampleBreakdown(foutput, "nominal", topology, false);

    std::vector<AnaFitParameters*> fitpara;

    FitParameters sigfitpara("par_fit");
    for(const auto& opt : parser.samples)
        sigfitpara.AddDetector(opt.detector, opt.binning, opt.det_offset);
    sigfitpara.InitEventMap(samples, 0);
    fitpara.push_back(&sigfitpara);

    FluxParameters fluxpara("par_flux");
    fluxpara.SetCovarianceMatrix(cov_flux);
    for(const auto& opt : parser.samples)
        fluxpara.AddDetector(opt.detector, enubins, opt.flux_offset);
    fluxpara.InitEventMap(samples, 0);
    fitpara.push_back(&fluxpara);

    TMatrixDSym* postfit_cov = (TMatrixDSym*)fpostfit -> Get("res_cov_matrix");
    TVectorD* postfit_params = (TVectorD*)fpostfit -> Get("res_vector");

    const int num_throws = 1E5;
    const int npar = postfit_cov -> GetNrows();
    TMatrixDSym cov_test(npar);
    cov_test.Zero();

    ToyThrower toy_thrower(*postfit_cov, seed, 1E-24);
    std::vector<std::vector<double> > throws;
    std::vector<double> toy(npar, 0);

    for(int i = 0; i < num_throws; ++i)
    {
        toy_thrower.Throw(toy);
        throws.push_back(toy);
    }

    TH1D h_throw("ht", "ht", 300, -3.0, 3.0);
    for(int t = 0; t < num_throws; ++t)
        h_throw.Fill(throws.at(t).at(10) + (*postfit_params)[10]);

    for(int t = 0; t < num_throws; ++t)
    {
        for(int i = 0; i < npar; ++i)
        {
            for(int j = 0; j < npar; ++j)
            {
                double x = throws.at(t).at(i);
                double y = throws.at(t).at(j);
                cov_test(i,j) += x * y / (1.0 * num_throws);
            }
        }
    }

    foutput -> cd();
    postfit_cov -> Write("postfit_cov");
    postfit_params -> Write("postfit_params");
    cov_test.Write("test_cov");
    h_throw.Write("h_throw");

    return 0;
}
