#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <sstream>
#include <string>
#include <unistd.h>

#include "AnySample.hh"
#include "AnyTreeMC.hh"
#include "FitParameters.hh"
#include "FluxParameters.hh"
#include "OptParser.hh"
#include "ToyThrower.hh"
#include "XsecExtractor.hh"
#include "XsecFitter.hh"

std::vector<std::vector<double> > MapThrow(const std::vector<double>& toy,
                                           const std::vector<double>& nom,
                                           const std::vector<AnaFitParameters*>& fit);

int main(int argc, char** argv)
{
    auto t_start_total = std::chrono::high_resolution_clock::now();
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
    const int num_throws = parser.num_throws;

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

    auto t_start_samples = std::chrono::high_resolution_clock::now();
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
    for(const auto& s : samples)
    {
        s -> GetSampleBreakdown(foutput, "nominal", topology, false);
        s -> FillEventHisto(2);

        std::string hist_name = s -> GetName() + "_prefit";
        s -> Write(foutput, hist_name, 0);
    }

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
    //fitpara.push_back(&fluxpara);
    auto t_end_samples = std::chrono::high_resolution_clock::now();

    TMatrixDSym* postfit_cov = (TMatrixDSym*)fpostfit -> Get("res_cov_matrix");
    TMatrixDSym* postfit_cor = (TMatrixDSym*)fpostfit -> Get("res_cor_matrix");
    TVectorD* postfit_param_root = (TVectorD*)fpostfit -> Get("res_vector");

    const int nfitbins = sigfitpara.GetNpar();
    TH1D h_postfit("h_postfit", "h_postfit", nfitbins, 0, nfitbins);

    auto t_start_throws = std::chrono::high_resolution_clock::now();
    std::cout << "[CalcXsec]: Throwing " << num_throws << " toys." << std::endl;
    const int npar = postfit_cov -> GetNrows();
    TMatrixD cov_test(npar, npar);
    TMatrixD cor_test(npar, npar);
    TMatrixD xsec_cov(nfitbins, nfitbins);
    TMatrixD xsec_cor(nfitbins, nfitbins);
    cov_test.Zero();
    xsec_cov.Zero();

    std::vector<double> postfit_param;
    for(int i = 0; i < npar; ++i)
        postfit_param.push_back((*postfit_param_root)[i]);

    XsecExtractor nd280_xsec("ND280", seed);
    XsecExtractor ingrid_xsec("INGRID", seed);

    std::vector<XsecExtractor> xsec_calc;
    for(const auto& det : parser.detectors)
    {
        XsecExtractor x(det.name, seed);
        x.SetNumTargets(det.ntargets_val, det.ntargets_err);
        xsec_calc.push_back(x);
    }

    TH1D h_nt("h_nt", "h_nt", 100, 2.70E29, 2.80E29);
    for(int i = 0; i < num_throws; ++i)
        h_nt.Fill(xsec_calc.at(0).ThrowNtargets());

    ToyThrower toy_thrower(*postfit_cov, seed, 1E-24);
    std::vector<TH1D> xsec_throws;
    xsec_throws.reserve(num_throws);
    std::vector<std::vector<double> > throws;
    std::vector<double> toy(npar, 0);

    int bin = 1;
    auto toy_param = MapThrow(toy, postfit_param, fitpara);
    for(int s = 0; s < samples.size(); ++s)
    {
        const unsigned int num_events = samples[s] -> GetN();
        for(unsigned int i = 0; i < num_events; ++i)
        {
            AnaEvent* ev = samples[s] -> GetEvent(i);
            ev -> SetEvWght(ev -> GetEvWghtMC());
            for(int f = 0; f < fitpara.size(); ++f)
            {
                const std::string det(samples[s] -> GetDetector());
                fitpara[f] -> ReWeight(ev, det, s, i, toy_param.at(f));
            }
        }

        std::string hist_name = samples[s] -> GetName() + "_postfit";
        samples[s] -> FillEventHisto(2);
        samples[s] -> Write(foutput, hist_name, 0);

        const auto h = samples[s] -> GetSignalHisto();
        for(int j = 1; j <= h -> GetNbinsX(); ++j)
            h_postfit.SetBinContent(bin++, h -> GetBinContent(j));
    }

    auto t_start_single = std::chrono::high_resolution_clock::now();
    auto t_end_single = std::chrono::high_resolution_clock::now();
    for(int t = 0; t < num_throws; ++t)
    {
        t_start_single = std::chrono::high_resolution_clock::now();
        if(t % 1000 == 0)
            std::cout << "[CalcXsec]: Throw " << t << "/" << num_throws << std::endl;

        toy_thrower.Throw(toy);
        throws.push_back(toy);
        toy_param = MapThrow(toy, postfit_param, fitpara);

        bin = 1;
        const auto h_name = "combined_throw_" + std::to_string(t);
        TH1D h_throw(h_name.c_str(), h_name.c_str(), nfitbins, 0, nfitbins);
        h_throw.SetDirectory(0);

        for(int s = 0; s < samples.size(); ++s)
        {
            const unsigned int num_events = samples[s] -> GetN();
            for(unsigned int i = 0; i < num_events; ++i)
            {
                AnaEvent* ev = samples[s] -> GetEvent(i);
                ev -> ResetEvWght();
                for(int f = 0; f < fitpara.size(); ++f)
                {
                    const std::string det(samples[s] -> GetDetector());
                    fitpara[f] -> ReWeight(ev, det, s, i, toy_param.at(f));
                }
            }

            samples[s] -> FillEventHisto(0);
            const auto h_pred = samples[s] -> GetSignalHisto();

            //std::string hist_name = samples[s] -> GetName() + "_throw" + std::to_string(t);
            //h_pred -> Write(hist_name.c_str());

            for(int j = 1; j <= h_pred -> GetNbinsX(); ++j)
                h_throw.SetBinContent(bin++, h_pred -> GetBinContent(j));
        }

        xsec_throws.push_back(h_throw);
        t_end_single = std::chrono::high_resolution_clock::now();
        //h_throw.Write();
    }
    std::cout << "[CalcXsec]: Throws Finished." << std::endl;
    auto t_end_throws = std::chrono::high_resolution_clock::now();

    TH1D h_throw("ht", "ht", 300, -3.0, 3.0);
    for(int t = 0; t < num_throws; ++t)
    {
        //for(int i = 0; i < sigfitpara.GetNpar(); ++i)
            h_throw.Fill(throws.at(t).at(0) + postfit_param.at(0));
    }

    TH1D h_mean("h_mean", "h_mean", nfitbins, 0, nfitbins);
    //TH1D h_throw("h_throw", "h_throw", 100, 100, 300);
    for(const auto& h : xsec_throws)
    {
        for(int i = 0; i < nfitbins; ++i)
            h_mean.Fill(i + 0.5, h.GetBinContent(i+1));
        //h_throw.Fill(h.GetBinContent(1));
    }
    h_mean.Scale(1.0 / (1.0 * num_throws));

    auto t_start_cov = std::chrono::high_resolution_clock::now();
    //for(int t = 0; t < num_throws; ++t)
    for(const auto& h : xsec_throws)
    {
        for(int i = 0; i < nfitbins; ++i)
        {
            for(int j = 0; j < nfitbins; ++j)
            {
                //const double x = throws.at(t).at(i);
                //const double y = throws.at(t).at(j);
                const double x = h.GetBinContent(i+1) - h_postfit.GetBinContent(i+1);
                const double y = h.GetBinContent(j+1) - h_postfit.GetBinContent(j+1);
                xsec_cov(i,j) += x * y / (1.0 * num_throws);
            }
        }
    }
    auto t_end_cov = std::chrono::high_resolution_clock::now();

    for(int i = 0; i < nfitbins; ++i)
    {
        for(int j = 0; j < nfitbins; ++j)
        {
            const double z = xsec_cov(i,j);
            const double x = xsec_cov(i,i);
            const double y = xsec_cov(j,j);
            xsec_cor(i,j) = z / (sqrt(x * y));
        }
    }

    for(int i = 1; i <= h_postfit.GetNbinsX(); ++i)
        h_postfit.SetBinError(i, sqrt(xsec_cov(i-1,i-1)));

    foutput -> cd();
    h_postfit.Write("h_postfit");
    postfit_cov -> Write("postfit_cov");
    postfit_cor -> Write("postfit_cor");
    postfit_param_root -> Write("postfit_params");
    xsec_cov.Write("xsec_cov");
    xsec_cor.Write("xsec_cor");
    h_throw.Write("h_throw");
    h_mean.Write("h_mean");
    h_nt.Write("h_ntargets");

    auto t_end_total = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> t_total = t_end_total - t_start_total;
    std::chrono::duration<double> t_samples = t_end_samples - t_start_samples;
    std::chrono::duration<double> t_throws = t_end_throws - t_start_throws;
    std::chrono::duration<double> t_single = t_end_single - t_start_single;
    std::chrono::duration<double> t_cov = t_end_cov - t_start_cov;

    std::cout << "------ Time -------" << std::endl;
    std::cout << "Total  : " << t_total.count() << std::endl;
    std::cout << "Samples: " << t_samples.count() << std::endl;
    std::cout << "Throws : " << t_throws.count() << std::endl;
    std::cout << "Single : " << t_single.count() << std::endl;
    std::cout << "Cov    : " << t_cov.count() << std::endl;

    return 0;
}

std::vector<std::vector<double> > MapThrow(const std::vector<double>& toy,
                                           const std::vector<double>& nom,
                                           const std::vector<AnaFitParameters*>& fit)
{
    std::vector<std::vector<double> > throw_vector;
    std::vector<double> param(toy.size(), 0);
    std::transform(toy.begin(), toy.end(), nom.begin(), param.begin(), std::plus<double>());
    for(int i = 0; i < param.size(); ++i)
    {
        if(param[i] < 0.0)
            param[i] = 0.0;
    }

    auto start = param.begin();
    auto end = param.begin();
    for(const auto& param_type : fit)
    {
        start = end;
        end = start + param_type -> GetNpar();
        throw_vector.emplace_back(std::vector<double>(start, end));
    }

    return throw_vector;
}
