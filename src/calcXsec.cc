#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
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

void ReweightEvents(std::vector<AnaSample*>& samples,
                    std::vector<AnaFitParameters*>& fitpara,
                    std::map<std::string, XsecExtractor>& xsec_calc,
                    std::map<std::string, TH1D>& h_postfit_map,
                    std::vector<std::vector<double>>& toy_param);

void ApplyNormalization(std::map<std::string, XsecExtractor>& xsec_calc,
                        std::map<std::string, TH1D>& h_postfit_map,
                        bool do_throw);

void CalcEfficiency(std::map<std::string, XsecExtractor>& xsec_calc,
                    std::map<std::string, TH1D>& h_selected,
                    std::map<std::string, TH1D>& h_true,
                    std::map<std::string, TH1D>& h_eff);

void ConcatHistograms(TH1D& h_postfit,
                      std::map<std::string, TH1D>& h_postfit_map,
                      const std::vector<std::string>& xsec_order);

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
    TTree* tdata_sel = (TTree*)(fdata->Get("selectedEvents"));
    TTree* tdata_tru = (TTree*)(fdata->Get("trueEvents"));

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
    std::vector<AnaSample*> samples_true;
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

        auto s = new AnySample(opt.cut_branch, opt.name, opt.detector, D1_edges, D2_edges, tdata_sel, false, opt.use_sample);
        s -> SetNorm(potD/potMC);
        samples.push_back(s);
    }

    std::cout << "[CalcXsec]: Reading and collecting events." << std::endl;
    AnyTreeMC selTree(fname_mc.c_str(), "selectedEvents");
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

    auto t_start_throws = std::chrono::high_resolution_clock::now();
    std::cout << "[CalcXsec]: Throwing " << num_throws << " toys." << std::endl;
    const int npar = postfit_cov -> GetNrows();
    const int nfitbins = sigfitpara.GetNpar();
    TMatrixD xsec_cov(nfitbins, nfitbins);
    TMatrixD xsec_cor(nfitbins, nfitbins);
    xsec_cov.Zero();

    std::vector<double> postfit_param;
    for(int i = 0; i < npar; ++i)
        postfit_param.push_back((*postfit_param_root)[i]);

    std::map<std::string, XsecExtractor> xsec_calc;
    std::vector<std::string> xsec_order;
    for(const auto& det : parser.detectors)
    {
        std::cout << "[CalcXsec]: Adding detector for cross section: " << det.name << std::endl;
        XsecExtractor x(det.name, det.binning, ++seed);
        x.SetNumTargets(det.ntargets_val, det.ntargets_err);
        x.SetFluxVar(det.flux_integral, det.flux_error);
        xsec_calc.emplace(std::make_pair(det.name, x));
        xsec_order.emplace_back(det.name);
    }

    std::map<std::string, TH1D> h_postfit_map;
    std::map<std::string, TH1D> h_postfit_map_true;
    std::map<std::string, TH1D> h_efficiency;
    for(const auto& det : xsec_calc)
    {
        const auto nbins = det.second.GetNbins();
        const auto name  = det.second.GetName();
        std::cout << "[XsecExtractor]: Adding " << name << " with " << nbins << " bins to postfit." << std::endl;
        h_postfit_map.emplace(std::make_pair(name, TH1D("", "", nbins, 0, nbins)));
        h_postfit_map_true.emplace(std::make_pair(name, TH1D("", "", nbins, 0, nbins)));
        h_efficiency.emplace(std::make_pair(name, TH1D("", "", nbins, 0, nbins)));
    }

    TH1D h_postfit("h_postfit", "h_postfit", nfitbins, 0, nfitbins);

    TH1D h_flux("h_flux", "h_flux", 200, 7E12, 2E13);
    for(int i = 0; i < num_throws; ++i)
        h_flux.Fill(xsec_calc.at("ND280").ThrowFlux());

    ToyThrower toy_thrower(*postfit_cov, seed, 1E-24);
    std::vector<TH1D> xsec_throws;
    xsec_throws.reserve(num_throws);
    std::vector<std::vector<double> > throws;
    std::vector<double> toy(npar, 0);

    auto toy_param = MapThrow(toy, postfit_param, fitpara);
    ReweightEvents(samples, fitpara, xsec_calc, h_postfit_map, toy_param);

    CalcEfficiency(xsec_calc, h_postfit_map, h_postfit_map_true, h_efficiency);
    ApplyNormalization(xsec_calc, h_postfit_map, false);
    ConcatHistograms(h_postfit, h_postfit_map, xsec_order);

    /*
    for(int s = 0; s < samples.size(); ++s)
    {
        const unsigned int num_events = samples[s] -> GetN();
        const std::string det(samples[s] -> GetDetector());
        for(unsigned int i = 0; i < num_events; ++i)
        {
            AnaEvent* ev = samples[s] -> GetEvent(i);
            ev -> SetEvWght(ev -> GetEvWghtMC());
            for(int f = 0; f < fitpara.size(); ++f)
            {
                fitpara[f] -> ReWeight(ev, det, s, i, toy_param.at(f));
            }

            if(ev -> isSignalEvent())
            {
                const auto idx = xsec_calc.at(det).GetAnyBinIndex(ev -> GetTrueD1(), ev -> GetTrueD2());
                h_postfit_map.at(det).Fill(idx + 0.5, ev -> GetEvWght());
            }
        }
    }

    for(auto& kv : h_postfit_map)
    {
        xsec_calc.at(kv.first).ApplyBinWidths(kv.second);
        xsec_calc.at(kv.first).ApplyNumTargets(kv.second, false);
        xsec_calc.at(kv.first).ApplyFluxInt(kv.second, false);
    }

    auto bin = 1;
    for(const auto& det : xsec_order)
    {
        const auto h = h_postfit_map.at(det);
        for(int j = 1; j <= h.GetNbinsX(); ++j)
            h_postfit.SetBinContent(bin++, h.GetBinContent(j));
        h_postfit_map.at(det).Reset();
    }
    */

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

        const auto h_name = "combined_throw_" + std::to_string(t);
        TH1D h_throw(h_name.c_str(), h_name.c_str(), nfitbins, 0, nfitbins);
        h_throw.SetDirectory(0);

        ReweightEvents(samples, fitpara, xsec_calc, h_postfit_map, toy_param);
        ApplyNormalization(xsec_calc, h_postfit_map, true);
        ConcatHistograms(h_throw, h_postfit_map, xsec_order);

        /*
        for(int s = 0; s < samples.size(); ++s)
        {
            const unsigned int num_events = samples[s] -> GetN();
            const std::string det(samples[s] -> GetDetector());
            for(unsigned int i = 0; i < num_events; ++i)
            {
                AnaEvent* ev = samples[s] -> GetEvent(i);
                ev -> ResetEvWght();
                for(int f = 0; f < fitpara.size(); ++f)
                {
                    fitpara[f] -> ReWeight(ev, det, s, i, toy_param.at(f));
                }

                if(ev -> isSignalEvent())
                {
                    const auto idx = xsec_calc.at(det).GetAnyBinIndex(ev -> GetTrueD1(), ev -> GetTrueD2());
                    h_postfit_map.at(det).Fill(idx + 0.5, ev -> GetEvWght());
                }
            }
        }

        for(auto& kv : h_postfit_map)
        {
            xsec_calc.at(kv.first).ApplyBinWidths(kv.second);
            xsec_calc.at(kv.first).ApplyNumTargets(kv.second, false);
            xsec_calc.at(kv.first).ApplyFluxInt(kv.second, true);
        }

        auto bin = 1;
        for(const auto& det : xsec_order)
        {
            const auto h = h_postfit_map.at(det);
            for(int j = 1; j <= h.GetNbinsX(); ++j)
                h_throw.SetBinContent(bin++, h.GetBinContent(j));
            h_postfit_map.at(det).Reset();
        }
        */
        xsec_throws.push_back(h_throw);
        t_end_single = std::chrono::high_resolution_clock::now();
    }
    std::cout << "[CalcXsec]: Throws Finished." << std::endl;
    auto t_end_throws = std::chrono::high_resolution_clock::now();

    TH1D h_mean("h_mean", "h_mean", nfitbins, 0, nfitbins);
    for(const auto& h : xsec_throws)
    {
        for(int i = 0; i < nfitbins; ++i)
            h_mean.Fill(i + 0.5, h.GetBinContent(i+1));
    }
    h_mean.Scale(1.0 / (1.0 * num_throws));

    auto t_start_cov = std::chrono::high_resolution_clock::now();
    for(const auto& h : xsec_throws)
    {
        for(int i = 0; i < nfitbins; ++i)
        {
            for(int j = 0; j < nfitbins; ++j)
            {
                const double x = h.GetBinContent(i+1) - h_postfit.GetBinContent(i+1);
                const double y = h.GetBinContent(j+1) - h_postfit.GetBinContent(j+1);
                xsec_cov(i,j) += x * y / (1.0 * num_throws);
            }
        }
    }

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
    auto t_end_cov = std::chrono::high_resolution_clock::now();

    foutput -> cd();
    h_postfit.Write("h_postfit");
    postfit_cov -> Write("postfit_cov");
    postfit_cor -> Write("postfit_cor");
    postfit_param_root -> Write("postfit_params");
    xsec_cov.Write("xsec_cov");
    xsec_cor.Write("xsec_cor");
    h_mean.Write("h_mean");
    h_flux.Write("h_flux");

    h_efficiency.at("ND280").Write("h_eff");

    foutput -> Close();
    delete foutput;

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

void ReweightEvents(std::vector<AnaSample*>& samples,
                    std::vector<AnaFitParameters*>& fitpara,
                    std::map<std::string, XsecExtractor>& xsec_calc,
                    std::map<std::string, TH1D>& h_postfit_map,
                    std::vector<std::vector<double>>& toy_param)
{
    for(int s = 0; s < samples.size(); ++s)
    {
        const unsigned int num_events = samples[s] -> GetN();
        const std::string det(samples[s] -> GetDetector());
        for(unsigned int i = 0; i < num_events; ++i)
        {
            AnaEvent* ev = samples[s] -> GetEvent(i);
            ev -> SetEvWght(ev -> GetEvWghtMC());
            for(int f = 0; f < fitpara.size(); ++f)
            {
                fitpara[f] -> ReWeight(ev, det, s, i, toy_param.at(f));
            }

            if(ev -> isSignalEvent())
            {
                const auto idx = xsec_calc.at(det).GetAnyBinIndex(ev -> GetTrueD1(), ev -> GetTrueD2());
                h_postfit_map.at(det).Fill(idx + 0.5, ev -> GetEvWght());
            }
        }
    }

}

void ApplyNormalization(std::map<std::string, XsecExtractor>& xsec_calc,
                        std::map<std::string, TH1D>& h_postfit_map,
                        bool do_throw)
{
    for(auto& kv : h_postfit_map)
    {
        xsec_calc.at(kv.first).ApplyBinWidths(kv.second);
        xsec_calc.at(kv.first).ApplyNumTargets(kv.second, do_throw);
        xsec_calc.at(kv.first).ApplyFluxInt(kv.second, do_throw);
    }
}

void CalcEfficiency(std::map<std::string, XsecExtractor>& xsec_calc,
                    std::map<std::string, TH1D>& h_selected,
                    std::map<std::string, TH1D>& h_true,
                    std::map<std::string, TH1D>& h_eff)
{
    for(const auto& kv : xsec_calc)
    {
        const auto det = kv.first;
        for(int i = 1; i <= h_eff.at(det).GetNbinsX(); ++i)
        {
            auto eff = h_selected.at(det).GetBinContent(i) / h_true.at(det).GetBinContent(i);
            std::cout << "Bin " << i << " Eff: " << eff << std::endl;
            h_eff.at(det).SetBinContent(i, eff);
        }
    }
}

void ConcatHistograms(TH1D& h_postfit,
                      std::map<std::string, TH1D>& h_postfit_map,
                      const std::vector<std::string>& xsec_order)
{
    auto bin = 1;
    for(const auto& det : xsec_order)
    {
        const auto h = h_postfit_map.at(det);
        for(int j = 1; j <= h.GetNbinsX(); ++j)
            h_postfit.SetBinContent(bin++, h.GetBinContent(j));
        h_postfit_map.at(det).Reset();
    }
}
