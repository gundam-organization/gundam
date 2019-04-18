#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <unistd.h>

#include <TCanvas.h>
#include <TChain.h>
#include <TFile.h>
#include <TGraph.h>
#include <TH1.h>
#include <TLegend.h>
#include <TSpline.h>
#include <TStyle.h>
#include <TTree.h>

#include "BinManager.h"
#include "XsecDial.h"

const std::string RESET("\033[0m");
const std::string RED("\033[31;1m");
const std::string GREEN("\033[92m");
const std::string COMMENT_CHAR("#");

const std::string TAG = GREEN + "[xsValidate]: " + RESET;
const std::string ERR = RED + "[ERROR]: " + RESET;

int main(int argc, char** argv)
{
    std::cout << "--------------------------------------------------------\n"
              << TAG << "Welcome to the Super-xsLLhSpline Validator.\n"
              << TAG << "Initializing the spline machinery..." << std::endl;

    std::string dial_name;
    std::string fname_input;
    std::string fname_output;
    std::string fname_binning;
    std::string fname_splines;
    std::string fname_dials;

    unsigned int nbins = 0;
    unsigned int dsteps = 7;
    unsigned int ntopo = 8;
    unsigned int nreac = 10;
    unsigned int nsamp = 10;

    double nominal{0};
    double error_neg{0};
    double error_pos{0};

    std::map<std::string, int> m_opt;
    bool limits = false;
    bool update = false;
    bool ratio_to_nominal = false;
    bool do_normalize_dials = false;

    char option;
    while((option = getopt(argc, argv, "i:o:b:s:r:n:d:S:Z:T:R:LDNUh")) != -1)
    {
        switch(option)
        {
            case 'i':
                fname_input = optarg;
                break;
            case 'o':
                fname_output = optarg;
                break;
            case 'b':
                fname_binning = optarg;
                break;
            case 's':
                fname_splines = optarg;
                break;
            case 'd':
                fname_dials = optarg;
                break;
            case 'r':
                dial_name = optarg;
                break;
            case 'n':
                dsteps = std::stoi(optarg);
                break;
            case 'S':
                m_opt["sam"] = std::stoi(optarg);
                break;
            case 'Z':
                m_opt["tar"] = std::stoi(optarg);
                break;
            case 'T':
                m_opt["top"] = std::stoi(optarg);
                break;
            case 'R':
                m_opt["rea"] = std::stoi(optarg);
                break;
            case 'L':
                limits = true;
                break;
            case 'D':
                ratio_to_nominal = true;
                break;
            case 'N':
                do_normalize_dials = true;
                break;
            case 'U':
                update = true;
                break;
            case 'h':
                std::cout << "USAGE: " << argv[0] << "\nOPTIONS\n"
                          << "-i : Input weights file (.root)\n"
                          << "-o : Output ROOT file\n"
                          << "-b : Binning file (.txt)\n"
                          << "-s : Splines file (.root)\n"
                          << "-r : Systematic parameter to validate\n"
                          << "-d : Dial value file (.txt)\n"
                          << "-n : Number of dial steps\n"
                          << "-S : Sample to validate\n"
                          << "-Z : Target to validate\n"
                          << "-T : Topology to validate\n"
                          << "-R : Reaction to validate\n"
                          << "-L : Read errors as upper and lower limits\n"
                          << "-D : Take ratio to nominal\n"
                          << "-N : Normalize dial values\n"
                          << "-U : Update output file\n"
                          << "-h : Display this help message\n";
            default:
                return 0;
        }
    }

    if(fname_input.empty() || fname_output.empty() || fname_binning.empty()
       || dial_name.empty() || fname_dials.empty() || fname_splines.empty())
    {
        std::cerr << ERR << "Missing one or more required arguments!" << std::endl;
        return 1;
    }

    std::cout << TAG << "Input weights file: " << fname_input << std::endl
              << TAG << "Input spline file: " << fname_splines << std::endl
              << TAG << "Input binning file: " << fname_binning << std::endl
              << TAG << "Input dial value file: " << fname_dials << std::endl
              << TAG << "Output ROOT file: " << fname_output << std::endl
              << TAG << "Validating splines for " << dial_name << std::endl
              << TAG << "Number of dial steps: " << dsteps << std::endl;

    std::cout << TAG << "Selected Events:\n";
    for(const auto& kv : m_opt)
        std::cout << kv.first << ": " << kv.second << std::endl;

    std::ifstream fdial(fname_dials, std::ios::in);
    if(!fdial.is_open())
    {
        std::cerr << ERR << "Failed to open " << fname_dials << std::endl;
        return 1;
    }
    else
    {
        std::string line;
        while(std::getline(fdial, line))
        {
            std::stringstream ss(line);
            std::string dial;
            double nom{0}, err_n{0}, err_p{0};

            ss >> dial >> nom >> err_n >> err_p;
            if(ss.str().front() == COMMENT_CHAR)
                continue;

            if(dial == dial_name)
            {
                nominal = nom;
                error_neg = err_n;
                error_pos = err_p;
                break;
            }
        }
        fdial.close();
    }

    std::cout << TAG << "Dial Parameter: " << dial_name << std::endl
              << TAG << "Nominal: " << nominal << std::endl
              << TAG << "Error -: " << error_neg << std::endl
              << TAG << "Error +: " << error_pos << std::endl;

    BinManager bin_manager(fname_binning);
    bin_manager.Print();
    nbins = bin_manager.GetNbins();

    std::cout << TAG << "Loaded binning file." << std::endl;

    XsecDial dial(dial_name, fname_binning, fname_splines);
    dial.SetVars(nominal, 0.05, 0.0, 2.0);
    //dial.SetDimensions(ntopo, nreac);
    dial.SetDimensions(std::vector<int>{nreac*nbins, nbins});
    dial.Print();

    TFile* f_splines = TFile::Open(fname_splines.c_str(), "READ");
    std::vector<TGraph*> v_splines;
    for(int t = 0; t < ntopo; ++t)
    {
        for(int r = 0; r < nreac; ++r)
        {
            for(int b = 0; b < nbins; ++b)
            {
                std::stringstream ss;
                ss << "spline_top" << t << "_reac" << r << "_bin" << b;
                TGraph* spline = (TGraph*)f_splines -> Get(ss.str().c_str());
                v_splines.push_back(spline);
            }
        }
    }

    std::cout << TAG << "Num. Splines: " << v_splines.size() << std::endl;

    int bin_index{0};
    int topology{0}, reaction{0}, sample{0}, target{0};
    float enu{0}, emu{0};
    float pmu{0}, cosmu{0}, q2{0};
    float pmu_reco{0}, cosmu_reco{0};
    float weight_nom{0};
    float weight_syst[dsteps];
    const double mu_mass{105.6583745};

    TChain chain("selectedEvents");
    chain.Add(fname_input.c_str());

    chain.SetBranchAddress("Enutrue", &enu);
    chain.SetBranchAddress("Pmutrue", &pmu);
    chain.SetBranchAddress("Pmureco", &pmu_reco);
    chain.SetBranchAddress("CTHmutrue", &cosmu);
    chain.SetBranchAddress("CTHmureco", &cosmu_reco);
    chain.SetBranchAddress("sample", &sample);
    chain.SetBranchAddress("topology", &topology);
    chain.SetBranchAddress("reaction", &reaction);
    chain.SetBranchAddress("target", &target);
    chain.SetBranchAddress("weight", &weight_nom);
    chain.SetBranchAddress("weight_syst", weight_syst);

    std::cout << TAG << "Tree opened and branches set; reading events." << std::endl;
    const unsigned int num_events = chain.GetEntries();
    std::map<std::string, int> m_evt;
    std::vector<TH1D> v_hists_sp;
    std::vector<TH1D> v_hists_rw;

    double dial_value[dsteps];
    for(unsigned int w = 0; w < dsteps; ++w)
    {
        int step = -1.0 * std::floor(dsteps / 2.0) + w;
        if(limits)
        {
            //double inc = (error_pos - error_neg) / (dsteps - 1);
            double lim = step < 0 ? error_neg : error_pos;
            double inc = 2 * std::abs(lim - nominal) / (dsteps - 1);
            dial_value[w] = nominal + step * inc;
        }
        else
        {
            if(step < 0)
                dial_value[w] = nominal + step * error_neg;
            else
                dial_value[w] = nominal + step * error_pos;
        }
        //std::cout.unsetf(std::ios::fixed);
        std::cout << "Dial Step: " << step << "; Value: " << dial_value[w] << std::endl;
    }

    const int hist_nbins{20};
    const double hist_low{0};
    const double hist_high{5000};

    for(int d = 0; d < dsteps; ++d)
    {
        std::stringstream ss;
        ss << dial_name << "_spline_step" << d;
        //std::cout << ss.str() << std::endl;
        v_hists_sp.emplace_back(TH1D(ss.str().c_str(), ss.str().c_str(), hist_nbins, hist_low, hist_high));

        ss.str("");
        ss << dial_name << "_reweight_step" << d;
        //std::cout << ss.str() << std::endl;
        v_hists_rw.emplace_back(TH1D(ss.str().c_str(), ss.str().c_str(), hist_nbins, hist_low, hist_high));

        for(int i = 0; i < num_events; ++i)
        {
            chain.GetEntry(i);

            m_evt["sam"] = sample;
            m_evt["top"] = topology;
            m_evt["tar"] = target;
            m_evt["rea"] = reaction;

            bool fill_event = true;
            for(const auto& pear : m_opt)
            {
                if(m_evt.at(pear.first) != pear.second)
                    fill_event = false;
            }
            if(fill_event == false)
                continue;

            emu = std::sqrt(pmu*pmu + mu_mass*mu_mass);
            q2 = 2.0 * enu * (emu - pmu * cosmu) - mu_mass * mu_mass;
            q2 = q2 / 1.0E6;

            unsigned int b = bin_manager.GetBinIndex(std::vector<double>{cosmu, pmu});
            unsigned int idx = topology * nsamp * nbins + reaction * nbins + b;

            //unsigned int dial_idx = dial.GetSplineIndex(topology, reaction, q2);
            unsigned int dial_idx = dial.GetSplineIndex(std::vector<int>{topology, reaction},
                                                        std::vector<double>{cosmu, pmu});
            double x_value = dial_value[d];
            if(do_normalize_dials)
                x_value = x_value / nominal;

            double dial_weight = dial.GetSplineValue(dial_idx, x_value);

            double weight = v_splines.at(idx) -> Eval(x_value);
            if(fill_event)
            {
                v_hists_sp.at(d).Fill(pmu, weight_nom * weight);
                v_hists_rw.at(d).Fill(pmu, weight_nom * weight_syst[d]);
            }

            if(idx != dial_idx)
            {
                std::cout << ERR << "Indexes do not match.\n"
                          << ERR << idx << " vs. " << dial_idx << std::endl;
            }

            if(weight != dial_weight)
            {
                std::cout << ERR << "Weights do not match.\n"
                          << ERR << weight << " vs. " << dial_weight << std::endl;
            }

            /*
            std::cout << "Q2 (GeV): " << q2 << std::endl
                      << "Q2 Bin  : " << b << std::endl
                      << "Nominal : " << weight_nom << std::endl
                      << "Weighted: " << weight << std::endl
                      << "Topo/Reac  : " << topology << " / " << reaction << std::endl
                      << "Spline IDX : " << idx << std::endl
                      << "Spline  : " << v_splines.at(idx) -> GetName() << std::endl;
            */
        }
    }

    std::cout << TAG << "Drawing histograms." << std::endl;
    int hist_color[] = {kBlack, kGreen, kRed, kBlue, kMagenta, kYellow, kCyan};
    std::string title = dial_name + "_spline_validation";

    gStyle -> SetOptStat(0);
    TCanvas canvas_sp(title.c_str(), title.c_str(), 1200, 900);
    v_hists_sp.back().Draw("axis");
    for(int i = 0; i < v_hists_sp.size(); ++i)
    {
        int j = -1.0 * std::floor(dsteps / 2.0) + i;
        v_hists_sp.at(i).SetLineWidth(2);
        if(std::signbit(j))
            v_hists_sp.at(i).SetLineStyle(kDashed);
        v_hists_sp.at(i).SetLineColor(hist_color[std::abs(j)]);
        v_hists_sp.at(i).Draw("hist same");
    }

    if(ratio_to_nominal)
    {
        auto hist_nom = v_hists_sp.at(std::floor(dsteps / 2.0));
        for(auto& hist : v_hists_sp)
            hist.Divide(&hist, &hist_nom, 1.0, 1.0);
    }

    TLegend legend_sp(0.55, 0.60, 0.85, 0.80);
    legend_sp.SetFillStyle(0);
    legend_sp.SetBorderSize(0);
    legend_sp.SetHeader("Dial Spline Validation");
    for(int i = 0; i < v_hists_sp.size(); ++i)
    {
        std::stringstream ss;
        ss << dial_name << "_value_" << dial_value[i];
        legend_sp.AddEntry(&v_hists_sp.at(i), ss.str().c_str(), "L");
    }
    legend_sp.Draw("same");

    title = dial_name + "_reweight_validation";
    TCanvas canvas_rw(title.c_str(), title.c_str(), 1200, 900);
    v_hists_rw.back().Draw("axis");
    for(int i = 0; i < v_hists_rw.size(); ++i)
    {
        int j = -1.0 * std::floor(dsteps / 2.0) + i;
        v_hists_rw.at(i).SetLineWidth(2);
        if(std::signbit(j))
            v_hists_rw.at(i).SetLineStyle(kDashed);
        v_hists_rw.at(i).SetLineColor(hist_color[std::abs(j)]);
        v_hists_rw.at(i).Draw("hist same");
    }

    if(ratio_to_nominal)
    {
        auto hist_nom = v_hists_rw.at(std::floor(dsteps / 2.0));
        for(auto& hist : v_hists_rw)
            hist.Divide(&hist, &hist_nom, 1.0, 1.0);
    }

    TLegend legend_rw(0.55, 0.60, 0.85, 0.80);
    legend_rw.SetFillStyle(0);
    legend_rw.SetBorderSize(0);
    legend_rw.SetHeader("Dial ReWeight Validation");
    for(int i = 0; i < v_hists_rw.size(); ++i)
    {
        std::stringstream ss;
        ss << dial_name << "_value_" << dial_value[i];
        legend_rw.AddEntry(&v_hists_rw.at(i), ss.str().c_str(), "L");
    }
    legend_rw.Draw("same");

    title = dial_name + "_ratio_validation";
    TCanvas canvas_ratio(title.c_str(), title.c_str(), 1200, 900);
    canvas_ratio.SetGrid();
    canvas_ratio.SetTicks();

    TLegend legend_ratio(0.55, 0.60, 0.85, 0.80);
    legend_ratio.SetFillStyle(0);
    legend_ratio.SetBorderSize(0);
    legend_ratio.SetHeader("Dial Validation");

    for(int d = 0; d < dsteps; ++d)
    {
        std::stringstream ss;
        ss << dial_name << "_ratio";
        for(const auto& pear : m_opt)
            ss << "_" << pear.first << pear.second;
        auto hist_ratio = (TH1D*)v_hists_rw.at(d).Clone(ss.str().c_str());
        hist_ratio -> Divide(&v_hists_sp.at(d), &v_hists_rw.at(d), 1.0, 1.0);
        hist_ratio -> GetYaxis() -> SetRangeUser(0.75, 1.25);
        hist_ratio -> SetTitle(ss.str().c_str());
        hist_ratio -> Draw("hist same");

        ss.str("");
        ss << dial_name << "_value_" << dial_value[d];
        legend_ratio.AddEntry(hist_ratio, ss.str().c_str(), "L");
    }
    legend_ratio.Draw("same");

    std::cout << TAG << "Saving histograms." << std::endl;
    TFile* f_output = TFile::Open(fname_output.c_str(), update ? "UPDATE" : "RECREATE");
    f_output -> cd();
    for(const auto& hist : v_hists_sp)
        hist.Write();
    for(const auto& hist : v_hists_rw)
        hist.Write();
    canvas_sp.Write();
    canvas_rw.Write();
    canvas_ratio.Write();
    f_output -> Close();

    std::cout << TAG << "Finished." << std::endl;
    return 0;
}
