#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

#include "TCanvas.h"
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TMatrixT.h"
#include "TMatrixTSym.h"
#include "TStyle.h"
#include "TTree.h"

#include "BinManager.hh"
#include "ColorOutput.hh"
#include "ProgressBar.hh"

int main(int argc, char** argv)
{
    const std::string TAG = color::GREEN_STR + "[xsDetVariation]: " + color::RESET_STR;
    const std::string ERR = color::RED_STR + color::BOLD_STR + "[ERROR]: " + color::RESET_STR;

    std::cout << "--------------------------------------------------------\n"
              << TAG << "Welcome to the Super-xsLLh Detector Variation Interface.\n"
              << TAG << "Initializing the variation machinery..." << std::endl;

    const unsigned int num_syst = 17;
    unsigned int syst_idx = 0;
    unsigned int num_toys = 0;
    double weight_cut = 1.0E5;
    bool do_single_variation = false;
    bool do_covariance = false;
    bool do_projection = false;
    bool do_print = false;
    std::string fname_input;
    std::string fname_output;
    std::string fname_binning;
    std::string variable_name;
    std::string variable_plot;
    std::string tree_name;
    std::string cut_samples;
    std::vector<std::string> var_names;

    const int num_samples = 10;
    const int cut_level[num_samples] = {7, 8, 9, 8, 7, 5, 4, 7, 8, 7};

    ProgressBar pbar(60, "#");
    pbar.SetRainbow();
    pbar.SetPrefix(std::string(TAG + "Reading Events "));

    char option;
    while((option = getopt(argc, argv, "i:o:b:t:v:s:n:p:c:w:CPh")) != -1)
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
            case 't':
                tree_name = optarg;
                break;
            case 'v':
                variable_name = optarg;
                break;
            case 's':
                syst_idx = std::stoi(optarg);
                do_single_variation = true;
                break;
            case 'n':
                num_toys = std::stoi(optarg);
                break;
            case 'p':
                variable_plot = optarg;
                do_projection = true;
                break;
            case 'w':
                weight_cut = std::stol(optarg);
                break;
            case 'c':
                cut_samples = optarg;
                break;
            case 'C':
                do_covariance = true;
                break;
            case 'P':
                do_print = true;
                break;
            case 'h':
                std::cout << "USAGE: " << argv[0] << "\nOPTIONS\n"
                          << "-i : Input HighLAND file (.root)\n"
                          << "-o : Output ROOT filename\n"
                          << "-b : Binning file (.txt)\n"
                          << "-t : Name of root tree\n"
                          << "-v : Name of variable(s)\n"
                          << "-s : Index of systematic\n"
                          << "-n : Number of toy throws\n"
                          << "-p : Variable to plot in 1D\n"
                          << "-w : Cut value for toy weights\n"
                          << "-c : Cut Branches (samples) to use\n"
                          << "-C : Calculate covariance matrix\n"
                          << "-P : Print plots to PDF\n"
                          << "-h : Display this help message\n";
            default:
                return 0;
        }
    }

    if(fname_input.empty() || fname_output.empty() || fname_binning.empty())
    {
        std::cout << ERR << "Missing necessary command line arguments.\n" << std::endl;
        return 1;
    }

    std::stringstream ss_var(variable_name);
    for(std::string s; std::getline(ss_var, s, ',');)
        var_names.emplace_back(s);

    std::vector<unsigned int> use_samples;
    std::stringstream ss_cut(cut_samples);
    for(std::string s; std::getline(ss_cut, s, ',');)
        use_samples.emplace_back(std::stoi(s));

    BinManager bin_manager(fname_binning);
    bin_manager.Print();
    const int nbins = bin_manager.GetNbins();
    const int nvars = var_names.size();
    const int ntoys = num_toys;

    std::cout << TAG << "Input HighLAND file: " << fname_input << std::endl
              << TAG << "Output ROOT file: " << fname_output << std::endl
              << TAG << "Input binning file: " << fname_binning << std::endl
              << TAG << "Name of ROOT tree: " << tree_name << std::endl
              << TAG << "Name of variable(s): " << variable_name << std::endl
              << TAG << "Number of toy throws: " << num_toys << std::endl
              << TAG << "Number of total bins: " << nbins << std::endl
              << TAG << "Toy Weight Cut: " << weight_cut << std::endl
              << TAG << "Calculating Covariance: " << std::boolalpha << do_covariance << std::endl
              << TAG << "Covariance Samples: " << cut_samples << std::endl;

    int sample;
    int accum_level[ntoys][num_samples];
    float hist_variables[nvars][ntoys];
    float weight_syst_total_noflux[ntoys];
    float weight_syst[ntoys][num_syst];

    TFile* file_input = TFile::Open(fname_input.c_str(), "READ");
    TTree* tree_event = (TTree*)file_input -> Get(tree_name.c_str());

    tree_event -> SetBranchAddress("accum_level", accum_level);
    tree_event -> SetBranchAddress("weight_syst", weight_syst);
    tree_event -> SetBranchAddress("weight_syst_total_noflux", weight_syst_total_noflux);

    for(unsigned int i = 0; i < nvars; ++i)
        tree_event -> SetBranchAddress(var_names[i].c_str(), hist_variables[i]);

    int var_plot = -1;
    if(do_projection)
    {
        auto it = std::find(var_names.begin(), var_names.end(), variable_plot);
        var_plot = std::distance(var_names.begin(), it);
    }

    std::cout << TAG << "Initalizing histograms." << std::endl;
    std::vector<std::vector<TH1F>> v_hists;
    std::vector<TH1F> v_avg;
    for(unsigned int s = 0; s < num_samples; ++s)
    {
        v_hists.emplace_back(std::vector<TH1F>());
        for(unsigned int t = 0; t < num_toys; ++t)
        {
            std::stringstream ss;
            ss << "sample" << s << "_toy" << t;
            if(do_projection)
            {
                std::vector<double> v_bins = bin_manager.GetBinVector(var_plot);
                v_hists.at(s).emplace_back(TH1F(ss.str().c_str(), ss.str().c_str(),
                                                v_bins.size()-1, &v_bins[0]));
                if(t == 0)
                {
                    ss.str(""); ss << "sample" << s << "_avg";
                    v_avg.emplace_back(TH1F(ss.str().c_str(), ss.str().c_str(),
                                                v_bins.size()-1, &v_bins[0]));
                }
            }
            else
            {
                v_hists.at(s).emplace_back(TH1F(ss.str().c_str(), ss.str().c_str(),
                                                nbins, 0, nbins));
                if(t == 0)
                {
                    ss.str(""); ss << "sample" << s << "_avg";
                    v_avg.emplace_back(TH1F(ss.str().c_str(), ss.str().c_str(),
                                                nbins, 0, nbins));
                }
            }
        }
    }

    unsigned int rejected_weights = 0;
    unsigned int total_weights = 0;
    const unsigned int num_events = tree_event -> GetEntries();
    for(unsigned int i = 0; i < num_events; ++i)
    {
        tree_event -> GetEntry(i);
        if(i % 2000 == 0 || i == (num_events-1))
            pbar.Print(i, num_events-1);

        for(unsigned int t = 0; t < num_toys; ++t)
        {
            for(unsigned int s = 0; s < num_samples; ++s)
            {
                if(accum_level[t][s] > cut_level[s])
                {
                    float idx = -1;
                    if(do_projection)
                        idx = hist_variables[var_plot][t];
                    else
                    {
                        std::vector<double> vars;
                        for(unsigned int v = 0; v < nvars; ++v)
                            vars.push_back(hist_variables[v][t]);
                        idx = bin_manager.GetBinIndex(vars);
                    }

                    float weight = do_single_variation ? weight_syst[t][syst_idx] : weight_syst_total_noflux[t];
                    if(weight > 0.0 && weight < weight_cut)
                    {
                        v_hists[s][t].Fill(idx, weight);
                        v_avg[s].Fill(idx, weight / num_toys);
                    }
                    else
                        rejected_weights++;
                    total_weights++;
                    break;
                }
            }
        }
    }

    double reject_fraction = (rejected_weights * 1.0) / total_weights;
    std::cout << TAG << "Finished processing events." << std::endl;
    std::cout << TAG << "Total weights: " << total_weights << std::endl;
    std::cout << TAG << "Rejected weights: " << rejected_weights << std::endl;
    std::cout << TAG << "Rejected fraction: " << reject_fraction << std::endl;

    //std::vector<bool> use_sample = {true, true, true, true, false, true, true, false, false, false};
    //const unsigned int num_elements = nbins * std::count(use_sample.begin(), use_sample.end(), true);
    const unsigned int num_elements = nbins * use_samples.size();
    TMatrixTSym<double> cov_mat(num_elements);
    TMatrixTSym<double> cor_mat(num_elements);
    cov_mat.Zero();
    cor_mat.Zero();

    if(do_covariance)
    {
        std::cout << TAG << "Calculating covariance matrix." << std::endl;
        std::vector<std::vector<float>> v_toys;
        std::vector<float> v_mean(num_elements, 0.0);

        for(unsigned int t = 0; t < num_toys; ++t)
        {
            std::vector<float> i_toy;
            for(unsigned int s = 0; s < num_samples; ++s)
            {
                if(std::count(use_samples.begin(), use_samples.end(), s) > 0)
                {
                    for(unsigned int b = 0; b < nbins; ++b)
                        i_toy.push_back(v_hists[s][t].GetBinContent(b+1));
                }
            }
            v_toys.emplace_back(i_toy);
        }

        for(unsigned int t = 0; t < num_toys; ++t)
        {
            for(unsigned int i = 0; i < num_elements; ++i)
                v_mean[i] += v_toys[t][i] / (1.0 * num_toys);
        }

        for(unsigned int t = 0; t < num_toys; ++t)
        {
            for(unsigned int i = 0; i < num_elements; ++i)
            {
                for(unsigned int j = 0; j < num_elements; ++j)
                {
                    if(v_mean[i] != 0 && v_mean[j] != 0)
                    {
                        cov_mat(i,j) += (1.0 - v_toys[t][i]/v_mean[i]) * (1.0 - v_toys[t][j]/v_mean[j])
                                       / (1.0 * num_toys);
                    }
                }
            }
        }

        for(unsigned int i = 0; i < num_elements; ++i)
        {
            if(cov_mat(i,i) <= 0.0)
                cov_mat(i,i) = 1.0;
        }

        for(unsigned int i = 0; i < num_elements; ++i)
        {
            for(unsigned int j = 0; j < num_elements; ++j)
            {
                double bin_i = cov_mat(i,i);
                double bin_j = cov_mat(j,j);
                cor_mat(i,j) = cov_mat(i,j) / std::sqrt(bin_i * bin_j);
                if(std::isnan(cor_mat(i,j)))
                    cor_mat(i,j) = 0;
            }
        }
    }

    std::cout << TAG << "Saving file." << std::endl;
    TFile* file_output = TFile::Open(fname_output.c_str(), "RECREATE");
    file_output -> cd();

    gStyle -> SetOptStat(0);
    for(unsigned int s = 0; s < num_samples; ++s)
    {
        std::stringstream ss;
        ss << "Sample" << s;
        TCanvas c(ss.str().c_str(), ss.str().c_str(), 1200, 900);
        //v_hists[s][0].Draw("axis");
        v_avg[s].Draw("axis");

        for(unsigned int t = 0; t < num_toys; ++t)
        {
            v_hists[s][t].SetLineColor(kRed);
            //v_hists[s][t].Scale(1, "width");
            v_hists[s][t].Draw("hist same");
        }

        v_avg[s].SetLineColor(kBlack);
        v_avg[s].SetLineWidth(2);
        //v_avg[s].Scale(1, "width");
        v_avg[s].GetYaxis() -> SetRangeUser(0, v_avg[s].GetMaximum()*1.50);
        v_avg[s].Draw("hist same");
        c.Write(ss.str().c_str());

        if(do_print)
            c.Print(std::string(ss.str() + ".pdf").c_str());
    }

    if(do_covariance)
    {
        cov_mat.Write("cov_mat");
        cor_mat.Write("cor_mat");
    }

    file_output -> Close();

    std::cout << TAG << "Finished." << std::endl;
    return 0;
}
