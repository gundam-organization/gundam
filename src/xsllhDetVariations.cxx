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

#include "json.hpp"
using json = nlohmann::json;

#include "BinManager.hh"
#include "ColorOutput.hh"
#include "ProgressBar.hh"

struct FileOptions
{
    std::string fname_input;
    std::string fname_binning;
    std::string tree_name;
    std::string detector;
    unsigned int num_samples;
    unsigned int num_toys;
    unsigned int num_syst;
    std::vector<int> samples;
    std::vector<int> cuts;
    BinManager bin_manager;
};

int main(int argc, char** argv)
{
    const std::string TAG = color::GREEN_STR + "[xsDetVariation]: " + color::RESET_STR;
    const std::string ERR = color::RED_STR + color::BOLD_STR + "[ERROR]: " + color::RESET_STR;

    std::cout << "--------------------------------------------------------\n"
              << TAG << "Welcome to the Super-xsLLh Detector Variation Interface.\n"
              << TAG << "Initializing the variation machinery..." << std::endl;

    ProgressBar pbar(60, "#");
    pbar.SetRainbow();
    pbar.SetPrefix(std::string(TAG + "Reading Events "));

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

    std::fstream f;
    f.open(json_file, std::ios::in);
    std::cout << TAG << "Opening " << json_file << std::endl;
    if(!f.is_open())
    {
        std::cout << ERR << "Unable to open JSON configure file." << std::endl;
        return 1;
    }

    json j;
    f >> j;

    bool do_projection = false;
    bool do_single_variation = j["single_syst"];
    bool do_covariance = j["covariance"];
    bool do_print = j["pdf_print"];

    unsigned int syst_idx = j["syst_idx"];
    double weight_cut = j["weight_cut"];

    std::string fname_output = j["fname_output"];
    std::string variable_name = j["var_names"];

    unsigned int num_load_samples = 10;
    unsigned int num_use_samples = 0;
    unsigned int num_samples = 10;

    std::vector<FileOptions> v_files;
    for(const auto& file : j["files"])
    {
        FileOptions f;
        f.fname_input = file["fname_input"];
        f.fname_binning = file["fname_binning"];
        f.tree_name = file["tree_name"];
        f.detector = file["detector"];
        f.num_toys = file["num_toys"];
        f.num_syst = file["num_syst"];
        f.num_samples = file["num_samples"];
        f.samples = file["samples"].get<std::vector<int>>();
        f.cuts = file["cuts"].get<std::vector<int>>();
        f.bin_manager.SetBinning(f.fname_binning);
        v_files.emplace_back(f);

        num_use_samples += f.samples.size();
    }

    std::string fname_input = v_files[0].fname_input;
    std::string fname_binning = v_files[0].fname_binning;
    std::string tree_name = v_files[0].tree_name;
    std::string variable_plot;
    std::string cut_samples;
    std::vector<std::string> var_names;

    std::stringstream ss_var(variable_name);
    for(std::string s; std::getline(ss_var, s, ',');)
        var_names.emplace_back(s);

    BinManager bin_manager(fname_binning);
    bin_manager.Print();
    const int nbins = bin_manager.GetNbins();
    const int nvars = var_names.size();
    const int ntoys = 500;
    const int num_toys = 500;

    std::cout << TAG << "Output ROOT file: " << fname_output << std::endl
              << TAG << "Name of variable(s): " << variable_name << std::endl
              << TAG << "Toy Weight Cut: " << weight_cut << std::endl
              << TAG << "Calculating Covariance: " << std::boolalpha << do_covariance << std::endl;

    int var_plot = -1;
    if(do_projection)
    {
        auto it = std::find(var_names.begin(), var_names.end(), variable_plot);
        var_plot = std::distance(var_names.begin(), it);
    }

    std::cout << TAG << "Initalizing histograms." << std::endl;
    std::vector<std::vector<TH1F>> v_hists;
    std::vector<TH1F> v_avg;
    for(const auto& file : v_files)
    {
        for(const auto& sam : file.samples)
        {
            std::vector<TH1F> v_temp;
            for(unsigned int t = 0; t < num_toys; ++t)
            {
                std::stringstream ss;
                ss << file.detector << "_sample" << sam << "_toy" << t;
                if(do_projection)
                {
                    std::vector<double> v_bins = file.bin_manager.GetBinVector(var_plot);
                    v_temp.emplace_back(TH1F(ss.str().c_str(), ss.str().c_str(),
                                v_bins.size()-1, &v_bins[0]));
                    if(t == 0)
                    {
                        ss.str(""); ss << file.detector << "_sample" << sam << "_avg";
                        v_avg.emplace_back(TH1F(ss.str().c_str(), ss.str().c_str(),
                                    v_bins.size()-1, &v_bins[0]));
                    }
                }
                else
                {
                    v_temp.emplace_back(TH1F(ss.str().c_str(), ss.str().c_str(),
                                nbins, 0, nbins));
                    if(t == 0)
                    {
                        ss.str(""); ss << file.detector << "_sample" << sam << "_avg";
                        v_avg.emplace_back(TH1F(ss.str().c_str(), ss.str().c_str(),
                                    nbins, 0, nbins));
                    }
                }
            }
            v_hists.emplace_back(v_temp);
        }
    }

    std::cout << TAG << "Finished initializing histograms" << std::endl
              << TAG << "Reading events from files..." << std::endl;

    unsigned int offset = 0;
    for(const auto& file : v_files)
    {
        int accum_level[file.num_toys][file.num_samples];
        float hist_variables[nvars][file.num_toys];
        float weight_syst_total_noflux[file.num_toys];
        float weight_syst[file.num_toys][file.num_syst];

        std::cout << TAG << "Opening file: " << file.fname_input << std::endl
                  << TAG << "Reading tree: " << file.tree_name << std::endl
                  << TAG << "Num Toys: " << file.num_toys << std::endl
                  << TAG << "Num Syst: " << file.num_syst << std::endl;

        TFile* file_input = TFile::Open(file.fname_input.c_str(), "READ");
        TTree* tree_event = (TTree*)file_input -> Get(file.tree_name.c_str());

        tree_event -> SetBranchAddress("accum_level", accum_level);
        tree_event -> SetBranchAddress("weight_syst", weight_syst);
        tree_event -> SetBranchAddress("weight_syst_total_noflux", weight_syst_total_noflux);
        for(unsigned int i = 0; i < nvars; ++i)
            tree_event -> SetBranchAddress(var_names[i].c_str(), hist_variables[i]);

        unsigned int rejected_weights = 0;
        unsigned int total_weights = 0;
        const unsigned int num_events = tree_event -> GetEntries();

        std::cout << TAG << "Number of events: " << num_events << std::endl;
        for(unsigned int i = 0; i < num_events; ++i)
        {
            tree_event -> GetEntry(i);
            if(i % 2000 == 0 || i == (num_events-1))
                pbar.Print(i, num_events-1);

            for(unsigned int t = 0; t < file.num_toys; ++t)
            {
                for(unsigned int s = 0; s < file.samples.size(); ++s)
                {
                    unsigned int si = file.samples[s];
                    if(accum_level[t][si] > file.cuts[si])
                    {
                        float idx = -1;
                        if(do_projection)
                            idx = hist_variables[var_plot][t];
                        else
                        {
                            std::vector<double> vars;
                            for(unsigned int v = 0; v < nvars; ++v)
                                vars.push_back(hist_variables[v][t]);
                            idx = file.bin_manager.GetBinIndex(vars);
                        }

                        float weight = do_single_variation ? weight_syst[t][syst_idx] : weight_syst_total_noflux[t];
                        if(weight > 0.0 && weight < weight_cut)
                        {
                            v_hists[s+offset][t].Fill(idx, weight);
                            v_avg[s+offset].Fill(idx, weight / num_toys);
                        }
                        else
                            rejected_weights++;
                        total_weights++;
                        break;
                    }
                }
            }
        }

        offset += file.samples.size();
        double reject_fraction = (rejected_weights * 1.0) / total_weights;
        std::cout << TAG << "Finished processing events." << std::endl;
        std::cout << TAG << "Total weights: " << total_weights << std::endl;
        std::cout << TAG << "Rejected weights: " << rejected_weights << std::endl;
        std::cout << TAG << "Rejected fraction: " << reject_fraction << std::endl;
    }

    const unsigned int num_elements = nbins * num_use_samples;
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
            for(unsigned int s = 0; s < num_use_samples; ++s)
            {
                for(unsigned int b = 0; b < nbins; ++b)
                    i_toy.push_back(v_hists[s][t].GetBinContent(b+1));
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

    std::cout << TAG << "Saving to output file." << std::endl;
    TFile* file_output = TFile::Open(fname_output.c_str(), "RECREATE");
    file_output -> cd();

    offset = 0;
    gStyle -> SetOptStat(0);
    for(const auto& file : v_files)
    {
        for(unsigned int s = 0; s < file.samples.size(); ++s)
        {
            std::stringstream ss;
            ss << file.detector << "_sample" << file.samples[s];
            TCanvas c(ss.str().c_str(), ss.str().c_str(), 1200, 900);
            //v_hists[s][0].Draw("axis");
            v_avg[s+offset].Draw("axis");

            for(unsigned int t = 0; t < num_toys; ++t)
            {
                v_hists[s+offset][t].SetLineColor(kRed);
                //v_hists[s+offset][t].Scale(1, "width");
                v_hists[s+offset][t].Draw("hist same");
            }

            v_avg[s+offset].SetLineColor(kBlack);
            v_avg[s+offset].SetLineWidth(2);
            //v_avg[s+offset].Scale(1, "width");
            v_avg[s+offset].GetYaxis() -> SetRangeUser(0, v_avg[s+offset].GetMaximum()*1.50);
            v_avg[s+offset].Draw("hist same");
            c.Write(ss.str().c_str());

            if(do_print)
                c.Print(std::string(ss.str() + ".pdf").c_str());
        }
        offset += file.samples.size();
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
