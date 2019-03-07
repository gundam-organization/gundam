#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
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
    std::string tree_name;
    std::string detector;
    unsigned int num_samples;
    unsigned int num_toys;
    unsigned int num_syst;
    std::vector<int> cuts;
    std::map<int, std::vector<int>> samples;
    std::vector<BinManager> bin_manager;
};

int main(int argc, char** argv)
{
    const std::string TAG = color::GREEN_STR + "[xsDetVariation]: " + color::RESET_STR;
    const std::string ERR = color::RED_STR + color::BOLD_STR + "[ERROR]: " + color::RESET_STR;

    std::cout << "--------------------------------------------------------\n"
              << TAG << color::RainbowText("Welcome to the Super-xsLLh Detector Variation Interface.\n")
              << TAG << color::RainbowText("Initializing the variation machinery...") << std::endl;

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
                std::cout << "USAGE: " << argv[0] << "\nOPTIONS:\n"
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

    bool do_projection  = j["projection"];
    bool do_single_syst = j["single_syst"];
    bool do_covariance  = j["covariance"];
    bool do_print       = j["pdf_print"];

    unsigned int syst_idx   = j["syst_idx"];
    const double weight_cut = j["weight_cut"];

    std::string fname_output  = j["fname_output"];
    std::string variable_plot = j["plot_variable"];
    std::string cov_mat_name  = j["covariance_name"];
    std::string cor_mat_name  = j["correlation_name"];

    std::vector<std::string> var_names = j["var_names"].get<std::vector<std::string>>();
    const int nvars = var_names.size();

    std::vector<BinManager> cov_bin_manager;
    std::map<std::string, std::string> temp_cov_binning = j["cov_sample_binning"];

    const unsigned int num_cov_samples = temp_cov_binning.size();
    cov_bin_manager.resize(num_cov_samples);
    for(const auto& kv : temp_cov_binning)
        cov_bin_manager.at(std::stoi(kv.first)) = std::move(BinManager(kv.second));

    unsigned int usable_toys = 0;
    std::vector<FileOptions> v_files;
    for(const auto& file : j["files"])
    {
        if(file["use"])
        {
            FileOptions f;
            f.fname_input = file["fname_input"];
            f.tree_name   = file["tree_name"];
            f.detector    = file["detector"];
            f.num_toys    = file["num_toys"];
            f.num_syst    = file["num_syst"];
            f.num_samples = file["num_samples"];
            f.cuts        = file["cuts"].get<std::vector<int>>();

            std::map<std::string, std::vector<int>> temp_json = file["samples"];
            for(const auto& kv : temp_json)
            {
                const int sam = std::stoi(kv.first);
                if(sam <= num_cov_samples)
                    f.samples.emplace(std::make_pair(sam, kv.second));
                else
                {
                    std::cout << ERR << "Invalid sample number: " << sam << std::endl;
                    return 64;
                }
            }

            v_files.emplace_back(f);

            if(f.num_toys < usable_toys || usable_toys == 0)
                usable_toys = f.num_toys;
        }
    }

    std::cout << TAG << "Output ROOT file: " << fname_output << std::endl
              << TAG << "Toy Weight Cut: " << weight_cut << std::endl
              << TAG << "Calculating Covariance: " << std::boolalpha << do_covariance << std::endl;

    std::cout << TAG << "Covariance Variables: ";
    for(const auto& var : var_names)
        std::cout << var << " ";
    std::cout << std::endl;

    int var_plot = -1;
    if(do_projection)
    {
        auto it  = std::find(var_names.begin(), var_names.end(), variable_plot);
        var_plot = std::distance(var_names.begin(), it);
    }

    std::cout << TAG << "Initalizing histograms." << std::endl;
    std::cout << TAG << "Using " << usable_toys << " toys." << std::endl;

    std::vector<std::vector<TH1F>> v_hists;
    std::vector<TH1F> v_avg;
    for(int i = 0; i < cov_bin_manager.size(); ++i)
    {
        BinManager bm   = cov_bin_manager.at(i);
        const int nbins = bm.GetNbins();
        std::vector<TH1F> v_temp;

        for(unsigned int t = 0; t < usable_toys; ++t)
        {
            std::stringstream ss;
            ss << "cov_sample" << i << "_toy" << t;
            if(do_projection)
            {
                std::vector<double> v_bins = bm.GetBinVector(var_plot);
                v_temp.emplace_back(
                    TH1F(ss.str().c_str(), ss.str().c_str(), v_bins.size() - 1, &v_bins[0]));
                if(t == 0)
                {
                    ss.str("");
                    ss << "cov_sample" << i << "_avg";
                    v_avg.emplace_back(
                        TH1F(ss.str().c_str(), ss.str().c_str(), v_bins.size() - 1, &v_bins[0]));
                }
            }
            else
            {
                v_temp.emplace_back(TH1F(ss.str().c_str(), ss.str().c_str(), nbins, 0, nbins));
                if(t == 0)
                {
                    ss.str("");
                    ss << "cov_sample" << i << "_avg";
                    v_avg.emplace_back(TH1F(ss.str().c_str(), ss.str().c_str(), nbins, 0, nbins));
                }
            }
        }
        v_hists.emplace_back(v_temp);
    }

    std::cout << TAG << "Finished initializing histograms" << std::endl
              << TAG << "Reading events from files..." << std::endl;

    for(const auto& file : v_files)
    {
        int NTOYS = 0;
        int accum_level[file.num_toys][file.num_samples];
        float hist_variables[nvars][file.num_toys];
        float weight_syst_total_noflux[file.num_toys];
        float weight_syst[file.num_toys][file.num_syst];

        std::cout << TAG << "Opening file: " << file.fname_input << std::endl
                  << TAG << "Reading tree: " << file.tree_name << std::endl
                  << TAG << "Num Toys: " << file.num_toys << std::endl
                  << TAG << "Num Syst: " << file.num_syst << std::endl;

        std::cout << TAG << "Branch to Sample mapping:" << std::endl;
        for(const auto& kv : file.samples)
        {
            std::cout << TAG << "Sample " << kv.first << ": ";
            for(const auto& b : kv.second)
                std::cout << b << " ";
            std::cout << std::endl;
        }

        TFile* file_input = TFile::Open(file.fname_input.c_str(), "READ");
        TTree* tree_event = (TTree*)file_input->Get(file.tree_name.c_str());

        tree_event->SetBranchAddress("NTOYS", &NTOYS);
        tree_event->SetBranchAddress("accum_level", accum_level);
        tree_event->SetBranchAddress("weight_syst", weight_syst);
        tree_event->SetBranchAddress("weight_syst_total_noflux", weight_syst_total_noflux);
        for(unsigned int i = 0; i < nvars; ++i)
            tree_event->SetBranchAddress(var_names[i].c_str(), hist_variables[i]);

        unsigned int rejected_weights = 0;
        unsigned int total_weights    = 0;
        const unsigned int num_events = tree_event->GetEntries();

        std::cout << TAG << "Number of events: " << num_events << std::endl;
        for(unsigned int i = 0; i < num_events; ++i)
        {
            tree_event->GetEntry(i);
            if(NTOYS != file.num_toys)
                std::cout << ERR << "Incorrect number of toys specified!" << std::endl;

            if(i % 2000 == 0 || i == (num_events - 1))
                pbar.Print(i, num_events - 1);

            for(unsigned int t = 0; t < usable_toys; ++t)
            {
                for(const auto& kv : file.samples)
                {
                    unsigned int s = kv.first;
                    for(const auto& branch : kv.second)
                    {
                        if(accum_level[t][branch] > file.cuts[branch])
                        {
                            int idx = -1;
                            if(do_projection)
                                idx = hist_variables[var_plot][t];
                            else
                            {
                                std::vector<double> vars;
                                for(unsigned int v = 0; v < nvars; ++v)
                                    vars.push_back(hist_variables[v][t]);
                                idx = cov_bin_manager[s].GetBinIndex(vars);
                            }

                            float weight = do_single_syst ? weight_syst[t][syst_idx]
                                                          : weight_syst_total_noflux[t];
                            if(weight > 0.0 && weight < weight_cut)
                            {
                                v_hists[s][t].Fill(idx, weight);
                                v_avg[s].Fill(idx, weight / file.num_toys);
                            }
                            else
                                rejected_weights++;
                            total_weights++;
                            break;
                        }
                    }
                }
            }
        }

        double reject_fraction = (rejected_weights * 1.0) / total_weights;
        std::cout << TAG << "Finished processing events." << std::endl;
        std::cout << TAG << "Total weights: " << total_weights << std::endl;
        std::cout << TAG << "Rejected weights: " << rejected_weights << std::endl;
        std::cout << TAG << "Rejected fraction: " << reject_fraction << std::endl;

        file_input->Close();
    }

    unsigned int num_elements = 0;
    TMatrixTSym<double> cov_mat(num_elements);
    TMatrixTSym<double> cor_mat(num_elements);

    if(do_covariance)
    {
        std::cout << TAG << "Calculating covariance matrix." << std::endl;
        std::vector<std::vector<float>> v_toys;

        for(unsigned int t = 0; t < usable_toys; ++t)
        {
            std::vector<float> i_toy;
            for(int s = 0; s < cov_bin_manager.size(); ++s)
            {
                const unsigned int nbins = cov_bin_manager[s].GetNbins();
                for(unsigned int b = 0; b < nbins; ++b)
                    i_toy.emplace_back(v_hists[s][t].GetBinContent(b + 1));
            }
            v_toys.emplace_back(i_toy);
        }

        std::cout << TAG << "Using " << usable_toys << " toys." << std::endl;
        num_elements = v_toys.at(0).size();
        std::vector<float> v_mean(num_elements, 0.0);
        cov_mat.ResizeTo(num_elements, num_elements);
        cor_mat.ResizeTo(num_elements, num_elements);
        cov_mat.Zero();
        cor_mat.Zero();

        for(unsigned int t = 0; t < usable_toys; ++t)
        {
            for(unsigned int i = 0; i < num_elements; ++i)
                v_mean[i] += v_toys[t][i] / (1.0 * usable_toys);
        }

        for(unsigned int t = 0; t < usable_toys; ++t)
        {
            for(unsigned int i = 0; i < num_elements; ++i)
            {
                for(unsigned int j = 0; j < num_elements; ++j)
                {
                    if(v_mean[i] != 0 && v_mean[j] != 0)
                    {
                        cov_mat(i, j) += (1.0 - v_toys[t][i] / v_mean[i])
                                         * (1.0 - v_toys[t][j] / v_mean[j]) / (1.0 * usable_toys);
                    }
                }
            }
        }

        for(unsigned int i = 0; i < num_elements; ++i)
        {
            if(cov_mat(i, i) <= 0.0)
                cov_mat(i, i) = 1.0;
        }

        for(unsigned int i = 0; i < num_elements; ++i)
        {
            for(unsigned int j = 0; j < num_elements; ++j)
            {
                double bin_i  = cov_mat(i, i);
                double bin_j  = cov_mat(j, j);
                cor_mat(i, j) = cov_mat(i, j) / std::sqrt(bin_i * bin_j);
                if(std::isnan(cor_mat(i, j)))
                    cor_mat(i, j) = 0;
            }
        }
    }

    std::cout << TAG << "Saving to output file." << std::endl;
    TFile* file_output = TFile::Open(fname_output.c_str(), "RECREATE");
    file_output->cd();

    gStyle->SetOptStat(0);
    for(int s = 0; s < cov_bin_manager.size(); ++s)
    {
        std::stringstream ss;
        ss << "cov_sample" << s;
        TCanvas c(ss.str().c_str(), ss.str().c_str(), 1200, 900);
        v_avg[s].Draw("axis");

        for(unsigned int t = 0; t < usable_toys; ++t)
        {
            v_hists[s][t].SetLineColor(kRed);
            if(do_projection)
                v_hists[s][t].Scale(1, "width");
            v_hists[s][t].Draw("hist same");
        }

        v_avg[s].SetLineColor(kBlack);
        v_avg[s].SetLineWidth(2);
        if(do_projection)
            v_avg[s].Scale(1, "width");
        v_avg[s].GetYaxis()->SetRangeUser(0, v_avg[s].GetMaximum() * 1.50);
        v_avg[s].Draw("hist same");
        c.Write(ss.str().c_str());

        if(do_print)
            c.Print(std::string(ss.str() + ".pdf").c_str());
    }

    if(do_covariance)
    {
        cov_mat.Write(cov_mat_name.c_str());
        cor_mat.Write(cor_mat_name.c_str());
    }

    file_output->Close();

    std::cout << TAG << "Finished." << std::endl;
    std::cout << TAG << "\u3042\u308a\u304c\u3068\u3046\u3054\u3056\u3044\u307e\u3057\u305f\uff01"
              << std::endl;
    return 0;
}
