#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>

#include <TCanvas.h>
#include <TFile.h>
#include <TGraph.h>
#include <TH1.h>
#include <TSpline.h>
#include <TTree.h>

#include "BinManager.h"

const std::string RESET("\033[0m");
const std::string RED("\033[31;1m");
const std::string GREEN("\033[92m");
const std::string COMMENT_CHAR("#");

const std::string TAG = GREEN + "[xsSplines]: " + RESET;
const std::string ERR = RED + "[ERROR]: " + RESET;

int main(int argc, char** argv)
{
    std::cout << "--------------------------------------------------------\n"
              << TAG << "Welcome to the Super-xsLLhSpline Generator.\n"
              << TAG << "Initializing the spline machinery..." << std::endl;

    std::string dial_name;
    std::string fname_input;
    std::string fname_output;
    std::string fname_binning;
    std::string fname_dials;

    unsigned int nbins = 0;
    unsigned int dsteps = 7;
    unsigned int ntopo = 8;
    unsigned int nreac = 10;
    unsigned int nsamp = 10;

    bool limits = false;
    bool do_normalize_dials = false;
    double nominal{0};
    double error_neg{0};
    double error_pos{0};

    char option;
    while((option = getopt(argc, argv, "i:o:b:s:r:n:d:LNh")) != -1)
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
            case 'r':
                dial_name = optarg;
                break;
            case 'n':
                dsteps = std::stoi(optarg);
                break;
            case 'd':
                fname_dials = optarg;
                break;
            case 'L':
                limits = true;
                break;
            case 'N':
                do_normalize_dials = true;
                break;
            case 'h':
                std::cout << "USAGE: " << argv[0] << "\nOPTIONS\n"
                          << "-i : Input weights file\n"
                          << "-o : Output ROOT file\n"
                          << "-b : Binning file (.txt)\n"
                          << "-r : Systematic parameter to reweight\n"
                          << "-n : Number of dial steps\n"
                          << "-d : Dial value file (.txt)\n"
                          << "-L : Read errors as upper and lower limits\n"
                          << "-h : Display this help message\n";
            default:
                return 0;
        }
    }

    if(fname_input.empty() || fname_output.empty() || fname_binning.empty()
       || dial_name.empty() || fname_dials.empty())
    {
        std::cerr << ERR << "Missing one or more required arguments!" << std::endl;
        return 1;
    }

    std::cout << TAG << "Input weights file: " << fname_input << std::endl
              << TAG << "Output spline file: " << fname_output << std::endl
              << TAG << "Input binning file: " << fname_binning << std::endl
              << TAG << "Input dial value file: " << fname_dials << std::endl
              << TAG << "Generating splines for " << dial_name << std::endl
              << TAG << "Number of dial steps: " << dsteps << std::endl;

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

    TH1F* h_dial_weights[ntopo][nreac][dsteps];
    TH1F* hist_nominal[ntopo][nreac];

    for(unsigned int t = 0; t < ntopo; t++)
    {
        for(unsigned int r = 0; r < nreac; r++)
        {
            std::stringstream ss;
            ss << "hist_nominal_top" << t << "_reac" << r;
            #ifdef DEBUG_MSG
            std::cout << ss.str() << std::endl;
            #endif
            hist_nominal[t][r] = new TH1F(ss.str().c_str(), ss.str().c_str(), nbins, 0, nbins);

            for(unsigned int w = 0; w < dsteps; ++w)
            {
                ss.str(std::string());
                ss << "hist_weight_dial" << w << "_top" << t << "_reac" << r;
                #ifdef DEBUS_MSG
                std::cout << ss.str() << std::endl;
                #endif
                h_dial_weights[t][r][w] = new TH1F(ss.str().c_str(), ss.str().c_str(), nbins, 0, nbins);
            }
        }
    }

    std::cout << TAG << "Histograms initialized." << std::endl;

    int bin_index{0};
    int topology{0}, reaction{0}, sample{0};
    float enu{0}, emu{0};
    float pmu{0}, cosmu{0}, q2{0};
    float weight_nom{0};
    float weight_syst[dsteps];
    const double mu_mass{105.6583745};

    TFile* file_input = TFile::Open(fname_input.c_str(), "READ");
    TTree* tree_event = (TTree*)file_input -> Get("selectedEvents");

    tree_event -> SetBranchAddress("Enutrue", &enu);
    tree_event -> SetBranchAddress("Pmutrue", &pmu);
    tree_event -> SetBranchAddress("CTHmutrue", &cosmu);
    tree_event -> SetBranchAddress("topology", &topology);
    tree_event -> SetBranchAddress("reaction", &reaction);
    tree_event -> SetBranchAddress("sample", &sample);
    tree_event -> SetBranchAddress("weight", &weight_nom);
    tree_event -> SetBranchAddress("weight_syst", weight_syst);

    std::cout << TAG << "Tree opened and branches set; reading events." << std::endl;

    unsigned int num_events = tree_event -> GetEntries();
    for(unsigned int i = 0; i < num_events; ++i)
    {
        tree_event -> GetEntry(i);

        emu = std::sqrt(pmu*pmu + mu_mass*mu_mass);
        q2 = 2.0 * enu * (emu - pmu * cosmu) - mu_mass * mu_mass;
        q2 = q2 / 1.0E6;

        //bin_index = bin_manager.GetBinIndex(std::vector<double>{q2});
        bin_index = bin_manager.GetBinIndex(std::vector<double>{cosmu, pmu});

        hist_nominal[topology][reaction] -> Fill(bin_index + 0.5, weight_nom);
        for(unsigned int w = 0; w < dsteps; ++w)
        {
            h_dial_weights[topology][reaction][w] -> Fill(bin_index + 0.5, weight_nom * weight_syst[w]);
        }

        /*
        std::cout << "Q2 (MeV): " << q2 << std::endl
                  << "Bin IDX : " << bin_index << std::endl
                  << "Nominal : " << weight_nom << std::endl
                  << "Weighted: " << weight_syst[0] << std::endl;
        */
    }
    file_input -> Close();

    std::cout << TAG << "Histograms filled." << std::endl;

    TFile* file_output = TFile::Open(fname_output.c_str(), "RECREATE");
    file_output -> cd();
    for(unsigned int t = 0; t < ntopo; ++t)
    {
        for(unsigned int r = 0; r < nreac; ++r)
        {

            //hist_nominal[t][r] -> Write();
            for(unsigned int w = 0; w < dsteps; ++w)
            {
                h_dial_weights[t][r][w] -> Divide(h_dial_weights[t][r][w], hist_nominal[t][r]);
                for(unsigned int b = 1; b <= nbins; ++b)
                {
                    if(h_dial_weights[t][r][w] -> GetBinContent(b) == 0)
                        h_dial_weights[t][r][w] -> SetBinContent(b, 1);
                }
                //h_dial_weights[t][r][w] -> Write();
            }
        }
    }
    std::cout << TAG << "Histograms normalized." << std::endl;

    double dial_value[dsteps];
    for(unsigned int w = 0; w < dsteps; ++w)
    {
        int step = -1.0 * std::floor(dsteps / 2.0) + w;
        if(limits)
        {
            double inc = (error_pos - error_neg) / (dsteps - 1);
            dial_value[w] = nominal + step * inc;
        }
        else
        {
            if(step < 0)
                dial_value[w] = nominal + step * error_neg;
            else
                dial_value[w] = nominal + step * error_pos;
        }
        std::cout.unsetf(std::ios::fixed);
        std::cout << "Dial Step: " << step << "; Value: " << dial_value[w] << std::endl;
    }

    TGraph* spline_array[ntopo][nreac][nbins];
    for(unsigned int t = 0; t < ntopo; ++t)
    {
        for(unsigned int r = 0; r < nreac; ++r)
        {
            for(unsigned int b = 0; b < nbins; ++b)
            {
                std::stringstream ss;
                ss << "spline_top" << t << "_reac" << r << "_bin" << b;
                spline_array[t][r][b] = new TGraph(dsteps);
                spline_array[t][r][b] -> SetName(ss.str().c_str());
                spline_array[t][r][b] -> SetTitle(ss.str().c_str());
                spline_array[t][r][b] -> SetMarkerStyle(20);
                spline_array[t][r][b] -> SetMarkerColor(2);

                for(unsigned int w = 0; w < dsteps; ++w)
                {
                    const double val = h_dial_weights[t][r][w] -> GetBinContent(b+1);

                    double x_value = dial_value[w];
                    if(do_normalize_dials)
                        x_value = x_value / nominal;
                    spline_array[t][r][b] -> SetPoint(w, x_value, val);
                }

                spline_array[t][r][b] -> Write();
            }
        }
    }
    file_output -> Close();
    std::cout << TAG << "Splines generated and saved." << std::endl;

    return 0;
}
