#include <chrono>
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
#include "PlotStyle.hh"

int main(int argc, char** argv)
{
    const std::string TAG = color::CYAN_STR + "[xsPlot]: " + color::RESET_STR;
    const std::string ERR = color::RED_STR + color::BOLD_STR + "[ERROR]: " + color::RESET_STR;

    std::cout << "------------------------------------------------\n"
              << TAG << color::RainbowText("Welcome to the Super-xsLLh Plot Maker.\n")
              << TAG << color::RainbowText("Initializing the plotting machinery...") << std::endl;

    const std::string xslf_env = std::getenv("XSLLHFITTER");
    if(xslf_env.empty())
    {
        std::cerr << ERR << "Environment variable \"XSLLHFITTER\" not set." << std::endl
                  << ERR << "Cannot determine source tree location." << std::endl;
        return 1;
    }

    bool read_fake_data = false;
    bool do_print_plots = false;
    std::string json_file;
    std::string xsec_filename;
    std::string fit_filename;
    std::string output_filename;
    std::string plot_extension = ".pdf";

    char option;
    while((option = getopt(argc, argv, "j:f:x:o:e:Ph")) != -1)
    {
        switch(option)
        {
            case 'j':
                json_file = optarg;
                break;
            case 'f':
                fit_filename = optarg;
                break;
            case 'x':
                xsec_filename = optarg;
                break;
            case 'o':
                output_filename = optarg;
                break;
            case 'e':
                plot_extension = optarg;
                break;
            case 'P':
                do_print_plots = true;
                break;
            case 'h':
                std::cout << "USAGE: " << argv[0] << "\nOPTIONS:\n"
                          << "-j : JSON input\n"
                          << "-f : Input fit file (overrides JSON config)\n"
                          << "-x : Input xsec file (overrides JSON config)\n"
                          << "-o : Output file (overrides JSON config)\n"
                          << "-e : File extension to use when printing plots\n"
                          << "-P : Print plots to directory\n"
                          << "-h : Print this usage guide\n";
            default:
                return 0;
        }
    }

    if(json_file.empty())
    {
        std::cout << ERR << "Missing required argument: -j" << std::endl;
        return 1;
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

    GlobalStyle xsllh_style;
    xsllh_style.ApplyStyle();

    HistStyle postfit_style;
    postfit_style.SetLineAtt(kBlack, kSolid, 2);
    postfit_style.SetAxisTitle("Bin Number","#frac{d#sigma}{dp_{#mu} dcos#theta_{#mu}} #frac{cm^{2}}{nucleon GeV/c}");

    HistStyle nominal_style;
    nominal_style.SetLineAtt(kBlue, kDashed, 2);

    HistStyle data_style;
    data_style.SetLineAtt(kRed, kSolid, 2);

    TFile* output_file = TFile::Open(output_filename.c_str(), "RECREATE");
    TFile* xsec_file = TFile::Open(xsec_filename.c_str(), "READ");
    for(const auto& s : j["xsec_plots"])
    {
        xsec_file->cd();
        std::string root_string = s["signal_name"];
        std::string hist_title = s["hist_title"];
        std::string binning_file = s["binning"];

        std::cout << TAG << "Building cross-section plot for " << root_string << std::endl;

        TH1D* hist_postfit = (TH1D*)xsec_file->Get((root_string + "_postfit").c_str());
        TH1D* hist_nominal = (TH1D*)xsec_file->Get((root_string + "_nominal").c_str());
        TH1D* hist_data = (TH1D*)xsec_file->Get((root_string + "_data").c_str());

        postfit_style.ApplyStyle(*hist_postfit);
        nominal_style.ApplyStyle(*hist_nominal);
        data_style.ApplyStyle(*hist_data);

        TCanvas c(root_string.c_str(), root_string.c_str(), 1200, 900);
        hist_postfit->Draw("e");
        hist_nominal->Draw("hist same");
        hist_data->Draw("hist same");

        TLegend l(0.65, 0.65, 0.85, 0.85);
        l.AddEntry(hist_postfit, "Post-fit", "lpe");
        l.AddEntry(hist_nominal, "Nominal MC", "l");
        l.AddEntry(hist_data, "Fake Data", "l");
        l.Draw();

        output_file->cd();
        c.Write();
        hist_postfit->Write();
        hist_nominal->Write();
        hist_data->Write();

        if(do_print_plots)
        {
            std::string save_string = root_string + plot_extension;
            c.Print(save_string.c_str());
        }

        delete hist_postfit;
        delete hist_nominal;
        delete hist_data;

        const unsigned int cos_bins = 9;
        for(unsigned int i = 0; i < cos_bins; ++i)
        {
            std::string canvas_name = root_string + "_cos_bin" + std::to_string(i);
            TCanvas temp_c(canvas_name.c_str(), canvas_name.c_str(), 1200, 900);

            std::cout << TAG << "Building kinematic plots for " << canvas_name << std::endl;

            xsec_file->cd();
            std::string hist_name = root_string + "_cos_bin" + std::to_string(i);
            TH1D* temp_postfit = (TH1D*)xsec_file->Get((hist_name + "_postfit").c_str());
            TH1D* temp_nominal = (TH1D*)xsec_file->Get((hist_name + "_nominal").c_str());
            TH1D* temp_data = (TH1D*)xsec_file->Get((hist_name + "_data").c_str());

            postfit_style.SetAxisTitle("p^{true}_{#mu} (GeV/c)", "#frac{d#sigma}{dp_{#mu} dcos#theta_{#mu}} #frac{cm^{2}}{nucleon GeV/c}");
            postfit_style.ScaleXbins(*temp_postfit, 0.001);
            nominal_style.ScaleXbins(*temp_nominal, 0.001);
            data_style.ScaleXbins(*temp_data, 0.001);

            postfit_style.ApplyStyle(*temp_postfit);
            nominal_style.ApplyStyle(*temp_nominal);
            data_style.ApplyStyle(*temp_data);

            temp_postfit->Draw("e");
            temp_nominal->Draw("hist same");
            temp_data->Draw("hist same");

            TLegend temp_l(0.65, 0.65, 0.85, 0.85);
            temp_l.AddEntry(temp_postfit, "Post-fit", "lpe");
            temp_l.AddEntry(temp_nominal, "Nominal MC", "l");
            temp_l.AddEntry(temp_data, "Fake Data", "l");
            temp_l.Draw();

            output_file->cd();
            temp_c.Write();
            temp_postfit->Write();
            temp_nominal->Write();
            temp_data->Write();

            if(do_print_plots)
            {
                std::string save_string = canvas_name + plot_extension;
                c.Print(save_string.c_str());
            }

            delete temp_postfit;
            delete temp_nominal;
            delete temp_data;
        }
    }

    xsec_file->Close();
    output_file->Close();

    std::cout << TAG << "Finished." << std::endl;
    std::cout << TAG << "\u3042\u308a\u304c\u3068\u3046\u3054\u3056\u3044\u307e\u3057\u305f\uff01"
              << std::endl;

    return 0;
}
