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

#include "nlohmann/json.hpp"
using json = nlohmann::json;

#include "BinManager.hh"
#include "CalcChisq.hh"
#include "ColorOutput.hh"
#include "PlotStyle.hh"

int main(int argc, char** argv)
{
    const std::string TAG = color::CYAN_STR + "[xsPlot]: " + color::RESET_STR;
    const std::string ERR = color::RED_STR + color::BOLD_STR + "[ERROR]: " + color::RESET_STR;

    std::cout << "------------------------------------------------\n"
              << TAG << color::RainbowText("Welcome to the Super-xsLLh Plot Maker.\n")
              << TAG << color::RainbowText("Initializing the plotting machinery...") << std::endl;

    const std::string xslf_env = std::getenv("XSLLHFITTER")? std::getenv("XSLLHFITTER"): "";
    if(xslf_env.empty())
    {
        std::cerr << ERR << "Environment variable \"XSLLHFITTER\" not set." << std::endl
                  << ERR << "Cannot determine source tree location." << std::endl;
        return 1;
    }

    bool do_print_plots = true;
    std::string json_file;
    std::string default_filename;
    std::string output_filename;
    std::string plot_extension;

    char option;
    while((option = getopt(argc, argv, "j:i:o:e:Ph")) != -1)
    {
        switch(option)
        {
            case 'j':
                json_file = optarg;
                break;
            case 'i':
                default_filename = optarg;
                break;
            case 'o':
                output_filename = optarg;
                break;
            case 'e':
                plot_extension = optarg;
                break;
            case 'P':
                do_print_plots = false;
                break;
            case 'h':
                std::cout << "USAGE: " << argv[0] << "\nOPTIONS:\n"
                          << "-j : JSON input\n"
                          << "-i : Default file for input (overrides JSON config)\n"
                          << "-o : Output file for plots (overrides JSON config)\n"
                          << "-e : File extension to use when printing plots\n"
                          << "-P : Disable printing plots\n"
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

    if(default_filename.empty())
        default_filename = j.value("default_root_file", "");

    if(plot_extension.empty())
        plot_extension = j.value("plot_extension", ".pdf");

    GlobalStyle xsllh_style;
    xsllh_style.ApplyStyle();

    if(output_filename.empty())
        output_filename = j["output_filename"];

    TFile* output_file = TFile::Open(output_filename.c_str(), "RECREATE");
    for(const auto& pj : j["plots"])
    {
        bool do_make_plot = pj.value("use", true);
        if(!do_make_plot)
            continue;

        std::string name = pj["name"];
        std::cout << "------------" << std::endl;
        std::cout << TAG << "Making " << name << std::endl;

        std::string canvas_name = name + "_canvas";
        TCanvas temp_canvas(canvas_name.c_str(), canvas_name.c_str(), 1200, 900);

        std::vector<int> log_axis = pj.value("log_axis", std::vector<int>{});
        if(!log_axis.empty())
        {
            temp_canvas.SetLogx(log_axis[0]);
            temp_canvas.SetLogy(log_axis[1]);
        }

        std::string leg_title = pj.value("legend_title", "");
        std::vector<double> leg_coords = pj["legend_coordinates"].get<std::vector<double>>();
        TLegend temp_legend(leg_coords[0], leg_coords[1], leg_coords[2], leg_coords[3]);
        temp_legend.SetFillStyle(0);
        temp_legend.SetHeader(leg_title.c_str());

        TFile* temp_file = nullptr;
        std::string input_file = pj.value("root_file", "");
        if(!input_file.empty())
        {
            std::cout << TAG << "Opening " << input_file << std::endl;
            temp_file = TFile::Open(input_file.c_str(), "READ");
        }
        else
        {
            std::cout << TAG << "Opening " << default_filename << std::endl;
            temp_file = TFile::Open(default_filename.c_str(), "READ");
        }

        if(!temp_file->IsOpen())
        {
            std::cout << TAG << "Unable to open ROOT file. Skipping plot." << std::endl;
            continue;
        }
        temp_file->cd();

        for(const auto& hj : pj["hists"])
        {
            bool do_use_plot = hj.value("use", true);
            if(!do_use_plot)
                continue;

            HistStyle temp_style;
            temp_style.SetLineAtt(hj.value("line_color", kBlack), hj.value("line_style", kSolid), hj.value("line_width", 2));
            temp_style.SetFillAtt(hj.value("fill_color", kBlack), hj.value("fill_style", 0));
            temp_style.SetMarkerAtt(hj.value("marker_color", kBlack), hj.value("marker_style", kFullCircle), hj.value("marker_size", 1));
            temp_style.SetAxisTitle(pj.value("x_axis", ""), pj.value("y_axis", ""));

            std::vector<double> axis_range = pj.value("axis_range", std::vector<double>{});
            if(!axis_range.empty())
                temp_style.SetAxisRange(axis_range[0], axis_range[1], axis_range[2], axis_range[3]);

            std::string hist_name = hj["name"];
            TH1D* temp_hist = nullptr;
            temp_file->cd();
            temp_file->GetObject(hist_name.c_str(), temp_hist);

            if(temp_hist == nullptr)
                continue;

            std::cout << TAG << "Adding " << hist_name << std::endl;
            temp_style.ApplyStyle(*temp_hist);

            double x_scale = pj.value("scale_x_axis", 0.0);
            if(x_scale > 0.0)
                temp_style.ScaleXbins(*temp_hist, x_scale);

            std::string error_name = hj.value("errors", "");
            TH1D* temp_errors = nullptr;

            if(!error_name.empty())
                temp_file->GetObject(error_name.c_str(), temp_errors);

            if(temp_errors != nullptr)
            {
                for(unsigned int b = 1; b <= temp_errors->GetNbinsX(); ++b)
                    temp_hist->SetBinError(b, temp_errors->GetBinContent(b));
            }

            std::string draw_str = hj["draw"];
            temp_hist->Draw(draw_str.c_str());

            std::string legend_entry = hj["legend_entry"];
            std::string legend_draw = hj.value("legend_draw", "l");
            temp_legend.AddEntry(temp_hist, legend_entry.c_str(), legend_draw.c_str());

            output_file->cd();
            temp_hist->Write();
        }

        json empty_json;
        json chisq_json = pj.value("chisq", empty_json);
        if(!chisq_json.empty())
        {
            for(const auto& entry : chisq_json)
            {
                std::cout << TAG << "Adding chi-square comparison." << std::endl;
                std::string h1_name = entry["hist_one"];
                std::string h2_name = entry["hist_two"];
                std::string label = entry["legend_label"];
                std::string cov_name = entry.value("covariance","");

                TH1D* h1 = nullptr;
                TH1D* h2 = nullptr;
                TMatrixDSym* cov_mat = nullptr;

                temp_file->cd();
                temp_file->GetObject(h1_name.c_str(), h1);
                temp_file->GetObject(h2_name.c_str(), h2);

                double chisq = 0;
                CalcChisq calc_chisq;
                if(!cov_name.empty())
                {
                    temp_file->GetObject(cov_name.c_str(), cov_mat);

                    calc_chisq.SetCovariance(*cov_mat);
                    chisq = calc_chisq.CalcChisqCov(*h1, *h2);
                }
                else
                {
                    chisq = calc_chisq.CalcChisqStat(*h1, *h2);
                }

                //label = label + std::to_string(chisq);
                std::stringstream ss;
                ss << std::setprecision(3) << std::fixed << label << chisq;
                label = ss.str();
                temp_legend.AddEntry((TObject*)nullptr, label.c_str(), "");
            }
        }

        temp_legend.Draw();

        output_file->cd();
        temp_canvas.Write();

        if(pj.value("print", false) && do_print_plots)
        {
            std::string save_str = pj["save_as"];
            save_str = save_str + plot_extension;
            temp_canvas.Print(save_str.c_str());
        }

        temp_file->Close();
    }

    output_file->Close();

    std::cout << TAG << "Finished." << std::endl;
    std::cout << TAG << "\u3042\u308a\u304c\u3068\u3046\u3054\u3056\u3044\u307e\u3057\u305f\uff01"
              << std::endl;

    return 0;
}
