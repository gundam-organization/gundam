#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <unistd.h>

#include "ColorOutput.hh"
#include "XsecCalc.hh"

int main(int argc, char** argv)
{
    const std::string TAG = color::CYAN_STR + "[CalcXsec]: " + color::RESET_STR;
    const std::string ERR = color::RED_STR + color::BOLD_STR
                            + "[ERROR]: " + color::RESET_STR;

    std::cout << "------------------------------------------------\n"
              << TAG << "Welcome to the Super-xsLLhFitter.\n"
              << TAG << "Initializing the fit machinery..." << std::endl;


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

    std::cout << TAG << "Constructing the extractor..." << std::endl;
    XsecCalc xsec(json_file);
    xsec.ReweightNominal();

    TH1D sel_signal = xsec.GetSelSignal(0);
    TH1D tru_signal = xsec.GetTruSignal(0);
    TFile* foutput = TFile::Open(xsec.GetOutputFileName().c_str(), "RECREATE");
    foutput->cd();
    sel_signal.Write("sel_signal");
    tru_signal.Write("tru_signal");
    foutput->Close();

    std::cout << TAG << "Finished." << std::endl;
    std::cout << TAG << "\u3042\u308a\u304c\u3068\u3046\u3054\u3056\u3044\u307e\u3057\u305f\uff01" << std::endl;
    return 0;
}
