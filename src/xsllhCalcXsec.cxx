#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>

#include "ColorOutput.hh"
#include "XsecCalc.hh"

int main(int argc, char** argv)
{
    const std::string TAG = color::CYAN_STR + "[CalcXsec]: " + color::RESET_STR;
    const std::string ERR = color::RED_STR + color::BOLD_STR + "[ERROR]: " + color::RESET_STR;

    std::cout << "------------------------------------------------\n"
              << TAG << "Welcome to the Super-xsLLhFitter.\n"
              << TAG << "Initializing the fit machinery..." << std::endl;

    const std::string xslf_env = std::getenv("XSLLHFITTER");
    if(xslf_env.empty())
    {
        std::cerr << ERR << "Environment variable \"XSLLHFITTER\" not set." << std::endl
                  << ERR << "Cannot determine source tree location." << std::endl;
        return 1;
    }

    std::string json_file;
    std::string input_file;
    std::string output_file;
    unsigned int num_toys{0};

    char option;
    while((option = getopt(argc, argv, "j:i:o:n:h")) != -1)
    {
        switch(option)
        {
            case 'j':
                json_file = optarg;
                break;
            case 'i':
                input_file = optarg;
                break;
            case 'o':
                output_file = optarg;
                break;
            case 'n':
                num_toys = std::stoi(optarg);
                break;
            case 'h':
                std::cout << "USAGE: " << argv[0] << "\nOPTIONS:\n"
                          << "-j : JSON input\n"
                          << "-i : Input file (overrides JSON config)\n"
                          << "-o : Output file (overrides JSON config)\n"
                          << "-n : Number of toys (overrides JSON config)\n";
            default:
                return 0;
        }
    }

    if(json_file.empty())
    {
        std::cout << ERR << "Missing required argument: -j" << std::endl;
        exit(1);
    }

    std::cout << TAG << "Constructing the extractor..." << std::endl;
    XsecCalc xsec(json_file);
    if(!input_file.empty())
        xsec.ReadFitFile(input_file);
    //xsec.ReweightNominal();
    xsec.ReweightBestFit();
    xsec.GenerateToys();
    if(!output_file.empty())
        xsec.SaveOutput(output_file);
    else
        xsec.SaveOutput();

    std::cout << TAG << "Finished." << std::endl;
    std::cout << TAG << "\u3042\u308a\u304c\u3068\u3046\u3054\u3056\u3044\u307e\u3057\u305f\uff01"
              << std::endl;
    return 0;
}
