#ifndef OPTPARSER_HH
#define OPTPARSER_HH

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "json.hpp"
using json = nlohmann::json;

struct SampleOpt
{
    int cut_branch;
    int flux_offset;
    int det_offset;
    bool use_sample;
    std::string name;
    std::string detector;
    std::string binning;
};

struct CovOpt
{
    std::string fname;
    std::string matrix;
    std::string binning;
};

struct DetOpt
{
    std::string name;
    std::string binning;
    std::string flux_file;
    std::vector<std::string> flux_hists;
    double ntargets_val;
    double ntargets_err;
};

class OptParser
{
    public:

        OptParser();
        bool ParseJSON(std::string json_file);
        bool ParseCLI(int argc, char** argv);

        std::string fname_data;
        std::string fname_mc;
        std::string fname_output;
        std::string fname_xsec;
        std::string input_dir;
        std::string xsLLh_env;

        int data_POT;
        int mc_POT;
        int rng_seed;
        int num_threads;
        int num_throws;

        std::vector<int> sample_signal;
        std::vector<std::string> sample_topology;

        CovOpt flux_cov;
        CovOpt det_cov;
        std::vector<SampleOpt> samples;
        std::vector<DetOpt> detectors;
};
#endif
